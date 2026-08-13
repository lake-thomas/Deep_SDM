#!/usr/bin/env python3
"""Replay exact NY/PA ground locations across completed August eastern candidates."""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.windows import Window

from august2026_batch_artifact_probe import (
    select_completed_candidates,
    write_csv_atomic,
)
from august2026_compare_artifact_probe import bootstrap_mean_ci
from august2026_fixed_env_artifact_probe import (
    cosine_distance,
    load_candidate,
    sha256,
    write_json_atomic,
)


EASTERN_HOSTS = {
    "ailanthus_altissima",
    "juglans_nigra",
    "prunus_serotina",
    "vitis_labrusca",
}
NY_2021 = Path(
    "/mnt/rsstu_dsfas_gsv_naip/Host_NAIP_SDM/NAIP_Archive/states/NY/2021/raw/"
    "m_4107701_nw_18_060_20211019.tif"
)
PA_2022 = Path(
    "/mnt/rsstu_dsfas_gsv_naip/Host_NAIP_SDM/NAIP_Archive/states/PA/2022/raw/"
    "m_4107701_nw_18_060_20221110.tif"
)
BASELINE_METRICS = Path(
    "/home/ncsucnr/projects/research_agent/agent_outputs/"
    "host_sdm_inference_artifacts_aug122026/results_exact_acquisition_overlap/"
    "exact_overlap_taxon_chip_metrics.csv"
)
BASELINE_METRICS_SHA256 = (
    "f014063bc36fbf9523168e56cd12b70aea1f6b7d3cd704cf43574c5d1c525438"
)


def read_centered_chips(
    path: Path,
    centers: list[tuple[float, float]],
    chip_size: int = 256,
) -> np.ndarray:
    chips = []
    with rasterio.open(path) as source:
        for chip_id, (x, y) in enumerate(centers):
            row, column = source.index(x, y)
            window = Window(
                column - chip_size // 2,
                row - chip_size // 2,
                chip_size,
                chip_size,
            )
            image = source.read([1, 2, 3, 4], window=window).astype(np.float32) / 255.0
            if image.shape != (4, chip_size, chip_size):
                raise RuntimeError(f"Incomplete exact-overlap chip {chip_id}: {path}")
            valid = np.isfinite(image).all(axis=0) & np.any(image != 0, axis=0)
            if valid.mean() < 0.95:
                raise RuntimeError(
                    f"Exact-overlap chip {chip_id} has valid fraction {valid.mean():.3f}"
                )
            chips.append(np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0))
    return np.stack(chips)


def predict_chips(
    model,
    chips: np.ndarray,
    mean_environment: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    features, probabilities = [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(chips), batch_size):
            image = torch.from_numpy(chips[start : start + batch_size]).to(device)
            environment = (
                torch.from_numpy(mean_environment)
                .to(device)
                .unsqueeze(0)
                .expand(len(image), -1)
            )
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                image_features = model.image_encoder(image)
                environment_features = model.env_encoder(environment)
                logits = model.classifier(
                    torch.cat([image_features, environment_features], dim=1)
                )
            features.append(image_features.float().cpu().numpy())
            probabilities.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(features), np.concatenate(probabilities).reshape(-1)


def compare_exact_overlap(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    if baseline["chip_id"].duplicated().any() or candidate["chip_id"].duplicated().any():
        raise ValueError("Exact-overlap chip IDs must be unique")
    if set(baseline["chip_id"]) != set(candidate["chip_id"]):
        raise ValueError("Candidate must replay the exact frozen overlap chip IDs")
    merged = baseline.merge(
        candidate,
        on="chip_id",
        how="inner",
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    for coordinate in ("x", "y"):
        difference = np.abs(
            merged[f"{coordinate}_baseline"] - merged[f"{coordinate}_candidate"]
        )
        if float(difference.max()) > 1e-6:
            raise ValueError(f"Candidate exact-overlap {coordinate} coordinates changed")
    baseline_signed = merged["pa_minus_ny_probability_baseline"].to_numpy(float)
    candidate_signed = merged["pa_minus_ny_probability_candidate"].to_numpy(float)
    baseline_absolute = np.abs(baseline_signed)
    candidate_absolute = np.abs(candidate_signed)
    absolute_reduction = baseline_absolute - candidate_absolute
    merged["absolute_probability_shift_reduction"] = absolute_reduction
    ci = bootstrap_mean_ci(absolute_reduction, seed=20260813)
    summary = {
        "pair_count": len(merged),
        "baseline_pa_minus_ny_probability_mean": float(baseline_signed.mean()),
        "candidate_pa_minus_ny_probability_mean": float(candidate_signed.mean()),
        "baseline_absolute_probability_shift_mean": float(baseline_absolute.mean()),
        "candidate_absolute_probability_shift_mean": float(candidate_absolute.mean()),
        "paired_absolute_probability_shift_reduction_mean": float(
            absolute_reduction.mean()
        ),
        "paired_absolute_probability_shift_reduction_bootstrap_95ci": ci,
        "absolute_shift_reduction_supported": ci[0] > 0,
    }
    return merged, summary


def reusable_output(output_dir: Path, candidate: dict, baseline_hash: str) -> bool:
    required = [
        output_dir / "COMPLETED",
        output_dir / "summary.json",
        output_dir / "exact_overlap_candidate_metrics.csv",
        output_dir / "paired_june_comparison.csv",
    ]
    if not all(path.is_file() for path in required):
        return False
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    return bool(
        summary.get("status") == "complete"
        and summary.get("checkpoint_sha256") == sha256(candidate["checkpoint_path"])
        and summary.get("config_sha256") == sha256(candidate["config_path_object"])
        and summary.get("dataset_sha256") == sha256(candidate["dataset_path"])
        and summary.get("baseline_metrics_sha256") == baseline_hash
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--host", choices=sorted(EASTERN_HOSTS), required=True)
    parser.add_argument("--experiment", action="append", default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--resume-completed", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


def main() -> None:
    args = parse_args()
    required = [args.registry, NY_2021, PA_2022, BASELINE_METRICS]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing exact-overlap inputs: {missing}")
    baseline_hash = sha256(BASELINE_METRICS)
    if baseline_hash != BASELINE_METRICS_SHA256:
        raise ValueError(
            "Frozen exact-overlap baseline hash changed: "
            f"expected {BASELINE_METRICS_SHA256}, found {baseline_hash}"
        )
    candidates = select_completed_candidates(
        args.registry, args.host, args.experiment
    )
    if args.output_root.exists() and any(args.output_root.iterdir()):
        if not args.resume_completed:
            raise FileExistsError(f"Refusing nonempty output root: {args.output_root}")
    else:
        args.output_root.mkdir(parents=True, exist_ok=True)

    pending, completed = [], []
    for candidate in candidates:
        output_dir = args.output_root / candidate["experiment"]
        if output_dir.exists():
            if args.resume_completed and reusable_output(
                output_dir, candidate, baseline_hash
            ):
                completed.append(candidate["experiment"])
                continue
            raise FileExistsError(f"Refusing partial or changed output: {output_dir}")
        pending.append(candidate)

    baseline_all = pd.read_csv(BASELINE_METRICS, low_memory=False)
    baseline = baseline_all[baseline_all["taxon"] == args.host].copy()
    if len(baseline) != 25:
        raise ValueError(f"Frozen {args.host} exact-overlap baseline is not 25 chips")
    baseline = baseline[
        [
            "chip_id",
            "x",
            "y",
            "ny_2021_probability",
            "pa_2022_probability",
            "pa_minus_ny_probability",
            "absolute_probability_change",
            "embedding_cosine_distance",
        ]
    ]
    centers = [
        (float(row.x), float(row.y))
        for row in baseline.sort_values("chip_id").itertuples(index=False)
    ]
    images = {
        "ny_2021": read_centered_chips(NY_2021, centers),
        "pa_2022": read_centered_chips(PA_2022, centers),
    }
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    code_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parent,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = {
        "status": "running_candidates" if pending else "complete",
        "host": args.host,
        "registry": str(args.registry.resolve()),
        "registry_sha256": sha256(args.registry),
        "baseline_metrics": str(BASELINE_METRICS.resolve()),
        "baseline_metrics_sha256": baseline_hash,
        "candidate_count": len(candidates),
        "completed_experiments": completed,
        "pending_experiments": [candidate["experiment"] for candidate in pending],
        "code_commit": code_commit,
    }
    write_json_atomic(args.output_root / "batch_status.json", status)
    completion_marker = args.output_root / "COMPLETED"
    if pending and completion_marker.is_file():
        completion_marker.unlink()

    for candidate in pending:
        status["active_experiment"] = candidate["experiment"]
        write_json_atomic(args.output_root / "batch_status.json", status)
        model, env_vars, mean_environment = load_candidate(
            candidate["checkpoint_path"],
            candidate["config_path_object"],
            candidate["dataset_path"],
            device,
        )
        features, probabilities = {}, {}
        for acquisition, chips in images.items():
            features[acquisition], probabilities[acquisition] = predict_chips(
                model, chips, mean_environment, device, args.batch_size
            )
        rows = []
        ordered_baseline = baseline.sort_values("chip_id")
        for index, row in enumerate(ordered_baseline.itertuples(index=False)):
            signed = float(
                probabilities["pa_2022"][index]
                - probabilities["ny_2021"][index]
            )
            rows.append(
                {
                    "chip_id": row.chip_id,
                    "x": row.x,
                    "y": row.y,
                    "ny_2021_probability": float(probabilities["ny_2021"][index]),
                    "pa_2022_probability": float(probabilities["pa_2022"][index]),
                    "pa_minus_ny_probability": signed,
                    "absolute_probability_change": abs(signed),
                    "embedding_cosine_distance": cosine_distance(
                        features["ny_2021"][index], features["pa_2022"][index]
                    ),
                }
            )
        candidate_metrics = pd.DataFrame(rows)
        paired, comparison = compare_exact_overlap(baseline, candidate_metrics)
        output_dir = args.output_root / candidate["experiment"]
        output_dir.mkdir(parents=True, exist_ok=False)
        write_csv_atomic(
            output_dir / "exact_overlap_candidate_metrics.csv", candidate_metrics
        )
        write_csv_atomic(output_dir / "paired_june_comparison.csv", paired)
        summary = {
            "status": "complete",
            "host": args.host,
            "candidate_experiment": candidate["experiment"],
            "device": str(device),
            **comparison,
            "candidate_mean_embedding_cosine_distance": float(
                candidate_metrics["embedding_cosine_distance"].mean()
            ),
            "checkpoint": str(candidate["checkpoint_path"].resolve()),
            "checkpoint_sha256": sha256(candidate["checkpoint_path"]),
            "config": str(candidate["config_path_object"].resolve()),
            "config_sha256": sha256(candidate["config_path_object"]),
            "dataset": str(candidate["dataset_path"].resolve()),
            "dataset_sha256": sha256(candidate["dataset_path"]),
            "baseline_metrics_sha256": baseline_hash,
            "environment_features": env_vars,
            "fixed_environment_source": "candidate_training_split_feature_means",
            "code_commit": code_commit,
            "guardrail": (
                "Same-ground observational replay isolates acquisition appearance more "
                "strongly than adjacent tiles, but real temporal change and "
                "georegistration differences remain possible. Positive paired absolute "
                "shift reduction favors the August candidate and is not a standalone "
                "production approval."
            ),
        }
        write_json_atomic(output_dir / "summary.json", summary)
        (output_dir / "COMPLETED").write_text("completed\n", encoding="utf-8")
        completed.append(candidate["experiment"])
        status["completed_experiments"] = completed
        status["pending_experiments"] = [
            item["experiment"]
            for item in pending
            if item["experiment"] not in completed
        ]
        del model, features, probabilities
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        write_json_atomic(args.output_root / "batch_status.json", status)

    status.pop("active_experiment", None)
    status["status"] = "complete"
    write_json_atomic(args.output_root / "batch_status.json", status)
    completion_marker.write_text("completed\n", encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
