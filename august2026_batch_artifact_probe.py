#!/usr/bin/env python3
"""Run one frozen artifact replay efficiently across completed August candidates."""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import inference_multimodal_production as inference

from august2026_compare_artifact_probe import (
    BASELINE_METRICS,
    BASELINE_SHA256,
    exact_pair_merge,
    load_baseline,
    paired_comparison,
)
from august2026_fixed_env_artifact_probe import (
    DEFAULT_PAIRS,
    DESCRIPTOR_COLUMNS,
    bootstrap_ci,
    embed_and_predict,
    load_candidate,
    pair_metrics,
    read_chips,
    sha256,
    tile_path,
    tile_state_map,
    write_json_atomic,
)
from august2026_summarize_experiments import read_json, summarize_row


HOST_DIAGNOSTIC = {
    "ailanthus_altissima": "ny",
    "juglans_nigra": "ny",
    "prunus_serotina": "ny",
    "vitis_labrusca": "ny",
    "notholithocarpus_densiflorus": "tanoak",
}


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    partial = path.with_name(f"{path.name}.partial")
    frame.to_csv(partial, index=False)
    os.replace(partial, path)


def add_tanoak_coastal_comparison(
    comparison_frame: pd.DataFrame,
    comparison: dict,
) -> dict:
    required = {"mid_lon_x", "mid_lon_y"}
    if not required.issubset(comparison_frame.columns):
        raise ValueError("Tanoak comparison lacks paired longitude fields")
    if not np.allclose(
        comparison_frame["mid_lon_x"],
        comparison_frame["mid_lon_y"],
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError("Tanoak baseline/candidate pair longitudes changed")
    coastal_frame = comparison_frame[
        comparison_frame["mid_lon_x"] <= -123.0
    ].copy()
    if coastal_frame.empty:
        raise ValueError("Tanoak coastal diagnostic contains no pairs")
    output = dict(comparison)
    output["coastal_west_of_123"] = paired_comparison(
        coastal_frame, seed=20260814
    )
    return output


def current_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parent,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def select_completed_candidates(
    registry_path: Path,
    host: str,
    experiments: list[str] | None,
) -> list[dict]:
    design_metadata_path = registry_path.parent / "design_metadata.json"
    if not design_metadata_path.is_file():
        raise FileNotFoundError(design_metadata_path)
    design_commit = read_json(design_metadata_path).get("code_commit")
    registry = pd.read_csv(registry_path, sep="\t")
    selected = registry[registry["host"] == host].copy()
    if experiments:
        selected = selected[selected["experiment"].isin(experiments)].copy()
        missing = set(experiments) - set(selected["experiment"])
        if missing:
            raise ValueError(f"Experiments are not present for {host}: {sorted(missing)}")
    if selected.empty:
        raise ValueError(f"Registry contains no selected candidates for {host}")

    candidates = []
    failures = []
    for row in selected.to_dict("records"):
        evidence = summarize_row(row, design_commit)
        if not evidence["evidence_gate_passed"]:
            failures.append(
                {
                    "experiment": row["experiment"],
                    "failures": evidence["evidence_gate_failures"],
                }
            )
            continue
        config_path = Path(row["config_path"])
        config = read_json(config_path)
        spec = config.get("august_experiment_spec") or {}
        if spec.get("host") != host:
            raise ValueError(
                f"Config host mismatch for {row['experiment']}: {spec.get('host')}"
            )
        candidates.append(
            {
                **row,
                "config": config,
                "checkpoint_path": Path(row["run_dir"]) / "checkpoints" / "best.pt",
                "config_path_object": config_path,
                "dataset_path": Path(config["csv_path"]),
                "configured_calibrator_path": (
                    Path(row["run_dir"]) / "platt_calibrator.json"
                    if str(config.get("postfit_calibration", "none")).lower()
                    == "platt"
                    else None
                ),
            }
        )
        calibrator_path = candidates[-1]["configured_calibrator_path"]
        if calibrator_path is not None and not calibrator_path.is_file():
            raise FileNotFoundError(
                f"Configured Platt calibrator is missing: {calibrator_path}"
            )
    if failures:
        raise RuntimeError(f"Candidate training evidence gate failed: {failures}")
    return candidates


def prepare_shared_inputs(diagnostic: str, pairs_path: Path):
    pairs = pd.read_csv(pairs_path, low_memory=False)
    states = tile_state_map(diagnostic, pairs)
    chips, raw_descriptors = {}, {}
    for tile, state in sorted(states.items()):
        path = tile_path(diagnostic, tile, state)
        if not path.is_file():
            raise FileNotFoundError(path)
        chips[tile], raw_descriptors[tile] = read_chips(path)

    raw_frame = pd.DataFrame.from_dict(raw_descriptors, orient="index")
    raw_mean = raw_frame[DESCRIPTOR_COLUMNS].mean(axis=0)
    raw_sd = raw_frame[DESCRIPTOR_COLUMNS].std(axis=0).replace(0, 1)
    raw_vectors = {
        tile: (
            (raw_frame.loc[tile, DESCRIPTOR_COLUMNS] - raw_mean) / raw_sd
        ).to_numpy(float)
        for tile in raw_frame.index
    }
    return pairs, states, chips, raw_frame, raw_vectors


def reusable_completed_probe(
    output_dir: Path,
    checkpoint_path: Path,
    config_path: Path,
    dataset_path: Path,
    pairs_path: Path,
    score_space: str = "raw_sigmoid_probability",
    platt_calibrator_sha256: str | None = None,
) -> bool:
    summary_path = output_dir / "summary.json"
    required = [
        output_dir / "COMPLETED",
        summary_path,
        output_dir / "pair_metrics.csv",
        output_dir / "june_baseline_comparison" / "COMPLETED",
    ]
    if not all(path.is_file() for path in required):
        return False
    summary = read_json(summary_path)
    expected = {
        "checkpoint_sha256": sha256(checkpoint_path),
        "config_sha256": sha256(config_path),
        "dataset_sha256": sha256(dataset_path),
        "pairs_sha256": sha256(pairs_path),
    }
    return (
        summary.get("status") == "complete"
        and summary.get("score_space") == score_space
        and summary.get("platt_calibrator_sha256") == platt_calibrator_sha256
        and all(
            summary.get(key) == value for key, value in expected.items()
        )
    )


def write_candidate_outputs(
    *,
    output_dir: Path,
    diagnostic: str,
    candidate: dict,
    pairs_path: Path,
    states: dict,
    metrics: pd.DataFrame,
    features: dict,
    probabilities: dict,
    env_vars: list[str],
    device: torch.device,
    code_commit: str,
    score_space: str,
    platt_calibrator_path: Path | None,
) -> dict:
    checkpoint_path = candidate["checkpoint_path"]
    config_path = candidate["config_path_object"]
    dataset_path = candidate["dataset_path"]
    summary = {
        "status": "complete",
        "diagnostic": diagnostic,
        "device": str(device),
        "pair_count": len(metrics),
        "unique_tile_count": len(states),
        "fixed_env_prediction_did_mean": float(
            metrics["fixed_env_prediction_did"].mean()
        ),
        "fixed_env_prediction_did_bootstrap_95ci": bootstrap_ci(
            metrics["fixed_env_prediction_did"], seed=20260813
        ),
        "fixed_env_cross_absdiff_mean": float(
            metrics["fixed_env_cross_absdiff"].mean()
        ),
        "fixed_env_control_absdiff_mean": float(
            metrics["fixed_env_control_absdiff"].mean()
        ),
        "embedding_distance_did_mean": float(metrics["embedding_distance_did"].mean()),
        "raw_descriptor_distance_did_mean": float(
            metrics["raw_descriptor_distance_did"].mean()
        ),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256(checkpoint_path),
        "config": str(config_path.resolve()),
        "config_sha256": sha256(config_path),
        "candidate_experiment": candidate["experiment"],
        "candidate_august_experiment_spec": candidate["config"].get(
            "august_experiment_spec"
        ),
        "dataset": str(dataset_path.resolve()),
        "dataset_sha256": sha256(dataset_path),
        "pairs": str(pairs_path.resolve()),
        "pairs_sha256": sha256(pairs_path),
        "environment_features": env_vars,
        "fixed_environment_source": "training_split_feature_means",
        "score_space": score_space,
        "platt_calibrator": (
            str(platt_calibrator_path.resolve()) if platt_calibrator_path else None
        ),
        "platt_calibrator_sha256": (
            sha256(platt_calibrator_path) if platt_calibrator_path else None
        ),
        "code_commit": code_commit,
        "guardrail": (
            "Matched observational acquisition-boundary replay with environmental "
            "predictors held fixed; acquisition, date, phenology, and geography remain "
            "confounded."
        ),
    }
    if diagnostic == "tanoak":
        coastal = metrics[metrics["mid_lon"] <= -123]
        summary["coastal_west_of_123"] = {
            "pair_count": len(coastal),
            "fixed_env_prediction_did_mean": float(
                coastal["fixed_env_prediction_did"].mean()
            ),
            "fixed_env_or_minus_ca_mean": float(
                coastal["fixed_env_or_minus_ca"].mean()
            ),
        }

    output_dir.mkdir(parents=True, exist_ok=False)
    write_csv_atomic(
        output_dir / "tile_fixed_env_probabilities.csv",
        pd.DataFrame(
            {
                "tile": list(probabilities),
                "fixed_env_probability": list(probabilities.values()),
            }
        ),
    )
    write_csv_atomic(output_dir / "pair_metrics.csv", metrics)
    np.savez_compressed(
        output_dir / "tile_mean_embeddings.npz",
        tile_names=np.asarray(sorted(features)),
        embeddings=np.stack([features[tile] for tile in sorted(features)]),
    )
    write_json_atomic(output_dir / "summary.json", summary)

    baseline_path = BASELINE_METRICS[diagnostic]
    baseline_hash = sha256(baseline_path)
    if baseline_hash != BASELINE_SHA256[diagnostic]:
        raise ValueError(
            f"Frozen {diagnostic} baseline hash changed: {baseline_hash}"
        )
    taxon = candidate["config"]["august_experiment_spec"]["host"]
    baseline = load_baseline(
        diagnostic,
        taxon if diagnostic == "ny" else None,
        baseline_path,
    )
    comparison_frame = exact_pair_merge(baseline, metrics)
    comparison = paired_comparison(comparison_frame)
    if diagnostic == "tanoak":
        comparison = add_tanoak_coastal_comparison(
            comparison_frame, comparison
        )
    comparison.update(
        {
            "status": "complete",
            "diagnostic": diagnostic,
            "taxon": taxon,
            "baseline_metrics": str(baseline_path.resolve()),
            "baseline_metrics_sha256": baseline_hash,
            "frozen_default_baseline_sha256": BASELINE_SHA256[diagnostic],
            "candidate_probe_dir": str(output_dir.resolve()),
            "candidate_summary_sha256": sha256(output_dir / "summary.json"),
            "candidate_pair_metrics_sha256": sha256(output_dir / "pair_metrics.csv"),
            "candidate_checkpoint_sha256": summary["checkpoint_sha256"],
            "candidate_dataset_sha256": summary["dataset_sha256"],
            "candidate_experiment": candidate["experiment"],
            "candidate_august_experiment_spec": candidate["config"].get(
                "august_experiment_spec"
            ),
            "code_commit": code_commit,
            "interpretation": (
                "Positive reductions favor the August candidate. This exact paired "
                "comparison is one acceptance-suite component and does not by itself "
                "select or approve a production model."
            ),
        }
    )
    comparison_dir = output_dir / "june_baseline_comparison"
    comparison_dir.mkdir()
    write_csv_atomic(
        comparison_dir / "paired_artifact_comparison.csv", comparison_frame
    )
    write_json_atomic(comparison_dir / "summary.json", comparison)
    (comparison_dir / "COMPLETED").write_text("completed\n", encoding="utf-8")
    (output_dir / "COMPLETED").write_text("completed\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--host", choices=sorted(HOST_DIAGNOSTIC), required=True)
    parser.add_argument("--experiment", action="append", default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--resume-completed", action="store_true")
    parser.add_argument(
        "--score-space",
        choices=["raw", "configured"],
        default="raw",
        help="Raw sigmoid scores, or the candidate's configured Platt score space.",
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


def main() -> None:
    args = parse_args()
    diagnostic = HOST_DIAGNOSTIC[args.host]
    pairs_path = args.pairs or DEFAULT_PAIRS[diagnostic]
    required = [args.registry, pairs_path, BASELINE_METRICS[diagnostic]]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing batch artifact-probe inputs: {missing}")
    candidates = select_completed_candidates(
        args.registry, args.host, args.experiment
    )
    if args.output_root.exists() and any(args.output_root.iterdir()):
        if not args.resume_completed:
            raise FileExistsError(
                f"Refusing to reuse nonempty output root: {args.output_root}"
            )
    else:
        args.output_root.mkdir(parents=True, exist_ok=True)

    pending = []
    skipped = []
    for candidate in candidates:
        calibrator_path = (
            candidate["configured_calibrator_path"]
            if args.score_space == "configured"
            else None
        )
        output_score_space = (
            "platt_calibrated_probability"
            if calibrator_path is not None
            else "raw_sigmoid_probability"
        )
        calibrator_hash = sha256(calibrator_path) if calibrator_path else None
        output_dir = args.output_root / candidate["experiment"]
        if output_dir.exists():
            if args.resume_completed and reusable_completed_probe(
                output_dir,
                candidate["checkpoint_path"],
                candidate["config_path_object"],
                candidate["dataset_path"],
                pairs_path,
                score_space=output_score_space,
                platt_calibrator_sha256=calibrator_hash,
            ):
                skipped.append(candidate["experiment"])
                continue
            raise FileExistsError(
                f"Refusing partial or mismatched candidate output: {output_dir}"
            )
        pending.append(candidate)

    code_commit = current_commit()
    status = {
        "status": "loading_shared_diagnostic_tiles" if pending else "complete",
        "diagnostic": diagnostic,
        "host": args.host,
        "registry": str(args.registry.resolve()),
        "registry_sha256": sha256(args.registry),
        "pairs": str(pairs_path.resolve()),
        "pairs_sha256": sha256(pairs_path),
        "candidate_count": len(candidates),
        "pending_experiments": [item["experiment"] for item in pending],
        "completed_experiments": skipped.copy(),
        "code_commit": code_commit,
        "requested_score_space": args.score_space,
    }
    write_json_atomic(args.output_root / "batch_status.json", status)
    if not pending:
        (args.output_root / "COMPLETED").write_text("completed\n", encoding="utf-8")
        print(json.dumps(status, indent=2))
        return
    completion_marker = args.output_root / "COMPLETED"
    if completion_marker.is_file():
        completion_marker.unlink()

    pairs, states, chips, raw_frame, raw_vectors = prepare_shared_inputs(
        diagnostic, pairs_path
    )
    write_csv_atomic(
        args.output_root / "shared_tile_raw_descriptors.csv",
        raw_frame.rename_axis("tile").reset_index(),
    )
    status["status"] = "running_candidates"
    status["unique_tile_count"] = len(states)
    write_json_atomic(args.output_root / "batch_status.json", status)

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    for candidate in pending:
        status["active_experiment"] = candidate["experiment"]
        write_json_atomic(args.output_root / "batch_status.json", status)
        model, env_vars, mean_environment = load_candidate(
            candidate["checkpoint_path"],
            candidate["config_path_object"],
            candidate["dataset_path"],
            device,
        )
        calibrator_path = (
            candidate["configured_calibrator_path"]
            if args.score_space == "configured"
            else None
        )
        platt_calibrator = (
            inference.validate_platt_calibrator(read_json(calibrator_path))
            if calibrator_path is not None
            else None
        )
        output_score_space = (
            "platt_calibrated_probability"
            if platt_calibrator is not None
            else "raw_sigmoid_probability"
        )
        features, probabilities = embed_and_predict(
            model,
            chips,
            mean_environment,
            device,
            args.batch_size,
            platt_calibrator=platt_calibrator,
        )
        metrics = pair_metrics(
            diagnostic, pairs, features, probabilities, raw_vectors
        )
        write_candidate_outputs(
            output_dir=args.output_root / candidate["experiment"],
            diagnostic=diagnostic,
            candidate=candidate,
            pairs_path=pairs_path,
            states=states,
            metrics=metrics,
            features=features,
            probabilities=probabilities,
            env_vars=env_vars,
            device=device,
            code_commit=code_commit,
            score_space=output_score_space,
            platt_calibrator_path=calibrator_path,
        )
        status["completed_experiments"].append(candidate["experiment"])
        status["pending_experiments"] = [
            item["experiment"]
            for item in pending
            if item["experiment"] not in status["completed_experiments"]
        ]
        del model, features, probabilities, metrics
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        write_json_atomic(args.output_root / "batch_status.json", status)

    status.pop("active_experiment", None)
    status["status"] = "complete"
    write_json_atomic(args.output_root / "batch_status.json", status)
    (args.output_root / "COMPLETED").write_text("completed\n", encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
