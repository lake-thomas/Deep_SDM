#!/usr/bin/env python3
"""Decompose Ailanthus/Juglans prediction correlation on common model support."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch

import inference_multimodal_production as inference
from august2026_batch_artifact_probe import select_completed_candidates
from august2026_fixed_env_artifact_probe import load_candidate, sha256, write_json_atomic


HOSTS = ("ailanthus_altissima", "juglans_nigra")


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    partial = path.with_name(f"{path.name}.partial")
    frame.to_csv(partial, index=False)
    os.replace(partial, path)


def resolve_chip_path(image_root: Path, value: str) -> Path:
    path = Path(str(value).replace("\\", os.sep))
    return path if path.is_absolute() else image_root / path


def select_support_rows(
    dataset_path: Path,
    image_root: Path,
    origin: str,
    env_vars: list[str],
    samples_per_class: int,
    seed: int,
) -> pd.DataFrame:
    columns = ["chip_path", "presence", "lat", "lon", *env_vars]
    frame = pd.read_csv(dataset_path, usecols=columns, low_memory=False)
    frame["presence"] = pd.to_numeric(frame["presence"], errors="raise").astype(int)
    if not set(frame["presence"]).issubset({0, 1}):
        raise ValueError(f"Nonbinary labels in {dataset_path}")
    selected = []
    for label, group in frame.groupby("presence", sort=True):
        count = min(samples_per_class, len(group))
        selected.append(group.sample(n=count, random_state=seed + int(label)))
    output = pd.concat(selected, ignore_index=True)
    output.insert(0, "origin_host", origin)
    output.insert(1, "origin_dataset", str(dataset_path.resolve()))
    output["resolved_chip_path"] = output["chip_path"].map(
        lambda value: str(resolve_chip_path(image_root, value).resolve())
    )
    missing = [path for path in output["resolved_chip_path"] if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing common-support chips: {missing[:5]}")
    numeric = output[env_vars].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError(f"Nonfinite environmental values in selected support: {dataset_path}")
    output[env_vars] = numeric
    return output


def read_image_batch(paths: list[str]) -> np.ndarray:
    images = []
    for value in paths:
        with rasterio.open(value) as source:
            image = source.read([1, 2, 3, 4]).astype(np.float32) / 255.0
        if image.ndim != 3 or image.shape[0] != 4:
            raise ValueError(f"Expected a four-band image chip: {value}")
        images.append(np.nan_to_num(np.clip(image, 0.0, 1.0)))
    return np.stack(images)


def encode_common_support(
    model,
    support: pd.DataFrame,
    env_vars: list[str],
    mean_environment: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    image_feature_batches = []
    environment_feature_batches = []
    full_logit_batches = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(support), batch_size):
            rows = support.iloc[start : start + batch_size]
            image = torch.from_numpy(
                read_image_batch(rows["resolved_chip_path"].tolist())
            ).to(device)
            environment = torch.from_numpy(
                rows[env_vars].to_numpy(dtype=np.float32)
            ).to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                image_features = model.image_encoder(image)
                environment_features = model.env_encoder(environment)
                logits = model.classifier(
                    torch.cat([image_features, environment_features], dim=1)
                )
            image_feature_batches.append(image_features.float().cpu())
            environment_feature_batches.append(environment_features.float().cpu())
            full_logit_batches.append(logits.float().cpu())
    image_features = torch.cat(image_feature_batches)
    environment_features = torch.cat(environment_feature_batches)
    full_logits = torch.cat(full_logit_batches)
    with torch.inference_mode():
        mean_environment_tensor = torch.from_numpy(mean_environment).to(device).unsqueeze(0)
        mean_environment_features = model.env_encoder(mean_environment_tensor).float().cpu()
        mean_image_features = image_features.mean(dim=0, keepdim=True)
        pathway_batches = {"image_varying": [], "environment_varying": []}
        for start in range(0, len(support), batch_size):
            image_batch = image_features[start : start + batch_size].to(device)
            env_batch = environment_features[start : start + batch_size].to(device)
            count = len(image_batch)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                image_logits = model.classifier(
                    torch.cat(
                        [
                            image_batch,
                            mean_environment_features.to(device).expand(count, -1),
                        ],
                        dim=1,
                    )
                )
                environment_logits = model.classifier(
                    torch.cat(
                        [mean_image_features.to(device).expand(count, -1), env_batch],
                        dim=1,
                    )
                )
            pathway_batches["image_varying"].append(image_logits.float().cpu())
            pathway_batches["environment_varying"].append(
                environment_logits.float().cpu()
            )
    logits = {
        "full": full_logits.numpy(),
        "image_varying": torch.cat(pathway_batches["image_varying"]).numpy(),
        "environment_varying": torch.cat(
            pathway_batches["environment_varying"]
        ).numpy(),
    }
    return {
        name: torch.sigmoid(torch.from_numpy(values)).numpy().reshape(-1)
        for name, values in logits.items()
    }


def residualize(outcome: np.ndarray, predictor: np.ndarray) -> np.ndarray:
    outcome = np.asarray(outcome, dtype=float)
    predictor = np.asarray(predictor, dtype=float)
    design = np.column_stack([np.ones(len(predictor)), predictor])
    coefficients = np.linalg.lstsq(design, outcome, rcond=None)[0]
    return outcome - design @ coefficients


def correlation_metrics(first: np.ndarray, second: np.ndarray) -> dict:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    valid = np.isfinite(first) & np.isfinite(second)
    first = first[valid]
    second = second[valid]
    if len(first) < 3 or np.std(first) == 0 or np.std(second) == 0:
        return {"n": len(first), "pearson": None, "spearman": None}
    return {
        "n": len(first),
        "pearson": float(np.corrcoef(first, second)[0, 1]),
        "spearman": float(
            np.corrcoef(
                pd.Series(first).rank(method="average"),
                pd.Series(second).rank(method="average"),
            )[0, 1]
        ),
    }


def build_correlation_table(predictions: pd.DataFrame) -> pd.DataFrame:
    subsets = {
        "all": np.ones(len(predictions), dtype=bool),
        "background": predictions["presence"].to_numpy(int) == 0,
        "presence": predictions["presence"].to_numpy(int) == 1,
        **{
            f"origin_{host}": predictions["origin_host"].to_numpy(str) == host
            for host in HOSTS
        },
    }
    rows = []
    for score_space in ("raw", "configured"):
        for subset_name, mask in subsets.items():
            for pathway in (
                "full",
                "image_varying",
                "environment_varying",
                "full_residual_after_environment",
            ):
                first = predictions.loc[
                    mask, f"{HOSTS[0]}_{score_space}_{pathway}"
                ].to_numpy(float)
                second = predictions.loc[
                    mask, f"{HOSTS[1]}_{score_space}_{pathway}"
                ].to_numpy(float)
                rows.append(
                    {
                        "score_space": score_space,
                        "subset": subset_name,
                        "pathway": pathway,
                        **correlation_metrics(first, second),
                    }
                )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-origin-class", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if args.samples_per_origin_class <= 0 or args.batch_size <= 0:
        parser.error("Sample and batch sizes must be positive")
    return args


def main() -> None:
    args = parse_args()
    if not args.registry.is_file():
        raise FileNotFoundError(args.registry)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to replace pathway audit: {args.output_dir}")
    candidates = {
        host: select_completed_candidates(args.registry, host, None)[0]
        for host in HOSTS
    }
    if any(len(select_completed_candidates(args.registry, host, None)) != 1 for host in HOSTS):
        raise ValueError("Pathway audit requires exactly one production candidate per host")
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    loaded = {}
    for host, candidate in candidates.items():
        loaded[host] = load_candidate(
            candidate["checkpoint_path"],
            candidate["config_path_object"],
            candidate["dataset_path"],
            device,
        )
    env_vars = loaded[HOSTS[0]][1]
    if loaded[HOSTS[1]][1] != env_vars:
        raise ValueError("Cross-host pathway audit requires identical environmental features")
    support = pd.concat(
        [
            select_support_rows(
                candidates[host]["dataset_path"],
                Path(candidates[host]["config"]["image_dir"]),
                host,
                env_vars,
                args.samples_per_origin_class,
                args.seed + index * 100,
            )
            for index, host in enumerate(HOSTS)
        ],
        ignore_index=True,
    )
    predictions = support[
        ["origin_host", "origin_dataset", "presence", "lat", "lon", "resolved_chip_path"]
    ].copy()
    calibrator_provenance = {}
    for host in HOSTS:
        model, _, mean_environment = loaded[host]
        host_predictions = encode_common_support(
            model,
            support,
            env_vars,
            mean_environment,
            device,
            args.batch_size,
        )
        calibrator_path = candidates[host]["configured_calibrator_path"]
        calibrator = (
            inference.validate_platt_calibrator(inference.load_json(calibrator_path))
            if calibrator_path is not None
            else None
        )
        calibrator_provenance[host] = {
            "configured_score_space": (
                "platt_calibrated_probability"
                if calibrator is not None
                else "raw_sigmoid_probability"
            ),
            "platt_calibrator": str(calibrator_path.resolve()) if calibrator_path else None,
            "platt_calibrator_sha256": sha256(calibrator_path) if calibrator_path else None,
        }
        for pathway, raw_values in host_predictions.items():
            predictions[f"{host}_raw_{pathway}"] = raw_values
            predictions[f"{host}_configured_{pathway}"] = (
                inference.apply_platt_probabilities(raw_values, calibrator)
            )
        for score_space in ("raw", "configured"):
            predictions[
                f"{host}_{score_space}_full_residual_after_environment"
            ] = residualize(
                predictions[f"{host}_{score_space}_full"].to_numpy(float),
                predictions[
                    f"{host}_{score_space}_environment_varying"
                ].to_numpy(float),
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    correlations = build_correlation_table(predictions)
    args.output_dir.mkdir(parents=True)
    prediction_path = args.output_dir / "common_support_predictions.csv"
    correlation_path = args.output_dir / "cross_host_pathway_correlations.csv"
    write_csv_atomic(prediction_path, predictions)
    write_csv_atomic(correlation_path, correlations)
    payload = {
        "status": "complete",
        "audit": "ailanthus_juglans_common_support_pathway_correlation",
        "registry": str(args.registry.resolve()),
        "registry_sha256": sha256(args.registry),
        "hosts": list(HOSTS),
        "support_row_count": len(support),
        "samples_per_origin_class_requested": args.samples_per_origin_class,
        "seed": args.seed,
        "device": str(device),
        "environment_features": env_vars,
        "candidates": {
            host: {
                "experiment": candidates[host]["experiment"],
                "checkpoint_sha256": sha256(candidates[host]["checkpoint_path"]),
                "dataset_sha256": sha256(candidates[host]["dataset_path"]),
                "image_root": str(
                    Path(candidates[host]["config"]["image_dir"]).resolve()
                ),
                **calibrator_provenance[host],
            }
            for host in HOSTS
        },
        "common_support_predictions": str(prediction_path.resolve()),
        "common_support_predictions_sha256": sha256(prediction_path),
        "cross_host_pathway_correlations": str(correlation_path.resolve()),
        "cross_host_pathway_correlations_sha256": sha256(correlation_path),
        "pathway_definitions": {
            "full": "Each chip is paired with its own environmental vector.",
            "image_varying": (
                "Image features vary while environmental features are fixed at each "
                "model's training-split mean vector."
            ),
            "environment_varying": (
                "Environmental features vary while image features are fixed at the "
                "model-specific mean embedding over the common support."
            ),
            "full_residual_after_environment": (
                "OLS residual of full score after controlling for the corresponding "
                "environment-varying score."
            ),
        },
        "guardrail": (
            "This is a controlled common-support decomposition, not a claim of causal "
            "feature attribution. Final aligned-raster correlation and visual review "
            "remain separate acceptance evidence."
        ),
    }
    write_json_atomic(args.output_dir / "summary.json", payload)
    (args.output_dir / "COMPLETED").write_text("completed\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
