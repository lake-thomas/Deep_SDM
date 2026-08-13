#!/usr/bin/env python3
"""Freeze a mild four-band augmentation recipe from exact-overlap NAIP evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_EXACT_OVERLAP = Path(
    "/home/ncsucnr/projects/research_agent/agent_outputs/"
    "host_sdm_inference_artifacts_aug122026/results_exact_acquisition_overlap/"
    "exact_overlap_raw_chip_stats.csv"
)
DEFAULT_EXACT_OVERLAP_SHA256 = (
    "0d2e1e48d31d6d433e031dc57cfe75d9215e57abf5a3fc6c04cb6e673a74253c"
)
QUANTILES = [0.25, 0.5, 0.75, 0.9, 0.95]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    with partial.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    os.replace(partial, path)


def quantile_payload(values) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("Calibration metric values must be nonempty and finite")
    return {f"q{int(q * 100):02d}": float(np.quantile(array, q)) for q in QUANTILES}


def derive_metrics(frame: pd.DataFrame) -> dict:
    required = {
        "chip_id",
        "acquisition",
        *(f"band_{index}_mean" for index in range(1, 5)),
        *(f"band_{index}_sd" for index in range(1, 5)),
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Exact-overlap evidence is missing columns: {sorted(missing)}")
    acquisitions = sorted(frame["acquisition"].unique())
    if acquisitions != ["ny_2021", "pa_2022"]:
        raise ValueError(f"Unexpected exact-overlap acquisitions: {acquisitions}")
    if frame.duplicated(["chip_id", "acquisition"]).any():
        raise ValueError("Exact-overlap chip/acquisition rows must be unique")
    first = frame[frame["acquisition"] == "ny_2021"].set_index("chip_id").sort_index()
    second = frame[frame["acquisition"] == "pa_2022"].set_index("chip_id").sort_index()
    if not first.index.equals(second.index) or len(first) < 10:
        raise ValueError("Exact-overlap acquisitions must share at least ten chip IDs")

    means = [f"band_{index}_mean" for index in range(1, 5)]
    deviations = [f"band_{index}_sd" for index in range(1, 5)]
    scene_first = first[means].mean(axis=1)
    scene_second = second[means].mean(axis=1)
    scene_ratio = scene_second / scene_first
    contrast_ratio = second[deviations].mean(axis=1) / first[deviations].mean(axis=1)
    first_ndvi = (
        (first["band_4_mean"] - first["band_1_mean"])
        / (first["band_4_mean"] + first["band_1_mean"])
    )
    second_ndvi = (
        (second["band_4_mean"] - second["band_1_mean"])
        / (second["band_4_mean"] + second["band_1_mean"])
    )
    relative_band_ratio = (
        second[means].to_numpy() / first[means].to_numpy()
    ) / scene_ratio.to_numpy()[:, None]
    arrays = {
        "absolute_scene_gain_change": np.abs(scene_ratio - 1.0),
        "absolute_contrast_ratio_change": np.abs(contrast_ratio - 1.0),
        "absolute_mean_ndvi_change": np.abs(second_ndvi - first_ndvi),
        "absolute_band_gain_residual_change": np.abs(
            relative_band_ratio - 1.0
        ).ravel(),
    }
    return {
        "paired_chip_count": len(first),
        "acquisition_order": acquisitions,
        "scene_ratio_median_pa_over_ny": float(scene_ratio.median()),
        "contrast_ratio_median_pa_over_ny": float(contrast_ratio.median()),
        "quantiles": {
            name: quantile_payload(values) for name, values in arrays.items()
        },
    }


def derive_recipe(metrics: dict) -> dict:
    quantiles = metrics["quantiles"]
    scene_gain = min(0.08, quantiles["absolute_scene_gain_change"]["q50"])
    contrast = min(0.08, quantiles["absolute_contrast_ratio_change"]["q50"])
    band_gain = min(
        0.02, quantiles["absolute_band_gain_residual_change"]["q25"]
    )
    max_ndvi = min(0.06, quantiles["absolute_mean_ndvi_change"]["q90"])
    return {
        "enabled": True,
        "probability": 0.8,
        "scene_gain": round(float(scene_gain), 4),
        "gamma": 0.06,
        "contrast": round(float(contrast), 4),
        "band_gain": round(float(band_gain), 4),
        "band_offset": 0.01,
        "max_mean_ndvi_shift": round(float(max_ndvi), 4),
        "attempts": 3,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exact-overlap-csv", type=Path, default=DEFAULT_EXACT_OVERLAP
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.exact_overlap_csv.is_file():
        raise FileNotFoundError(args.exact_overlap_csv)
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace calibration output: {args.output}")
    code_root = Path(__file__).resolve().parent
    code_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=code_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    code_dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=code_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    if code_dirty:
        raise RuntimeError("Refusing to freeze calibration from a dirty code tree")
    source_hash = sha256(args.exact_overlap_csv)
    if (
        args.exact_overlap_csv.resolve() == DEFAULT_EXACT_OVERLAP.resolve()
        and source_hash != DEFAULT_EXACT_OVERLAP_SHA256
    ):
        raise ValueError(
            "Frozen exact-overlap evidence hash changed: "
            f"expected {DEFAULT_EXACT_OVERLAP_SHA256}, found {source_hash}"
        )
    metrics = derive_metrics(pd.read_csv(args.exact_overlap_csv, low_memory=False))
    recipe = derive_recipe(metrics)
    output = {
        "status": "complete",
        "purpose": "August 2026 mild correlated four-band augmentation screen",
        "code_root": str(code_root),
        "code_commit": code_commit,
        "code_dirty": code_dirty,
        "source_exact_overlap_csv": str(args.exact_overlap_csv.resolve()),
        "source_exact_overlap_sha256": source_hash,
        "frozen_default_source_sha256": DEFAULT_EXACT_OVERLAP_SHA256,
        "evidence": metrics,
        "radiometric_augmentation": recipe,
        "selection_rules": {
            "scene_gain": "minimum of 0.08 cap and paired median absolute scene-gain change",
            "contrast": "minimum of 0.08 cap and paired median absolute contrast-ratio change",
            "band_gain": "minimum of 0.02 cap and paired q25 band-relative change",
            "max_mean_ndvi_shift": "minimum of 0.06 cap and paired q90 mean-NDVI change",
            "gamma": "fixed conservative 0.06 range; chip means do not identify gamma",
            "band_offset": "fixed conservative 0.01 range, equivalent to 2.55 8-bit DN",
        },
        "guardrails": [
            "The source is 25 exact-overlap chips from one NY 2021/PA 2022 comparison.",
            "The recipe is a bounded augmentation arm, not a universal radiometric correction.",
            "Augmentation is training-only and must be judged with calibration, BlockCV, and seam diagnostics.",
            "Phenology and acquisition remain confounded outside the exact-overlap footprint.",
        ],
    }
    write_json_atomic(args.output, output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
