#!/usr/bin/env python3
"""Compute streaming correlation diagnostics for two aligned prediction rasters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from august2026_summarize_experiments import sha256, write_json_atomic


def _same_transform(first, second, tolerance: float = 1e-9) -> bool:
    return bool(np.allclose(tuple(first), tuple(second), atol=tolerance, rtol=0))


def aligned_raster_metrics(
    first_path: Path,
    second_path: Path,
    sample_size: int = 1_000_000,
    seed: int = 20260813,
) -> dict:
    random = np.random.default_rng(seed)
    reservoir_first = np.empty(0, dtype=np.float32)
    reservoir_second = np.empty(0, dtype=np.float32)
    reservoir_keys = np.empty(0, dtype=np.float64)
    totals = {
        "n": 0,
        "sum_first": 0.0,
        "sum_second": 0.0,
        "sum_first_sq": 0.0,
        "sum_second_sq": 0.0,
        "sum_product": 0.0,
        "sum_abs_difference": 0.0,
        "sum_squared_difference": 0.0,
    }
    with rasterio.open(first_path) as first, rasterio.open(second_path) as second:
        mismatches = {}
        if (first.width, first.height) != (second.width, second.height):
            mismatches["shape"] = {
                "first": [first.height, first.width],
                "second": [second.height, second.width],
            }
        if first.crs != second.crs:
            mismatches["crs"] = {"first": str(first.crs), "second": str(second.crs)}
        if not _same_transform(first.transform, second.transform):
            mismatches["transform"] = {
                "first": list(first.transform),
                "second": list(second.transform),
            }
        if mismatches:
            raise ValueError(f"Prediction rasters are not aligned: {mismatches}")
        for _, window in first.block_windows(1):
            first_array = first.read(1, window=window, masked=True).filled(np.nan)
            second_array = second.read(1, window=window, masked=True).filled(np.nan)
            valid = np.isfinite(first_array) & np.isfinite(second_array)
            if not valid.any():
                continue
            first_values = first_array[valid].astype(np.float64, copy=False)
            second_values = second_array[valid].astype(np.float64, copy=False)
            difference = first_values - second_values
            totals["n"] += len(first_values)
            totals["sum_first"] += float(first_values.sum())
            totals["sum_second"] += float(second_values.sum())
            totals["sum_first_sq"] += float(np.dot(first_values, first_values))
            totals["sum_second_sq"] += float(np.dot(second_values, second_values))
            totals["sum_product"] += float(np.dot(first_values, second_values))
            totals["sum_abs_difference"] += float(np.abs(difference).sum())
            totals["sum_squared_difference"] += float(np.dot(difference, difference))
            keys = random.random(len(first_values))
            reservoir_first = np.concatenate(
                [reservoir_first, first_values.astype(np.float32)]
            )
            reservoir_second = np.concatenate(
                [reservoir_second, second_values.astype(np.float32)]
            )
            reservoir_keys = np.concatenate([reservoir_keys, keys])
            if len(reservoir_keys) > sample_size:
                retained = np.argpartition(reservoir_keys, -sample_size)[-sample_size:]
                reservoir_first = reservoir_first[retained]
                reservoir_second = reservoir_second[retained]
                reservoir_keys = reservoir_keys[retained]
        first_tags = first.tags()
        second_tags = second.tags()
        geometry = {
            "width": first.width,
            "height": first.height,
            "crs": str(first.crs),
            "transform": list(first.transform),
            "bounds": list(first.bounds),
        }
    count = totals["n"]
    if count < 3:
        raise ValueError("Aligned rasters have fewer than three jointly valid pixels")
    covariance_numerator = (
        totals["sum_product"]
        - totals["sum_first"] * totals["sum_second"] / count
    )
    first_variance_numerator = (
        totals["sum_first_sq"] - totals["sum_first"] ** 2 / count
    )
    second_variance_numerator = (
        totals["sum_second_sq"] - totals["sum_second"] ** 2 / count
    )
    denominator = np.sqrt(first_variance_numerator * second_variance_numerator)
    if denominator <= 0:
        pearson = None
    else:
        pearson = float(covariance_numerator / denominator)
    first_rank = pd.Series(reservoir_first).rank(method="average")
    second_rank = pd.Series(reservoir_second).rank(method="average")
    spearman = (
        float(np.corrcoef(first_rank, second_rank)[0, 1])
        if first_rank.nunique() > 1 and second_rank.nunique() > 1
        else None
    )
    sample_difference = reservoir_first.astype(float) - reservoir_second.astype(float)
    return {
        "joint_valid_pixel_count": count,
        "pearson": pearson,
        "spearman_reservoir_sample": spearman,
        "reservoir_sample_count": len(reservoir_first),
        "first_mean": totals["sum_first"] / count,
        "second_mean": totals["sum_second"] / count,
        "mean_absolute_difference": totals["sum_abs_difference"] / count,
        "root_mean_squared_difference": np.sqrt(
            totals["sum_squared_difference"] / count
        ),
        "sample_difference_quantiles": {
            str(quantile): float(np.quantile(sample_difference, quantile))
            for quantile in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
        },
        "geometry": geometry,
        "first_tags": first_tags,
        "second_tags": second_tags,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-raster", type=Path, required=True)
    parser.add_argument("--second-raster", type=Path, required=True)
    parser.add_argument("--first-label", required=True)
    parser.add_argument("--second-label", required=True)
    parser.add_argument("--sample-size", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.sample_size <= 2:
        parser.error("--sample-size must exceed two")
    return args


def main() -> None:
    args = parse_args()
    for path in (args.first_raster, args.second_raster):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace raster audit: {args.output}")
    metrics = aligned_raster_metrics(
        args.first_raster,
        args.second_raster,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    payload = {
        "status": "complete",
        "audit": "aligned_prediction_raster_correlation",
        "first_label": args.first_label,
        "second_label": args.second_label,
        "first_raster": str(args.first_raster.resolve()),
        "first_raster_sha256": sha256(args.first_raster),
        "second_raster": str(args.second_raster.resolve()),
        "second_raster_sha256": sha256(args.second_raster),
        "sample_seed": args.seed,
        **metrics,
        "guardrail": (
            "Cross-host map similarity may reflect shared ecology, shared predictors, "
            "shared acquisition exposure, or model behavior. Interpret alongside the "
            "dataset-overlap and common-support pathway audits."
        ),
    }
    write_json_atomic(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
