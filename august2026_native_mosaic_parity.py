#!/usr/bin/env python3
"""Compare deterministic native-tile samples with final production mosaics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform

from august2026_prepare_mosaic_inputs import index_outputs
from august2026_summarize_experiments import sha256, write_json_atomic


SAMPLE_FRACTIONS = (0.25, 0.50, 0.75)
DEFAULT_MAX_TILES = 123
MAX_PREDICTION_MEAN_ABSOLUTE_DIFFERENCE = 0.025
MAX_PREDICTION_ABSOLUTE_MEAN_SIGNED_DIFFERENCE = 0.010
MAX_UNCERTAINTY_MEAN_ABSOLUTE_DIFFERENCE = 0.005


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    partial = path.with_name(f"{path.name}.partial")
    frame.to_csv(partial, index=False)
    os.replace(partial, path)


def stable_tile_sample(paths: list[Path], maximum: int) -> list[Path]:
    """Return a deterministic, order-independent sample spread across the manifest."""
    if maximum <= 0:
        raise ValueError("maximum must be positive")
    ranked = sorted(
        paths,
        key=lambda path: (
            hashlib.sha256(path.stem.encode("utf-8")).hexdigest(),
            path.stem,
        ),
    )
    return ranked[:maximum]


def sample_points(source: rasterio.io.DatasetReader) -> list[tuple[float, float]]:
    bounds = source.bounds
    return [
        (
            bounds.left + x_fraction * (bounds.right - bounds.left),
            bounds.bottom + y_fraction * (bounds.top - bounds.bottom),
        )
        for y_fraction in SAMPLE_FRACTIONS
        for x_fraction in SAMPLE_FRACTIONS
    ]


def sampled_values(
    native: rasterio.io.DatasetReader,
    mosaic: rasterio.io.DatasetReader,
) -> tuple[np.ndarray, np.ndarray]:
    if native.crs is None or mosaic.crs is None:
        raise ValueError("Native and mosaic rasters must carry CRS metadata")
    points = sample_points(native)
    native_values = np.asarray(
        [value[0] for value in native.sample(points)], dtype=float
    )
    xs, ys = transform(
        native.crs,
        mosaic.crs,
        [point[0] for point in points],
        [point[1] for point in points],
    )
    mosaic_values = np.asarray(
        [value[0] for value in mosaic.sample(zip(xs, ys))], dtype=float
    )
    for values, source in ((native_values, native), (mosaic_values, mosaic)):
        if source.nodata is not None:
            values[np.isclose(values, float(source.nodata), rtol=0.0, atol=1e-8)] = np.nan
        values[~np.isfinite(values)] = np.nan
    return native_values, mosaic_values


def compare_layer(
    tiles: list[Path],
    output_index: dict[str, Path],
    mosaic_path: Path,
    layer: str,
) -> list[dict]:
    rows = []
    with rasterio.open(mosaic_path) as mosaic:
        for tile in tiles:
            native_path = output_index[tile.stem]
            with rasterio.open(native_path) as native:
                native_values, mosaic_values = sampled_values(native, mosaic)
            paired = np.isfinite(native_values) & np.isfinite(mosaic_values)
            paired_count = int(paired.sum())
            if paired_count == 0:
                native_mean = mosaic_mean = signed = absolute = pointwise_absolute = None
            else:
                native_mean = float(native_values[paired].mean())
                mosaic_mean = float(mosaic_values[paired].mean())
                differences = mosaic_values[paired] - native_values[paired]
                signed = float(differences.mean())
                # Keep the June control's tile-mean statistic for a directly
                # comparable gate, while also exposing cancellation within tiles.
                absolute = abs(signed)
                pointwise_absolute = float(np.abs(differences).mean())
            rows.append(
                {
                    "tile": tile.stem,
                    "layer": layer,
                    "paired_sample_count": paired_count,
                    "native_mean": native_mean,
                    "mosaic_mean": mosaic_mean,
                    "signed_mosaic_minus_native": signed,
                    "absolute_mosaic_minus_native": absolute,
                    "pointwise_mean_absolute_difference": pointwise_absolute,
                    "native_path": str(native_path.resolve()),
                }
            )
    return rows


def layer_summary(frame: pd.DataFrame, layer: str) -> dict:
    selected = frame[frame["layer"] == layer].copy()
    valid = selected[selected["paired_sample_count"] > 0]
    absolute = pd.to_numeric(valid["absolute_mosaic_minus_native"], errors="coerce")
    signed = pd.to_numeric(valid["signed_mosaic_minus_native"], errors="coerce")
    pointwise = pd.to_numeric(
        valid["pointwise_mean_absolute_difference"], errors="coerce"
    )
    finite = np.isfinite(absolute) & np.isfinite(signed) & np.isfinite(pointwise)
    absolute = absolute[finite]
    signed = signed[finite]
    return {
        "selected_tile_count": int(len(selected)),
        "valid_tile_count": int(len(absolute)),
        "mean_absolute_difference": float(absolute.mean()) if len(absolute) else None,
        "median_absolute_difference": float(absolute.median()) if len(absolute) else None,
        "mean_signed_difference": float(signed.mean()) if len(signed) else None,
        "max_absolute_difference": float(absolute.max()) if len(absolute) else None,
        "mean_pointwise_absolute_difference": (
            float(pointwise[np.isfinite(pointwise)].mean())
            if np.isfinite(pointwise).any()
            else None
        ),
    }


def validate_summary(prediction: dict, uncertainty: dict, expected_count: int) -> list[str]:
    failures = []
    for layer, summary in (("prediction", prediction), ("uncertainty", uncertainty)):
        if summary["selected_tile_count"] != expected_count:
            failures.append(f"{layer}_selected_tile_count")
        if summary["valid_tile_count"] != expected_count:
            failures.append(f"{layer}_valid_tile_count")
        for key in (
            "mean_absolute_difference",
            "median_absolute_difference",
            "mean_signed_difference",
            "max_absolute_difference",
            "mean_pointwise_absolute_difference",
        ):
            value = summary[key]
            if value is None or not np.isfinite(value):
                failures.append(f"{layer}_{key}")
    if not failures:
        if (
            prediction["mean_absolute_difference"]
            > MAX_PREDICTION_MEAN_ABSOLUTE_DIFFERENCE
        ):
            failures.append("prediction_mean_absolute_difference")
        if (
            abs(prediction["mean_signed_difference"])
            > MAX_PREDICTION_ABSOLUTE_MEAN_SIGNED_DIFFERENCE
        ):
            failures.append("prediction_absolute_mean_signed_difference")
        if (
            uncertainty["mean_absolute_difference"]
            > MAX_UNCERTAINTY_MEAN_ABSOLUTE_DIFFERENCE
        ):
            failures.append("uncertainty_mean_absolute_difference")
    return failures


def run_parity(
    tile_manifest: Path,
    prediction_output_root: Path,
    uncertainty_output_root: Path,
    prediction_mosaic: Path,
    uncertainty_mosaic: Path,
    host: str,
    region: str,
    output_dir: Path,
    max_tiles: int = DEFAULT_MAX_TILES,
) -> dict:
    manifest_tiles = [
        Path(line.strip())
        for line in tile_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not manifest_tiles or len(manifest_tiles) != len(set(manifest_tiles)):
        raise ValueError("Tile manifest must be nonempty and unique")
    prediction_index = index_outputs(prediction_output_root, "prediction")
    uncertainty_index = index_outputs(uncertainty_output_root, "uncertainty")
    expected_stems = {tile.stem for tile in manifest_tiles}
    for label, index in (
        ("prediction", prediction_index),
        ("uncertainty", uncertainty_index),
    ):
        if set(index) != expected_stems:
            raise ValueError(
                f"{label} native output set differs from the tile manifest"
            )
    selected = stable_tile_sample(manifest_tiles, min(max_tiles, len(manifest_tiles)))
    rows = compare_layer(selected, prediction_index, prediction_mosaic, "prediction")
    rows.extend(
        compare_layer(selected, uncertainty_index, uncertainty_mosaic, "uncertainty")
    )
    frame = pd.DataFrame(rows)
    prediction = layer_summary(frame, "prediction")
    uncertainty = layer_summary(frame, "uncertainty")
    failures = validate_summary(prediction, uncertainty, len(selected))
    output_dir.mkdir(parents=True, exist_ok=False)
    detail_path = output_dir / "native_mosaic_parity.csv"
    write_csv_atomic(detail_path, frame)
    payload = {
        "status": "complete",
        "production_ready": not failures,
        "host": host,
        "region": region,
        "tile_manifest": str(tile_manifest.resolve()),
        "tile_manifest_sha256": sha256(tile_manifest),
        "manifest_tile_count": len(manifest_tiles),
        "selected_tile_count": len(selected),
        "selection_method": "lowest SHA256(tile stem), order independent",
        "samples_per_tile": len(SAMPLE_FRACTIONS) ** 2,
        "prediction_mosaic": str(prediction_mosaic.resolve()),
        "prediction_mosaic_sha256": sha256(prediction_mosaic),
        "uncertainty_mosaic": str(uncertainty_mosaic.resolve()),
        "uncertainty_mosaic_sha256": sha256(uncertainty_mosaic),
        "prediction": prediction,
        "uncertainty": uncertainty,
        "thresholds": {
            "prediction_mean_absolute_difference_maximum": (
                MAX_PREDICTION_MEAN_ABSOLUTE_DIFFERENCE
            ),
            "prediction_absolute_mean_signed_difference_maximum": (
                MAX_PREDICTION_ABSOLUTE_MEAN_SIGNED_DIFFERENCE
            ),
            "uncertainty_mean_absolute_difference_maximum": (
                MAX_UNCERTAINTY_MEAN_ABSOLUTE_DIFFERENCE
            ),
            "basis": (
                "Conservative pre-registered limits relative to the June 123-tile "
                "assembly control, whose worst host prediction mean absolute "
                "difference was 0.0121, worst absolute signed mean was 0.0019, "
                "and worst uncertainty mean absolute difference was 0.00084."
            ),
        },
        "detail": str(detail_path.resolve()),
        "detail_sha256": sha256(detail_path),
        "failure_count": len(failures),
        "failures": failures,
        "interpretation": (
            "This is an assembly/resampling parity control. It does not test model "
            "acquisition sensitivity or replace matched-boundary diagnostics."
        ),
    }
    write_json_atomic(output_dir / "summary.json", payload)
    marker = "COMPLETED" if not failures else "REJECTED"
    (output_dir / marker).write_text(f"{marker.lower()}\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-manifest", type=Path, required=True)
    parser.add_argument("--prediction-output-root", type=Path, required=True)
    parser.add_argument("--uncertainty-output-root", type=Path, required=True)
    parser.add_argument("--prediction-mosaic", type=Path, required=True)
    parser.add_argument("--uncertainty-mosaic", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--max-tiles", type=int, default=DEFAULT_MAX_TILES)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required_files = [
        args.tile_manifest,
        args.prediction_mosaic,
        args.uncertainty_mosaic,
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    for path in (args.prediction_output_root, args.uncertainty_output_root):
        if not path.is_dir():
            missing.append(str(path))
    if missing:
        raise FileNotFoundError(f"Missing native/mosaic parity inputs: {missing}")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to replace parity output: {args.output_dir}")
    payload = run_parity(
        args.tile_manifest,
        args.prediction_output_root,
        args.uncertainty_output_root,
        args.prediction_mosaic,
        args.uncertainty_mosaic,
        args.host,
        args.region,
        args.output_dir,
        max_tiles=args.max_tiles,
    )
    if payload["failures"]:
        raise RuntimeError(f"Native/mosaic parity failed: {payload['failures']}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
