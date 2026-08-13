#!/usr/bin/env python3
"""Validate completeness, geometry, tags, masks, and sampled ranges of tiled inference."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling

from august2026_summarize_experiments import sha256, write_json_atomic


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    partial = path.with_name(f"{path.name}.partial")
    frame.to_csv(partial, index=False)
    os.replace(partial, path)


def _sample(source, size: int = 64) -> np.ndarray:
    return source.read(
        1,
        out_shape=(min(size, source.height), min(size, source.width)),
        masked=True,
        resampling=Resampling.nearest,
    )


def validate_triplet(
    input_tile: Path,
    prediction_path: Path,
    uncertainty_path: Path,
    count_path: Path,
    expected_score_space: str,
    expected_calibrator_sha256: str | None,
    product: str,
) -> dict:
    failures = []
    with (
        rasterio.open(input_tile) as source,
        rasterio.open(prediction_path) as prediction,
        rasterio.open(uncertainty_path) as uncertainty,
        rasterio.open(count_path) as count,
    ):
        geometry = (source.width, source.height, source.crs, source.transform)
        for label, raster in (
            ("prediction", prediction),
            ("uncertainty", uncertainty),
            ("count", count),
        ):
            if (raster.width, raster.height, raster.crs, raster.transform) != geometry:
                failures.append(f"{label}_geometry")
        if prediction.dtypes[0] != "float32":
            failures.append("prediction_dtype")
        if uncertainty.dtypes[0] != "float32":
            failures.append("uncertainty_dtype")
        if count.dtypes[0] != "uint16":
            failures.append("count_dtype")
        expected_tags = {
            "blend_window": "hann",
            "blend_edge_floor": "0.05",
            "score_space": expected_score_space,
        }
        if expected_calibrator_sha256 is not None:
            expected_tags["platt_calibrator_sha256"] = expected_calibrator_sha256
        for label, raster in (
            ("prediction", prediction),
            ("uncertainty", uncertainty),
            ("count", count),
        ):
            tags = raster.tags()
            for key, value in expected_tags.items():
                found = tags.get(key)
                if key == "blend_edge_floor":
                    try:
                        matches = abs(float(found) - float(value)) <= 1e-12
                    except (TypeError, ValueError):
                        matches = False
                else:
                    matches = found == value
                if not matches:
                    failures.append(f"{label}_tag_{key}")
            if expected_calibrator_sha256 is None and tags.get(
                "platt_calibrator_sha256"
            ):
                failures.append(f"{label}_unexpected_calibrator")
        prediction_sample = _sample(prediction)
        uncertainty_sample = _sample(uncertainty)
        count_sample = _sample(count)
        prediction_valid = ~np.ma.getmaskarray(prediction_sample)
        uncertainty_valid = ~np.ma.getmaskarray(uncertainty_sample)
        count_values = np.asarray(count_sample.filled(0), dtype=float)
        count_valid = count_values > 0
        if not np.array_equal(prediction_valid, uncertainty_valid):
            failures.append("prediction_uncertainty_mask")
        if not np.array_equal(prediction_valid, count_valid):
            failures.append("prediction_count_mask")
        prediction_values = np.asarray(prediction_sample.compressed(), dtype=float)
        uncertainty_values = np.asarray(uncertainty_sample.compressed(), dtype=float)
        if not len(prediction_values):
            failures.append("no_sampled_valid_predictions")
        elif prediction_values.min() < 0 or prediction_values.max() > 1:
            failures.append("prediction_range")
        if len(uncertainty_values):
            if uncertainty_values.min() < 0 or uncertainty_values.max() > 0.5 + 1e-6:
                failures.append("uncertainty_range")
            if product == "deterministic" and uncertainty_values.max() > 1e-7:
                failures.append("deterministic_uncertainty_nonzero")
        if count_values.max() > 16:
            failures.append("count_exceeds_stride64_support")
    return {
        "tile": str(input_tile),
        "prediction": str(prediction_path),
        "uncertainty": str(uncertainty_path),
        "count": str(count_path),
        "passed": not failures,
        "failures": ";".join(failures),
        "sampled_prediction_min": (
            float(prediction_values.min()) if len(prediction_values) else None
        ),
        "sampled_prediction_max": (
            float(prediction_values.max()) if len(prediction_values) else None
        ),
        "sampled_uncertainty_max": (
            float(uncertainty_values.max()) if len(uncertainty_values) else None
        ),
        "sampled_count_max": float(count_values.max()),
    }


def output_index(output_root: Path, suffix: str) -> dict[str, Path]:
    paths = list(output_root.rglob(f"*_{suffix}.tif"))
    index = {}
    duplicates = []
    ending = f"_{suffix}"
    for path in paths:
        stem = path.stem[: -len(ending)]
        if stem in index:
            duplicates.append(stem)
        index[stem] = path
    if duplicates:
        raise ValueError(f"Duplicate output tile stems: {sorted(set(duplicates))[:10]}")
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--expected-score-space", required=True)
    parser.add_argument("--expected-calibrator-sha256", default=None)
    parser.add_argument("--product", choices=["deterministic", "uncertainty"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.tile_manifest.is_file():
        raise FileNotFoundError(args.tile_manifest)
    if not args.output_root.is_dir():
        raise FileNotFoundError(args.output_root)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to replace inference QA: {args.output_dir}")
    tiles = [Path(line.strip()) for line in args.tile_manifest.read_text().splitlines() if line.strip()]
    if not tiles or len(tiles) != len(set(tiles)):
        raise ValueError("Tile manifest must be nonempty and unique")
    missing_inputs = [str(path) for path in tiles if not path.is_file()]
    if missing_inputs:
        raise FileNotFoundError(f"Missing input tiles: {missing_inputs[:10]}")
    indices = {
        suffix: output_index(args.output_root, suffix)
        for suffix in ("prediction", "uncertainty", "count")
    }
    expected_stems = {path.stem for path in tiles}
    failures = []
    for suffix, index in indices.items():
        missing = expected_stems - set(index)
        extra = set(index) - expected_stems
        if missing:
            failures.append(f"missing_{suffix}={len(missing)}")
        if extra:
            failures.append(f"extra_{suffix}={len(extra)}")
    rows = []
    if not failures:
        for tile in tiles:
            rows.append(
                validate_triplet(
                    tile,
                    indices["prediction"][tile.stem],
                    indices["uncertainty"][tile.stem],
                    indices["count"][tile.stem],
                    args.expected_score_space,
                    args.expected_calibrator_sha256,
                    args.product,
                )
            )
        failed_tiles = [row for row in rows if not row["passed"]]
        if failed_tiles:
            failures.append(f"failed_tile_triplets={len(failed_tiles)}")
    args.output_dir.mkdir(parents=True)
    detail_path = args.output_dir / "tile_output_qa.csv"
    write_csv_atomic(detail_path, pd.DataFrame(rows))
    payload = {
        "status": "complete",
        "production_ready": not failures,
        "host": args.host,
        "region": args.region,
        "product": args.product,
        "tile_manifest": str(args.tile_manifest.resolve()),
        "tile_manifest_sha256": sha256(args.tile_manifest),
        "expected_tile_count": len(tiles),
        "prediction_count": len(indices["prediction"]),
        "uncertainty_count": len(indices["uncertainty"]),
        "count_raster_count": len(indices["count"]),
        "expected_score_space": args.expected_score_space,
        "expected_calibrator_sha256": args.expected_calibrator_sha256,
        "failure_count": len(failures),
        "failures": failures,
        "tile_output_qa": str(detail_path.resolve()),
        "tile_output_qa_sha256": sha256(detail_path),
        "sampling_note": (
            "All triplets receive metadata/tag/geometry checks. Raster values and "
            "valid masks are checked on a 64x64 nearest-neighbor grid per tile."
        ),
    }
    write_json_atomic(args.output_dir / "summary.json", payload)
    (args.output_dir / "COMPLETED").write_text("completed\n", encoding="utf-8")
    if failures:
        raise RuntimeError(f"Inference output QA failed: {failures}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
