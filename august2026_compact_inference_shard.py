#!/usr/bin/env python3
"""Create an atomic, QA-verified LERC_ZSTD copy of one inference shard."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import pandas as pd
import rasterio

from august2026_summarize_experiments import sha256, write_json_atomic


FLOAT_SUFFIXES = ("prediction", "uncertainty")
ALL_SUFFIXES = (*FLOAT_SUFFIXES, "count")


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    partial = path.with_name(f"{path.name}.partial")
    frame.to_csv(partial, index=False)
    os.replace(partial, path)


def manifest_stems(path: Path) -> list[str]:
    rows = [Path(line.strip()).stem for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or len(rows) != len(set(rows)):
        raise ValueError("Tile manifest must be nonempty and unique")
    return rows


def source_index(source_dir: Path, suffix: str) -> dict[str, Path]:
    ending = f"_{suffix}"
    index = {}
    for path in source_dir.glob(f"*_{suffix}.tif"):
        stem = path.stem[: -len(ending)]
        if stem in index:
            raise ValueError(f"Duplicate {suffix} stem: {stem}")
        index[stem] = path
    return index


def validate_pair(source_path: Path, compact_path: Path, suffix: str, max_z_error: float) -> dict:
    maximum = 0.0
    absolute_sum = 0.0
    valid_count = 0
    mask_equal = True
    with rasterio.open(source_path) as source, rasterio.open(compact_path) as compact:
        geometry_equal = (
            source.width, source.height, source.crs, source.transform
        ) == (
            compact.width, compact.height, compact.crs, compact.transform
        )
        metadata_equal = (
            source.dtypes == compact.dtypes
            and source.nodata == compact.nodata
            and source.descriptions == compact.descriptions
            and source.tags() == compact.tags()
        )
        compression = compact.compression.value if compact.compression is not None else None
        image_structure = compact.tags(ns="IMAGE_STRUCTURE")
        for _, window in source.block_windows(1):
            left = source.read(1, window=window, masked=True)
            right = compact.read(1, window=window, masked=True)
            left_mask = np.ma.getmaskarray(left)
            right_mask = np.ma.getmaskarray(right)
            if not np.array_equal(left_mask, right_mask):
                mask_equal = False
                continue
            valid = ~left_mask
            if np.any(valid):
                delta = np.abs(
                    np.asarray(left.data[valid], dtype=np.float64)
                    - np.asarray(right.data[valid], dtype=np.float64)
                )
                maximum = max(maximum, float(delta.max()))
                absolute_sum += float(delta.sum())
                valid_count += int(delta.size)
    if suffix in FLOAT_SUFFIXES:
        codec_ok = compression == "LERC_ZSTD"
        try:
            recorded_error = float(image_structure.get("MAX_Z_ERROR", "nan"))
        except ValueError:
            recorded_error = float("nan")
        error_ok = maximum <= max_z_error + 5e-8 and abs(recorded_error - max_z_error) <= 1e-12
    else:
        codec_ok = True
        recorded_error = 0.0
        error_ok = maximum == 0.0
    passed = geometry_equal and metadata_equal and mask_equal and codec_ok and error_ok
    return {
        "geometry_equal": geometry_equal,
        "metadata_equal": metadata_equal,
        "mask_equal": mask_equal,
        "compression": compression,
        "recorded_max_z_error": recorded_error,
        "measured_mae": absolute_sum / valid_count if valid_count else 0.0,
        "measured_max_abs_error": maximum,
        "valid_pixel_count": valid_count,
        "passed": passed,
    }


def compact_one(
    source_path: Path,
    output_path: Path,
    suffix: str,
    max_z_error: float,
    gdal_translate: str,
) -> dict:
    partial = output_path.with_name(f"{output_path.name}.partial.tif")
    if output_path.exists() or partial.exists():
        raise FileExistsError(f"Refusing existing compact output: {output_path}")
    started = time.monotonic()
    source_hash = sha256(source_path)
    if suffix in FLOAT_SUFFIXES:
        command = [
            gdal_translate, "-q", "-of", "GTiff",
            "-co", "TILED=YES", "-co", "BIGTIFF=IF_SAFER",
            "-co", "COMPRESS=LERC_ZSTD",
            "-co", f"MAX_Z_ERROR={max_z_error:.12g}",
            str(source_path), str(partial),
        ]
        subprocess.run(command, check=True)
    else:
        shutil.copy2(source_path, partial)
    validation = validate_pair(source_path, partial, suffix, max_z_error)
    if not validation["passed"]:
        raise RuntimeError(f"Compaction validation failed for {source_path}: {validation}")
    os.replace(partial, output_path)
    source_bytes = source_path.stat().st_size
    compact_bytes = output_path.stat().st_size
    return {
        "stem": source_path.name.removesuffix(f"_{suffix}.tif"),
        "suffix": suffix,
        "source": str(source_path.resolve()),
        "source_sha256": source_hash,
        "source_bytes": source_bytes,
        "compact": str(output_path.resolve()),
        "compact_sha256": sha256(output_path),
        "compact_bytes": compact_bytes,
        "size_ratio": compact_bytes / source_bytes,
        "elapsed_seconds": time.monotonic() - started,
        **validation,
    }


def compact_shard(
    tile_manifest: Path,
    source_dir: Path,
    output_dir: Path,
    max_z_error: float = 1e-5,
    workers: int = 4,
    gdal_translate: str | None = None,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to replace compact shard: {output_dir}")
    if not 0 < max_z_error <= 1e-3:
        raise ValueError("max_z_error must be in (0, 1e-3]")
    if workers < 1:
        raise ValueError("workers must be positive")
    executable = gdal_translate or shutil.which("gdal_translate")
    if not executable:
        raise FileNotFoundError("gdal_translate is required")
    stems = manifest_stems(tile_manifest)
    indices = {suffix: source_index(source_dir, suffix) for suffix in ALL_SUFFIXES}
    expected = set(stems)
    for suffix, index in indices.items():
        if set(index) != expected:
            raise ValueError(
                f"Source {suffix} set differs from manifest: "
                f"missing={len(expected - set(index))}, extra={len(set(index) - expected)}"
            )
    output_dir.mkdir(parents=True)
    started = time.monotonic()
    jobs = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for stem in stems:
            for suffix in ALL_SUFFIXES:
                source = indices[suffix][stem]
                output = output_dir / source.name
                jobs.append(
                    executor.submit(
                        compact_one, source, output, suffix, max_z_error, executable
                    )
                )
        rows = [future.result() for future in as_completed(jobs)]
    rows.sort(key=lambda row: (row["stem"], row["suffix"]))
    if not all(row["passed"] for row in rows):
        raise RuntimeError("One or more compact outputs failed validation")
    detail = output_dir / "compaction_manifest.csv"
    write_csv_atomic(detail, pd.DataFrame(rows))
    source_bytes = sum(row["source_bytes"] for row in rows)
    compact_bytes = sum(row["compact_bytes"] for row in rows)
    payload = {
        "status": "complete",
        "stage": "native-inference-shard-storage-compaction",
        "tile_manifest": str(tile_manifest.resolve()),
        "tile_manifest_sha256": sha256(tile_manifest),
        "source_dir": str(source_dir.resolve()),
        "expected_tile_count": len(stems),
        "output_raster_count": len(rows),
        "float_compression": "LERC_ZSTD",
        "max_z_error": max_z_error,
        "workers": workers,
        "source_bytes": source_bytes,
        "compact_bytes": compact_bytes,
        "size_ratio": compact_bytes / source_bytes,
        "bytes_saved": source_bytes - compact_bytes,
        "elapsed_seconds": time.monotonic() - started,
        "detail": str(detail.resolve()),
        "detail_sha256": sha256(detail),
        "all_outputs_passed": True,
        "guardrail": (
            "The source shard is retained. Removing a QA-approved source shard is a separate, "
            "explicit retention-policy action."
        ),
    }
    write_json_atomic(output_dir / "summary.json", payload)
    (output_dir / "COMPLETED").write_text("complete\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-manifest", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-z-error", type=float, default=1e-5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gdal-translate", default=None)
    args = parser.parse_args()
    payload = compact_shard(
        args.tile_manifest,
        args.source_dir,
        args.output_dir,
        max_z_error=args.max_z_error,
        workers=args.workers,
        gdal_translate=args.gdal_translate,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
