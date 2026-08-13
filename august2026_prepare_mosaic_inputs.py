#!/usr/bin/env python3
"""Freeze exact, zone-grouped raster inputs for one production mosaic."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from august2026_summarize_experiments import sha256, write_json_atomic


ZONE_PATTERN = re.compile(r"_[ns][ew]_(\d{2})_\d{3}(?:_|$)", re.IGNORECASE)


def index_outputs(output_root: Path, suffix: str) -> dict[str, Path]:
    ending = f"_{suffix}"
    index = {}
    for path in output_root.rglob(f"*_{suffix}.tif"):
        stem = path.stem[: -len(ending)]
        if stem in index:
            raise ValueError(f"Duplicate output stem for {suffix}: {stem}")
        index[stem] = path.resolve()
    return index


def prepare_inputs(
    tile_manifest: Path,
    inference_output_root: Path,
    suffix: str,
    output_dir: Path,
) -> dict:
    tiles = [Path(line.strip()) for line in tile_manifest.read_text().splitlines() if line.strip()]
    if not tiles or len(tiles) != len(set(tiles)):
        raise ValueError("Tile manifest must be nonempty and unique")
    outputs = index_outputs(inference_output_root, suffix)
    expected_stems = {path.stem for path in tiles}
    missing = expected_stems - set(outputs)
    extra = set(outputs) - expected_stems
    if missing or extra:
        raise ValueError(
            f"Mosaic output set differs: missing={len(missing)}, extra={len(extra)}"
        )
    by_zone: dict[str, list[str]] = {}
    for tile in tiles:
        match = ZONE_PATTERN.search(tile.stem)
        if match is None:
            raise ValueError(f"Cannot derive UTM zone from tile name: {tile.name}")
        by_zone.setdefault(match.group(1), []).append(str(outputs[tile.stem]))
    output_dir.mkdir(parents=True, exist_ok=False)
    zone_manifests = {}
    for zone, paths in sorted(by_zone.items()):
        path = output_dir / f"utm{zone}_{suffix}_rasters.txt"
        path.write_text("\n".join(paths) + "\n", encoding="utf-8")
        zone_manifests[zone] = {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "raster_count": len(paths),
        }
    payload = {
        "status": "complete",
        "tile_manifest": str(tile_manifest.resolve()),
        "tile_manifest_sha256": sha256(tile_manifest),
        "inference_output_root": str(inference_output_root.resolve()),
        "suffix": suffix,
        "expected_raster_count": len(tiles),
        "zone_count": len(zone_manifests),
        "zone_manifests": zone_manifests,
    }
    write_json_atomic(output_dir / "mosaic_input_metadata.json", payload)
    (output_dir / "COMPLETED").write_text("completed\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-manifest", type=Path, required=True)
    parser.add_argument("--inference-output-root", type=Path, required=True)
    parser.add_argument("--suffix", choices=["prediction", "uncertainty", "count"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.tile_manifest.is_file():
        raise FileNotFoundError(args.tile_manifest)
    if not args.inference_output_root.is_dir():
        raise FileNotFoundError(args.inference_output_root)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to replace mosaic inputs: {args.output_dir}")
    payload = prepare_inputs(
        args.tile_manifest,
        args.inference_output_root,
        args.suffix,
        args.output_dir,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
