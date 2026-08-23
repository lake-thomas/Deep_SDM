#!/usr/bin/env python3
"""Project serial DGX production time from completed state-domain smoke shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd


HOSTS = {
    "ailanthus_altissima", "juglans_nigra", "prunus_serotina",
    "vitis_labrusca", "notholithocarpus_densiflorus",
}
PRODUCTS = {"deterministic", "uncertainty"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def line_count(path: Path) -> int:
    rows = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or len(rows) != len(set(rows)):
        raise ValueError(f"Manifest must be nonempty and unique: {path}")
    return len(rows)


def synthesize_projection(
    status_paths: list[Path], region_manifests: dict[str, Path], output_dir: Path
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to replace runtime projection: {output_dir}")
    manifest_counts = {}
    manifest_hashes = {}
    for region, path in region_manifests.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        manifest_counts[region] = line_count(path)
        manifest_hashes[region] = sha256(path)
    records = []
    observed_shards = set()
    for path in status_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        status = json.loads(path.read_text(encoding="utf-8"))
        key = (status.get("host"), status.get("product"), str(status.get("shard_id", "")))
        if key in observed_shards:
            raise ValueError(f"Duplicate smoke shard status: {key}")
        observed_shards.add(key)
        if status.get("status") != "completed" or status.get("returncode") != 0:
            raise RuntimeError(f"Smoke shard did not complete: {path}")
        host, product = status.get("host"), status.get("product")
        region = status.get("region")
        if host not in HOSTS or product not in PRODUCTS or region not in region_manifests:
            raise ValueError(f"Unknown smoke identity: {host}:{region}:{product}")
        tile_list = Path(status.get("tile_list", ""))
        if not tile_list.is_file() or sha256(tile_list) != status.get("tile_list_sha256"):
            raise ValueError(f"Smoke tile list changed: {path}")
        tile_count = int(status.get("expected_tile_count", -1))
        if tile_count != line_count(tile_list):
            raise ValueError(f"Smoke status tile count changed: {path}")
        elapsed = float(status.get("elapsed_seconds", -1))
        if elapsed <= 0:
            raise ValueError(f"Smoke elapsed time must be positive: {path}")
        records.append({
            "host": host, "region": region, "product": product,
            "shard_id": key[2], "smoke_tile_count": tile_count,
            "elapsed_seconds": elapsed, "status_path": str(path.resolve()),
            "status_sha256": sha256(path),
        })
    frame = pd.DataFrame(records)
    expected = {(host, product) for host in HOSTS for product in PRODUCTS}
    observed = set(zip(frame["host"], frame["product"])) if len(frame) else set()
    if observed != expected:
        raise ValueError("Runtime projection requires both products for all five hosts")
    rows = []
    for (host, region, product), group in frame.groupby(
        ["host", "region", "product"], sort=True
    ):
        smoke_tiles = int(group["smoke_tile_count"].sum())
        elapsed = float(group["elapsed_seconds"].sum())
        seconds_per_tile = elapsed / smoke_tiles
        full_tiles = manifest_counts[region]
        projected_seconds = seconds_per_tile * full_tiles
        rows.append({
            "host": host, "region": region, "product": product,
            "smoke_shard_count": len(group), "smoke_tile_count": smoke_tiles,
            "smoke_elapsed_seconds": elapsed,
            "observed_seconds_per_tile": seconds_per_tile,
            "full_manifest_tile_count": full_tiles,
            "projected_seconds": projected_seconds,
            "projected_hours": projected_seconds / 3600.0,
            "projected_days": projected_seconds / 86400.0,
        })
    projection = pd.DataFrame(rows).sort_values(["product", "host"])
    output_dir.mkdir(parents=True)
    detail_path = output_dir / "runtime_projection_by_host_product.csv"
    partial = detail_path.with_name(f"{detail_path.name}.partial")
    projection.to_csv(partial, index=False)
    os.replace(partial, detail_path)
    total = float(projection["projected_seconds"].sum())
    payload = {
        "status": "complete",
        "stage": "local-dgx-full-inference-runtime-projection",
        "host_count": len(HOSTS),
        "product_count": len(PRODUCTS),
        "smoke_status_count": len(status_paths),
        "region_full_manifest_tile_counts": manifest_counts,
        "region_full_manifest_sha256": manifest_hashes,
        "projected_serial_seconds": total,
        "projected_serial_hours": total / 3600.0,
        "projected_serial_days": total / 86400.0,
        "detail": str(detail_path.resolve()),
        "detail_sha256": sha256(detail_path),
        "interpretation": (
            "Conservative first-order projection from state-domain smoke shards. "
            "It includes per-shard model startup and should be compared with early "
            "full-shard throughput before scheduling the remaining queue."
        ),
    }
    summary = output_dir / "summary.json"
    summary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (output_dir / "COMPLETED").write_text("complete\n", encoding="utf-8")
    return payload


def parse_region_manifest(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError("--region-manifest must be REGION=PATH")
    region, path = raw.split("=", 1)
    if not region or not path:
        raise ValueError("--region-manifest must be REGION=PATH")
    return region, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", type=Path, action="append", required=True)
    parser.add_argument("--region-manifest", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    pairs = [parse_region_manifest(raw) for raw in args.region_manifest]
    manifests = dict(pairs)
    if len(manifests) != len(pairs):
        raise ValueError("Duplicate --region-manifest region")
    print(json.dumps(synthesize_projection(args.status, manifests, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
