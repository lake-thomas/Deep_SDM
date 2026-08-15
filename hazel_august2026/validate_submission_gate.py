#!/usr/bin/env python3
"""Validate frozen Hazel bundles and inference-QA gates before submission."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def validate_bundle(bundle_root: Path, registry: Path | None = None) -> dict:
    root = bundle_root.resolve()
    metadata_path = root / "bundle_metadata.json"
    marker = root / "COMPLETED"
    if not metadata_path.is_file() or not marker.is_file():
        raise FileNotFoundError("Frozen Hazel bundle lacks metadata or COMPLETED marker")
    metadata = read_json(metadata_path)
    if (
        metadata.get("status") != "complete"
        or metadata.get("bundle_type")
        != "accepted_august2026_hazel_hann_inference"
    ):
        raise RuntimeError("Hazel bundle metadata is not approved for August inference")
    hashes = metadata.get("file_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Hazel bundle contains no frozen file hashes")
    for relative, expected in hashes.items():
        candidate = (root / relative).resolve()
        if root not in candidate.parents:
            raise ValueError(f"Bundle hash path escapes the bundle: {relative}")
        if not candidate.is_file() or sha256(candidate) != expected:
            raise ValueError(f"Frozen bundle file changed: {relative}")
    if registry is not None:
        if not registry.is_file():
            raise FileNotFoundError(registry)
        expected_registry = metadata.get("source_registry_sha256")
        if not expected_registry or sha256(registry) != expected_registry:
            raise ValueError("Submitted production registry differs from frozen bundle")
        bundled_registry = root / "production_experiment_registry.tsv"
        if (
            not bundled_registry.is_file()
            or sha256(bundled_registry) != expected_registry
        ):
            raise ValueError("Bundled production registry differs from source registry")
    return metadata


def validate_inference_qa(
    summary_path: Path,
    tile_manifest: Path,
    host: str,
    region: str,
    product: str,
    expected_tile_count: int,
) -> dict:
    marker = summary_path.parent / "COMPLETED"
    if not summary_path.is_file() or not marker.is_file():
        raise FileNotFoundError(f"Inference QA package is incomplete: {summary_path}")
    summary = read_json(summary_path)
    expected = {
        "status": "complete",
        "production_ready": True,
        "host": host,
        "region": region,
        "product": product,
        "tile_manifest_sha256": sha256(tile_manifest),
        "expected_tile_count": expected_tile_count,
    }
    observed = {key: summary.get(key) for key in expected}
    if observed != expected:
        raise ValueError(
            f"Inference QA identity or gate failed for {product}: "
            f"expected={expected}, observed={observed}"
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    bundle = subparsers.add_parser("bundle")
    bundle.add_argument("--bundle-root", type=Path, required=True)
    bundle.add_argument("--registry", type=Path)
    qa = subparsers.add_parser("qa-pair")
    qa.add_argument("--bundle-root", type=Path, required=True)
    qa.add_argument("--tile-manifest", type=Path, required=True)
    qa.add_argument("--host", required=True)
    qa.add_argument("--region", required=True)
    qa.add_argument("--expected-tile-count", type=int, required=True)
    qa.add_argument("--deterministic-summary", type=Path, required=True)
    qa.add_argument("--uncertainty-summary", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_bundle(args.bundle_root, getattr(args, "registry", None))
    if args.command == "qa-pair":
        if args.expected_tile_count <= 0 or not args.tile_manifest.is_file():
            raise ValueError("QA pair requires a positive count and tile manifest")
        validate_inference_qa(
            args.deterministic_summary,
            args.tile_manifest,
            args.host,
            args.region,
            "deterministic",
            args.expected_tile_count,
        )
        validate_inference_qa(
            args.uncertainty_summary,
            args.tile_manifest,
            args.host,
            args.region,
            "uncertainty",
            args.expected_tile_count,
        )
    print(json.dumps({"status": "complete", "command": args.command}, indent=2))


if __name__ == "__main__":
    main()
