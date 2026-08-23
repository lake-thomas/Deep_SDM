#!/usr/bin/env python3
"""Validate a frozen local-DGX August inference bundle and registry identity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

BUNDLE_TYPE = "accepted_august2026_dgx_hann_inference"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_bundle(root: Path, registry: Path | None = None) -> dict:
    root = root.resolve()
    metadata_path = root / "bundle_metadata.json"
    if not metadata_path.is_file() or not (root / "COMPLETED").is_file():
        raise FileNotFoundError("DGX bundle lacks metadata or COMPLETED marker")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "complete" or metadata.get("bundle_type") != BUNDLE_TYPE:
        raise RuntimeError("Bundle is not approved for local-DGX August inference")
    hashes = metadata.get("file_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("DGX bundle contains no frozen file hashes")
    for relative, expected in hashes.items():
        candidate = (root / relative).resolve()
        if root not in candidate.parents:
            raise ValueError(f"Bundle path escapes root: {relative}")
        if not candidate.is_file() or sha256(candidate) != expected:
            raise ValueError(f"Frozen DGX bundle file changed: {relative}")
    if registry is not None:
        if not registry.is_file() or sha256(registry) != metadata.get("source_registry_sha256"):
            raise ValueError("DGX production registry differs from frozen bundle")
    return metadata


def validate_inference_qa(
    summary_path: Path,
    tile_manifest: Path,
    host: str,
    region: str,
    product: str,
    expected_tile_count: int,
    expected_base_seed: int,
    expected_mc_samples: int,
    expected_output_root: Path,
) -> dict:
    marker = summary_path.parent / "COMPLETED"
    if not summary_path.is_file() or not marker.is_file():
        raise FileNotFoundError(f"Inference QA package is incomplete: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {
        "status": "complete",
        "production_ready": True,
        "host": host,
        "region": region,
        "product": product,
        "tile_manifest_sha256": sha256(tile_manifest),
        "expected_tile_count": expected_tile_count,
        "base_seed": expected_base_seed,
        "expected_mc_samples": expected_mc_samples,
        "output_root": str(expected_output_root.resolve()),
    }
    observed = {key: summary.get(key) for key in expected}
    if observed != expected:
        raise ValueError(
            f"DGX inference QA identity or gate failed for {product}: "
            f"expected={expected}, observed={observed}"
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--tile-manifest", type=Path)
    parser.add_argument("--host")
    parser.add_argument("--region")
    parser.add_argument("--expected-tile-count", type=int)
    parser.add_argument("--deterministic-summary", type=Path)
    parser.add_argument("--uncertainty-summary", type=Path)
    parser.add_argument("--active-output-root", type=Path)
    args = parser.parse_args()
    payload = validate_bundle(args.bundle_root, args.registry)
    qa_args = (
        args.tile_manifest, args.host, args.region, args.expected_tile_count,
        args.deterministic_summary, args.uncertainty_summary,
        args.active_output_root,
    )
    if any(value is not None for value in qa_args):
        if any(value is None for value in qa_args):
            parser.error("DGX QA-pair validation requires every QA argument")
        if args.expected_tile_count <= 0 or not args.tile_manifest.is_file():
            raise ValueError("QA pair requires a positive count and tile manifest")
        defaults = payload.get("inference_defaults") or {}
        for product, summary in (
            ("deterministic", args.deterministic_summary),
            ("uncertainty", args.uncertainty_summary),
        ):
            validate_inference_qa(
                summary, args.tile_manifest, args.host, args.region, product,
                args.expected_tile_count, defaults["base_seed"],
                defaults[f"{product}_mc_samples"],
                args.active_output_root / args.region / args.host / product,
            )
    print(json.dumps({
        "status": "complete", "bundle_type": payload["bundle_type"],
        "qa_pair_validated": any(value is not None for value in qa_args),
    }, indent=2))


if __name__ == "__main__":
    main()
