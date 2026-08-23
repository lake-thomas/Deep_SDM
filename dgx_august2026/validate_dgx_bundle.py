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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path)
    args = parser.parse_args()
    payload = validate_bundle(args.bundle_root, args.registry)
    print(json.dumps({"status": "complete", "bundle_type": payload["bundle_type"]}, indent=2))


if __name__ == "__main__":
    main()
