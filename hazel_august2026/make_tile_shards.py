#!/usr/bin/env python3
"""Split a frozen tile manifest into deterministic Hazel inference shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=1000)
    args = parser.parse_args()
    if args.shard_size <= 0:
        parser.error("--shard-size must be positive")
    return args


def main() -> None:
    args = parse_args()
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing nonempty shard directory: {args.output_dir}")
    tiles = [line.strip() for line in args.manifest.read_text().splitlines() if line.strip()]
    if not tiles or len(tiles) != len(set(tiles)):
        raise ValueError("Tile manifest must be nonempty and unique")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, start in enumerate(range(0, len(tiles), args.shard_size)):
        shard = args.output_dir / f"shard_{index:05d}.txt"
        shard.write_text("\n".join(tiles[start : start + args.shard_size]) + "\n")
        rows.append(
            {
                "shard_id": f"{index:05d}",
                "tile_count": min(args.shard_size, len(tiles) - start),
                "path": str(shard.resolve()),
                "sha256": sha256(shard),
            }
        )
    payload = {
        "status": "complete",
        "source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": sha256(args.manifest),
        "tile_count": len(tiles),
        "shard_size": args.shard_size,
        "shard_count": len(rows),
        "shards": rows,
    }
    (args.output_dir / "shard_metadata.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "COMPLETED").write_text("completed\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
