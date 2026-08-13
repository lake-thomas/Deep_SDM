#!/usr/bin/env python3
"""Build frozen full and smoke NAIP tile manifests on Hazel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DEFAULT_ARCHIVE_ROOT = Path(
    "/rsstu/users/r/rkmeente/DSFAS_GSV_NAIP/Host_NAIP_SDM/NAIP_Archive"
)
STATE_YEARS = {
    "east_of_mississippi": {
        "CT": "2021", "ME": "2021", "MA": "2021", "NH": "2021",
        "RI": "2021", "VT": "2021", "DE": "2021", "MD": "2021",
        "NJ": "2019", "NY": "2021", "PA": "2022", "VA": "2021",
        "AL": "2021", "FL": "2021", "GA": "2021", "MS": "2021",
        "NC": "2022", "SC": "2021", "TN": "2021", "WV": "2022",
        "IL": "2021", "IN": "2020", "KY": "2020", "MI": "2022",
        "OH": "2021", "WI": "2020", "MN": "2021",
    },
    "tanoak_ca_or": {"CA": "2022", "OR": "2020"},
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", choices=sorted(STATE_YEARS), required=True)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke-per-state", type=int, default=2)
    args = parser.parse_args()
    if args.smoke_per_state <= 0:
        parser.error("--smoke-per-state must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing nonempty manifest directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    full, smoke, state_counts, missing = [], [], {}, []
    for state, year in STATE_YEARS[args.region].items():
        raw_dir = args.archive_root / "states" / state / year / "raw"
        tiles = sorted(raw_dir.glob("*.tif"))
        state_counts[state] = len(tiles)
        if not tiles:
            missing.append(str(raw_dir))
            continue
        full.extend(str(path.resolve()) for path in tiles)
        smoke.extend(str(path.resolve()) for path in tiles[: args.smoke_per_state])
    if missing:
        raise FileNotFoundError(f"Required archive state/year directories are empty: {missing}")
    full_path = args.output_dir / f"{args.region}_all_tiles.txt"
    smoke_path = args.output_dir / f"{args.region}_smoke_tiles.txt"
    full_path.write_text("\n".join(full) + "\n", encoding="utf-8")
    smoke_path.write_text("\n".join(smoke) + "\n", encoding="utf-8")
    payload = {
        "status": "complete",
        "region": args.region,
        "state_years": STATE_YEARS[args.region],
        "state_tile_counts": state_counts,
        "full_tile_count": len(full),
        "smoke_tile_count": len(smoke),
        "full_manifest": str(full_path.resolve()),
        "full_manifest_sha256": file_sha256(full_path),
        "smoke_manifest": str(smoke_path.resolve()),
        "smoke_manifest_sha256": file_sha256(smoke_path),
    }
    (args.output_dir / "manifest_metadata.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "COMPLETED").write_text("completed\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
