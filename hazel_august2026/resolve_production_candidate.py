#!/usr/bin/env python3
"""Resolve one accepted August production candidate on Hazel."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
from pathlib import Path


LOCAL_PREFIX = "/mnt/rsstu_dsfas_gsv_naip/Host_NAIP_SDM"
DEFAULT_HPC_ROOT = Path("/rsstu/users/r/rkmeente/DSFAS_GSV_NAIP/Host_NAIP_SDM")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remap(path: str | Path, hpc_root: Path) -> Path:
    text = str(path)
    return (
        hpc_root / text.removeprefix(LOCAL_PREFIX).lstrip("/")
        if text.startswith(LOCAL_PREFIX)
        else Path(text)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--hpc-root", type=Path, default=DEFAULT_HPC_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.registry.open(newline="", encoding="utf-8") as stream:
        matches = [
            row for row in csv.DictReader(stream, delimiter="\t")
            if row["host"] == args.host
        ]
    if len(matches) != 1:
        raise ValueError(f"Expected one production registry row for {args.host}")
    row = matches[0]
    run_dir = remap(row["run_dir"], args.hpc_root)
    checkpoint = run_dir / "checkpoints" / "best.pt"
    config = run_dir / "config.input.json"
    for path in (checkpoint, config):
        if not path.is_file():
            raise FileNotFoundError(path)
    config_payload = json.loads(config.read_text(encoding="utf-8"))
    calibration = str(config_payload.get("postfit_calibration", "none")).lower()
    calibrator = run_dir / "platt_calibrator.json" if calibration == "platt" else None
    if calibrator is not None and not calibrator.is_file():
        raise FileNotFoundError(calibrator)
    print(f"CHECKPOINT_PATH={shlex.quote(str(checkpoint))}")
    print(f"CONFIG_PATH={shlex.quote(str(config))}")
    print(
        "CALIBRATOR_PATH="
        + shlex.quote(str(calibrator) if calibrator is not None else "")
    )
    print(f"EXPERIMENT={shlex.quote(row['experiment'])}")
    print(f"SCORE_SPACE={shlex.quote(row['approved_score_space'])}")
    print(
        "CALIBRATOR_SHA256="
        + shlex.quote(sha256(calibrator) if calibrator is not None else "")
    )


if __name__ == "__main__":
    main()
