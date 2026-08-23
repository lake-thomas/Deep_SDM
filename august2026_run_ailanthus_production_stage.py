#!/usr/bin/env python3
"""Run the replacement Ailanthus production fit and its complete acceptance gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from august2026_ailanthus_production_train import DATE_TAG, write_design
from august2026_experiments import read_json, run_experiments, sha256
from august2026_production_train import PRODUCTION_OUTPUT_ROOT
from august2026_run_ailanthus_replacement_acceptance import run_acceptance


def validate_blockcv_handoff(
    production_selection: Path, blockcv_evidence_dir: Path
) -> None:
    selection = read_json(production_selection)
    expected_dir = selection.get("blockcv_evidence_dir")
    if expected_dir is None or Path(expected_dir).resolve() != blockcv_evidence_dir.resolve():
        raise ValueError("BlockCV evidence directory differs from production selection")
    summary = blockcv_evidence_dir / "summary.json"
    if not summary.is_file():
        raise FileNotFoundError(summary)
    if selection.get("blockcv_evidence_summary_sha256") != sha256(summary):
        raise ValueError("BlockCV evidence summary identity changed")


def run_stage(
    production_selection: Path,
    blockcv_evidence_dir: Path,
    device: str,
    batch_size: int,
    resume_completed: bool,
) -> dict:
    validate_blockcv_handoff(production_selection, blockcv_evidence_dir)
    registry, rows = write_design(production_selection)
    run_experiments(registry, rows, max_runs=None)
    selection_hash = sha256(production_selection)
    output_root = (
        PRODUCTION_OUTPUT_ROOT
        / f"production_ailanthus_replacement_acceptance_"
        f"{selection_hash[:10]}_{DATE_TAG}"
    )
    return run_acceptance(
        registry,
        blockcv_evidence_dir,
        output_root,
        device,
        batch_size,
        resume_completed,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-selection", type=Path, required=True)
    parser.add_argument("--blockcv-evidence-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--resume-completed", action="store_true")
    args = parser.parse_args()
    if not args.production_selection.is_file():
        parser.error(f"production selection does not exist: {args.production_selection}")
    if not args.blockcv_evidence_dir.is_dir():
        parser.error(f"BlockCV evidence does not exist: {args.blockcv_evidence_dir}")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


def main() -> None:
    args = parse_args()
    print(json.dumps(run_stage(
        args.production_selection,
        args.blockcv_evidence_dir,
        args.device,
        args.batch_size,
        args.resume_completed,
    ), indent=2))


if __name__ == "__main__":
    main()
