#!/usr/bin/env python3
"""Run the frozen evidence chain for the Ailanthus modality-dropout screen."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from august2026_ailanthus_robustness_evidence import synthesize_evidence


PYTHON = Path("/home/ncsucnr/miniforge3/envs/host-naip-sdm/bin/python")
CODE_ROOT = Path(__file__).resolve().parent
EXPECTED_CANDIDATE_COUNT = 3
EXPECTED_STAGE = "ailanthus-image-modality-dropout-screen"


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=CODE_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--resume-completed", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if not args.registry.is_file():
        parser.error(f"registry does not exist: {args.registry}")
    return args


def main() -> None:
    args = parse_args()
    matched_root = args.output_root / "matched_boundary_raw"
    exact_root = args.output_root / "exact_overlap_raw"
    evidence_root = args.output_root / "evidence"
    resume = ["--resume-completed"] if args.resume_completed else []

    run([
        str(PYTHON), "august2026_summarize_experiments.py",
        "--registry", str(args.registry),
    ])
    run([
        str(PYTHON), "august2026_batch_artifact_probe.py",
        "--registry", str(args.registry),
        "--host", "ailanthus_altissima",
        "--output-root", str(matched_root),
        "--device", args.device,
        "--batch-size", str(args.batch_size),
        *resume,
    ])
    run([
        str(PYTHON), "august2026_exact_overlap_probe.py",
        "--registry", str(args.registry),
        "--host", "ailanthus_altissima",
        "--output-root", str(exact_root),
        "--device", args.device,
        "--batch-size", str(args.batch_size),
        *resume,
    ])
    synthesize_evidence(
        args.registry,
        matched_root,
        exact_root,
        evidence_root,
        expected_candidate_count=EXPECTED_CANDIDATE_COUNT,
        expected_stage=EXPECTED_STAGE,
        evidence_filename="ailanthus_modality_dropout_candidate_evidence.csv",
    )


if __name__ == "__main__":
    main()
