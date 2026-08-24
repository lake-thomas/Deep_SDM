#!/usr/bin/env python3
"""Run and synthesize the complete Ailanthus replacement BlockCV stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from august2026_ailanthus_blockcv import DATE_TAG, write_design
from august2026_blockcv_evidence import synthesize_blockcv
from august2026_experiments import OUTPUT_BASE, run_experiments, sha256
from august2026_summarize_experiments import summarize_registry


HOST = "ailanthus_altissima"
EXPECTED_EXPERIMENT_COUNT = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    args = parser.parse_args()
    if not args.selection_manifest.is_file():
        parser.error(f"selection manifest does not exist: {args.selection_manifest}")
    return args


def run_stage(selection_manifest: Path) -> dict:
    registry, rows = write_design(selection_manifest)
    run_experiments(registry, rows, max_runs=None)
    status = summarize_registry(registry, allow_incomplete=False)
    if (
        status["experiment_count"] != EXPECTED_EXPERIMENT_COUNT
        or status["complete_gate_count"] != EXPECTED_EXPERIMENT_COUNT
        or not status["all_experiments_evidence_complete"]
    ):
        raise RuntimeError(
            f"Ailanthus BlockCV training evidence is incomplete: {status}"
        )
    metrics = Path(status["metrics_csv"])
    if not metrics.is_file():
        raise FileNotFoundError(metrics)

    selection_hash = sha256(selection_manifest)
    output_dir = (
        OUTPUT_BASE
        / f"stage05b_ailanthus_blockcv_acceptance_{selection_hash[:10]}_{DATE_TAG}"
        / "blockcv_evidence"
    )
    if output_dir.exists():
        raise FileExistsError(f"Refusing to replace BlockCV evidence: {output_dir}")
    evidence = synthesize_blockcv(
        selection_manifest,
        metrics,
        output_dir,
        expected_hosts={HOST},
        expected_selection_type="ailanthus_recipe_for_blockcv",
    )
    return evidence


def main() -> None:
    args = parse_args()
    print(json.dumps(run_stage(args.selection_manifest), indent=2))


if __name__ == "__main__":
    main()
