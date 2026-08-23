#!/usr/bin/env python3
"""Synthesize standard BlockCV evidence for the Ailanthus replacement only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from august2026_blockcv_evidence import synthesize_blockcv


HOST = "ailanthus_altissima"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--training-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.selection_manifest, args.training_metrics):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to replace evidence: {args.output_dir}")
    status = synthesize_blockcv(
        args.selection_manifest,
        args.training_metrics,
        args.output_dir,
        expected_hosts={HOST},
        expected_selection_type="ailanthus_recipe_for_blockcv",
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
