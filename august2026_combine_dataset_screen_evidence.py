#!/usr/bin/env python3
"""Combine disjoint Vitis/Tanoak and control dataset-screen evidence packages."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from august2026_summarize_experiments import read_json, sha256, write_json_atomic


HOSTS = frozenset({
    "ailanthus_altissima",
    "juglans_nigra",
    "prunus_serotina",
    "vitis_labrusca",
    "notholithocarpus_densiflorus",
})


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    partial = path.with_name(f"{path.name}.partial")
    frame.to_csv(partial, index=False)
    os.replace(partial, path)


def combine_evidence(source_dirs: list[Path], output_dir: Path) -> dict:
    if len(source_dirs) < 2:
        raise ValueError("At least two dataset-screen evidence packages are required")
    frames = []
    sources = []
    observed_hosts: set[str] = set()
    for source_dir in source_dirs:
        summary_path = source_dir / "summary.json"
        evidence_path = source_dir / "combined_candidate_evidence.csv"
        completed_path = source_dir / "COMPLETED"
        for path in (summary_path, evidence_path, completed_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        summary = read_json(summary_path)
        if summary.get("status") != "complete":
            raise RuntimeError(f"Dataset-screen evidence is incomplete: {source_dir}")
        frame = pd.read_csv(evidence_path, low_memory=False)
        required_columns = {
            "host", "arm", "background_target", "experiment", "evidence_gate_passed",
            "pareto_nondominated_within_host",
        }
        missing = required_columns - set(frame.columns)
        if frame.empty or missing:
            raise ValueError(f"Invalid dataset evidence {source_dir}: missing={missing}")
        source_hosts = set(frame["host"])
        overlap = observed_hosts & source_hosts
        if overlap:
            raise ValueError(f"Dataset evidence packages overlap hosts: {sorted(overlap)}")
        observed_hosts |= source_hosts
        frames.append(frame)
        sources.append(
            {
                "directory": str(source_dir.resolve()),
                "summary": str(summary_path.resolve()),
                "summary_sha256": sha256(summary_path),
                "evidence": str(evidence_path.resolve()),
                "evidence_sha256": sha256(evidence_path),
                "hosts": sorted(source_hosts),
                "candidate_count": len(frame),
            }
        )
    if observed_hosts != HOSTS:
        raise ValueError(
            f"Combined dataset evidence must contain five hosts; found {sorted(observed_hosts)}"
        )
    combined = pd.concat(frames, ignore_index=True, sort=False)
    if combined["experiment"].duplicated().any():
        raise ValueError("Combined dataset evidence has duplicate experiments")
    combined = combined.sort_values(
        ["host", "background_target", "arm", "experiment"], kind="mergesort"
    ).reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=False)
    combined_path = output_dir / "combined_candidate_evidence.csv"
    write_csv_atomic(combined_path, combined)
    payload = {
        "status": "complete",
        "stage": "five-host-dataset-screen",
        "host_count": len(observed_hosts),
        "hosts": sorted(observed_hosts),
        "candidate_count": len(combined),
        "source_packages": sources,
        "combined_candidate_evidence": str(combined_path.resolve()),
        "combined_candidate_evidence_sha256": sha256(combined_path),
        "selection_guardrail": (
            "This join changes no candidate metric or Pareto flag. It only creates the "
            "single five-host evidence table required for explicit dataset selection."
        ),
    }
    write_json_atomic(output_dir / "summary.json", payload)
    (output_dir / "COMPLETED").write_text("completed\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = combine_evidence(args.source_dir, args.output_dir)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
