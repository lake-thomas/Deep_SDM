#!/usr/bin/env python3
"""Freeze evidence-backed August dataset choices for the loss screen."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from august2026_summarize_experiments import read_json, sha256, write_json_atomic


def parse_choice(value: str) -> tuple[str, str, int]:
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("choice must be HOST:ARM:BACKGROUND_TARGET")
    host, arm, target_text = parts
    if arm not in {"random", "domain_balanced"}:
        raise argparse.ArgumentTypeError("choice arm must be random or domain_balanced")
    try:
        target = int(target_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("choice target must be an integer") from error
    if target <= 0:
        raise argparse.ArgumentTypeError("choice target must be positive")
    return host, arm, target


def parse_rationale(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("rationale must be HOST=TEXT")
    host, text = value.split("=", 1)
    if not host or not text.strip():
        raise argparse.ArgumentTypeError("rationale host and text must be nonempty")
    return host, text.strip()


def strict_bool(value, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"{name} is not boolean: {value}")
    return normalized == "true"


def freeze_selection(
    evidence_dir: Path,
    choices: list[tuple[str, str, int]],
    rationales: dict[str, str],
    output_path: Path,
    code_commit: str,
) -> dict:
    status_path = evidence_dir / "summary.json"
    combined_path = evidence_dir / "combined_candidate_evidence.csv"
    required = [status_path, combined_path, evidence_dir / "COMPLETED"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Dataset selection evidence is incomplete: {missing}")
    evidence_status = read_json(status_path)
    if evidence_status.get("status") != "complete":
        raise RuntimeError("Dataset-screen synthesis is not complete")
    if not choices:
        raise ValueError("At least one dataset choice is required")
    hosts = [choice[0] for choice in choices]
    if len(hosts) != len(set(hosts)):
        raise ValueError("Only one dataset choice is allowed per host")
    if set(hosts) != set(rationales):
        raise ValueError("Exactly one rationale is required for each selected host")

    evidence = pd.read_csv(combined_path, low_memory=False)
    selections = []
    for host, arm, target in choices:
        matched = evidence[
            (evidence["host"] == host)
            & (evidence["arm"] == arm)
            & (pd.to_numeric(evidence["background_target"]) == target)
        ]
        if len(matched) != 1:
            raise ValueError(
                f"Choice must resolve to one evidence row: {host}:{arm}:{target}"
            )
        row = matched.iloc[0]
        if not strict_bool(row["evidence_gate_passed"], "evidence_gate_passed"):
            raise RuntimeError(f"Training evidence gate failed for {row['experiment']}")
        if not strict_bool(
            row["pareto_nondominated_within_host"],
            "pareto_nondominated_within_host",
        ):
            raise RuntimeError(
                f"Refusing dominated dataset choice without a revised design: {row['experiment']}"
            )
        config_path = Path(row["config_path"])
        checkpoint_path = Path(row["run_dir"]) / "checkpoints" / "best.pt"
        if not config_path.is_file() or not checkpoint_path.is_file():
            raise FileNotFoundError(f"Selected run artifacts are missing: {row['experiment']}")
        config = read_json(config_path)
        dataset_path = Path(config["csv_path"])
        if not dataset_path.is_file():
            raise FileNotFoundError(dataset_path)
        expected = {
            "config": (sha256(config_path), row["config_sha256"]),
            "checkpoint": (sha256(checkpoint_path), row["best_checkpoint_sha256"]),
            "dataset": (sha256(dataset_path), row["training_dataset_csv_sha256"]),
        }
        mismatches = [
            name for name, (actual, recorded) in expected.items() if actual != recorded
        ]
        if mismatches:
            raise ValueError(
                f"Selected evidence hashes changed for {row['experiment']}: {mismatches}"
            )
        selections.append(
            {
                "host": host,
                "arm": arm,
                "background_target": target,
                "source_experiment": row["experiment"],
                "source_config": str(config_path.resolve()),
                "source_config_sha256": expected["config"][0],
                "source_checkpoint": str(checkpoint_path.resolve()),
                "source_checkpoint_sha256": expected["checkpoint"][0],
                "dataset": str(dataset_path.resolve()),
                "dataset_sha256": expected["dataset"][0],
                "image_dir": config["image_dir"],
                "artifact_comparison_summary": row["artifact_comparison_summary"],
                "artifact_relative_signed_did_reduction": row[
                    "artifact_relative_signed_did_reduction"
                ],
                "artifact_candidate_absolute_did_mean": row[
                    "artifact_candidate_absolute_did_mean"
                ],
                "test_roc_auc_all_thresholds": row["test_roc_auc_all_thresholds"],
                "test_brier_score": row["test_brier_score"],
                "rationale": rationales[host],
            }
        )
    payload = {
        "status": "complete",
        "selection_type": "dataset_for_loss_screen",
        "code_commit": code_commit,
        "source_evidence_dir": str(evidence_dir.resolve()),
        "source_evidence_summary": str(status_path.resolve()),
        "source_evidence_summary_sha256": sha256(status_path),
        "source_combined_evidence": str(combined_path.resolve()),
        "source_combined_evidence_sha256": sha256(combined_path),
        "selection_count": len(selections),
        "selections": selections,
        "guardrail": (
            "These choices authorize only the isolated loss screen. They do not approve "
            "a production recipe or replace BlockCV and artifact acceptance."
        ),
    }
    write_json_atomic(output_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--select", action="append", type=parse_choice, required=True)
    parser.add_argument("--rationale", action="append", type=parse_rationale, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace selection manifest: {args.output}")
    rationales = dict(args.rationale)
    if len(rationales) != len(args.rationale):
        raise ValueError("Duplicate rationale host")
    code_root = Path(__file__).resolve().parent
    code_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=code_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=code_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("Refusing to freeze a dataset selection from a dirty code tree")
    payload = freeze_selection(
        args.evidence_dir, args.select, rationales, args.output, code_commit
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
