#!/usr/bin/env python3
"""Freeze evidence-backed loss choices for the August augmentation screen."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

from august2026_freeze_dataset_selection import (
    parse_rationale,
    strict_bool,
)
from august2026_summarize_experiments import read_json, sha256, write_json_atomic


LOSS_CHOICES = {"ordinary_bce", "pos_weight_bce", "balanced_batches"}


def parse_choice(value: str) -> tuple[str, str]:
    parts = value.split(":")
    if len(parts) != 2 or parts[1] not in LOSS_CHOICES:
        raise argparse.ArgumentTypeError(
            "choice must be HOST:ordinary_bce|pos_weight_bce|balanced_batches"
        )
    return parts[0], parts[1]


def freeze_loss_selection(
    evidence_dir: Path,
    choices: list[tuple[str, str]],
    rationales: dict[str, str],
    output_path: Path,
    code_commit: str,
) -> dict:
    status_path = evidence_dir / "summary.json"
    combined_path = evidence_dir / "combined_loss_candidate_evidence.csv"
    required = [status_path, combined_path, evidence_dir / "COMPLETED"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Loss selection evidence is incomplete: {missing}")
    status = read_json(status_path)
    if status.get("status") != "complete" or status.get("stage") != "loss-screen":
        raise RuntimeError("Loss evidence package is not complete")
    hosts = [host for host, _ in choices]
    if not choices or len(hosts) != len(set(hosts)):
        raise ValueError("Exactly one loss choice is required per selected host")
    if set(hosts) != set(rationales):
        raise ValueError("Exactly one rationale is required for each selected host")

    evidence = pd.read_csv(combined_path, low_memory=False)
    selections = []
    for host, strategy in choices:
        matched = evidence[
            (evidence["host"] == host) & (evidence["loss_strategy"] == strategy)
        ]
        if len(matched) != 1:
            raise ValueError(f"Loss choice must resolve to one row: {host}:{strategy}")
        row = matched.iloc[0]
        if not strict_bool(row["evidence_gate_passed"], "evidence_gate_passed"):
            raise RuntimeError(f"Training evidence gate failed for {row['experiment']}")
        if not strict_bool(
            row["pareto_nondominated_within_host"],
            "pareto_nondominated_within_host",
        ):
            raise RuntimeError(f"Refusing dominated loss choice: {row['experiment']}")
        config_path = Path(row["config_path"])
        checkpoint_path = Path(row["run_dir"]) / "checkpoints" / "best.pt"
        config = read_json(config_path)
        dataset_path = Path(config["csv_path"])
        paths = [config_path, checkpoint_path, dataset_path]
        if not all(path.is_file() for path in paths):
            raise FileNotFoundError(f"Selected loss artifacts are missing: {paths}")
        recorded = {
            "config": row["config_sha256"],
            "checkpoint": row["best_checkpoint_sha256"],
            "dataset": row["training_dataset_csv_sha256"],
        }
        actual = {
            "config": sha256(config_path),
            "checkpoint": sha256(checkpoint_path),
            "dataset": sha256(dataset_path),
        }
        if actual != recorded:
            raise ValueError(f"Selected loss artifact hashes changed: {host}:{strategy}")
        selections.append(
            {
                "host": host,
                "arm": row["arm"],
                "background_target": int(row["background_target"]),
                "loss_strategy": strategy,
                "source_experiment": row["experiment"],
                "source_config": str(config_path.resolve()),
                "source_config_sha256": actual["config"],
                "source_checkpoint": str(checkpoint_path.resolve()),
                "source_checkpoint_sha256": actual["checkpoint"],
                "dataset": str(dataset_path.resolve()),
                "dataset_sha256": actual["dataset"],
                "image_dir": config["image_dir"],
                "artifact_comparison_summary": row[
                    "artifact_comparison_summary"
                ],
                "artifact_candidate_absolute_did_mean": row[
                    "artifact_candidate_absolute_did_mean"
                ],
                "test_roc_auc_all_thresholds": row["test_roc_auc_all_thresholds"],
                "test_brier_score": row["test_brier_score"],
                **(
                    {
                        "exact_overlap_summary": row["exact_overlap_summary"],
                        "exact_overlap_candidate_absolute_shift_mean": row[
                            "exact_overlap_candidate_absolute_shift_mean"
                        ],
                    }
                    if "exact_overlap_summary" in row.index
                    and pd.notna(row["exact_overlap_summary"])
                    else {}
                ),
                "rationale": rationales[host],
            }
        )
    payload = {
        "status": "complete",
        "selection_type": "loss_for_augmentation_screen",
        "code_commit": code_commit,
        "source_evidence_dir": str(evidence_dir.resolve()),
        "source_evidence_summary": str(status_path.resolve()),
        "source_evidence_summary_sha256": sha256(status_path),
        "source_combined_evidence": str(combined_path.resolve()),
        "source_combined_evidence_sha256": sha256(combined_path),
        "selection_count": len(selections),
        "selections": selections,
        "guardrail": (
            "These choices authorize only the augmentation/regularization factorial. "
            "They do not approve a production recipe or replace BlockCV and final "
            "artifact acceptance."
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
        raise FileExistsError(f"Refusing to replace loss selection: {args.output}")
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
        raise RuntimeError("Refusing to freeze a loss selection from a dirty tree")
    payload = freeze_loss_selection(
        args.evidence_dir, args.select, rationales, args.output, code_commit
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
