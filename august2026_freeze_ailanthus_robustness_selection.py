#!/usr/bin/env python3
"""Freeze one artifact-qualified Ailanthus recipe for replacement BlockCV."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

from august2026_summarize_experiments import read_json, sha256, write_json_atomic


HOST = "ailanthus_altissima"


def strict_bool(value, name: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"{name} is not boolean: {value}")
    return normalized == "true"


def freeze_selection(
    evidence_dir: Path,
    experiment: str,
    rationale: str,
    output_path: Path,
    code_commit: str,
) -> dict:
    summary_path = evidence_dir / "summary.json"
    completed_path = evidence_dir / "COMPLETED"
    required = [summary_path, completed_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Robustness evidence is incomplete: {missing}")
    summary = read_json(summary_path)
    evidence_path = Path(summary.get("candidate_evidence", ""))
    if (
        summary.get("status") != "complete"
        or summary.get("stage") != "ailanthus-robustness-screen"
        or not evidence_path.is_file()
        or sha256(evidence_path) != summary.get("candidate_evidence_sha256")
    ):
        raise RuntimeError("Robustness evidence identity is invalid")
    evidence = pd.read_csv(evidence_path, low_memory=False)
    matched = evidence[evidence["experiment"] == experiment]
    if len(matched) != 1:
        raise ValueError("Ailanthus choice must resolve to exactly one evidence row")
    row = matched.iloc[0]
    for column in (
        "evidence_gate_passed",
        "advancement_gate_passed",
        "pareto_nondominated_within_screen",
    ):
        if not strict_bool(row[column], column):
            raise RuntimeError(f"Selected recipe failed {column}: {experiment}")

    config_path = Path(row["config_path"])
    run_dir = Path(row["run_dir"])
    checkpoint_path = run_dir / "checkpoints" / "best.pt"
    config = read_json(config_path)
    dataset_path = Path(config["csv_path"])
    actual = {
        "config": sha256(config_path),
        "checkpoint": sha256(checkpoint_path),
        "dataset": sha256(dataset_path),
    }
    recorded = {
        "config": row["config_sha256"],
        "checkpoint": row["best_checkpoint_sha256"],
        "dataset": row["training_dataset_csv_sha256"],
    }
    if actual != recorded:
        raise ValueError("Selected Ailanthus training artifacts changed")
    spec = config.get("august_experiment_spec") or {}
    selection = {
        "host": HOST,
        "arm": row["arm"],
        "background_target": int(row["background_target"]),
        "loss_strategy": row["loss_strategy"],
        "spatial_augmentation": row["spatial_augmentation"],
        "radiometric_augmentation": config["radiometric_augmentation"],
        "dropout": float(row["dropout"]),
        "label_smoothing": float(row["label_smoothing"]),
        "seed": int(spec.get("seed", config.get("seed", 42))),
        "source_experiment": experiment,
        "source_config": str(config_path.resolve()),
        "source_config_sha256": actual["config"],
        "source_checkpoint": str(checkpoint_path.resolve()),
        "source_checkpoint_sha256": actual["checkpoint"],
        "uniform_dataset": str(dataset_path.resolve()),
        "uniform_dataset_sha256": actual["dataset"],
        "image_dir": config["image_dir"],
        "artifact_comparison_summary": row["artifact_comparison_summary"],
        "exact_overlap_summary": row["exact_overlap_summary"],
        "test_roc_auc_all_thresholds": float(row["test_roc_auc_all_thresholds"]),
        "test_brier_score": float(row["test_brier_score"]),
        "rationale": rationale,
    }
    payload = {
        "status": "complete",
        "selection_type": "ailanthus_recipe_for_blockcv",
        "code_commit": code_commit,
        "source_evidence_dir": str(evidence_dir.resolve()),
        "source_evidence_summary": str(summary_path.resolve()),
        "source_evidence_summary_sha256": sha256(summary_path),
        "source_candidate_evidence": str(evidence_path.resolve()),
        "source_candidate_evidence_sha256": sha256(evidence_path),
        "selection_count": 1,
        "selections": [selection],
        "guardrail": (
            "This manifest authorizes only five Ailanthus held-out BlockCV fits. "
            "It does not authorize production training or inference and leaves the "
            "four accepted hosts frozen."
        ),
    }
    write_json_atomic(output_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--rationale", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace selection: {args.output}")
    code_root = Path(__file__).resolve().parent
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=code_root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if subprocess.run(
        ["git", "status", "--porcelain"], cwd=code_root, check=True,
        capture_output=True, text=True,
    ).stdout.strip():
        raise RuntimeError("Refusing to freeze selection from a dirty tree")
    print(json.dumps(freeze_selection(
        args.evidence_dir, args.experiment, args.rationale, args.output, commit
    ), indent=2))


if __name__ == "__main__":
    main()
