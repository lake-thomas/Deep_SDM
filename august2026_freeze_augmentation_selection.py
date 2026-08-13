#!/usr/bin/env python3
"""Freeze evidence-backed August recipes for five-fold held-out BlockCV."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

from august2026_freeze_dataset_selection import parse_rationale, strict_bool
from august2026_summarize_experiments import read_json, sha256, write_json_atomic


def parse_choice(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("choice must be HOST:EXPERIMENT")
    host, experiment = value.split(":", 1)
    if not host or not experiment:
        raise argparse.ArgumentTypeError("choice host and experiment must be nonempty")
    return host, experiment


def freeze_augmentation_selection(
    evidence_dir: Path,
    choices: list[tuple[str, str]],
    rationales: dict[str, str],
    output_path: Path,
    code_commit: str,
) -> dict:
    status_path = evidence_dir / "summary.json"
    combined_path = evidence_dir / "combined_augmentation_candidate_evidence.csv"
    required = [status_path, combined_path, evidence_dir / "COMPLETED"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Augmentation evidence is incomplete: {missing}")
    status = read_json(status_path)
    if status.get("status") != "complete" or status.get("stage") != "augmentation-screen":
        raise RuntimeError("Augmentation evidence package is not complete")
    hosts = [host for host, _ in choices]
    if not choices or len(hosts) != len(set(hosts)):
        raise ValueError("Exactly one augmentation recipe is required per host")
    if set(hosts) != set(rationales):
        raise ValueError("Exactly one rationale is required per host")

    evidence = pd.read_csv(combined_path, low_memory=False)
    selections = []
    for host, experiment in choices:
        matched = evidence[
            (evidence["host"] == host) & (evidence["experiment"] == experiment)
        ]
        if len(matched) != 1:
            raise ValueError(f"Recipe choice must resolve to one row: {host}:{experiment}")
        row = matched.iloc[0]
        if not strict_bool(row["evidence_gate_passed"], "evidence_gate_passed"):
            raise RuntimeError(f"Training evidence failed for {experiment}")
        if not strict_bool(
            row["pareto_nondominated_within_host"],
            "pareto_nondominated_within_host",
        ):
            raise RuntimeError(f"Refusing dominated augmentation recipe: {experiment}")
        config_path = Path(row["config_path"])
        checkpoint_path = Path(row["run_dir"]) / "checkpoints" / "best.pt"
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
            raise ValueError(f"Selected augmentation artifacts changed: {experiment}")
        spec = config.get("august_experiment_spec") or {}
        selections.append(
            {
                "host": host,
                "arm": row["arm"],
                "background_target": int(row["background_target"]),
                "loss_strategy": row["loss_strategy"],
                "spatial_augmentation": "dihedral",
                "radiometric_augmentation": config.get(
                    "radiometric_augmentation", False
                ),
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
                "artifact_comparison_summary": row[
                    "artifact_comparison_summary"
                ],
                "test_roc_auc_all_thresholds": row["test_roc_auc_all_thresholds"],
                "test_brier_score": row["test_brier_score"],
                **(
                    {"exact_overlap_summary": row["exact_overlap_summary"]}
                    if "exact_overlap_summary" in row.index
                    and pd.notna(row["exact_overlap_summary"])
                    else {}
                ),
                "rationale": rationales[host],
            }
        )
    payload = {
        "status": "complete",
        "selection_type": "recipe_for_blockcv",
        "code_commit": code_commit,
        "source_evidence_dir": str(evidence_dir.resolve()),
        "source_evidence_summary": str(status_path.resolve()),
        "source_evidence_summary_sha256": sha256(status_path),
        "source_combined_evidence": str(combined_path.resolve()),
        "source_combined_evidence_sha256": sha256(combined_path),
        "selection_count": len(selections),
        "selections": selections,
        "guardrail": (
            "These recipes are approved only for five held-out BlockCV fits. Final "
            "production training, calibration choice, inference, and map acceptance "
            "remain separate gates."
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
        raise FileExistsError(f"Refusing to replace BlockCV selection: {args.output}")
    rationales = dict(args.rationale)
    if len(rationales) != len(args.rationale):
        raise ValueError("Duplicate rationale host")
    code_root = Path(__file__).resolve().parent
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=code_root, check=True,
        capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=code_root, check=True,
        capture_output=True, text=True
    ).stdout.strip()
    if dirty:
        raise RuntimeError("Refusing to freeze BlockCV selection from a dirty tree")
    payload = freeze_augmentation_selection(
        args.evidence_dir, args.select, rationales, args.output, commit
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
