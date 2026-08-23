#!/usr/bin/env python3
"""Synthesize the gated Ailanthus robustness-screen evidence package."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from august2026_dataset_screen_evidence import (
    load_artifact_rows,
    load_exact_overlap_rows,
    pareto_nondominated,
)
from august2026_loss_screen_evidence import strict_evidence_gate
from august2026_summarize_experiments import read_json, sha256, write_json_atomic


HOST = "ailanthus_altissima"
EXPECTED_CANDIDATE_COUNT = 5
MATCHED_BOUNDARY_MAXIMUM = 0.09378977404583405
EXACT_OVERLAP_MAXIMUM = 0.4089618235671861
MINIMUM_UNIFORM_AUC = 0.95
PARETO_OBJECTIVES = {
    "test_roc_auc_all_thresholds": "max",
    "test_brier_score": "min",
    "artifact_candidate_absolute_did_mean": "min",
    "exact_overlap_candidate_absolute_shift_mean": "min",
}


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    frame.to_csv(partial, index=False)
    os.replace(partial, path)


def add_advancement_gates(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the frozen screen gates without selecting a candidate."""
    output = frame.copy()
    output["matched_boundary_maximum"] = MATCHED_BOUNDARY_MAXIMUM
    output["exact_overlap_maximum"] = EXACT_OVERLAP_MAXIMUM
    output["minimum_uniform_auc"] = MINIMUM_UNIFORM_AUC
    output["matched_boundary_gate_passed"] = (
        pd.to_numeric(output["artifact_candidate_absolute_did_mean"], errors="coerce")
        <= MATCHED_BOUNDARY_MAXIMUM
    )
    output["exact_overlap_gate_passed"] = (
        pd.to_numeric(
            output["exact_overlap_candidate_absolute_shift_mean"], errors="coerce"
        )
        <= EXACT_OVERLAP_MAXIMUM
    )
    output["uniform_auc_gate_passed"] = (
        pd.to_numeric(output["test_roc_auc_all_thresholds"], errors="coerce")
        >= MINIMUM_UNIFORM_AUC
    )
    calibration_columns = [
        "test_calibration_binary_log_loss",
        "test_calibration_calibration_slope",
        "test_calibration_calibration_intercept",
        "test_calibration_expected_calibration_error",
    ]
    calibration = output[calibration_columns].apply(pd.to_numeric, errors="coerce")
    output["finite_calibration_gate_passed"] = np.isfinite(calibration).all(axis=1)
    output["advancement_gate_passed"] = output[
        [
            "matched_boundary_gate_passed",
            "exact_overlap_gate_passed",
            "uniform_auc_gate_passed",
            "finite_calibration_gate_passed",
        ]
    ].all(axis=1)
    output["pareto_nondominated_within_screen"] = pareto_nondominated(
        output, PARETO_OBJECTIVES
    )
    return output


def synthesize_evidence(
    registry: Path,
    matched_root: Path,
    exact_overlap_root: Path,
    output_dir: Path,
    expected_candidate_count: int = EXPECTED_CANDIDATE_COUNT,
    expected_stage: str = "ailanthus-robustness-screen",
    evidence_filename: str = "ailanthus_robustness_candidate_evidence.csv",
) -> dict:
    metadata_path = registry.parent / "design_metadata.json"
    training_path = registry.parent / "analysis" / "experiment_metrics.csv"
    required = [registry, metadata_path, training_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing robustness-screen inputs: {missing}")
    metadata = read_json(metadata_path)
    if metadata.get("stage") != expected_stage:
        raise RuntimeError(f"Registry is not the frozen {expected_stage} screen")

    training = pd.read_csv(training_path, low_memory=False)
    if (
        len(training) != expected_candidate_count
        or training["experiment"].duplicated().any()
        or set(training["stage"]) != {expected_stage}
        or set(training["host"]) != {HOST}
    ):
        raise ValueError("Ailanthus robustness training matrix is incomplete")
    strict_evidence_gate(training)

    matched = load_artifact_rows([matched_root])
    exact = load_exact_overlap_rows([exact_overlap_root])
    experiments = set(training["experiment"])
    if set(matched["experiment"]) != experiments or set(exact["experiment"]) != experiments:
        raise ValueError("Training and artifact experiment sets differ")
    combined = training.merge(matched, on="experiment", validate="one_to_one")
    combined = combined.merge(exact, on="experiment", validate="one_to_one")
    identity_checks = {
        "matched checkpoint": combined["best_checkpoint_sha256"]
        == combined["artifact_checkpoint_sha256"],
        "matched dataset": combined["training_dataset_csv_sha256"]
        == combined["artifact_dataset_sha256"],
        "exact checkpoint": combined["best_checkpoint_sha256"]
        == combined["exact_overlap_checkpoint_sha256"],
        "exact dataset": combined["training_dataset_csv_sha256"]
        == combined["exact_overlap_dataset_sha256"],
    }
    failed = [name for name, values in identity_checks.items() if not values.all()]
    if failed:
        raise ValueError(f"Robustness evidence identity checks failed: {failed}")
    combined = add_advancement_gates(combined)

    output_dir.mkdir(parents=True, exist_ok=False)
    evidence_path = output_dir / evidence_filename
    write_csv_atomic(evidence_path, combined)
    passing = combined.loc[combined["advancement_gate_passed"], "experiment"].tolist()
    status = {
        "status": "complete",
        "stage": expected_stage,
        "registry": str(registry.resolve()),
        "registry_sha256": sha256(registry),
        "design_metadata": str(metadata_path.resolve()),
        "design_metadata_sha256": sha256(metadata_path),
        "training_metrics": str(training_path.resolve()),
        "training_metrics_sha256": sha256(training_path),
        "matched_artifact_root": str(matched_root.resolve()),
        "exact_overlap_root": str(exact_overlap_root.resolve()),
        "candidate_count": len(combined),
        "advancement_candidate_count": len(passing),
        "advancement_candidates": passing,
        "candidate_evidence": str(evidence_path.resolve()),
        "candidate_evidence_sha256": sha256(evidence_path),
        "gates": {
            "matched_boundary_absolute_did_maximum": MATCHED_BOUNDARY_MAXIMUM,
            "exact_overlap_absolute_shift_maximum": EXACT_OVERLAP_MAXIMUM,
            "minimum_uniform_auc": MINIMUM_UNIFORM_AUC,
            "finite_raw_calibration_metrics_required": True,
        },
        "selection_guardrail": (
            "No recipe is selected automatically. Only advancement-passing candidates "
            "may be considered for Ailanthus-only held-out BlockCV; Pareto status and "
            "recipe simplicity guide the documented human-readable choice."
        ),
    }
    write_json_atomic(output_dir / "summary.json", status)
    (output_dir / "COMPLETED").write_text("completed\n", encoding="utf-8")
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--matched-root", type=Path, required=True)
    parser.add_argument("--exact-overlap-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to replace evidence: {args.output_dir}")
    print(json.dumps(synthesize_evidence(
        args.registry, args.matched_root, args.exact_overlap_root, args.output_dir
    ), indent=2))


if __name__ == "__main__":
    main()
