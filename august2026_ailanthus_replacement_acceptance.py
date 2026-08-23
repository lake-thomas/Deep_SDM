#!/usr/bin/env python3
"""Gate one replacement Ailanthus production fit before release assembly."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from august2026_dataset_screen_evidence import (
    load_artifact_rows,
    load_exact_overlap_rows,
)
from august2026_final_model_acceptance import artifact_nonregression_gate
from august2026_loss_screen_evidence import strict_evidence_gate
from august2026_summarize_experiments import (
    read_json,
    sha256,
    summarize_registry,
    write_json_atomic,
)


HOST = "ailanthus_altissima"


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    frame.to_csv(partial, index=False)
    os.replace(partial, path)


def synthesize_acceptance(
    registry: Path,
    blockcv_evidence_dir: Path,
    raw_matched_root: Path,
    configured_matched_root: Path,
    raw_exact_root: Path,
    configured_exact_root: Path,
    output_dir: Path,
) -> dict:
    training_status = summarize_registry(registry, allow_incomplete=False)
    training_path = Path(training_status["metrics_csv"])
    training = pd.read_csv(training_path, low_memory=False)
    if (
        len(training) != 1
        or set(training["host"]) != {HOST}
        or set(training["stage"]) != {"production-fit"}
    ):
        raise ValueError("Replacement registry must contain one Ailanthus production fit")
    strict_evidence_gate(training)
    row = training.iloc[0]

    blockcv_path = blockcv_evidence_dir / "summary.json"
    if not blockcv_path.is_file() or not (blockcv_evidence_dir / "COMPLETED").is_file():
        raise FileNotFoundError("Replacement BlockCV evidence is incomplete")
    blockcv = read_json(blockcv_path)
    if (
        blockcv.get("status") != "complete"
        or blockcv.get("production_ready") is not True
        or blockcv.get("host_count") != 1
    ):
        raise RuntimeError("Replacement BlockCV evidence failed")

    config = read_json(Path(row["config_path"]))
    selection_gate = config.get("selection_gate") or {}
    selection_path = Path(selection_gate.get("selection_manifest", ""))
    if (
        not selection_path.is_file()
        or selection_gate.get("selection_manifest_sha256") != sha256(selection_path)
    ):
        raise ValueError("Replacement production selection identity changed")
    selection = read_json(selection_path)
    if (
        selection.get("selection_type")
        != "ailanthus_blockcv_approved_recipe_for_production_fit"
        or selection.get("blockcv_evidence_summary_sha256") != sha256(blockcv_path)
    ):
        raise ValueError("Replacement selection does not reference supplied BlockCV")

    raw_matched = load_artifact_rows([raw_matched_root])
    configured_matched = load_artifact_rows(
        [configured_matched_root], require_raw_score_space=False
    )
    raw_exact = load_exact_overlap_rows([raw_exact_root])
    configured_exact = load_exact_overlap_rows(
        [configured_exact_root], require_raw_score_space=False
    )
    frames = [raw_matched, configured_matched, raw_exact, configured_exact]
    experiment = row["experiment"]
    if any(len(frame) != 1 or set(frame["experiment"]) != {experiment} for frame in frames):
        raise ValueError("Replacement artifact experiment identity differs from training")

    raw = raw_matched.iloc[0]
    configured = configured_matched.iloc[0]
    exact_raw = raw_exact.iloc[0]
    exact_configured = configured_exact.iloc[0]
    checkpoint_hash = row["best_checkpoint_sha256"]
    dataset_hash = row["training_dataset_csv_sha256"]
    failures = []
    for label, evidence_row, checkpoint_column, dataset_column in (
        ("raw_matched", raw, "artifact_checkpoint_sha256", "artifact_dataset_sha256"),
        (
            "configured_matched", configured,
            "artifact_checkpoint_sha256", "artifact_dataset_sha256",
        ),
        (
            "raw_exact", exact_raw,
            "exact_overlap_checkpoint_sha256", "exact_overlap_dataset_sha256",
        ),
        (
            "configured_exact", exact_configured,
            "exact_overlap_checkpoint_sha256", "exact_overlap_dataset_sha256",
        ),
    ):
        if evidence_row[checkpoint_column] != checkpoint_hash:
            failures.append(f"{label}_checkpoint_hash")
        if evidence_row[dataset_column] != dataset_hash:
            failures.append(f"{label}_dataset_hash")

    if raw["artifact_score_space"] != "raw_sigmoid_probability":
        failures.append("raw_matched_score_space")
    if pd.notna(raw["artifact_platt_calibrator_sha256"]):
        failures.append("raw_matched_unexpected_calibrator")
    if exact_raw["exact_overlap_score_space"] != "raw_sigmoid_probability":
        failures.append("raw_exact_score_space")
    if pd.notna(exact_raw["exact_overlap_platt_calibrator_sha256"]):
        failures.append("raw_exact_unexpected_calibrator")

    expected_space = (
        "platt_calibrated_probability"
        if row["postfit_calibration"] == "platt"
        else "raw_sigmoid_probability"
    )
    expected_calibrator = (
        row.get("platt_calibrator_sha256")
        if row["postfit_calibration"] == "platt"
        else None
    )
    if pd.isna(expected_calibrator):
        expected_calibrator = None
    observed_matched_calibrator = configured["artifact_platt_calibrator_sha256"]
    observed_exact_calibrator = configured_exact.iloc[0][
        "exact_overlap_platt_calibrator_sha256"
    ]
    if pd.isna(observed_matched_calibrator):
        observed_matched_calibrator = None
    if pd.isna(observed_exact_calibrator):
        observed_exact_calibrator = None
    if configured["artifact_score_space"] != expected_space:
        failures.append("configured_matched_score_space")
    if exact_configured.iloc[0]["exact_overlap_score_space"] != expected_space:
        failures.append("configured_exact_score_space")
    if observed_matched_calibrator != expected_calibrator:
        failures.append("configured_matched_calibrator_hash")
    if observed_exact_calibrator != expected_calibrator:
        failures.append("configured_exact_calibrator_hash")

    raw_matched_gate = artifact_nonregression_gate(
        HOST,
        raw["artifact_baseline_absolute_did_mean"],
        raw["artifact_candidate_absolute_did_mean"],
    )
    configured_matched_gate = artifact_nonregression_gate(
        HOST,
        configured["artifact_baseline_absolute_did_mean"],
        configured["artifact_candidate_absolute_did_mean"],
    )
    raw_exact_gate = artifact_nonregression_gate(
        HOST,
        exact_raw["exact_overlap_baseline_absolute_shift_mean"],
        exact_raw["exact_overlap_candidate_absolute_shift_mean"],
    )
    configured_exact_gate = artifact_nonregression_gate(
        HOST,
        exact_configured.iloc[0]["exact_overlap_baseline_absolute_shift_mean"],
        exact_configured.iloc[0]["exact_overlap_candidate_absolute_shift_mean"],
    )
    for label, gate in (
        ("raw_matched_boundary_regression", raw_matched_gate),
        ("configured_matched_boundary_regression", configured_matched_gate),
        ("raw_exact_overlap_regression", raw_exact_gate),
        ("configured_exact_overlap_regression", configured_exact_gate),
    ):
        if not gate["passed"]:
            failures.append(label)

    record = {
        "host": HOST,
        "experiment": experiment,
        "postfit_calibration": row["postfit_calibration"],
        "expected_release_score_space": expected_space,
        "raw_matched_candidate": raw["artifact_candidate_absolute_did_mean"],
        "raw_matched_allowed_maximum": raw_matched_gate["allowed_maximum"],
        "configured_matched_candidate": configured[
            "artifact_candidate_absolute_did_mean"
        ],
        "configured_matched_allowed_maximum": configured_matched_gate[
            "allowed_maximum"
        ],
        "raw_exact_candidate": exact_raw[
            "exact_overlap_candidate_absolute_shift_mean"
        ],
        "raw_exact_allowed_maximum": raw_exact_gate["allowed_maximum"],
        "configured_exact_candidate": exact_configured.iloc[0][
            "exact_overlap_candidate_absolute_shift_mean"
        ],
        "configured_exact_allowed_maximum": configured_exact_gate["allowed_maximum"],
        "acceptance_passed": not failures,
        "failures": ";".join(failures),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    host_path = output_dir / "ailanthus_replacement_acceptance.csv"
    write_csv_atomic(host_path, pd.DataFrame([record]))
    status = {
        "status": "complete",
        "stage": "ailanthus-replacement-final-acceptance",
        "production_ready_for_release_assembly": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "registry": str(registry.resolve()),
        "registry_sha256": sha256(registry),
        "training_metrics": str(training_path.resolve()),
        "training_metrics_sha256": sha256(training_path),
        "blockcv_evidence_dir": str(blockcv_evidence_dir.resolve()),
        "blockcv_summary_sha256": sha256(blockcv_path),
        "production_selection_manifest": str(selection_path.resolve()),
        "production_selection_manifest_sha256": sha256(selection_path),
        "raw_matched_root": str(raw_matched_root.resolve()),
        "configured_matched_root": str(configured_matched_root.resolve()),
        "raw_exact_root": str(raw_exact_root.resolve()),
        "configured_exact_root": str(configured_exact_root.resolve()),
        "host_acceptance": str(host_path.resolve()),
        "host_acceptance_sha256": sha256(host_path),
        "guardrail": (
            "Passing authorizes assembly with the four already accepted August fits. "
            "Five-host pathway/correlation review, Hann inference, raster QA, and "
            "visual acceptance remain required."
        ),
    }
    write_json_atomic(output_dir / "summary.json", status)
    (output_dir / "COMPLETED").write_text(
        "completed\n" if not failures else "completed_with_failures\n",
        encoding="utf-8",
    )
    if failures:
        raise RuntimeError(f"Ailanthus replacement acceptance failed: {failures}")
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--blockcv-evidence-dir", type=Path, required=True)
    parser.add_argument("--raw-matched-root", type=Path, required=True)
    parser.add_argument("--configured-matched-root", type=Path, required=True)
    parser.add_argument("--raw-exact-root", type=Path, required=True)
    parser.add_argument("--configured-exact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to replace acceptance: {args.output_dir}")
    print(json.dumps(synthesize_acceptance(
        args.registry,
        args.blockcv_evidence_dir,
        args.raw_matched_root,
        args.configured_matched_root,
        args.raw_exact_root,
        args.configured_exact_root,
        args.output_dir,
    ), indent=2))


if __name__ == "__main__":
    main()
