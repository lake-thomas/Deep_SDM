#!/usr/bin/env python3
"""Validate and summarize one frozen August experiment registry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    with partial.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    os.replace(partial, path)


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    frame.to_csv(partial, index=False)
    os.replace(partial, path)


def _prefixed(payload: dict, prefix: str) -> dict:
    return {f"{prefix}_{key}": value for key, value in payload.items()}


def summarize_row(row: dict, design_commit: str | None) -> dict:
    run_dir = Path(row["run_dir"])
    config_path = Path(row["config_path"])
    config = read_json(config_path)
    postfit_calibration = str(config.get("postfit_calibration", "none")).lower()
    result = dict(row)
    result["run_complete"] = False
    result["evidence_gate_passed"] = False
    result["evidence_gate_failures"] = ""
    failures = []

    required = {
        "run_status": run_dir / "run_status.json",
        "runtime_provenance": run_dir / "runtime_provenance.json",
        "training_design": run_dir / "training_design.json",
        "training_history": run_dir / "training_history.csv",
        "validation_calibration": run_dir / "validation_raw_calibration_metrics.json",
        "test_calibration": run_dir / "test_raw_calibration_metrics.json",
        "validation_threshold": run_dir / "validation_threshold.json",
        "best_checkpoint": run_dir / "checkpoints" / "best.pt",
    }
    if postfit_calibration != "platt":
        required["test_metrics"] = run_dir / "model_summary_metrics.csv"
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        result["evidence_gate_failures"] = "missing:" + ",".join(missing)
        return result

    run_status = read_json(required["run_status"])
    runtime = read_json(required["runtime_provenance"])
    training_design = read_json(required["training_design"])
    if run_status.get("status") != "completed":
        failures.append(f"run_status={run_status.get('status')}")
    else:
        result["run_complete"] = True
    if runtime.get("git_dirty") is not False:
        failures.append(f"runtime_git_dirty={runtime.get('git_dirty')}")
    if design_commit and runtime.get("git_commit") != design_commit:
        failures.append("runtime_commit_mismatch")
    if runtime.get("config_sha256") != sha256(config_path):
        failures.append("config_hash_mismatch")
    dataset_path = Path(config["csv_path"])
    if training_design.get("dataset_csv_sha256") != sha256(dataset_path):
        failures.append("dataset_hash_mismatch")

    history = pd.read_csv(required["training_history"])
    if history.empty or "val_loss" not in history:
        failures.append("invalid_training_history")
    else:
        best_index = history["val_loss"].astype(float).idxmin()
        result["epochs_completed"] = len(history)
        result["best_epoch_zero_based"] = int(best_index)
        result["best_epoch_one_based"] = int(best_index) + 1
        result["best_val_loss"] = float(history.loc[best_index, "val_loss"])
        result["final_val_loss"] = float(history.iloc[-1]["val_loss"])

    if postfit_calibration != "platt":
        test_metrics = pd.read_csv(required["test_metrics"])
        if len(test_metrics) != 1:
            failures.append("invalid_test_metrics")
        else:
            result.update(_prefixed(test_metrics.iloc[0].to_dict(), "test"))
    result.update(
        _prefixed(read_json(required["validation_calibration"]), "validation_calibration")
    )
    result.update(_prefixed(read_json(required["test_calibration"]), "test_calibration"))
    result.update(_prefixed(read_json(required["validation_threshold"]), "threshold"))
    result.update(_prefixed(training_design, "training"))
    result["postfit_calibration"] = postfit_calibration
    if postfit_calibration == "platt":
        calibrated_required = {
            "platt_calibrator": run_dir / "platt_calibrator.json",
            "test_platt_metrics": run_dir / "test_metrics_platt.json",
            "test_platt_calibration": run_dir / "test_platt_calibration_metrics.json",
            "test_platt_predictions": run_dir / "prediction_results_platt.csv",
        }
        missing_calibrated = [
            name for name, path in calibrated_required.items() if not path.is_file()
        ]
        if missing_calibrated:
            failures.append("missing_platt:" + ",".join(missing_calibrated))
        else:
            configured_test_metrics = read_json(
                calibrated_required["test_platt_metrics"]
            )
            if not isinstance(configured_test_metrics, dict) or not configured_test_metrics:
                failures.append("invalid_test_metrics")
            else:
                result.update(_prefixed(configured_test_metrics, "test"))
                result.update(_prefixed(configured_test_metrics, "test_platt"))
            result.update(
                _prefixed(
                    read_json(calibrated_required["platt_calibrator"]),
                    "platt_calibrator",
                )
            )
            result.update(
                _prefixed(
                    read_json(calibrated_required["test_platt_calibration"]),
                    "test_platt_calibration",
                )
            )
            result["platt_calibrator_sha256"] = sha256(
                calibrated_required["platt_calibrator"]
            )
            result["test_platt_predictions_sha256"] = sha256(
                calibrated_required["test_platt_predictions"]
            )
    elif postfit_calibration != "none":
        failures.append(f"unknown_postfit_calibration={postfit_calibration}")
    result["best_checkpoint_sha256"] = sha256(required["best_checkpoint"])
    result["evidence_gate_passed"] = not failures
    result["evidence_gate_failures"] = ";".join(failures)
    return result


DELTA_METRICS = [
    "best_val_loss",
    "test_roc_auc_all_thresholds",
    "test_average_precision_all_thresholds",
    "test_brier_score",
    "test_mcc_at_threshold",
    "test_calibration_binary_log_loss",
    "test_calibration_calibration_slope",
    "test_calibration_expected_calibration_error",
]


def arm_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    complete = summary[summary["evidence_gate_passed"] == True].copy()  # noqa: E712
    rows = []
    group_columns = ["stage", "host", "background_target", "split_design"]
    for keys, group in complete.groupby(group_columns, dropna=False):
        by_arm = group.set_index("arm")
        if not {"random", "domain_balanced"}.issubset(by_arm.index):
            continue
        record = dict(zip(group_columns, keys))
        record["interpretation"] = (
            "domain_balanced minus random; descriptive because background rows differ"
        )
        for metric in DELTA_METRICS:
            if metric not in by_arm:
                continue
            random_value = float(by_arm.loc["random", metric])
            balanced_value = float(by_arm.loc["domain_balanced", metric])
            record[f"random_{metric}"] = random_value
            record[f"domain_balanced_{metric}"] = balanced_value
            record[f"delta_{metric}"] = balanced_value - random_value
        rows.append(record)
    return pd.DataFrame(rows)


def summarize_registry(registry: Path, allow_incomplete: bool = False) -> dict:
    design_metadata_path = registry.parent / "design_metadata.json"
    design_metadata = read_json(design_metadata_path) if design_metadata_path.is_file() else {}
    design_commit = design_metadata.get("code_commit")
    registry_frame = pd.read_csv(registry, sep="\t", dtype={"background_target": int})
    rows = [summarize_row(row, design_commit) for row in registry_frame.to_dict("records")]
    summary = pd.DataFrame(rows)
    incomplete = summary[summary["evidence_gate_passed"] != True]  # noqa: E712
    if not allow_incomplete and not incomplete.empty:
        details = incomplete[["experiment", "evidence_gate_failures"]].to_dict("records")
        raise RuntimeError(f"Experiment evidence gate is incomplete: {details}")

    analysis_root = registry.parent / "analysis"
    write_csv_atomic(analysis_root / "experiment_metrics.csv", summary)
    deltas = arm_deltas(summary)
    write_csv_atomic(analysis_root / "domain_balanced_minus_random.csv", deltas)
    status = {
        "registry": str(registry),
        "experiment_count": len(summary),
        "complete_gate_count": int(summary["evidence_gate_passed"].sum()),
        "incomplete_gate_count": len(incomplete),
        "all_experiments_evidence_complete": incomplete.empty,
        "metrics_csv": str(analysis_root / "experiment_metrics.csv"),
        "arm_delta_csv": str(analysis_root / "domain_balanced_minus_random.csv"),
        "selection_guardrail": (
            "No production candidate is selected here. Uniform-split fit metrics must "
            "be combined with bounded seam diagnostics and held-out BlockCV."
        ),
    }
    write_json_atomic(analysis_root / "summary_status.json", status)
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    status = summarize_registry(args.registry, allow_incomplete=args.allow_incomplete)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
