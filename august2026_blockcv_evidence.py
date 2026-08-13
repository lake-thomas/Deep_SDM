#!/usr/bin/env python3
"""Validate and summarize five-fold August held-out BlockCV evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from august2026_combine_dataset_screen_evidence import HOSTS
from august2026_blockcv import blockcv_paths
from august2026_loss_screen_evidence import strict_evidence_gate
from august2026_summarize_experiments import read_json, sha256, write_json_atomic


FOLDS = {1, 2, 3, 4, 5}
MIN_FOLD_ROC_AUC = 0.50
MIN_MEDIAN_ROC_AUC = 0.75
METRICS = {
    "test_roc_auc_all_thresholds": "roc_auc",
    "test_average_precision_all_thresholds": "average_precision",
    "test_brier_score": "raw_brier_score",
    "test_calibration_binary_log_loss": "raw_binary_log_loss",
    "test_calibration_expected_calibration_error": "raw_ece",
    "test_calibration_calibration_intercept": "raw_calibration_intercept",
    "test_calibration_calibration_slope": "raw_calibration_slope",
    "test_platt_calibration_brier_score": "platt_brier_score",
    "test_platt_calibration_binary_log_loss": "platt_binary_log_loss",
    "test_platt_calibration_expected_calibration_error": "platt_ece",
    "test_platt_calibration_calibration_intercept": "platt_calibration_intercept",
    "test_platt_calibration_calibration_slope": "platt_calibration_slope",
}


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    frame.to_csv(partial, index=False)
    os.replace(partial, path)


def summarize_values(values: pd.Series, prefix: str) -> dict:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if len(numeric) != 5 or not np.isfinite(numeric).all():
        raise ValueError(f"BlockCV metric {prefix} must have five finite values")
    return {
        f"{prefix}_mean": float(np.mean(numeric)),
        f"{prefix}_sd": float(np.std(numeric, ddof=1)),
        f"{prefix}_min": float(np.min(numeric)),
        f"{prefix}_q25": float(np.quantile(numeric, 0.25)),
        f"{prefix}_median": float(np.median(numeric)),
        f"{prefix}_q75": float(np.quantile(numeric, 0.75)),
        f"{prefix}_max": float(np.max(numeric)),
    }


def calibration_recommendation(group: pd.DataFrame) -> dict:
    raw_brier = pd.to_numeric(group["test_brier_score"], errors="coerce").to_numpy(float)
    platt_brier = pd.to_numeric(
        group["test_platt_calibration_brier_score"], errors="coerce"
    ).to_numpy(float)
    raw_log = pd.to_numeric(
        group["test_calibration_binary_log_loss"], errors="coerce"
    ).to_numpy(float)
    platt_log = pd.to_numeric(
        group["test_platt_calibration_binary_log_loss"], errors="coerce"
    ).to_numpy(float)
    arrays = [raw_brier, platt_brier, raw_log, platt_log]
    if any(len(values) != 5 or not np.isfinite(values).all() for values in arrays):
        raise ValueError("Calibration recommendation requires five finite paired folds")
    brier_nonworse = platt_brier <= raw_brier + 1e-12
    log_nonworse = platt_log <= raw_log + 1e-12
    paired_both_nonworse = brier_nonworse & log_nonworse
    recommend_platt = bool(
        np.median(platt_brier) < np.median(raw_brier)
        and np.median(platt_log) < np.median(raw_log)
        and paired_both_nonworse.sum() >= 3
    )
    return {
        "calibration_recommendation": "platt" if recommend_platt else "raw",
        "raw_brier_median": float(np.median(raw_brier)),
        "platt_brier_median": float(np.median(platt_brier)),
        "platt_minus_raw_brier_median": float(
            np.median(platt_brier) - np.median(raw_brier)
        ),
        "raw_binary_log_loss_median": float(np.median(raw_log)),
        "platt_binary_log_loss_median": float(np.median(platt_log)),
        "platt_minus_raw_binary_log_loss_median": float(
            np.median(platt_log) - np.median(raw_log)
        ),
        "folds_platt_nonworse_brier": int(brier_nonworse.sum()),
        "folds_platt_nonworse_log_loss": int(log_nonworse.sum()),
        "folds_platt_nonworse_both": int(paired_both_nonworse.sum()),
        "rule": (
            "Recommend Platt only when median Brier and median binary log loss both "
            "improve and both metrics are non-worse in at least 3/5 paired folds."
        ),
    }


def synthesize_blockcv(
    selection_manifest: Path,
    training_metrics_path: Path,
    output_dir: Path,
) -> dict:
    selection = read_json(selection_manifest)
    if (
        selection.get("status") != "complete"
        or selection.get("selection_type") != "recipe_for_blockcv"
    ):
        raise RuntimeError("Selection manifest is not approved for BlockCV evidence")
    selections = selection.get("selections") or []
    if not selections or len(selections) != selection.get("selection_count"):
        raise ValueError("BlockCV selection manifest count is invalid")

    metrics = pd.read_csv(training_metrics_path, low_memory=False)
    if metrics.empty or metrics["experiment"].duplicated().any():
        raise ValueError("BlockCV metrics must contain unique experiments")
    if set(metrics["stage"]) != {"blockcv"}:
        raise ValueError("Metrics are not from the frozen BlockCV stage")
    strict_evidence_gate(metrics)
    if set(metrics["postfit_calibration"]) != {"platt"}:
        raise ValueError("Every BlockCV fold must carry full Platt comparison evidence")
    required = set(METRICS) | {"host", "cv_round", "training_dataset_csv_sha256"}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"BlockCV metrics are missing columns: {sorted(missing)}")

    summary_rows = []
    hard_failures = []
    expected_hosts = {item["host"] for item in selections}
    if expected_hosts != HOSTS:
        raise ValueError("BlockCV evidence requires exactly the five production hosts")
    if set(metrics["host"]) != expected_hosts:
        raise ValueError("BlockCV host set differs from the selected recipes")
    for item in selections:
        host = item["host"]
        group = metrics[metrics["host"] == host].sort_values("cv_round")
        observed_folds = set(pd.to_numeric(group["cv_round"]).astype(int))
        if len(group) != 5 or observed_folds != FOLDS:
            hard_failures.append(
                f"{host}: expected folds {sorted(FOLDS)}, found {sorted(observed_folds)}"
            )
            continue
        for row in group.itertuples(index=False):
            config = read_json(Path(row.config_path))
            expected_manifest, _, _ = blockcv_paths(item, int(row.cv_round))
            if Path(config["csv_path"]).resolve() != expected_manifest.resolve():
                hard_failures.append(
                    f"{host}: fold {row.cv_round} uses an unexpected manifest"
                )
            if sha256(Path(config["csv_path"])) != row.training_dataset_csv_sha256:
                hard_failures.append(
                    f"{host}: fold {row.cv_round} manifest hash mismatch"
                )

        record = {"host": host, "fold_count": len(group)}
        for source_column, label in METRICS.items():
            record.update(summarize_values(group[source_column], label))
        record.update(calibration_recommendation(group))
        roc = pd.to_numeric(group["test_roc_auc_all_thresholds"], errors="coerce")
        fold_above_chance = bool((roc > MIN_FOLD_ROC_AUC).all())
        median_gate = bool(float(roc.median()) >= MIN_MEDIAN_ROC_AUC)
        record["all_folds_roc_auc_above_chance"] = fold_above_chance
        record["median_roc_auc_gate_passed"] = median_gate
        record["blockcv_production_gate_passed"] = fold_above_chance and median_gate
        if not fold_above_chance:
            hard_failures.append(f"{host}: one or more fold ROC AUC values are <= 0.50")
        if not median_gate:
            hard_failures.append(
                f"{host}: median ROC AUC {float(roc.median()):.6f} is below 0.75"
            )
        summary_rows.append(record)

    summary_frame = pd.DataFrame(summary_rows)
    output_dir.mkdir(parents=True, exist_ok=False)
    fold_path = output_dir / "blockcv_fold_metrics.csv"
    summary_path = output_dir / "blockcv_host_summary.csv"
    write_csv_atomic(fold_path, metrics)
    write_csv_atomic(summary_path, summary_frame)
    status = {
        "status": "complete",
        "production_ready": not hard_failures,
        "selection_manifest": str(selection_manifest.resolve()),
        "selection_manifest_sha256": sha256(selection_manifest),
        "training_metrics": str(training_metrics_path.resolve()),
        "training_metrics_sha256": sha256(training_metrics_path),
        "host_count": len(expected_hosts),
        "expected_fold_count_per_host": 5,
        "hard_failure_count": len(hard_failures),
        "hard_failures": hard_failures,
        "thresholds": {
            "all_fold_roc_auc_strictly_greater_than": MIN_FOLD_ROC_AUC,
            "minimum_median_roc_auc": MIN_MEDIAN_ROC_AUC,
        },
        "fold_metrics": str(fold_path),
        "host_summary": str(summary_path),
        "calibration_guardrail": (
            "Calibration recommendation is paired across spatial folds and changes "
            "score space, not ranking. Final production calibration must be fit on the "
            "uniform model's separate validation subset and recorded explicitly."
        ),
    }
    write_json_atomic(output_dir / "summary.json", status)
    (output_dir / "COMPLETED").write_text("completed\n", encoding="utf-8")
    if hard_failures:
        raise RuntimeError(f"BlockCV production gate failed: {hard_failures}")
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--training-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in [args.selection_manifest, args.training_metrics]:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to reuse BlockCV evidence output: {args.output_dir}")
    status = synthesize_blockcv(
        args.selection_manifest, args.training_metrics, args.output_dir
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
