#!/usr/bin/env python3
"""Combine the reused baseline with August augmentation-factorial evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from august2026_augmentation_screen import augmentation_combinations
from august2026_dataset_screen_evidence import (
    add_pareto_flags,
    load_artifact_rows,
    load_exact_overlap_rows,
)
from august2026_loss_screen_evidence import strict_evidence_gate
from august2026_summarize_experiments import read_json, sha256, write_json_atomic


FACTOR_LEVELS = {
    "radiometric_enabled": (False, True),
    "dropout": (0.25, 0.35),
    "label_smoothing": (0.0, 0.05),
}
EFFECT_METRICS = {
    "best_val_loss": -1,
    "test_roc_auc_all_thresholds": 1,
    "test_average_precision_all_thresholds": 1,
    "test_brier_score": -1,
    "test_mcc_at_threshold": 1,
    "test_calibration_binary_log_loss": -1,
    "test_calibration_expected_calibration_error": -1,
    "test_calibration_slope_absolute_error": -1,
    "test_calibration_intercept_absolute_error": -1,
    "artifact_candidate_absolute_did_mean": -1,
    "artifact_coastal_candidate_absolute_did_mean": -1,
    "exact_overlap_candidate_absolute_shift_mean": -1,
}


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    frame.to_csv(partial, index=False)
    os.replace(partial, path)


def _combination_tuple(spec: dict) -> tuple[bool, float, float]:
    radiometric = spec.get("radiometric_augmentation", False)
    enabled = bool(radiometric.get("enabled", True)) if isinstance(radiometric, dict) else bool(radiometric)
    return enabled, float(spec["dropout"]), float(spec["label_smoothing"])


def add_calibration_error_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["test_calibration_slope_absolute_error"] = (
        pd.to_numeric(
            output["test_calibration_calibration_slope"], errors="coerce"
        )
        - 1.0
    ).abs()
    output["test_calibration_intercept_absolute_error"] = pd.to_numeric(
        output["test_calibration_calibration_intercept"], errors="coerce"
    ).abs()
    return output


def factorial_effect_contrasts(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    factors = list(FACTOR_LEVELS)
    rows = []
    for host, host_frame in frame.groupby("host", sort=True):
        if len(host_frame) != 8:
            raise ValueError(f"Factorial contrast requires eight candidates for {host}")
        observed = {
            (
                bool(row.radiometric_enabled),
                float(row.dropout),
                float(row.label_smoothing),
            )
            for row in host_frame.itertuples(index=False)
        }
        expected = {
            (radiometric, dropout, smoothing)
            for radiometric in FACTOR_LEVELS["radiometric_enabled"]
            for dropout in FACTOR_LEVELS["dropout"]
            for smoothing in FACTOR_LEVELS["label_smoothing"]
        }
        if observed != expected:
            raise ValueError(f"Factorial cells changed for {host}")
        for factor in factors:
            low_level, high_level = FACTOR_LEVELS[factor]
            matched = [item for item in factors if item != factor]
            for matched_levels, subset in host_frame.groupby(
                matched, sort=True, dropna=False
            ):
                if len(subset) != 2:
                    raise ValueError(
                        f"Factorial pair is incomplete for {host}:{factor}"
                    )
                low = subset[subset[factor] == low_level]
                high = subset[subset[factor] == high_level]
                if len(low) != 1 or len(high) != 1:
                    raise ValueError(
                        f"Factor levels changed for {host}:{factor}"
                    )
                low = low.iloc[0]
                high = high.iloc[0]
                if not isinstance(matched_levels, tuple):
                    matched_levels = (matched_levels,)
                record = {
                    "host": host,
                    "factor": factor,
                    "low_level": low_level,
                    "high_level": high_level,
                    "low_experiment": low["experiment"],
                    "high_experiment": high["experiment"],
                }
                record.update(
                    {
                        f"matched_{name}": value
                        for name, value in zip(matched, matched_levels)
                    }
                )
                for metric, direction in EFFECT_METRICS.items():
                    low_value = pd.to_numeric(
                        pd.Series([low.get(metric)]), errors="coerce"
                    ).iloc[0]
                    high_value = pd.to_numeric(
                        pd.Series([high.get(metric)]), errors="coerce"
                    ).iloc[0]
                    delta = (
                        float(high_value - low_value)
                        if pd.notna(low_value) and pd.notna(high_value)
                        else float("nan")
                    )
                    record[f"low_{metric}"] = low_value
                    record[f"high_{metric}"] = high_value
                    record[f"delta_{metric}"] = delta
                    record[f"improvement_{metric}"] = delta * direction
                rows.append(record)
    pairwise = pd.DataFrame(rows)
    aggregates = []
    for (host, factor), group in pairwise.groupby(["host", "factor"], sort=True):
        record = {
            "host": host,
            "factor": factor,
            "matched_pair_count": len(group),
            "interpretation": (
                "Positive improvement values favor the high factor level; each pair "
                "holds the other two factorial settings fixed."
            ),
        }
        for metric in EFFECT_METRICS:
            values = pd.to_numeric(
                group[f"improvement_{metric}"], errors="coerce"
            )
            finite = values[np.isfinite(values)]
            record[f"{metric}_finite_pair_count"] = len(finite)
            record[f"{metric}_mean_improvement"] = (
                float(finite.mean()) if len(finite) else None
            )
            record[f"{metric}_median_improvement"] = (
                float(finite.median()) if len(finite) else None
            )
            record[f"{metric}_favorable_pair_count"] = int((finite > 0).sum())
        aggregates.append(record)
    return pairwise, pd.DataFrame(aggregates)


def synthesize_augmentation_evidence(
    selection_manifest: Path,
    training_metrics_path: Path,
    artifact_roots: list[Path],
    output_dir: Path,
    exact_overlap_roots: list[Path] | None = None,
) -> dict:
    selection = read_json(selection_manifest)
    if (
        selection.get("status") != "complete"
        or selection.get("selection_type") != "loss_for_augmentation_screen"
    ):
        raise RuntimeError("Selection manifest is not approved for augmentation evidence")
    selections = selection.get("selections") or []
    if not selections or len(selections) != selection.get("selection_count"):
        raise ValueError("Augmentation selection manifest count is invalid")

    source_path = Path(selection["source_combined_evidence"])
    if sha256(source_path) != selection["source_combined_evidence_sha256"]:
        raise ValueError("Source loss evidence hash changed")
    source = pd.read_csv(source_path, low_memory=False)
    baseline_experiments = {item["source_experiment"] for item in selections}
    baselines = source[source["experiment"].isin(baseline_experiments)].copy()
    if set(baselines["experiment"]) != baseline_experiments:
        raise ValueError("Not all selected loss baselines are present")
    baselines["augmentation_candidate_origin"] = "reused_loss_baseline"
    baselines["radiometric_enabled"] = False

    training = pd.read_csv(training_metrics_path, low_memory=False)
    if training.empty or training["experiment"].duplicated().any():
        raise ValueError("Augmentation training summary must contain unique experiments")
    if set(training["stage"]) != {"augmentation-screen"}:
        raise ValueError("New metrics are not the frozen augmentation-screen stage")
    strict_evidence_gate(training)
    radiometric_enabled = []
    for row in training.itertuples(index=False):
        config = read_json(Path(row.config_path))
        spec = config.get("august_experiment_spec") or {}
        radiometric_enabled.append(_combination_tuple(spec)[0])
    training["radiometric_enabled"] = radiometric_enabled

    artifacts = load_artifact_rows(artifact_roots)
    if set(training["experiment"]) != set(artifacts["experiment"]):
        raise ValueError("Augmentation training and artifact experiment sets differ")
    new = training.merge(artifacts, on="experiment", how="inner", validate="one_to_one")
    if not (new["best_checkpoint_sha256"] == new["artifact_checkpoint_sha256"]).all():
        raise ValueError("Augmentation checkpoint hashes disagree")
    if not (
        new["training_dataset_csv_sha256"] == new["artifact_dataset_sha256"]
    ).all():
        raise ValueError("Augmentation dataset hashes disagree")
    if not (new["host"] == new["artifact_host"]).all():
        raise ValueError("Augmentation host identities disagree")

    exact_overlap_roots = exact_overlap_roots or []
    expected_exact = set(
        new.loc[new["host"] != "notholithocarpus_densiflorus", "experiment"]
    )
    if expected_exact and not exact_overlap_roots:
        raise ValueError("Eastern augmentation candidates require exact-overlap evidence")
    if exact_overlap_roots:
        exact = load_exact_overlap_rows(exact_overlap_roots)
        if set(exact["experiment"]) != expected_exact:
            raise ValueError("Augmentation exact-overlap experiment set differs")
        new = new.merge(exact, on="experiment", how="left", validate="one_to_one")
        eastern = new["host"] != "notholithocarpus_densiflorus"
        if not (
            new.loc[eastern, "host"] == new.loc[eastern, "exact_overlap_host"]
        ).all():
            raise ValueError("Augmentation exact-overlap host identities disagree")
        if not (
            new.loc[eastern, "best_checkpoint_sha256"]
            == new.loc[eastern, "exact_overlap_checkpoint_sha256"]
        ).all():
            raise ValueError("Augmentation exact-overlap checkpoint hashes disagree")
        if not (
            new.loc[eastern, "training_dataset_csv_sha256"]
            == new.loc[eastern, "exact_overlap_dataset_sha256"]
        ).all():
            raise ValueError("Augmentation exact-overlap dataset hashes disagree")
    new["augmentation_candidate_origin"] = "new_factorial_run"

    expected_combinations = {
        (
            bool(item["radiometric"]),
            float(item["dropout"]),
            float(item["label_smoothing"]),
        )
        for item in augmentation_combinations()
    }
    for item in selections:
        host = item["host"]
        host_new = new[new["host"] == host]
        observed = {
            (
                bool(row.radiometric_enabled),
                float(row.dropout),
                float(row.label_smoothing),
            )
            for row in host_new.itertuples(index=False)
        }
        if len(host_new) != 7 or observed != expected_combinations:
            raise ValueError(f"Augmentation factorial is incomplete for {host}")
        if not (
            host_new["training_dataset_csv_sha256"] == item["dataset_sha256"]
        ).all():
            raise ValueError(f"Augmentation candidates changed dataset for {host}")
        if set(host_new["loss_strategy"]) != {item["loss_strategy"]}:
            raise ValueError(f"Augmentation candidates changed loss for {host}")

    combined = pd.concat([baselines, new], ignore_index=True, sort=False)
    combined = add_calibration_error_columns(combined)
    combined = add_pareto_flags(combined)
    pairwise_effects, aggregate_effects = factorial_effect_contrasts(combined)
    if len(pairwise_effects) != len(selections) * 12 or len(
        aggregate_effects
    ) != len(selections) * 3:
        raise RuntimeError("Augmentation factorial effect matrix is incomplete")
    output_dir.mkdir(parents=True, exist_ok=False)
    combined_path = output_dir / "combined_augmentation_candidate_evidence.csv"
    pairwise_path = output_dir / "matched_factorial_effects.csv"
    aggregate_path = output_dir / "factorial_effect_summary.csv"
    write_csv_atomic(combined_path, combined)
    write_csv_atomic(pairwise_path, pairwise_effects)
    write_csv_atomic(aggregate_path, aggregate_effects)
    status = {
        "status": "complete",
        "stage": "augmentation-screen",
        "selection_manifest": str(selection_manifest.resolve()),
        "selection_manifest_sha256": sha256(selection_manifest),
        "source_loss_evidence": str(source_path.resolve()),
        "source_loss_evidence_sha256": sha256(source_path),
        "augmentation_training_metrics": str(training_metrics_path.resolve()),
        "augmentation_training_metrics_sha256": sha256(training_metrics_path),
        "artifact_roots": [str(path.resolve()) for path in artifact_roots],
        "exact_overlap_roots": [str(path.resolve()) for path in exact_overlap_roots],
        "host_count": len(selections),
        "candidate_count": len(combined),
        "expected_candidates_per_host": 8,
        "combined_augmentation_candidate_evidence": str(combined_path),
        "combined_augmentation_candidate_evidence_sha256": sha256(combined_path),
        "matched_factorial_effects": str(pairwise_path.resolve()),
        "matched_factorial_effects_sha256": sha256(pairwise_path),
        "factorial_effect_summary": str(aggregate_path.resolve()),
        "factorial_effect_summary_sha256": sha256(aggregate_path),
        "factorial_effect_pair_count": len(pairwise_effects),
        "factorial_effect_group_count": len(aggregate_effects),
        "selection_guardrail": (
            "No augmentation recipe is selected automatically. Each host has one "
            "reused baseline plus the seven non-baseline factorial arms. BlockCV and "
            "final artifact/map acceptance remain required."
        ),
    }
    write_json_atomic(output_dir / "summary.json", status)
    (output_dir / "COMPLETED").write_text("completed\n", encoding="utf-8")
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--training-metrics", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, action="append", required=True)
    parser.add_argument("--exact-overlap-root", type=Path, action="append", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required = [
        args.selection_manifest,
        args.training_metrics,
        *args.artifact_root,
        *(args.exact_overlap_root or []),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing augmentation evidence inputs: {missing}")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to reuse augmentation evidence: {args.output_dir}")
    status = synthesize_augmentation_evidence(
        args.selection_manifest,
        args.training_metrics,
        args.artifact_root,
        args.output_dir,
        exact_overlap_roots=args.exact_overlap_root,
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
