#!/usr/bin/env python3
"""Synthesize August dataset-screen fit, calibration, and seam evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from august2026_summarize_experiments import read_json, sha256, write_json_atomic


OBJECTIVES = {
    "test_roc_auc_all_thresholds": "max",
    "test_brier_score": "min",
    "artifact_candidate_absolute_did_mean": "min",
}
DELTA_COLUMNS = [
    "test_roc_auc_all_thresholds",
    "test_average_precision_all_thresholds",
    "test_brier_score",
    "test_calibration_binary_log_loss",
    "artifact_candidate_signed_did_mean",
    "artifact_candidate_absolute_did_mean",
    "artifact_relative_signed_did_reduction",
]


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    frame.to_csv(partial, index=False)
    os.replace(partial, path)


def strict_boolean(series: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    normalized = series.astype(str).str.strip().str.lower()
    invalid = ~normalized.isin({"true", "false"})
    if invalid.any():
        raise ValueError(f"{name} contains non-boolean values")
    return normalized == "true"


def pareto_nondominated(frame: pd.DataFrame, objectives: dict[str, str]) -> np.ndarray:
    missing = set(objectives) - set(frame.columns)
    if missing:
        raise ValueError(f"Pareto objective columns are missing: {sorted(missing)}")
    values = frame[list(objectives)].apply(pd.to_numeric, errors="coerce").to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Pareto objective values must be finite")
    direction = np.asarray(
        [1.0 if objectives[column] == "max" else -1.0 for column in objectives]
    )
    utility = values * direction
    nondominated = np.ones(len(frame), dtype=bool)
    for index in range(len(frame)):
        no_worse = np.all(utility >= utility[index], axis=1)
        strictly_better = np.any(utility > utility[index], axis=1)
        if np.any(no_worse & strictly_better):
            nondominated[index] = False
    return nondominated


def load_artifact_rows(artifact_roots: list[Path]) -> pd.DataFrame:
    rows = []
    for root in artifact_roots:
        status_path = root / "batch_status.json"
        completed_path = root / "COMPLETED"
        if not status_path.is_file() or not completed_path.is_file():
            raise FileNotFoundError(f"Incomplete artifact batch root: {root}")
        status = read_json(status_path)
        if status.get("status") != "complete":
            raise RuntimeError(f"Artifact batch is not complete: {status_path}")
        experiments = status.get("completed_experiments") or []
        if len(experiments) != status.get("candidate_count"):
            raise RuntimeError(f"Artifact batch candidate count disagrees: {status_path}")
        for experiment in experiments:
            candidate_root = root / experiment
            probe_path = candidate_root / "summary.json"
            comparison_path = (
                candidate_root / "june_baseline_comparison" / "summary.json"
            )
            required = [
                candidate_root / "COMPLETED",
                probe_path,
                candidate_root / "june_baseline_comparison" / "COMPLETED",
                comparison_path,
            ]
            if not all(path.is_file() for path in required):
                raise FileNotFoundError(
                    f"Incomplete candidate artifact evidence: {candidate_root}"
                )
            probe = read_json(probe_path)
            comparison = read_json(comparison_path)
            if probe.get("status") != "complete" or comparison.get("status") != "complete":
                raise RuntimeError(f"Candidate artifact status failed: {candidate_root}")
            if probe.get("candidate_experiment") != experiment:
                raise ValueError(f"Artifact experiment identity mismatch: {candidate_root}")
            if (
                probe.get("checkpoint_sha256")
                != comparison.get("candidate_checkpoint_sha256")
                or probe.get("dataset_sha256")
                != comparison.get("candidate_dataset_sha256")
            ):
                raise ValueError(f"Artifact probe/comparison hashes disagree: {candidate_root}")
            signed_ci = comparison["paired_signed_did_reduction_bootstrap_95ci"]
            absolute_ci = comparison["paired_absolute_did_reduction_bootstrap_95ci"]
            rows.append(
                {
                    "experiment": experiment,
                    "artifact_host": comparison["taxon"],
                    "artifact_diagnostic": comparison["diagnostic"],
                    "artifact_pair_count": comparison["pair_count"],
                    "artifact_baseline_signed_did_mean": comparison[
                        "baseline_signed_did_mean"
                    ],
                    "artifact_candidate_signed_did_mean": comparison[
                        "candidate_signed_did_mean"
                    ],
                    "artifact_paired_signed_did_reduction_mean": comparison[
                        "paired_signed_did_reduction_mean"
                    ],
                    "artifact_paired_signed_did_reduction_ci_low": signed_ci[0],
                    "artifact_paired_signed_did_reduction_ci_high": signed_ci[1],
                    "artifact_relative_signed_did_reduction": comparison[
                        "relative_signed_did_reduction"
                    ],
                    "artifact_baseline_absolute_did_mean": comparison[
                        "baseline_absolute_did_mean"
                    ],
                    "artifact_candidate_absolute_did_mean": comparison[
                        "candidate_absolute_did_mean"
                    ],
                    "artifact_paired_absolute_did_reduction_mean": comparison[
                        "paired_absolute_did_reduction_mean"
                    ],
                    "artifact_paired_absolute_did_reduction_ci_low": absolute_ci[0],
                    "artifact_paired_absolute_did_reduction_ci_high": absolute_ci[1],
                    "artifact_comparison_status": comparison["comparison_status"],
                    "artifact_signed_reduction_supported": signed_ci[0] > 0,
                    "artifact_absolute_reduction_supported": absolute_ci[0] > 0,
                    "artifact_checkpoint_sha256": comparison[
                        "candidate_checkpoint_sha256"
                    ],
                    "artifact_dataset_sha256": comparison["candidate_dataset_sha256"],
                    "artifact_score_space": probe.get("score_space"),
                    "artifact_platt_calibrator_sha256": probe.get(
                        "platt_calibrator_sha256"
                    ),
                    "artifact_probe_summary": str(probe_path.resolve()),
                    "artifact_comparison_summary": str(comparison_path.resolve()),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty or frame["experiment"].duplicated().any():
        raise ValueError("Artifact evidence must contain unique candidate experiments")
    return frame


def load_exact_overlap_rows(exact_roots: list[Path]) -> pd.DataFrame:
    rows = []
    for root in exact_roots:
        status_path = root / "batch_status.json"
        if not status_path.is_file() or not (root / "COMPLETED").is_file():
            raise FileNotFoundError(f"Incomplete exact-overlap batch root: {root}")
        status = read_json(status_path)
        if status.get("status") != "complete":
            raise RuntimeError(f"Exact-overlap batch is not complete: {status_path}")
        experiments = status.get("completed_experiments") or []
        if len(experiments) != status.get("candidate_count"):
            raise RuntimeError(f"Exact-overlap candidate count disagrees: {status_path}")
        for experiment in experiments:
            candidate_root = root / experiment
            summary_path = candidate_root / "summary.json"
            if not summary_path.is_file() or not (candidate_root / "COMPLETED").is_file():
                raise FileNotFoundError(
                    f"Incomplete exact-overlap candidate output: {candidate_root}"
                )
            summary = read_json(summary_path)
            if summary.get("status") != "complete":
                raise RuntimeError(
                    f"Exact-overlap candidate status failed: {candidate_root}"
                )
            ci = summary[
                "paired_absolute_probability_shift_reduction_bootstrap_95ci"
            ]
            rows.append(
                {
                    "experiment": experiment,
                    "exact_overlap_host": summary["host"],
                    "exact_overlap_pair_count": summary["pair_count"],
                    "exact_overlap_baseline_signed_shift_mean": summary[
                        "baseline_pa_minus_ny_probability_mean"
                    ],
                    "exact_overlap_candidate_signed_shift_mean": summary[
                        "candidate_pa_minus_ny_probability_mean"
                    ],
                    "exact_overlap_baseline_absolute_shift_mean": summary[
                        "baseline_absolute_probability_shift_mean"
                    ],
                    "exact_overlap_candidate_absolute_shift_mean": summary[
                        "candidate_absolute_probability_shift_mean"
                    ],
                    "exact_overlap_paired_absolute_shift_reduction_mean": summary[
                        "paired_absolute_probability_shift_reduction_mean"
                    ],
                    "exact_overlap_paired_absolute_shift_reduction_ci_low": ci[0],
                    "exact_overlap_paired_absolute_shift_reduction_ci_high": ci[1],
                    "exact_overlap_absolute_shift_reduction_supported": summary[
                        "absolute_shift_reduction_supported"
                    ],
                    "exact_overlap_checkpoint_sha256": summary["checkpoint_sha256"],
                    "exact_overlap_dataset_sha256": summary["dataset_sha256"],
                    "exact_overlap_score_space": summary.get("score_space"),
                    "exact_overlap_platt_calibrator_sha256": summary.get(
                        "platt_calibrator_sha256"
                    ),
                    "exact_overlap_summary": str(summary_path.resolve()),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty or frame["experiment"].duplicated().any():
        raise ValueError("Exact-overlap evidence must contain unique candidate experiments")
    return frame


def add_pareto_flags(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["pareto_nondominated_within_host"] = False
    for _, indices in output.groupby("host", sort=False).groups.items():
        output.loc[indices, "pareto_nondominated_within_host"] = pareto_nondominated(
            output.loc[indices], OBJECTIVES
        )
    return output


def incremental_depth_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (host, arm), group in frame.groupby(["host", "arm"], sort=False):
        ordered = group.sort_values("background_target")
        previous = None
        for record in ordered.to_dict("records"):
            output = {
                "host": host,
                "arm": arm,
                "background_target": record["background_target"],
                "previous_background_target": (
                    previous["background_target"] if previous else None
                ),
            }
            for column in DELTA_COLUMNS:
                output[column] = record[column]
                output[f"increment_from_previous_{column}"] = (
                    record[column] - previous[column] if previous else None
                )
            rows.append(output)
            previous = record
    return pd.DataFrame(rows)


def domain_balanced_minus_random(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (host, target), group in frame.groupby(
        ["host", "background_target"], sort=False
    ):
        by_arm = group.set_index("arm")
        if not {"random", "domain_balanced"}.issubset(by_arm.index):
            continue
        output = {
            "host": host,
            "background_target": target,
            "interpretation": (
                "domain_balanced minus random; descriptive because sampled rows differ"
            ),
        }
        for column in DELTA_COLUMNS:
            output[f"random_{column}"] = by_arm.loc["random", column]
            output[f"domain_balanced_{column}"] = by_arm.loc[
                "domain_balanced", column
            ]
            output[f"delta_{column}"] = (
                by_arm.loc["domain_balanced", column] - by_arm.loc["random", column]
            )
        rows.append(output)
    return pd.DataFrame(rows)


def synthesize(
    training_metrics_path: Path,
    artifact_roots: list[Path],
    output_dir: Path,
    exact_overlap_roots: list[Path] | None = None,
) -> dict:
    training = pd.read_csv(training_metrics_path, low_memory=False)
    if training.empty or training["experiment"].duplicated().any():
        raise ValueError("Training summary must contain unique experiments")
    evidence_passed = strict_boolean(
        training["evidence_gate_passed"], "evidence_gate_passed"
    )
    if not evidence_passed.all():
        failed = training.loc[
            ~evidence_passed,
            ["experiment", "evidence_gate_failures"],
        ].to_dict("records")
        raise RuntimeError(f"Training evidence is incomplete: {failed}")
    stages = set(training["stage"])
    allowed_stages = {"dataset-screen", "control-dataset-screen"}
    if len(stages) != 1 or not stages.issubset(allowed_stages):
        raise ValueError(
            "Evidence synthesis requires one frozen dataset-screen stage, found "
            f"{sorted(stages)}"
        )

    artifacts = load_artifact_rows(artifact_roots)
    training_experiments = set(training["experiment"])
    artifact_experiments = set(artifacts["experiment"])
    if training_experiments != artifact_experiments:
        raise ValueError(
            "Training and artifact experiment sets differ: "
            f"missing_artifacts={sorted(training_experiments - artifact_experiments)}, "
            f"unexpected_artifacts={sorted(artifact_experiments - training_experiments)}"
        )
    combined = training.merge(
        artifacts, on="experiment", how="inner", validate="one_to_one"
    )
    if not (combined["best_checkpoint_sha256"] == combined["artifact_checkpoint_sha256"]).all():
        raise ValueError("Training and artifact checkpoint hashes disagree")
    if not (
        combined["training_dataset_csv_sha256"]
        == combined["artifact_dataset_sha256"]
    ).all():
        raise ValueError("Training and artifact dataset hashes disagree")
    if not (combined["host"] == combined["artifact_host"]).all():
        raise ValueError("Training and artifact host identities disagree")
    exact_overlap_roots = exact_overlap_roots or []
    if exact_overlap_roots:
        exact = load_exact_overlap_rows(exact_overlap_roots)
        expected_exact = set(
            combined.loc[
                combined["host"] != "notholithocarpus_densiflorus", "experiment"
            ]
        )
        found_exact = set(exact["experiment"])
        if expected_exact != found_exact:
            raise ValueError(
                "Exact-overlap experiment set differs for eastern candidates: "
                f"missing={sorted(expected_exact - found_exact)}, "
                f"unexpected={sorted(found_exact - expected_exact)}"
            )
        combined = combined.merge(
            exact, on="experiment", how="left", validate="one_to_one"
        )
        eastern = combined["host"] != "notholithocarpus_densiflorus"
        if not (
            combined.loc[eastern, "host"]
            == combined.loc[eastern, "exact_overlap_host"]
        ).all():
            raise ValueError("Training and exact-overlap host identities disagree")
        if not (
            combined.loc[eastern, "best_checkpoint_sha256"]
            == combined.loc[eastern, "exact_overlap_checkpoint_sha256"]
        ).all():
            raise ValueError("Training and exact-overlap checkpoint hashes disagree")
        if not (
            combined.loc[eastern, "training_dataset_csv_sha256"]
            == combined.loc[eastern, "exact_overlap_dataset_sha256"]
        ).all():
            raise ValueError("Training and exact-overlap dataset hashes disagree")
    combined = add_pareto_flags(combined)

    output_dir.mkdir(parents=True, exist_ok=False)
    combined_path = output_dir / "combined_candidate_evidence.csv"
    depth_path = output_dir / "background_depth_incremental_deltas.csv"
    arm_path = output_dir / "domain_balanced_minus_random_combined.csv"
    write_csv_atomic(combined_path, combined)
    write_csv_atomic(depth_path, incremental_depth_deltas(combined))
    write_csv_atomic(arm_path, domain_balanced_minus_random(combined))
    status = {
        "status": "complete",
        "training_metrics": str(training_metrics_path.resolve()),
        "training_metrics_sha256": sha256(training_metrics_path),
        "artifact_roots": [str(path.resolve()) for path in artifact_roots],
        "exact_overlap_roots": [
            str(path.resolve()) for path in exact_overlap_roots
        ],
        "experiment_count": len(combined),
        "hosts": sorted(combined["host"].unique()),
        "pareto_candidate_count_by_host": {
            host: int(group["pareto_nondominated_within_host"].sum())
            for host, group in combined.groupby("host")
        },
        "all_signed_reduction_supported_count": int(
            combined["artifact_signed_reduction_supported"].sum()
        ),
        "all_absolute_reduction_supported_count": int(
            combined["artifact_absolute_reduction_supported"].sum()
        ),
        "combined_candidate_evidence": str(combined_path),
        "background_depth_incremental_deltas": str(depth_path),
        "domain_balanced_minus_random_combined": str(arm_path),
        "selection_guardrail": (
            "No model is selected automatically. Pareto membership uses only frozen "
            "uniform-split AUC, Brier score, and absolute matched-pair DID. Finalists "
            "still require loss/augmentation isolation, held-out BlockCV, calibration, "
            "full acceptance diagnostics, and visual review."
        ),
    }
    write_json_atomic(output_dir / "summary.json", status)
    (output_dir / "COMPLETED").write_text("completed\n", encoding="utf-8")
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-metrics", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, action="append", required=True)
    parser.add_argument("--exact-overlap-root", type=Path, action="append", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required = [
        args.training_metrics,
        *args.artifact_root,
        *(args.exact_overlap_root or []),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing dataset-screen evidence inputs: {missing}")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to reuse evidence output: {args.output_dir}")
    status = synthesize(
        args.training_metrics,
        args.artifact_root,
        args.output_dir,
        exact_overlap_roots=args.exact_overlap_root,
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
