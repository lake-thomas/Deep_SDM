#!/usr/bin/env python3
"""Combine ordinary-BCE baselines with August loss-screen candidate evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from august2026_dataset_screen_evidence import (
    add_pareto_flags,
    load_artifact_rows,
    load_exact_overlap_rows,
)
from august2026_summarize_experiments import read_json, sha256, write_json_atomic


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    frame.to_csv(partial, index=False)
    os.replace(partial, path)


def strict_evidence_gate(frame: pd.DataFrame) -> None:
    values = frame["evidence_gate_passed"]
    if pd.api.types.is_bool_dtype(values):
        passed = values
    else:
        normalized = values.astype(str).str.strip().str.lower()
        if not normalized.isin({"true", "false"}).all():
            raise ValueError("Loss training evidence gate has non-boolean values")
        passed = normalized == "true"
    if not passed.all():
        failed = frame.loc[
            ~passed, ["experiment", "evidence_gate_failures"]
        ].to_dict("records")
        raise RuntimeError(f"Loss training evidence is incomplete: {failed}")


def synthesize_loss_evidence(
    selection_manifest: Path,
    training_metrics_path: Path,
    artifact_roots: list[Path],
    output_dir: Path,
    exact_overlap_roots: list[Path] | None = None,
) -> dict:
    selection = read_json(selection_manifest)
    if (
        selection.get("status") != "complete"
        or selection.get("selection_type") != "dataset_for_loss_screen"
    ):
        raise RuntimeError("Selection manifest is not approved for loss evidence")
    selections = selection.get("selections") or []
    if not selections or len(selections) != selection.get("selection_count"):
        raise ValueError("Loss selection manifest count is invalid")

    source_evidence_path = Path(selection["source_combined_evidence"])
    if sha256(source_evidence_path) != selection["source_combined_evidence_sha256"]:
        raise ValueError("Source dataset-screen evidence hash changed")
    source = pd.read_csv(source_evidence_path, low_memory=False)
    baseline_experiments = {item["source_experiment"] for item in selections}
    baselines = source[source["experiment"].isin(baseline_experiments)].copy()
    if set(baselines["experiment"]) != baseline_experiments:
        raise ValueError("Not all selected ordinary-BCE baselines are in source evidence")
    baselines["loss_candidate_origin"] = "reused_ordinary_bce_baseline"

    training = pd.read_csv(training_metrics_path, low_memory=False)
    if training.empty or training["experiment"].duplicated().any():
        raise ValueError("Loss training summary must contain unique experiments")
    if set(training["stage"]) != {"loss-screen"}:
        raise ValueError("New training metrics are not the frozen loss-screen stage")
    strict_evidence_gate(training)
    artifacts = load_artifact_rows(artifact_roots)
    if set(training["experiment"]) != set(artifacts["experiment"]):
        raise ValueError("Loss training and matched-artifact experiment sets differ")
    new = training.merge(
        artifacts, on="experiment", how="inner", validate="one_to_one"
    )
    if not (new["best_checkpoint_sha256"] == new["artifact_checkpoint_sha256"]).all():
        raise ValueError("Loss training and artifact checkpoint hashes disagree")
    if not (
        new["training_dataset_csv_sha256"] == new["artifact_dataset_sha256"]
    ).all():
        raise ValueError("Loss training and artifact dataset hashes disagree")
    if not (new["host"] == new["artifact_host"]).all():
        raise ValueError("Loss training and artifact host identities disagree")

    exact_overlap_roots = exact_overlap_roots or []
    expected_exact = set(
        new.loc[new["host"] != "notholithocarpus_densiflorus", "experiment"]
    )
    if expected_exact and not exact_overlap_roots:
        raise ValueError("Eastern loss candidates require exact-overlap evidence")
    if exact_overlap_roots:
        exact = load_exact_overlap_rows(exact_overlap_roots)
        if set(exact["experiment"]) != expected_exact:
            raise ValueError("Loss exact-overlap experiment set differs")
        new = new.merge(exact, on="experiment", how="left", validate="one_to_one")
        eastern = new["host"] != "notholithocarpus_densiflorus"
        if not (
            new.loc[eastern, "host"] == new.loc[eastern, "exact_overlap_host"]
        ).all():
            raise ValueError("Loss exact-overlap host identities disagree")
        if not (
            new.loc[eastern, "best_checkpoint_sha256"]
            == new.loc[eastern, "exact_overlap_checkpoint_sha256"]
        ).all():
            raise ValueError("Loss exact-overlap checkpoint hashes disagree")
        if not (
            new.loc[eastern, "training_dataset_csv_sha256"]
            == new.loc[eastern, "exact_overlap_dataset_sha256"]
        ).all():
            raise ValueError("Loss exact-overlap dataset hashes disagree")
    new["loss_candidate_origin"] = "new_loss_screen_run"

    for item in selections:
        host = item["host"]
        host_new = new[new["host"] == host]
        if len(host_new) != 2 or set(host_new["loss_strategy"]) != {
            "pos_weight_bce",
            "balanced_batches",
        }:
            raise ValueError(f"Loss screen does not contain two isolated arms for {host}")
        expected_dataset_hash = item["dataset_sha256"]
        if not (host_new["training_dataset_csv_sha256"] == expected_dataset_hash).all():
            raise ValueError(f"Loss candidates changed selected dataset for {host}")
        if host != "notholithocarpus_densiflorus":
            host_baseline = baselines[baselines["host"] == host]
            required_exact = {
                "exact_overlap_summary",
                "exact_overlap_candidate_absolute_shift_mean",
            }
            if not required_exact.issubset(host_baseline.columns) or host_baseline[
                "exact_overlap_summary"
            ].isna().any():
                raise ValueError(f"Eastern baseline lacks exact-overlap evidence: {host}")

    combined = pd.concat([baselines, new], ignore_index=True, sort=False)
    combined = add_pareto_flags(combined)
    output_dir.mkdir(parents=True, exist_ok=False)
    combined_path = output_dir / "combined_loss_candidate_evidence.csv"
    write_csv_atomic(combined_path, combined)
    status = {
        "status": "complete",
        "stage": "loss-screen",
        "selection_manifest": str(selection_manifest.resolve()),
        "selection_manifest_sha256": sha256(selection_manifest),
        "source_dataset_evidence": str(source_evidence_path.resolve()),
        "source_dataset_evidence_sha256": sha256(source_evidence_path),
        "loss_training_metrics": str(training_metrics_path.resolve()),
        "loss_training_metrics_sha256": sha256(training_metrics_path),
        "artifact_roots": [str(path.resolve()) for path in artifact_roots],
        "exact_overlap_roots": [
            str(path.resolve()) for path in exact_overlap_roots
        ],
        "host_count": len(selections),
        "candidate_count": len(combined),
        "expected_candidates_per_host": 3,
        "combined_loss_candidate_evidence": str(combined_path),
        "combined_loss_candidate_evidence_sha256": sha256(combined_path),
        "selection_guardrail": (
            "No loss strategy is selected automatically. Each host has the reused "
            "ordinary-BCE baseline plus two isolated loss arms. Pareto membership "
            "does not replace BlockCV, calibration review, or acquisition diagnostics."
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
        raise FileNotFoundError(f"Missing loss-screen evidence inputs: {missing}")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to reuse loss evidence output: {args.output_dir}")
    status = synthesize_loss_evidence(
        args.selection_manifest,
        args.training_metrics,
        args.artifact_root,
        args.output_dir,
        exact_overlap_roots=args.exact_overlap_root,
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
