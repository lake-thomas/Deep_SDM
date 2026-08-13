#!/usr/bin/env python3
"""Combine the reused baseline with August augmentation-factorial evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from august2026_augmentation_screen import augmentation_combinations
from august2026_dataset_screen_evidence import (
    add_pareto_flags,
    load_artifact_rows,
    load_exact_overlap_rows,
)
from august2026_loss_screen_evidence import strict_evidence_gate
from august2026_summarize_experiments import read_json, sha256, write_json_atomic


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    frame.to_csv(partial, index=False)
    os.replace(partial, path)


def _combination_tuple(spec: dict) -> tuple[bool, float, float]:
    radiometric = spec.get("radiometric_augmentation", False)
    enabled = bool(radiometric.get("enabled", True)) if isinstance(radiometric, dict) else bool(radiometric)
    return enabled, float(spec["dropout"]), float(spec["label_smoothing"])


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
    combined = add_pareto_flags(combined)
    output_dir.mkdir(parents=True, exist_ok=False)
    combined_path = output_dir / "combined_augmentation_candidate_evidence.csv"
    write_csv_atomic(combined_path, combined)
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
