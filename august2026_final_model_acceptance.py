#!/usr/bin/env python3
"""Synthesize pre-registered final acceptance evidence for five production models."""

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
)
from august2026_loss_screen_evidence import strict_evidence_gate
from august2026_summarize_experiments import read_json, sha256, write_json_atomic


HOSTS = {
    "ailanthus_altissima",
    "juglans_nigra",
    "prunus_serotina",
    "vitis_labrusca",
    "notholithocarpus_densiflorus",
}
EASTERN_HOSTS = HOSTS - {"notholithocarpus_densiflorus"}
SEAM_POSITIVE_HOSTS = {
    "prunus_serotina",
    "vitis_labrusca",
    "notholithocarpus_densiflorus",
}
CONTROL_ABSOLUTE_TOLERANCE = 0.0025
CONTROL_RELATIVE_TOLERANCE = 0.10


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    partial = path.with_name(f"{path.name}.partial")
    frame.to_csv(partial, index=False)
    os.replace(partial, path)


def artifact_nonregression_gate(
    host: str,
    baseline: float,
    candidate: float,
) -> dict:
    values = np.asarray([baseline, candidate], dtype=float)
    if not np.isfinite(values).all() or baseline < 0 or candidate < 0:
        return {
            "passed": False,
            "allowed_maximum": None,
            "failure": "nonfinite_or_negative_artifact_metric",
        }
    if host in SEAM_POSITIVE_HOSTS:
        allowed = baseline
        rule = "candidate_mean_must_not_exceed_june_mean"
    else:
        allowed = baseline + max(
            CONTROL_ABSOLUTE_TOLERANCE,
            CONTROL_RELATIVE_TOLERANCE * baseline,
        )
        rule = "low_seam_control_with_fixed_absolute_or_relative_tolerance"
    return {
        "passed": bool(candidate <= allowed + 1e-12),
        "allowed_maximum": float(allowed),
        "failure": None if candidate <= allowed + 1e-12 else "artifact_regression",
        "rule": rule,
    }


def validate_support_audits(
    dataset_overlap_summary: Path,
    pathway_summary: Path,
    training: pd.DataFrame,
) -> list[str]:
    failures = []
    overlap = read_json(dataset_overlap_summary)
    pathway = read_json(pathway_summary)
    if overlap.get("status") != "complete":
        failures.append("dataset overlap audit is incomplete")
    if pathway.get("status") != "complete":
        failures.append("pathway correlation audit is incomplete")
    if set(pathway.get("hosts") or []) != {
        "ailanthus_altissima",
        "juglans_nigra",
    }:
        failures.append("pathway audit host identities changed")
    training_by_host = training.set_index("host")
    expected_datasets = {
        host: training_by_host.loc[host, "training_dataset_csv_sha256"]
        for host in ("ailanthus_altissima", "juglans_nigra")
    }
    observed_overlap = {
        overlap.get("first_label"): overlap.get("first_dataset_sha256"),
        overlap.get("second_label"): overlap.get("second_dataset_sha256"),
    }
    if observed_overlap != expected_datasets:
        failures.append("dataset overlap audit does not use both production datasets")
    candidates = pathway.get("candidates") or {}
    for host, dataset_hash in expected_datasets.items():
        if (candidates.get(host) or {}).get("dataset_sha256") != dataset_hash:
            failures.append(f"pathway audit dataset hash differs for {host}")
        if (candidates.get(host) or {}).get("checkpoint_sha256") != training_by_host.loc[
            host, "best_checkpoint_sha256"
        ]:
            failures.append(f"pathway audit checkpoint hash differs for {host}")
    return failures


def resolve_production_selection(
    training: pd.DataFrame, blockcv_evidence_dir: Path, blockcv: dict
) -> tuple[Path, dict]:
    """Resolve one shared or one explicitly combined production selection."""
    selection_paths = set()
    for row in training.itertuples(index=False):
        config = read_json(Path(row.config_path))
        gate = config.get("selection_gate") or {}
        selection_path = Path(gate.get("selection_manifest", ""))
        if not selection_path.is_file():
            raise FileNotFoundError(
                f"Production config selection manifest is missing: {selection_path}"
            )
        if gate.get("selection_manifest_sha256") != sha256(selection_path):
            raise ValueError(f"Production selection hash changed: {selection_path}")
        selection_paths.add(selection_path.resolve())
    if len(selection_paths) == 1:
        production_selection_path = next(iter(selection_paths))
    else:
        combined_raw = blockcv.get("combined_production_selection_manifest")
        if not combined_raw:
            raise ValueError(
                "Production fits use multiple selection manifests without an "
                "identity-checked combined replacement manifest"
            )
        production_selection_path = Path(combined_raw)
        if not production_selection_path.is_absolute():
            production_selection_path = blockcv_evidence_dir / production_selection_path
        if not production_selection_path.is_file():
            raise FileNotFoundError(production_selection_path)
        combined = read_json(production_selection_path)
        sources = combined.get("source_production_selection_manifests") or []
        observed_sources = {}
        for source in sources:
            source_path = Path(source.get("path", "")).resolve()
            if source_path in observed_sources:
                raise ValueError("Combined production selection has duplicate sources")
            observed_sources[source_path] = source.get("sha256")
        expected_sources = {
            path: sha256(path) for path in selection_paths
        }
        if observed_sources != expected_sources:
            raise ValueError("Combined production selection source identities changed")
        if (
            combined.get("status") != "complete"
            or combined.get("selection_type")
            != "blockcv_approved_recipe_for_production_fit"
            or combined.get("selection_count") != len(HOSTS)
            or {item.get("host") for item in combined.get("selections", [])}
            != HOSTS
        ):
            raise ValueError("Combined production selection host set is invalid")
    production_selection = read_json(production_selection_path)
    if (
        production_selection.get("blockcv_evidence_summary_sha256")
        != sha256(blockcv_evidence_dir / "summary.json")
    ):
        raise ValueError("Production selection does not reference supplied BlockCV evidence")
    return production_selection_path, production_selection


def synthesize_acceptance(
    training_metrics: Path,
    blockcv_evidence_dir: Path,
    raw_artifact_roots: list[Path],
    configured_artifact_roots: list[Path],
    raw_exact_roots: list[Path],
    configured_exact_roots: list[Path],
    dataset_overlap_summary: Path,
    pathway_summary: Path,
    output_dir: Path,
) -> dict:
    training = pd.read_csv(training_metrics, low_memory=False)
    if training.empty or training["experiment"].duplicated().any():
        raise ValueError("Production metrics must contain unique experiments")
    if set(training["stage"]) != {"production-fit"} or set(training["host"]) != HOSTS:
        raise ValueError("Production metrics must contain exactly the five production hosts")
    strict_evidence_gate(training)
    if len(training) != 5:
        raise ValueError("Final acceptance requires exactly five production fits")
    blockcv = read_json(blockcv_evidence_dir / "summary.json")
    if blockcv.get("status") != "complete" or not blockcv.get("production_ready"):
        raise RuntimeError("Frozen BlockCV evidence is not production-ready")
    if not (blockcv_evidence_dir / "COMPLETED").is_file():
        raise FileNotFoundError(blockcv_evidence_dir / "COMPLETED")
    production_selection_path, production_selection = resolve_production_selection(
        training, blockcv_evidence_dir, blockcv
    )

    raw_artifacts = load_artifact_rows(raw_artifact_roots)
    configured_artifacts = load_artifact_rows(
        configured_artifact_roots, require_raw_score_space=False
    )
    raw_exact = load_exact_overlap_rows(raw_exact_roots)
    configured_exact = load_exact_overlap_rows(
        configured_exact_roots, require_raw_score_space=False
    )
    experiment_set = set(training["experiment"])
    for label, frame, expected_count in (
        ("raw artifact", raw_artifacts, 5),
        ("configured artifact", configured_artifacts, 5),
        ("raw exact overlap", raw_exact, 4),
        ("configured exact overlap", configured_exact, 4),
    ):
        if len(frame) != expected_count:
            raise ValueError(f"{label} evidence count is {len(frame)}, expected {expected_count}")
    if set(raw_artifacts["experiment"]) != experiment_set:
        raise ValueError("Raw artifact experiment set differs from production fits")
    if set(configured_artifacts["experiment"]) != experiment_set:
        raise ValueError("Configured artifact experiment set differs from production fits")
    eastern_experiments = set(training.loc[training["host"].isin(EASTERN_HOSTS), "experiment"])
    if set(raw_exact["experiment"]) != eastern_experiments:
        raise ValueError("Raw exact-overlap experiment set differs")
    if set(configured_exact["experiment"]) != eastern_experiments:
        raise ValueError("Configured exact-overlap experiment set differs")

    combined = training.merge(raw_artifacts, on="experiment", validate="one_to_one")
    configured_artifacts = configured_artifacts.add_prefix("configured_").rename(
        columns={"configured_experiment": "experiment"}
    )
    combined = combined.merge(configured_artifacts, on="experiment", validate="one_to_one")
    raw_exact = raw_exact.add_prefix("raw_").rename(columns={"raw_experiment": "experiment"})
    configured_exact = configured_exact.add_prefix("configured_").rename(
        columns={"configured_experiment": "experiment"}
    )
    combined = combined.merge(raw_exact, on="experiment", how="left", validate="one_to_one")
    combined = combined.merge(
        configured_exact, on="experiment", how="left", validate="one_to_one"
    )
    failures = []
    records = []
    for row in combined.to_dict("records"):
        host = row["host"]
        checkpoint_hash = row["best_checkpoint_sha256"]
        dataset_hash = row["training_dataset_csv_sha256"]
        row_failures = []
        for prefix in ("artifact", "configured_artifact"):
            if row[f"{prefix}_checkpoint_sha256"] != checkpoint_hash:
                row_failures.append(f"{prefix}_checkpoint_hash")
            if row[f"{prefix}_dataset_sha256"] != dataset_hash:
                row_failures.append(f"{prefix}_dataset_hash")
        if row["artifact_score_space"] != "raw_sigmoid_probability":
            row_failures.append("raw_artifact_score_space")
        if pd.notna(row["artifact_platt_calibrator_sha256"]):
            row_failures.append("raw_artifact_unexpected_calibrator")
        expected_configured_space = (
            "platt_calibrated_probability"
            if row["postfit_calibration"] == "platt"
            else "raw_sigmoid_probability"
        )
        if row["configured_artifact_score_space"] != expected_configured_space:
            row_failures.append("configured_artifact_score_space")
        expected_calibrator_hash = (
            row.get("platt_calibrator_sha256")
            if row["postfit_calibration"] == "platt"
            else None
        )
        if pd.isna(expected_calibrator_hash):
            expected_calibrator_hash = None
        if row["postfit_calibration"] == "platt" and expected_calibrator_hash is None:
            row_failures.append("missing_training_calibrator_hash")
        observed_calibrator_hash = row[
            "configured_artifact_platt_calibrator_sha256"
        ]
        if pd.isna(observed_calibrator_hash):
            observed_calibrator_hash = None
        if observed_calibrator_hash != expected_calibrator_hash:
            row_failures.append("configured_artifact_calibrator_hash")
        raw_gate = artifact_nonregression_gate(
            host,
            row["artifact_baseline_absolute_did_mean"],
            row["artifact_candidate_absolute_did_mean"],
        )
        configured_gate = artifact_nonregression_gate(
            host,
            row["configured_artifact_baseline_absolute_did_mean"],
            row["configured_artifact_candidate_absolute_did_mean"],
        )
        if not raw_gate["passed"]:
            row_failures.append("raw_matched_boundary_regression")
        if not configured_gate["passed"]:
            row_failures.append("configured_matched_boundary_regression")
        raw_exact_gate = None
        configured_exact_gate = None
        raw_coastal_gate = None
        configured_coastal_gate = None
        if host == "notholithocarpus_densiflorus":
            raw_coastal_values = (
                row.get("artifact_coastal_baseline_absolute_did_mean"),
                row.get("artifact_coastal_candidate_absolute_did_mean"),
            )
            configured_coastal_values = (
                row.get("configured_artifact_coastal_baseline_absolute_did_mean"),
                row.get("configured_artifact_coastal_candidate_absolute_did_mean"),
            )
            raw_coastal_count = row.get("artifact_coastal_pair_count")
            configured_coastal_count = row.get(
                "configured_artifact_coastal_pair_count"
            )
            if not pd.notna(raw_coastal_count) or raw_coastal_count <= 0:
                row_failures.append("raw_tanoak_coastal_pair_count")
            if (
                not pd.notna(configured_coastal_count)
                or configured_coastal_count <= 0
            ):
                row_failures.append("configured_tanoak_coastal_pair_count")
            raw_coastal_gate = artifact_nonregression_gate(
                host, *raw_coastal_values
            )
            configured_coastal_gate = artifact_nonregression_gate(
                host, *configured_coastal_values
            )
            if not raw_coastal_gate["passed"]:
                row_failures.append("raw_tanoak_coastal_regression")
            if not configured_coastal_gate["passed"]:
                row_failures.append("configured_tanoak_coastal_regression")
        if host in EASTERN_HOSTS:
            for prefix in ("raw_exact_overlap", "configured_exact_overlap"):
                if row[f"{prefix}_checkpoint_sha256"] != checkpoint_hash:
                    row_failures.append(f"{prefix}_checkpoint_hash")
                if row[f"{prefix}_dataset_sha256"] != dataset_hash:
                    row_failures.append(f"{prefix}_dataset_hash")
            if row["raw_exact_overlap_score_space"] != "raw_sigmoid_probability":
                row_failures.append("raw_exact_overlap_score_space")
            if pd.notna(row["raw_exact_overlap_platt_calibrator_sha256"]):
                row_failures.append("raw_exact_overlap_unexpected_calibrator")
            if row["configured_exact_overlap_score_space"] != expected_configured_space:
                row_failures.append("configured_exact_overlap_score_space")
            observed_exact_calibrator_hash = row[
                "configured_exact_overlap_platt_calibrator_sha256"
            ]
            if pd.isna(observed_exact_calibrator_hash):
                observed_exact_calibrator_hash = None
            if observed_exact_calibrator_hash != expected_calibrator_hash:
                row_failures.append("configured_exact_overlap_calibrator_hash")
            raw_exact_gate = artifact_nonregression_gate(
                host,
                row["raw_exact_overlap_baseline_absolute_shift_mean"],
                row["raw_exact_overlap_candidate_absolute_shift_mean"],
            )
            configured_exact_gate = artifact_nonregression_gate(
                host,
                row["configured_exact_overlap_baseline_absolute_shift_mean"],
                row["configured_exact_overlap_candidate_absolute_shift_mean"],
            )
            if not raw_exact_gate["passed"]:
                row_failures.append("raw_exact_overlap_regression")
            if not configured_exact_gate["passed"]:
                row_failures.append("configured_exact_overlap_regression")
        records.append(
            {
                "host": host,
                "experiment": row["experiment"],
                "postfit_calibration": row["postfit_calibration"],
                "expected_release_score_space": expected_configured_space,
                "raw_matched_baseline": row["artifact_baseline_absolute_did_mean"],
                "raw_matched_candidate": row["artifact_candidate_absolute_did_mean"],
                "raw_matched_allowed_maximum": raw_gate["allowed_maximum"],
                "raw_matched_passed": raw_gate["passed"],
                "configured_matched_candidate": row[
                    "configured_artifact_candidate_absolute_did_mean"
                ],
                "configured_matched_allowed_maximum": configured_gate[
                    "allowed_maximum"
                ],
                "configured_matched_passed": configured_gate["passed"],
                "raw_tanoak_coastal_candidate": (
                    row.get("artifact_coastal_candidate_absolute_did_mean")
                    if raw_coastal_gate
                    else None
                ),
                "raw_tanoak_coastal_allowed_maximum": (
                    raw_coastal_gate["allowed_maximum"] if raw_coastal_gate else None
                ),
                "raw_tanoak_coastal_passed": (
                    raw_coastal_gate["passed"] if raw_coastal_gate else None
                ),
                "configured_tanoak_coastal_candidate": (
                    row.get("configured_artifact_coastal_candidate_absolute_did_mean")
                    if configured_coastal_gate
                    else None
                ),
                "configured_tanoak_coastal_allowed_maximum": (
                    configured_coastal_gate["allowed_maximum"]
                    if configured_coastal_gate
                    else None
                ),
                "configured_tanoak_coastal_passed": (
                    configured_coastal_gate["passed"]
                    if configured_coastal_gate
                    else None
                ),
                "raw_exact_baseline": (
                    row.get("raw_exact_overlap_baseline_absolute_shift_mean")
                    if host in EASTERN_HOSTS
                    else None
                ),
                "raw_exact_candidate": (
                    row.get("raw_exact_overlap_candidate_absolute_shift_mean")
                    if host in EASTERN_HOSTS
                    else None
                ),
                "raw_exact_allowed_maximum": (
                    raw_exact_gate["allowed_maximum"] if raw_exact_gate else None
                ),
                "raw_exact_passed": raw_exact_gate["passed"] if raw_exact_gate else None,
                "configured_exact_candidate": (
                    row.get("configured_exact_overlap_candidate_absolute_shift_mean")
                    if host in EASTERN_HOSTS
                    else None
                ),
                "configured_exact_allowed_maximum": (
                    configured_exact_gate["allowed_maximum"]
                    if configured_exact_gate
                    else None
                ),
                "configured_exact_passed": (
                    configured_exact_gate["passed"] if configured_exact_gate else None
                ),
                "acceptance_passed": not row_failures,
                "failures": ";".join(row_failures),
            }
        )
        failures.extend(f"{host}: {failure}" for failure in row_failures)
    failures.extend(
        validate_support_audits(
            dataset_overlap_summary, pathway_summary, training
        )
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    host_path = output_dir / "final_host_acceptance.csv"
    write_csv_atomic(host_path, pd.DataFrame(records))
    payload = {
        "status": "complete",
        "production_ready_for_inference": not failures,
        "host_count": len(records),
        "host_acceptance_pass_count": sum(row["acceptance_passed"] for row in records),
        "failure_count": len(failures),
        "failures": failures,
        "training_metrics": str(training_metrics.resolve()),
        "training_metrics_sha256": sha256(training_metrics),
        "blockcv_evidence_dir": str(blockcv_evidence_dir.resolve()),
        "blockcv_summary_sha256": sha256(blockcv_evidence_dir / "summary.json"),
        "production_selection_manifest": str(production_selection_path),
        "production_selection_manifest_sha256": sha256(production_selection_path),
        "raw_artifact_roots": [str(path.resolve()) for path in raw_artifact_roots],
        "configured_artifact_roots": [
            str(path.resolve()) for path in configured_artifact_roots
        ],
        "raw_exact_overlap_roots": [str(path.resolve()) for path in raw_exact_roots],
        "configured_exact_overlap_roots": [
            str(path.resolve()) for path in configured_exact_roots
        ],
        "dataset_overlap_summary": str(dataset_overlap_summary.resolve()),
        "dataset_overlap_summary_sha256": sha256(dataset_overlap_summary),
        "pathway_summary": str(pathway_summary.resolve()),
        "pathway_summary_sha256": sha256(pathway_summary),
        "host_acceptance": str(host_path.resolve()),
        "host_acceptance_sha256": sha256(host_path),
        "pre_registered_artifact_rules": {
            "seam_positive_hosts": sorted(SEAM_POSITIVE_HOSTS),
            "seam_positive_maximum": "June mean absolute shift/DID",
            "low_seam_controls": sorted(HOSTS - SEAM_POSITIVE_HOSTS),
            "control_absolute_tolerance": CONTROL_ABSOLUTE_TOLERANCE,
            "control_relative_tolerance": CONTROL_RELATIVE_TOLERANCE,
            "tanoak_coastal_west_of_123_must_pass": True,
            "raw_model_and_configured_release_must_both_pass": True,
        },
        "guardrail": (
            "This gate approves candidates to enter full Hann inference. It does not "
            "replace structural raster QA, aligned-map correlation, fixed-scale visual "
            "review, or collaborator-facing limitations."
        ),
    }
    write_json_atomic(output_dir / "summary.json", payload)
    (output_dir / "COMPLETED").write_text("completed\n", encoding="utf-8")
    if failures:
        raise RuntimeError(f"Final model acceptance failed: {failures}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-metrics", type=Path, required=True)
    parser.add_argument("--blockcv-evidence-dir", type=Path, required=True)
    parser.add_argument("--raw-artifact-root", type=Path, action="append", required=True)
    parser.add_argument(
        "--configured-artifact-root", type=Path, action="append", required=True
    )
    parser.add_argument("--raw-exact-root", type=Path, action="append", required=True)
    parser.add_argument(
        "--configured-exact-root", type=Path, action="append", required=True
    )
    parser.add_argument("--dataset-overlap-summary", type=Path, required=True)
    parser.add_argument("--pathway-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required = [
        args.training_metrics,
        args.blockcv_evidence_dir / "summary.json",
        args.dataset_overlap_summary,
        args.pathway_summary,
        *args.raw_artifact_root,
        *args.configured_artifact_root,
        *args.raw_exact_root,
        *args.configured_exact_root,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing final acceptance inputs: {missing}")
    payload = synthesize_acceptance(
        args.training_metrics,
        args.blockcv_evidence_dir,
        args.raw_artifact_root,
        args.configured_artifact_root,
        args.raw_exact_root,
        args.configured_exact_root,
        args.dataset_overlap_summary,
        args.pathway_summary,
        args.output_dir,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
