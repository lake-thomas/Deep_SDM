#!/usr/bin/env python3
"""Version and rebuild August manifests from immutable materialized chips.

This utility never extracts imagery. It reconstructs model manifests from the
canonical successful materialization records and existing source-safe split
assignments, validates the complete staged package, archives the previous
manifest revision, and only then promotes the staged package atomically.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd

from host_dataset_builder.candidate_manifests import (
    _builder_source_hashes,
    _git_state,
    _sha256,
)
from host_dataset_builder.materialize_candidates import (
    _atomic_json,
    build_model_manifests,
)


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def refinalize(
    candidate_package: Path,
    output_root: Path,
    revision_id: str,
    rationale: str,
) -> dict:
    candidate_package = candidate_package.resolve()
    output_root = output_root.resolve()
    state_path = output_root / "materialization_run.json"
    metadata_path = candidate_package / "manifest_metadata.json"
    records_path = output_root / "materialization_records.csv"
    assignments_path = output_root / "actual_source_safe_split_assignments.csv"
    assignment_audit_path = output_root / "candidate_to_actual_split_audit.csv"
    required = [
        state_path,
        metadata_path,
        records_path,
        assignments_path,
        assignment_audit_path,
        output_root / "model_manifests",
        output_root / "model_manifest_summary.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Refinalization inputs are missing: {missing}")

    state = _read_json(state_path)
    if state.get("status") != "complete" or not state.get("manifest_status", {}).get(
        "production_ready_manifests"
    ):
        raise RuntimeError("Existing materialization is not production-ready")
    identity = state.get("identity", {})
    if Path(identity.get("candidate_package", "")).resolve() != candidate_package:
        raise RuntimeError("Candidate package differs from the materialization identity")
    if identity.get("candidate_metadata_sha256") != _sha256(metadata_path):
        raise RuntimeError("Candidate metadata hash differs from the materialization identity")

    revision_root = output_root / "manifest_revisions" / revision_id
    staging_root = output_root / f".manifest_revision_staging_{revision_id}"
    if revision_root.exists() or staging_root.exists():
        raise FileExistsError(
            f"Refusing existing revision/staging path: {revision_root}; {staging_root}"
        )

    metadata = _read_json(metadata_path)
    records = pd.read_csv(records_path, low_memory=False)
    successes = records[records["materialization_status"] == "success"].copy()
    assignments = pd.read_csv(assignments_path, low_memory=False)
    assignment_audit = pd.read_csv(assignment_audit_path, low_memory=False)
    successful_components = assignment_audit[["sample_id", "actual_source_component"]]
    if successes["sample_id"].duplicated().any():
        raise RuntimeError("Canonical successful records contain duplicate sample IDs")
    if set(successes["sample_id"]) != set(assignments["sample_id"]):
        raise RuntimeError("Successful records and source-safe assignments differ")
    if set(successes["sample_id"]) != set(successful_components["sample_id"]):
        raise RuntimeError("Successful records and component audit differ")

    n_folds = int(identity["n_folds"])
    seed = int(identity["seed"])
    staging_root.mkdir(parents=True, exist_ok=False)
    try:
        manifest_status = build_model_manifests(
            candidate_package=candidate_package,
            output_root=staging_root,
            extraction_candidates=successful_components,
            successful_records=successes,
            assignments=assignments,
            metadata=metadata,
            n_folds=n_folds,
            seed=seed,
            dataset_root=output_root,
        )
        if not manifest_status.get("production_ready_manifests"):
            raise RuntimeError("Staged manifest package did not pass its hard QA gate")

        previous_state_sha256 = _sha256(state_path)
        previous_summary_sha256 = _sha256(output_root / "model_manifest_summary.csv")
        revision_root.mkdir(parents=True, exist_ok=False)
        shutil.copy2(state_path, revision_root / "materialization_run_before_revision.json")
        os.replace(
            output_root / "model_manifests",
            revision_root / "model_manifests",
        )
        os.replace(
            output_root / "model_manifest_summary.csv",
            revision_root / "model_manifest_summary.csv",
        )
        for qa_name in [
            "materialization_attrition_audit",
            "candidate_builder_provenance",
        ]:
            qa_path = output_root / "qa" / qa_name
            if qa_path.exists():
                archived_qa_root = revision_root / "qa"
                archived_qa_root.mkdir(parents=True, exist_ok=True)
                os.replace(qa_path, archived_qa_root / qa_name)

        os.replace(staging_root / "model_manifests", output_root / "model_manifests")
        os.replace(
            staging_root / "model_manifest_summary.csv",
            output_root / "model_manifest_summary.csv",
        )
        staging_root.rmdir()

        completed_at = pd.Timestamp.now(tz="America/New_York").isoformat()
        revision = {
            "status": "complete",
            "revision_id": revision_id,
            "completed_at": completed_at,
            "rationale": rationale,
            "algorithm": "same_source_domain_then_global_nested_rank",
            "chip_extraction_repeated": False,
            "candidate_package": str(candidate_package),
            "candidate_metadata_sha256": _sha256(metadata_path),
            "canonical_materialization_records": str(records_path),
            "canonical_materialization_records_sha256": _sha256(records_path),
            "source_safe_assignments": str(assignments_path),
            "source_safe_assignments_sha256": _sha256(assignments_path),
            "previous_materialization_state_sha256": previous_state_sha256,
            "previous_manifest_summary_sha256": previous_summary_sha256,
            "archived_revision": str(revision_root),
            "new_manifest_summary_sha256": _sha256(
                output_root / "model_manifest_summary.csv"
            ),
            "builder_git": _git_state(Path(__file__).resolve().parent),
            "builder_source_sha256": _builder_source_hashes(
                Path(__file__).resolve().parent / "host_dataset_builder"
            ),
            "manifest_status": manifest_status,
        }
        _atomic_json(revision, revision_root / "revision_metadata.json")
        _atomic_json(revision, output_root / "manifest_revision.json")
        state["manifest_status"] = manifest_status
        state["manifests_refinalized_at"] = completed_at
        state["manifest_revision"] = revision
        _atomic_json(state, state_path)
        return revision
    except Exception:
        # Retain a completed staging tree for diagnosis. Before promotion the
        # canonical package is untouched; after promotion the archived revision
        # provides a complete recovery point.
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-package", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--revision-id", required=True)
    parser.add_argument("--rationale", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = refinalize(
        candidate_package=args.candidate_package,
        output_root=args.output_root,
        revision_id=args.revision_id,
        rationale=args.rationale,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
