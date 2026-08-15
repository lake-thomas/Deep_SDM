#!/usr/bin/env python3
"""Validate 25 frozen BlockCV fits and synthesize production-gate evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from august2026_blockcv_evidence import synthesize_blockcv
from august2026_run_augmentation_acceptance import current_clean_commit
from august2026_summarize_experiments import (
    read_json,
    sha256,
    summarize_registry,
    write_json_atomic,
)


EXPECTED_EXPERIMENT_COUNT = 25


def _resume_status(
    status_path: Path,
    registry: Path,
    selection_manifest: Path,
    training_metrics: Path,
    code_commit: str,
) -> dict:
    status = read_json(status_path)
    expected = {
        "registry_sha256": sha256(registry),
        "selection_manifest_sha256": sha256(selection_manifest),
        "training_metrics_sha256": sha256(training_metrics),
        "code_commit": code_commit,
    }
    observed = {key: status.get(key) for key in expected}
    if observed != expected:
        raise ValueError(
            f"BlockCV acceptance resume identity changed: expected={expected}, "
            f"observed={observed}"
        )
    return status


def _reuse_evidence(
    evidence_root: Path,
    selection_manifest: Path,
    training_metrics: Path,
) -> dict:
    summary_path = evidence_root / "summary.json"
    marker = evidence_root / "COMPLETED"
    if not summary_path.is_file() or not marker.is_file():
        raise FileNotFoundError("Existing BlockCV evidence package is incomplete")
    evidence = read_json(summary_path)
    expected = {
        "status": "complete",
        "production_ready": True,
        "selection_manifest_sha256": sha256(selection_manifest),
        "training_metrics_sha256": sha256(training_metrics),
        "host_count": 5,
        "expected_fold_count_per_host": 5,
    }
    observed = {key: evidence.get(key) for key in expected}
    if observed != expected:
        raise ValueError(
            f"Existing BlockCV evidence is not reusable: expected={expected}, "
            f"observed={observed}"
        )
    for key in ("fold_metrics", "host_summary"):
        path = Path(evidence.get(key, ""))
        if not path.is_file():
            raise FileNotFoundError(f"BlockCV evidence lacks {key}: {path}")
    return evidence


def run_blockcv_acceptance(
    registry: Path,
    selection_manifest: Path,
    output_root: Path,
    resume_completed: bool = False,
) -> dict:
    selection = read_json(selection_manifest)
    if (
        selection.get("status") != "complete"
        or selection.get("selection_type") != "recipe_for_blockcv"
        or selection.get("selection_count") != 5
    ):
        raise RuntimeError("Selection manifest is not approved for five-host BlockCV")

    # This both materializes the canonical metrics table and fails before creating
    # an acceptance root unless every frozen fit passes traceability checks.
    training_status = summarize_registry(registry, allow_incomplete=False)
    if (
        training_status["experiment_count"] != EXPECTED_EXPERIMENT_COUNT
        or training_status["complete_gate_count"] != EXPECTED_EXPERIMENT_COUNT
        or not training_status["all_experiments_evidence_complete"]
    ):
        raise RuntimeError(f"BlockCV training gate is incomplete: {training_status}")
    training_metrics = Path(training_status["metrics_csv"])

    code_root = Path(__file__).resolve().parent
    code_commit = current_clean_commit(code_root)
    status_path = output_root / "acceptance_status.json"
    evidence_root = output_root / "blockcv_evidence"
    if output_root.exists() and any(output_root.iterdir()):
        if not resume_completed or not status_path.is_file():
            raise FileExistsError(f"Refusing nonempty BlockCV acceptance root: {output_root}")
        status = _resume_status(
            status_path,
            registry,
            selection_manifest,
            training_metrics,
            code_commit,
        )
    else:
        output_root.mkdir(parents=True, exist_ok=True)
        status = {
            "status": "running",
            "stage": "blockcv-acceptance",
            "registry": str(registry.resolve()),
            "registry_sha256": sha256(registry),
            "selection_manifest": str(selection_manifest.resolve()),
            "selection_manifest_sha256": sha256(selection_manifest),
            "training_metrics": str(training_metrics.resolve()),
            "training_metrics_sha256": sha256(training_metrics),
            "code_commit": code_commit,
        }
        write_json_atomic(status_path, status)

    if evidence_root.exists():
        if not resume_completed:
            raise FileExistsError(f"Refusing existing BlockCV evidence: {evidence_root}")
        evidence = _reuse_evidence(
            evidence_root, selection_manifest, training_metrics
        )
    else:
        evidence = synthesize_blockcv(
            selection_manifest, training_metrics, evidence_root
        )

    status.update(
        {
            "status": "complete",
            "production_ready": evidence["production_ready"],
            "blockcv_evidence": str((evidence_root / "summary.json").resolve()),
            "blockcv_evidence_sha256": sha256(evidence_root / "summary.json"),
        }
    )
    write_json_atomic(status_path, status)
    (output_root / "COMPLETED").write_text("completed\n", encoding="utf-8")
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume-completed", action="store_true")
    args = parser.parse_args()
    for path in (args.registry, args.selection_manifest):
        if not path.is_file():
            parser.error(f"Missing required input: {path}")
    return args


def main() -> None:
    args = parse_args()
    status = run_blockcv_acceptance(
        args.registry,
        args.selection_manifest,
        args.output_root,
        resume_completed=args.resume_completed,
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
