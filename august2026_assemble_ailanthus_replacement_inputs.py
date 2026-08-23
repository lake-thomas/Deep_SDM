#!/usr/bin/env python3
"""Assemble one accepted Ailanthus replacement with four frozen production hosts."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

from august2026_combine_dataset_screen_evidence import HOSTS
from august2026_loss_screen_evidence import strict_evidence_gate
from august2026_summarize_experiments import (
    read_json,
    sha256,
    write_json_atomic,
)


AILANTHUS = "ailanthus_altissima"


def write_frame(path: Path, frame: pd.DataFrame, sep: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    frame.to_csv(partial, sep=sep, index=False)
    partial.replace(path)


def checked_pointer(payload: dict, path_key: str, hash_key: str) -> Path:
    path = Path(payload.get(path_key, ""))
    if not path.is_file() or sha256(path) != payload.get(hash_key):
        raise ValueError(f"Source pointer changed: {path_key}")
    return path.resolve()


def exact_host_frame(frame: pd.DataFrame, label: str, rows_per_host: int) -> None:
    if set(frame["host"]) != HOSTS:
        raise ValueError(f"{label} host set changed")
    counts = frame.groupby("host").size().to_dict()
    if counts != {host: rows_per_host for host in HOSTS}:
        raise ValueError(f"{label} row counts changed: {counts}")


def combine_by_host(
    original: pd.DataFrame, replacement: pd.DataFrame, label: str,
    rows_per_host: int,
) -> pd.DataFrame:
    if set(replacement["host"]) != {AILANTHUS}:
        raise ValueError(f"{label} replacement is not Ailanthus-only")
    combined = pd.concat(
        [original[original["host"] != AILANTHUS], replacement],
        ignore_index=True, sort=False,
    )
    exact_host_frame(combined, label, rows_per_host)
    return combined.sort_values(["host"] + (["fold"] if "fold" in combined else []))


def combine_selections(original: dict, replacement: dict, label: str) -> list[dict]:
    old = original.get("selections") or []
    new = replacement.get("selections") or []
    if {item.get("host") for item in old} != HOSTS:
        raise ValueError(f"Original {label} selection host set changed")
    if len(new) != 1 or new[0].get("host") != AILANTHUS:
        raise ValueError(f"Replacement {label} selection is not Ailanthus-only")
    combined = [item for item in old if item.get("host") != AILANTHUS] + new
    if {item.get("host") for item in combined} != HOSTS or len(combined) != len(HOSTS):
        raise ValueError(f"Combined {label} selection host set changed")
    return sorted(combined, key=lambda item: item["host"])


def assemble_inputs(
    original_registry: Path,
    original_acceptance_path: Path,
    replacement_acceptance_path: Path,
    output_dir: Path,
    code_commit: str,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Refusing existing replacement assembly: {output_dir}")
    original_acceptance = read_json(original_acceptance_path)
    replacement_acceptance = read_json(replacement_acceptance_path)
    if (
        original_acceptance.get("status") != "complete"
        or original_acceptance.get("production_ready_for_inference") is not False
        or original_acceptance.get("host_acceptance_pass_count") != 4
        or not original_acceptance.get("failures")
        or any(
            not str(failure).startswith(f"{AILANTHUS}:")
            for failure in original_acceptance.get("failures", [])
        )
    ):
        raise RuntimeError("Original acceptance is not the four-pass Ailanthus failure")
    if (
        replacement_acceptance.get("status") != "complete"
        or replacement_acceptance.get("stage")
        != "ailanthus-replacement-final-acceptance"
        or replacement_acceptance.get("production_ready_for_release_assembly") is not True
    ):
        raise RuntimeError("Replacement Ailanthus acceptance did not pass")

    original_metrics_path = checked_pointer(
        original_acceptance, "training_metrics", "training_metrics_sha256"
    )
    original_blockcv_summary = (
        Path(original_acceptance["blockcv_evidence_dir"]) / "summary.json"
    ).resolve()
    if sha256(original_blockcv_summary) != original_acceptance["blockcv_summary_sha256"]:
        raise ValueError("Original BlockCV summary changed")
    original_production_selection_path = checked_pointer(
        original_acceptance,
        "production_selection_manifest",
        "production_selection_manifest_sha256",
    )
    replacement_registry = checked_pointer(
        replacement_acceptance, "registry", "registry_sha256"
    )
    replacement_metrics_path = checked_pointer(
        replacement_acceptance, "training_metrics", "training_metrics_sha256"
    )
    replacement_blockcv_summary = (
        Path(replacement_acceptance["blockcv_evidence_dir"]) / "summary.json"
    ).resolve()
    if sha256(replacement_blockcv_summary) != replacement_acceptance["blockcv_summary_sha256"]:
        raise ValueError("Replacement BlockCV summary changed")
    replacement_production_selection_path = checked_pointer(
        replacement_acceptance,
        "production_selection_manifest",
        "production_selection_manifest_sha256",
    )
    original_production_selection = read_json(original_production_selection_path)
    new_production_selection = read_json(replacement_production_selection_path)
    if (
        original_production_selection.get("blockcv_evidence_summary_sha256")
        != sha256(original_blockcv_summary)
        or new_production_selection.get("blockcv_evidence_summary_sha256")
        != sha256(replacement_blockcv_summary)
    ):
        raise ValueError("Production selections do not reference supplied BlockCV evidence")

    original_registry_frame = pd.read_csv(original_registry, sep="\t", low_memory=False)
    replacement_registry_frame = pd.read_csv(replacement_registry, sep="\t", low_memory=False)
    original_metrics = pd.read_csv(original_metrics_path, low_memory=False)
    replacement_metrics = pd.read_csv(replacement_metrics_path, low_memory=False)
    exact_host_frame(original_registry_frame, "original production registry", 1)
    exact_host_frame(original_metrics, "original production metrics", 1)
    if set(original_registry_frame["experiment"]) != set(original_metrics["experiment"]):
        raise ValueError("Original registry and metrics experiments differ")
    strict_evidence_gate(original_metrics)
    strict_evidence_gate(replacement_metrics)
    combined_registry = combine_by_host(
        original_registry_frame, replacement_registry_frame,
        "combined production registry", 1,
    )

    old_blockcv = read_json(original_blockcv_summary)
    new_blockcv = read_json(replacement_blockcv_summary)
    for payload, host_count in ((old_blockcv, 5), (new_blockcv, 1)):
        if (
            payload.get("status") != "complete"
            or payload.get("production_ready") is not True
            or payload.get("host_count") != host_count
        ):
            raise RuntimeError("Source BlockCV evidence is not production-ready")
    old_blockcv_selection_path = checked_pointer(
        old_blockcv, "selection_manifest", "selection_manifest_sha256"
    )
    new_blockcv_selection_path = checked_pointer(
        new_production_selection,
        "blockcv_selection_manifest",
        "blockcv_selection_manifest_sha256",
    )
    old_blockcv_selection = read_json(old_blockcv_selection_path)
    new_blockcv_selection = read_json(new_blockcv_selection_path)
    combined_blockcv_selections = combine_selections(
        old_blockcv_selection, new_blockcv_selection, "BlockCV"
    )

    old_blockcv_selection_from_production = checked_pointer(
        original_production_selection,
        "blockcv_selection_manifest",
        "blockcv_selection_manifest_sha256",
    )
    if old_blockcv_selection_from_production != old_blockcv_selection_path:
        raise ValueError("Original BlockCV selection pointers differ")

    old_fold_path = checked_pointer(
        original_production_selection,
        "blockcv_fold_metrics", "blockcv_fold_metrics_sha256",
    )
    new_fold_path = checked_pointer(
        new_production_selection,
        "blockcv_fold_metrics", "blockcv_fold_metrics_sha256",
    )
    old_host_path = checked_pointer(
        original_production_selection,
        "blockcv_host_summary", "blockcv_host_summary_sha256",
    )
    new_host_path = checked_pointer(
        new_production_selection,
        "blockcv_host_summary", "blockcv_host_summary_sha256",
    )
    for payload, fold_path_value, host_path_value in (
        (old_blockcv, old_fold_path, old_host_path),
        (new_blockcv, new_fold_path, new_host_path),
    ):
        if (
            Path(payload["fold_metrics"]).resolve() != fold_path_value
            or Path(payload["host_summary"]).resolve() != host_path_value
        ):
            raise ValueError("BlockCV evidence pointers differ")
    old_fold = pd.read_csv(old_fold_path, low_memory=False)
    new_fold = pd.read_csv(new_fold_path, low_memory=False)
    old_host = pd.read_csv(old_host_path, low_memory=False)
    new_host = pd.read_csv(new_host_path, low_memory=False)
    old_training_path = checked_pointer(
        old_blockcv, "training_metrics", "training_metrics_sha256"
    )
    new_training_path = checked_pointer(
        new_blockcv, "training_metrics", "training_metrics_sha256"
    )
    old_training = pd.read_csv(old_training_path, low_memory=False)
    new_training = pd.read_csv(new_training_path, low_memory=False)
    combined_fold = combine_by_host(old_fold, new_fold, "combined BlockCV folds", 5)
    combined_host = combine_by_host(old_host, new_host, "combined BlockCV hosts", 1)
    combined_training = combine_by_host(
        old_training, new_training, "combined BlockCV training metrics", 5
    )
    host_gates = combined_host["blockcv_production_gate_passed"].map(
        lambda value: value is True or str(value).strip().lower() == "true"
    )
    if not host_gates.all():
        raise RuntimeError("A combined BlockCV host failed its production gate")

    combined_production_selections = combine_selections(
        original_production_selection, new_production_selection, "production"
    )
    output_dir.mkdir(parents=True)
    registry_path = output_dir / "production" / "experiment_registry.tsv"
    write_frame(registry_path, combined_registry, sep="\t")
    write_json_atomic(registry_path.parent / "design_metadata.json", {
        "status": "complete", "stage": "five-host-production-registry-assembly",
        "code_commit": None,
        "assembly_code_commit": code_commit,
        "source_original_registry": str(original_registry.resolve()),
        "source_original_registry_sha256": sha256(original_registry),
        "source_replacement_registry": str(replacement_registry),
        "source_replacement_registry_sha256": sha256(replacement_registry),
        "note": "No single training commit exists because four frozen fits and one later replacement are assembled without retraining.",
    })

    blockcv_dir = output_dir / "blockcv_evidence"
    blockcv_selection_path = output_dir / "selection" / "combined_blockcv_selection.json"
    production_selection_path = output_dir / "selection" / "production_selection.json"
    write_json_atomic(blockcv_selection_path, {
        "status": "complete",
        "selection_type": "combined_ailanthus_replacement_recipe_for_blockcv_evidence",
        "code_commit": code_commit,
        "source_blockcv_selection_manifests": [
            {"path": str(path), "sha256": sha256(path)}
            for path in (old_blockcv_selection_path, new_blockcv_selection_path)
        ],
        "selection_count": len(combined_blockcv_selections),
        "selections": combined_blockcv_selections,
    })
    fold_path = blockcv_dir / "blockcv_fold_metrics.csv"
    host_path = blockcv_dir / "blockcv_host_summary.csv"
    blockcv_training_path = blockcv_dir / "experiment_metrics.csv"
    write_frame(fold_path, combined_fold)
    write_frame(host_path, combined_host)
    write_frame(blockcv_training_path, combined_training)
    blockcv_summary_path = blockcv_dir / "summary.json"
    write_json_atomic(blockcv_summary_path, {
        "status": "complete", "production_ready": True,
        "selection_manifest": str(blockcv_selection_path.resolve()),
        "selection_manifest_sha256": sha256(blockcv_selection_path),
        "training_metrics": str(blockcv_training_path.resolve()),
        "training_metrics_sha256": sha256(blockcv_training_path),
        "host_count": len(HOSTS), "expected_fold_count_per_host": 5,
        "hard_failure_count": 0, "hard_failures": [],
        "thresholds": old_blockcv.get("thresholds"),
        "fold_metrics": str(fold_path.resolve()),
        "fold_metrics_sha256": sha256(fold_path),
        "host_summary": str(host_path.resolve()),
        "host_summary_sha256": sha256(host_path),
        "combined_production_selection_manifest": str(production_selection_path.resolve()),
        "calibration_guardrail": old_blockcv.get("calibration_guardrail"),
        "source_blockcv_summaries": [
            {"path": str(path), "sha256": sha256(path)}
            for path in (original_blockcv_summary, replacement_blockcv_summary)
        ],
    })
    write_json_atomic(production_selection_path, {
        "status": "complete",
        "selection_type": "blockcv_approved_recipe_for_production_fit",
        "code_commit": code_commit,
        "blockcv_evidence_dir": str(blockcv_dir.resolve()),
        "blockcv_evidence_summary": str(blockcv_summary_path.resolve()),
        "blockcv_evidence_summary_sha256": sha256(blockcv_summary_path),
        "blockcv_host_summary": str(host_path.resolve()),
        "blockcv_host_summary_sha256": sha256(host_path),
        "blockcv_fold_metrics": str(fold_path.resolve()),
        "blockcv_fold_metrics_sha256": sha256(fold_path),
        "blockcv_selection_manifest": str(blockcv_selection_path.resolve()),
        "blockcv_selection_manifest_sha256": sha256(blockcv_selection_path),
        "source_production_selection_manifests": [
            {"path": str(path), "sha256": sha256(path)}
            for path in (
                original_production_selection_path,
                replacement_production_selection_path,
            )
        ],
        "selection_count": len(combined_production_selections),
        "selections": combined_production_selections,
        "guardrail": (
            "This manifest authorizes a fresh complete five-host final acceptance "
            "rerun. It does not itself authorize inference."
        ),
    })
    (blockcv_dir / "COMPLETED").write_text("complete\n", encoding="utf-8")
    payload = {
        "status": "complete", "stage": "ailanthus-replacement-five-host-assembly",
        "code_commit": code_commit,
        "production_registry": str(registry_path.resolve()),
        "production_registry_sha256": sha256(registry_path),
        "blockcv_evidence_dir": str(blockcv_dir.resolve()),
        "blockcv_summary_sha256": sha256(blockcv_summary_path),
        "production_selection_manifest": str(production_selection_path.resolve()),
        "production_selection_manifest_sha256": sha256(production_selection_path),
        "original_acceptance": str(original_acceptance_path.resolve()),
        "original_acceptance_sha256": sha256(original_acceptance_path),
        "replacement_acceptance": str(replacement_acceptance_path.resolve()),
        "replacement_acceptance_sha256": sha256(replacement_acceptance_path),
        "next_gate": "Run the complete five-host final acceptance suite on these assembled inputs.",
    }
    write_json_atomic(output_dir / "summary.json", payload)
    (output_dir / "COMPLETED").write_text("complete\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-registry", type=Path, required=True)
    parser.add_argument("--original-acceptance", type=Path, required=True)
    parser.add_argument("--replacement-acceptance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in (
        args.original_registry, args.original_acceptance, args.replacement_acceptance
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    code_root = Path(__file__).resolve().parent
    if subprocess.run(
        ["git", "status", "--porcelain"], cwd=code_root, check=True,
        capture_output=True, text=True,
    ).stdout.strip():
        raise RuntimeError("Refusing replacement assembly from a dirty tree")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=code_root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    print(json.dumps(assemble_inputs(
        args.original_registry, args.original_acceptance,
        args.replacement_acceptance, args.output_dir, commit,
    ), indent=2))


if __name__ == "__main__":
    main()
