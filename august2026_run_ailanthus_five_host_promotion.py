#!/usr/bin/env python3
"""Assemble an accepted Ailanthus replacement and rerun five-host acceptance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from august2026_ailanthus_production_train import DATE_TAG
from august2026_assemble_ailanthus_replacement_inputs import assemble_inputs
from august2026_experiments import read_json, sha256
from august2026_production_train import PRODUCTION_OUTPUT_ROOT
from august2026_run_augmentation_acceptance import current_clean_commit
from august2026_run_final_model_acceptance import run_final_acceptance


CODE_ROOT = Path(__file__).resolve().parent


def validate_existing_assembly(
    output_dir: Path,
    original_registry: Path,
    original_acceptance: Path,
    replacement_acceptance: Path,
) -> dict:
    summary_path = output_dir / "summary.json"
    marker = output_dir / "COMPLETED"
    if not summary_path.is_file() or not marker.is_file():
        raise FileNotFoundError("Existing five-host assembly is incomplete")
    payload = read_json(summary_path)
    if (
        payload.get("status") != "complete"
        or payload.get("original_acceptance_sha256") != sha256(original_acceptance)
        or payload.get("replacement_acceptance_sha256")
        != sha256(replacement_acceptance)
    ):
        raise ValueError("Existing five-host assembly identity changed")
    design_metadata = output_dir / "production" / "design_metadata.json"
    if not design_metadata.is_file():
        raise FileNotFoundError("Existing five-host assembly lacks design metadata")
    design = read_json(design_metadata)
    if (
        Path(design.get("source_original_registry", "")).resolve()
        != original_registry.resolve()
        or design.get("source_original_registry_sha256") != sha256(original_registry)
    ):
        raise ValueError("Existing five-host assembly original registry changed")
    for path_key, hash_key in (
        ("production_registry", "production_registry_sha256"),
        ("production_selection_manifest", "production_selection_manifest_sha256"),
    ):
        path = Path(payload[path_key])
        if not path.is_file() or sha256(path) != payload[hash_key]:
            raise ValueError(f"Existing five-host assembly artifact changed: {path_key}")
    blockcv_summary = Path(payload["blockcv_evidence_dir"]) / "summary.json"
    if not blockcv_summary.is_file() or sha256(blockcv_summary) != payload[
        "blockcv_summary_sha256"
    ]:
        raise ValueError("Existing combined BlockCV summary changed")
    return payload


def run_stage(
    original_registry: Path,
    original_acceptance: Path,
    replacement_acceptance: Path,
    assembly_dir: Path,
    device: str,
    batch_size: int,
    samples_per_origin_class: int,
    resume_completed: bool,
) -> dict:
    commit = current_clean_commit(CODE_ROOT)
    if assembly_dir.exists():
        if not resume_completed:
            raise FileExistsError(f"Refusing existing assembly: {assembly_dir}")
        assembly = validate_existing_assembly(
            assembly_dir,
            original_registry,
            original_acceptance,
            replacement_acceptance,
        )
    else:
        assembly = assemble_inputs(
            original_registry,
            original_acceptance,
            replacement_acceptance,
            assembly_dir,
            commit,
        )
    assembly_summary = assembly_dir / "summary.json"
    acceptance_root = (
        PRODUCTION_OUTPUT_ROOT
        / f"final_model_acceptance_ailanthus_replacement_"
        f"{sha256(assembly_summary)[:10]}_{DATE_TAG}"
    )
    return run_final_acceptance(
        Path(assembly["production_registry"]),
        Path(assembly["blockcv_evidence_dir"]),
        acceptance_root,
        device,
        batch_size,
        samples_per_origin_class,
        resume_completed=resume_completed,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-registry", type=Path, required=True)
    parser.add_argument("--original-acceptance", type=Path, required=True)
    parser.add_argument("--replacement-acceptance", type=Path, required=True)
    parser.add_argument("--assembly-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--samples-per-origin-class", type=int, default=512)
    parser.add_argument("--resume-completed", action="store_true")
    args = parser.parse_args()
    for path in (
        args.original_registry,
        args.original_acceptance,
        args.replacement_acceptance,
    ):
        if not path.is_file():
            parser.error(f"required input does not exist: {path}")
    if args.batch_size <= 0 or args.samples_per_origin_class <= 0:
        parser.error("batch and common-support sample sizes must be positive")
    return args


def main() -> None:
    args = parse_args()
    print(json.dumps(run_stage(
        args.original_registry,
        args.original_acceptance,
        args.replacement_acceptance,
        args.assembly_dir,
        args.device,
        args.batch_size,
        args.samples_per_origin_class,
        args.resume_completed,
    ), indent=2))


if __name__ == "__main__":
    main()
