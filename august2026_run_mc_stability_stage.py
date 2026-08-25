#!/usr/bin/env python3
"""Run, resume, and synthesize the accepted five-host MC stability screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from august2026_experiments import read_json, sha256
from august2026_mc_stability_evidence import PASS_LEVELS, synthesize_evidence
from august2026_mc_stability_screen import (
    materialize_analysis_manifest,
    run_design,
    versioned_roots,
    write_design,
)
from august2026_run_augmentation_acceptance import current_clean_commit


CODE_ROOT = Path(__file__).resolve().parent


def existing_design(
    production_registry: Path,
    final_acceptance: Path,
    design_root: Path,
    output_root: Path,
) -> tuple[Path, list[dict]]:
    registry = design_root / "experiment_registry.tsv"
    metadata_path = design_root / "design_metadata.json"
    if not registry.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("Existing MC stability design is incomplete")
    metadata = read_json(metadata_path)
    expected = {
        "production_registry_sha256": sha256(production_registry),
        "final_acceptance_sha256": sha256(final_acceptance),
        "code_commit": current_clean_commit(CODE_ROOT),
        "design_root": str(design_root.resolve()),
        "output_root": str(output_root.resolve()),
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("MC stability resume identity changed")
    rows = pd.read_csv(registry, sep="\t", keep_default_na=False).to_dict("records")
    return registry, rows


def existing_evidence(manifest: Path, evidence_dir: Path) -> dict:
    summary = evidence_dir / "summary.json"
    marker = evidence_dir / "COMPLETED"
    if not summary.is_file() or not marker.is_file():
        raise FileNotFoundError("Existing MC stability evidence is incomplete")
    payload = read_json(summary)
    if (
        payload.get("status") != "complete"
        or payload.get("manifest_sha256") != sha256(manifest)
        or payload.get("recommended_mc_samples") not in PASS_LEVELS
    ):
        raise ValueError("Existing MC stability evidence identity changed")
    evidence = Path(payload.get("evidence", ""))
    if not evidence.is_file() or sha256(evidence) != payload.get("evidence_sha256"):
        raise ValueError("Existing MC stability table changed")
    return payload


def run_stage(
    production_registry: Path,
    final_acceptance: Path,
    resume_completed: bool,
    design_root: Path | None = None,
    output_root: Path | None = None,
) -> dict:
    resolved_design, resolved_output = versioned_roots(final_acceptance)
    design_root = design_root or resolved_design
    output_root = output_root or resolved_output
    registry_path = design_root / "experiment_registry.tsv"
    if registry_path.exists():
        if not resume_completed:
            raise FileExistsError(f"Refusing existing MC stability design: {registry_path}")
        registry, rows = existing_design(
            production_registry, final_acceptance, design_root, output_root
        )
    else:
        registry, rows = write_design(
            production_registry, final_acceptance,
            design_root=design_root, output_root=output_root,
        )
    run_design(registry, rows, resume_completed=resume_completed)
    manifest = materialize_analysis_manifest(registry)
    evidence_dir = output_root / "evidence"
    if evidence_dir.exists():
        if not resume_completed:
            raise FileExistsError(f"Refusing existing MC evidence: {evidence_dir}")
        return existing_evidence(manifest, evidence_dir)
    return synthesize_evidence(manifest, evidence_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-registry", type=Path, required=True)
    parser.add_argument("--final-acceptance", type=Path, required=True)
    parser.add_argument("--resume-completed", action="store_true")
    args = parser.parse_args()
    for path in (args.production_registry, args.final_acceptance):
        if not path.is_file():
            parser.error(f"required input does not exist: {path}")
    return args


def main() -> None:
    args = parse_args()
    print(json.dumps(run_stage(
        args.production_registry,
        args.final_acceptance,
        args.resume_completed,
    ), indent=2))


if __name__ == "__main__":
    main()
