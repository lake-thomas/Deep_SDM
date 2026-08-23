#!/usr/bin/env python3
"""Freeze and optionally run the bounded Ailanthus image-modality-dropout screen."""

from __future__ import annotations

import argparse
import subprocess
from copy import deepcopy
from pathlib import Path

import pandas as pd

from august2026_ailanthus_consistency_screen import (
    ROBUSTNESS_EVIDENCE_DIR,
    validated_base,
)
from august2026_experiments import (
    DESIGN_BASE,
    OUTPUT_BASE,
    read_json,
    run_experiments,
    sha256,
    write_json_atomic,
    write_tsv_atomic,
)


DATE_TAG = "20260823"
HOST = "ailanthus_altissima"
MODALITY_DROPOUT_PROBABILITIES = (0.10, 0.25, 0.50)
CONSISTENCY_EVIDENCE_DIR = Path(
    "/mnt/rsstu_dsfas_gsv_naip/Host_NAIP_SDM/Outputs_Host_NAIP_SDMs/"
    "experimental_model_runs_august2026/"
    "stage06_ailanthus_consistency_edc283deaf_20260823/acceptance/evidence"
)


def experiment_name(probability: float) -> str:
    encoded = int(round(probability * 100))
    return (
        f"stage07_{HOST}_db_bg20000_posw_dih_rad2_d025_s005_"
        f"moddrop{encoded:02d}_uniform_{DATE_TAG}"
    )


def strict_bool(value, name: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"{name} is not boolean: {value}")
    return normalized == "true"


def validated_failed_consistency(evidence_dir: Path) -> str:
    summary_path = evidence_dir / "summary.json"
    evidence_path = evidence_dir / "ailanthus_consistency_candidate_evidence.csv"
    marker = evidence_dir / "COMPLETED"
    for path in (summary_path, evidence_path, marker):
        if not path.is_file():
            raise FileNotFoundError(path)
    summary = read_json(summary_path)
    if (
        summary.get("status") != "complete"
        or summary.get("stage") != "ailanthus-radiometric-consistency-screen"
        or summary.get("candidate_count") != 3
        or summary.get("advancement_candidate_count") != 0
        or summary.get("candidate_evidence_sha256") != sha256(evidence_path)
    ):
        raise RuntimeError("The Ailanthus consistency screen is not a clean failure")
    frame = pd.read_csv(evidence_path, low_memory=False)
    if len(frame) != 3 or frame["experiment"].duplicated().any():
        raise ValueError("Ailanthus consistency evidence matrix is incomplete")
    required_true = [
        "exact_overlap_gate_passed",
        "uniform_auc_gate_passed",
        "finite_calibration_gate_passed",
    ]
    for column in required_true:
        if not frame[column].map(lambda value: strict_bool(value, column)).all():
            raise ValueError("Consistency candidates failed a non-boundary prerequisite")
    if frame["matched_boundary_gate_passed"].map(
        lambda value: strict_bool(value, "matched_boundary_gate_passed")
    ).any():
        raise ValueError("A consistency candidate unexpectedly passed the boundary gate")
    return sha256(summary_path)


def build_config(
    source: dict,
    base: dict,
    probability: float,
    stage_root: Path,
    consistency_evidence_dir: Path,
    consistency_summary_hash: str,
) -> dict:
    config = deepcopy(source)
    name = experiment_name(probability)
    spec = {
        "stage": "ailanthus-image-modality-dropout-screen",
        "host": HOST,
        "arm": "domain_balanced",
        "background_target": 20000,
        "epochs": 15,
        "loss_strategy": "pos_weight_bce",
        "spatial_augmentation": "dihedral",
        "radiometric_recipe": "rad2",
        "dropout": 0.25,
        "label_smoothing": 0.05,
        "image_modality_dropout_probability": probability,
        "split_design": "uniform",
        "seed": int(source.get("seed", 42)),
    }
    config.update({
        "experiment": name,
        "output_dir": str(stage_root),
        "epochs": 15,
        "radiometric_consistency": {"enabled": False, "weight": 0.0},
        "image_modality_dropout_probability": probability,
        "august_experiment_spec": spec,
        "modality_dropout_selection_gate": {
            "failed_consistency_evidence_dir": str(
                consistency_evidence_dir.resolve()
            ),
            "failed_consistency_summary_sha256": consistency_summary_hash,
            "base_experiment": base["experiment"],
            "base_checkpoint_sha256": base["best_checkpoint_sha256"],
            "base_matched_boundary_absolute_did": float(
                base["artifact_candidate_absolute_did_mean"]
            ),
        },
        "scientific_note": (
            "Ailanthus-only structural screen after stronger augmentation and "
            "paired-view radiometric consistency both failed the unchanged "
            "matched-boundary gate. During training only, one Bernoulli mask per "
            "sample drops the complete image embedding with inverted-dropout "
            "rescaling; validation, test, and inference use the full image pathway."
        ),
    })
    return config


def write_design(
    consistency_evidence_dir: Path,
    robustness_evidence_dir: Path,
) -> tuple[Path, list[dict]]:
    consistency_hash = validated_failed_consistency(consistency_evidence_dir)
    base, source, robustness_hash = validated_base(robustness_evidence_dir)
    code_root = Path(__file__).resolve().parent
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=code_root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=code_root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("Refusing to freeze modality-dropout screen from a dirty tree")
    label = f"stage07_ailanthus_moddrop_{consistency_hash[:10]}_{DATE_TAG}"
    stage_root = OUTPUT_BASE / label
    design_root = DESIGN_BASE / label
    rows = []
    for probability in MODALITY_DROPOUT_PROBABILITIES:
        config = build_config(
            source,
            base,
            probability,
            stage_root,
            consistency_evidence_dir,
            consistency_hash,
        )
        name = config["experiment"]
        config_path = design_root / "configs" / f"{name}.json"
        if config_path.exists():
            if read_json(config_path) != config:
                raise FileExistsError(f"Refusing changed config: {config_path}")
        else:
            write_json_atomic(config_path, config)
        rows.append({
            **config["august_experiment_spec"],
            "experiment": name,
            "config_path": str(config_path),
            "config_sha256": sha256(config_path),
            "run_dir": str(stage_root / name),
            "status": "pending",
            "note": "",
        })
    registry = design_root / "experiment_registry.tsv"
    write_tsv_atomic(registry, rows, list(rows[0]))
    write_json_atomic(design_root / "design_metadata.json", {
        "status": "complete",
        "date_tag": DATE_TAG,
        "stage": "ailanthus-image-modality-dropout-screen",
        "code_root": str(code_root),
        "code_commit": commit,
        "code_dirty": False,
        "failed_consistency_evidence_dir": str(consistency_evidence_dir.resolve()),
        "failed_consistency_summary_sha256": consistency_hash,
        "failed_robustness_evidence_dir": str(robustness_evidence_dir.resolve()),
        "failed_robustness_summary_sha256": robustness_hash,
        "base_experiment": base["experiment"],
        "base_matched_boundary_absolute_did": float(
            base["artifact_candidate_absolute_did_mean"]
        ),
        "image_modality_dropout_probabilities": list(
            MODALITY_DROPOUT_PROBABILITIES
        ),
        "stage_output_root": str(stage_root),
        "experiment_count": len(rows),
        "registry": str(registry),
        "advancement_gate": (
            "Unchanged Ailanthus matched-boundary maximum 0.0937897740, exact-"
            "overlap non-regression, uniform ROC AUC at least 0.95, and finite "
            "raw calibration. No automatic selection."
        ),
    })
    return registry, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--failed-consistency-evidence-dir",
        type=Path,
        default=CONSISTENCY_EVIDENCE_DIR,
    )
    parser.add_argument(
        "--failed-robustness-evidence-dir",
        type=Path,
        default=ROBUSTNESS_EVIDENCE_DIR,
    )
    parser.add_argument("--run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry, rows = write_design(
        args.failed_consistency_evidence_dir,
        args.failed_robustness_evidence_dir,
    )
    print(f"Wrote gated Ailanthus modality-dropout design: {registry}")
    if args.run:
        run_experiments(registry, rows, max_runs=None)


if __name__ == "__main__":
    main()
