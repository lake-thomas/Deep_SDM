#!/usr/bin/env python3
"""Freeze and optionally run the paired-view Ailanthus consistency screen."""

from __future__ import annotations

import argparse
import subprocess
from copy import deepcopy
from pathlib import Path

import pandas as pd

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
BASE_EXPERIMENT = (
    "stage05a_ailanthus_altissima_db_bg20000_posw_dih_rad2_"
    "d025_s005_uniform_20260823"
)
CONSISTENCY_WEIGHTS = (0.5, 2.0, 5.0)
ROBUSTNESS_EVIDENCE_DIR = Path(
    "/mnt/rsstu_dsfas_gsv_naip/Host_NAIP_SDM/Outputs_Host_NAIP_SDMs/"
    "experimental_model_runs_august2026/"
    "stage05a_ailanthus_robustness_acceptance_346327dd4f_20260823/evidence"
)


def experiment_name(weight: float) -> str:
    encoded = int(round(weight * 100))
    return (
        f"stage06_{HOST}_db_bg20000_posw_dih_rad2_d025_s005_"
        f"consw{encoded:03d}_uniform_{DATE_TAG}"
    )


def validated_base(evidence_dir: Path) -> tuple[dict, dict, str]:
    summary_path = evidence_dir / "summary.json"
    evidence_path = evidence_dir / "ailanthus_robustness_candidate_evidence.csv"
    marker = evidence_dir / "COMPLETED"
    for path in (summary_path, evidence_path, marker):
        if not path.is_file():
            raise FileNotFoundError(path)
    summary = read_json(summary_path)
    if (
        summary.get("status") != "complete"
        or summary.get("advancement_candidate_count") != 0
        or summary.get("candidate_evidence_sha256") != sha256(evidence_path)
    ):
        raise RuntimeError("The bounded Ailanthus robustness screen is not a clean failure")
    frame = pd.read_csv(evidence_path, low_memory=False)
    if len(frame) != 5 or frame["experiment"].duplicated().any():
        raise ValueError("Ailanthus robustness evidence matrix is incomplete")
    matched = pd.to_numeric(
        frame["artifact_candidate_absolute_did_mean"], errors="raise"
    )
    base_rows = frame[frame["experiment"] == BASE_EXPERIMENT]
    if len(base_rows) != 1 or int(matched.idxmin()) != int(base_rows.index[0]):
        raise ValueError("The paired-view base is not the best matched-boundary candidate")
    base = base_rows.iloc[0].to_dict()
    required = {
        "host": HOST,
        "arm": "domain_balanced",
        "background_target": 20000,
        "loss_strategy": "pos_weight_bce",
        "spatial_augmentation": "dihedral",
        "radiometric_recipe": "rad2",
        "dropout": 0.25,
        "label_smoothing": 0.05,
        "exact_overlap_gate_passed": True,
        "uniform_auc_gate_passed": True,
        "finite_calibration_gate_passed": True,
        "matched_boundary_gate_passed": False,
    }
    mismatches = {
        key: {"expected": value, "found": base.get(key)}
        for key, value in required.items()
        if base.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Paired-view base evidence changed: {mismatches}")
    config_path = Path(base["config_path"])
    if not config_path.is_file() or sha256(config_path) != base["config_sha256"]:
        raise ValueError("Paired-view base config changed")
    source = read_json(config_path)
    if not isinstance(source.get("radiometric_augmentation"), dict):
        raise ValueError("Paired-view base lacks bounded radiometric augmentation")
    return base, source, sha256(summary_path)


def build_config(
    source: dict,
    base: dict,
    weight: float,
    stage_root: Path,
    evidence_dir: Path,
    evidence_summary_hash: str,
) -> dict:
    config = deepcopy(source)
    name = experiment_name(weight)
    spec = {
        "stage": "ailanthus-radiometric-consistency-screen",
        "host": HOST,
        "arm": "domain_balanced",
        "background_target": 20000,
        "epochs": 15,
        "loss_strategy": "pos_weight_bce",
        "spatial_augmentation": "dihedral",
        "radiometric_recipe": "rad2",
        "dropout": 0.25,
        "label_smoothing": 0.05,
        "radiometric_consistency_weight": weight,
        "split_design": "uniform",
        "seed": int(source.get("seed", 42)),
    }
    config.update({
        "experiment": name,
        "output_dir": str(stage_root),
        "epochs": 15,
        "radiometric_consistency": {"enabled": True, "weight": weight},
        "august_experiment_spec": spec,
        "consistency_selection_gate": {
            "failed_robustness_evidence_dir": str(evidence_dir.resolve()),
            "failed_robustness_summary_sha256": evidence_summary_hash,
            "base_experiment": BASE_EXPERIMENT,
            "base_checkpoint_sha256": base["best_checkpoint_sha256"],
            "base_matched_boundary_absolute_did": float(
                base["artifact_candidate_absolute_did_mean"]
            ),
        },
        "scientific_note": (
            "Ailanthus-only paired-view structural screen after stronger bounded "
            "augmentation and dropout failed the unchanged matched-boundary gate. "
            "Both training views share one exact dihedral orientation and receive "
            "independent bounded four-band radiometric perturbations; validation, "
            "test, and checkpoint selection remain unaugmented and unsmoothed."
        ),
    })
    return config


def write_design(evidence_dir: Path) -> tuple[Path, list[dict]]:
    base, source, evidence_hash = validated_base(evidence_dir)
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
        raise RuntimeError("Refusing to freeze consistency screen from a dirty tree")
    label = f"stage06_ailanthus_consistency_{evidence_hash[:10]}_{DATE_TAG}"
    stage_root = OUTPUT_BASE / label
    design_root = DESIGN_BASE / label
    rows = []
    for weight in CONSISTENCY_WEIGHTS:
        config = build_config(
            source, base, weight, stage_root, evidence_dir, evidence_hash
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
        "stage": "ailanthus-radiometric-consistency-screen",
        "code_root": str(code_root),
        "code_commit": commit,
        "code_dirty": False,
        "failed_robustness_evidence_dir": str(evidence_dir.resolve()),
        "failed_robustness_summary_sha256": evidence_hash,
        "base_experiment": BASE_EXPERIMENT,
        "base_matched_boundary_absolute_did": float(
            base["artifact_candidate_absolute_did_mean"]
        ),
        "consistency_weights": list(CONSISTENCY_WEIGHTS),
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
        "--failed-robustness-evidence-dir",
        type=Path,
        default=ROBUSTNESS_EVIDENCE_DIR,
    )
    parser.add_argument("--run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry, rows = write_design(args.failed_robustness_evidence_dir)
    print(f"Wrote gated Ailanthus consistency design: {registry}")
    if args.run:
        run_experiments(registry, rows, max_runs=None)


if __name__ == "__main__":
    main()
