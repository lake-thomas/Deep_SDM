#!/usr/bin/env python3
"""Freeze and optionally run the post-acceptance Ailanthus robustness screen."""

from __future__ import annotations

import argparse
import subprocess
from copy import deepcopy
from pathlib import Path

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
EXPECTED_SELECTION_SHA256 = (
    "346327dd4f70052ba00677fb2e85cbfac4c81b041b1856a25ea44d6221c858ad"
)
FINAL_ACCEPTANCE_SUMMARY = Path(
    "/mnt/rsstu_dsfas_gsv_naip/Host_NAIP_SDM/Outputs_Host_NAIP_SDMs/"
    "production_model_runs_august2026/final_model_acceptance_346327dd4f_20260823/"
    "final_model_acceptance/summary.json"
)
MILD_CALIBRATION = DESIGN_BASE / "radiometric_augmentation_calibration_20260813.json"

# These bounds use the paired-acquisition q75 scene/contrast changes, median
# band-relative change, and q95 NDVI change recorded in the frozen calibration.
# Gamma and offset are conservative engineering bounds because chip means do
# not identify gamma and the paired evidence does not directly identify offset.
STRONG_RADIOMETRIC = {
    "enabled": True,
    "probability": 0.9,
    "scene_gain": 0.1447,
    "gamma": 0.10,
    "contrast": 0.1572,
    "band_gain": 0.0712,
    "band_offset": 0.02,
    "max_mean_ndvi_shift": 0.0630,
    "attempts": 5,
}


def candidate_recipes(mild_recipe: dict) -> list[dict]:
    return [
        {
            "recipe_label": "rad2",
            "radiometric_augmentation": deepcopy(STRONG_RADIOMETRIC),
            "dropout": 0.25,
            "label_smoothing": 0.05,
        },
        {
            "recipe_label": "rad2",
            "radiometric_augmentation": deepcopy(STRONG_RADIOMETRIC),
            "dropout": 0.35,
            "label_smoothing": 0.0,
        },
        {
            "recipe_label": "rad2",
            "radiometric_augmentation": deepcopy(STRONG_RADIOMETRIC),
            "dropout": 0.35,
            "label_smoothing": 0.05,
        },
        {
            "recipe_label": "rad2",
            "radiometric_augmentation": deepcopy(STRONG_RADIOMETRIC),
            "dropout": 0.50,
            "label_smoothing": 0.05,
        },
        {
            "recipe_label": "rad1",
            "radiometric_augmentation": deepcopy(mild_recipe),
            "dropout": 0.50,
            "label_smoothing": 0.05,
        },
    ]


def experiment_name(recipe: dict) -> str:
    dropout = int(round(float(recipe["dropout"]) * 100))
    smoothing = int(round(float(recipe["label_smoothing"]) * 100))
    return (
        f"stage05a_{HOST}_db_bg20000_posw_dih_{recipe['recipe_label']}_"
        f"d{dropout:03d}_s{smoothing:03d}_uniform_{DATE_TAG}"
    )


def validated_inputs(selection_manifest: Path) -> tuple[dict, dict, str, str]:
    selection_hash = sha256(selection_manifest)
    if selection_hash != EXPECTED_SELECTION_SHA256:
        raise ValueError("Production selection manifest identity changed")
    selection_payload = read_json(selection_manifest)
    if (
        selection_payload.get("status") != "complete"
        or selection_payload.get("selection_type")
        != "blockcv_approved_recipe_for_production_fit"
    ):
        raise RuntimeError("Selection manifest is not BlockCV-approved")
    matches = [
        item for item in selection_payload.get("selections", [])
        if item.get("host") == HOST
    ]
    if len(matches) != 1:
        raise ValueError("Expected exactly one Ailanthus production selection")
    selection = matches[0]
    required = {
        "arm": "domain_balanced",
        "background_target": 20000,
        "loss_strategy": "pos_weight_bce",
        "spatial_augmentation": "dihedral",
    }
    mismatches = {
        key: {"expected": value, "found": selection.get(key)}
        for key, value in required.items()
        if selection.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Ailanthus production selection changed: {mismatches}")

    source_config_path = Path(selection["source_config"])
    dataset_path = Path(selection["uniform_dataset"])
    for path, expected_hash in (
        (source_config_path, selection["source_config_sha256"]),
        (dataset_path, selection["uniform_dataset_sha256"]),
        (FINAL_ACCEPTANCE_SUMMARY, None),
        (MILD_CALIBRATION, None),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        if expected_hash is not None and sha256(path) != expected_hash:
            raise ValueError(f"Frozen Ailanthus input changed: {path}")
    final_acceptance = read_json(FINAL_ACCEPTANCE_SUMMARY)
    expected_failures = {
        f"{HOST}: raw_matched_boundary_regression",
        f"{HOST}: configured_matched_boundary_regression",
    }
    if (
        final_acceptance.get("status") != "complete"
        or final_acceptance.get("production_ready_for_inference") is not False
        or set(final_acceptance.get("failures", [])) != expected_failures
    ):
        raise RuntimeError("Ailanthus-only screen is not authorized by final acceptance")
    mild_payload = read_json(MILD_CALIBRATION)
    if mild_payload.get("status") != "complete":
        raise RuntimeError("Mild radiometric calibration is incomplete")
    return (
        selection,
        read_json(source_config_path),
        selection_hash,
        sha256(FINAL_ACCEPTANCE_SUMMARY),
    )


def build_config(
    source: dict,
    selection: dict,
    recipe: dict,
    stage_root: Path,
    selection_manifest: Path,
    selection_hash: str,
    acceptance_hash: str,
) -> dict:
    config = deepcopy(source)
    name = experiment_name(recipe)
    spec = {
        "stage": "ailanthus-robustness-screen",
        "host": HOST,
        "arm": "domain_balanced",
        "background_target": 20000,
        "epochs": 15,
        "loss_strategy": "pos_weight_bce",
        "spatial_augmentation": "dihedral",
        "radiometric_recipe": recipe["recipe_label"],
        "dropout": recipe["dropout"],
        "label_smoothing": recipe["label_smoothing"],
        "split_design": "uniform",
        "seed": int(source.get("seed", 42)),
    }
    config.update(
        {
            "experiment": name,
            "output_dir": str(stage_root),
            "epochs": 15,
            "loss_strategy": "pos_weight_bce",
            "spatial_augmentation": "dihedral",
            "radiometric_augmentation": deepcopy(
                recipe["radiometric_augmentation"]
            ),
            "dropout": recipe["dropout"],
            "label_smoothing": recipe["label_smoothing"],
            "postfit_calibration": "none",
            "august_experiment_spec": spec,
            "selection_gate": {
                "production_selection_manifest": str(selection_manifest.resolve()),
                "production_selection_manifest_sha256": selection_hash,
                "selected_ailanthus_recipe": selection,
                "failed_final_acceptance_summary": str(
                    FINAL_ACCEPTANCE_SUMMARY.resolve()
                ),
                "failed_final_acceptance_summary_sha256": acceptance_hash,
                "mild_radiometric_calibration": str(MILD_CALIBRATION.resolve()),
                "mild_radiometric_calibration_sha256": sha256(MILD_CALIBRATION),
            },
            "scientific_note": (
                "Post-acceptance Ailanthus-only uniform screen. It retains the "
                "BlockCV-approved domain-balanced 20,000-background manifest, "
                "positive-weighted BCE, and dihedral augmentation while testing "
                "broader paired-acquisition-bounded radiometric perturbations and "
                "regularization. Advancement requires the unchanged matched-boundary "
                "and exact-overlap gates before Ailanthus-only BlockCV."
            ),
        }
    )
    return config


def write_design(selection_manifest: Path) -> tuple[Path, list[dict]]:
    selection, source, selection_hash, acceptance_hash = validated_inputs(
        selection_manifest
    )
    mild_recipe = read_json(MILD_CALIBRATION)["radiometric_augmentation"]
    code_root = Path(__file__).resolve().parent
    code_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=code_root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=code_root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("Refusing to freeze Ailanthus design from a dirty tree")

    label = f"stage05a_ailanthus_robustness_{selection_hash[:10]}_{DATE_TAG}"
    stage_root = OUTPUT_BASE / label
    design_root = DESIGN_BASE / label
    config_root = design_root / "configs"
    rows = []
    for recipe in candidate_recipes(mild_recipe):
        config = build_config(
            source, selection, recipe, stage_root, selection_manifest,
            selection_hash, acceptance_hash,
        )
        name = config["experiment"]
        config_path = config_root / f"{name}.json"
        if config_path.exists():
            if read_json(config_path) != config:
                raise FileExistsError(f"Refusing changed config: {config_path}")
        else:
            write_json_atomic(config_path, config)
        rows.append(
            {
                **config["august_experiment_spec"],
                "experiment": name,
                "config_path": str(config_path),
                "config_sha256": sha256(config_path),
                "run_dir": str(stage_root / name),
                "status": "pending",
                "note": "",
            }
        )
    registry = design_root / "experiment_registry.tsv"
    write_tsv_atomic(registry, rows, list(rows[0]))
    write_json_atomic(
        design_root / "design_metadata.json",
        {
            "status": "complete",
            "date_tag": DATE_TAG,
            "stage": "ailanthus-robustness-screen",
            "code_root": str(code_root),
            "code_commit": code_commit,
            "code_dirty": False,
            "production_selection_manifest": str(selection_manifest.resolve()),
            "production_selection_manifest_sha256": selection_hash,
            "failed_final_acceptance_summary": str(
                FINAL_ACCEPTANCE_SUMMARY.resolve()
            ),
            "failed_final_acceptance_summary_sha256": acceptance_hash,
            "stage_output_root": str(stage_root),
            "experiment_count": len(rows),
            "registry": str(registry),
            "advancement_gate": (
                "Unchanged Ailanthus raw matched-boundary maximum 0.0937897740, "
                "plus exact-overlap non-regression, discrimination, and calibration."
            ),
        },
    )
    return registry, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None)
    args = parser.parse_args()
    if args.max_runs is not None and args.max_runs <= 0:
        parser.error("--max-runs must be positive")
    return args


def main() -> None:
    args = parse_args()
    if not args.selection_manifest.is_file():
        raise FileNotFoundError(args.selection_manifest)
    registry, rows = write_design(args.selection_manifest)
    print(f"Wrote gated Ailanthus robustness design: {registry}")
    if args.run:
        run_experiments(registry, rows, max_runs=args.max_runs)


if __name__ == "__main__":
    main()
