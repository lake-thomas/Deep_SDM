#!/usr/bin/env python3
"""Generate and optionally run the gated August augmentation factorial."""

from __future__ import annotations

import argparse
import itertools
import subprocess
from copy import deepcopy
from pathlib import Path

from august2026_experiments import (
    DATE_TAG,
    DESIGN_BASE,
    OUTPUT_BASE,
    read_json,
    run_experiments,
    sha256,
    write_json_atomic,
    write_tsv_atomic,
)


RADIOMETRIC_CALIBRATION = (
    DESIGN_BASE / "radiometric_augmentation_calibration_20260813.json"
)
RADIOMETRIC_CALIBRATION_SHA256 = (
    "78fcd253c529a492565cf4107595fe1861a9bf675c740227a47920f43a8e3486"
)


def augmentation_combinations() -> list[dict]:
    combinations = []
    for radiometric, dropout, smoothing in itertools.product(
        [False, True], [0.25, 0.35], [0.0, 0.05]
    ):
        if not radiometric and dropout == 0.25 and smoothing == 0.0:
            continue
        combinations.append(
            {
                "radiometric": radiometric,
                "dropout": dropout,
                "label_smoothing": smoothing,
            }
        )
    return combinations


def experiment_name(selection: dict, combination: dict) -> str:
    arm = "db" if selection["arm"] == "domain_balanced" else "rnd"
    loss = {
        "ordinary_bce": "bce",
        "pos_weight_bce": "posw",
        "balanced_batches": "balbatch",
    }[selection["loss_strategy"]]
    rad = "rad1" if combination["radiometric"] else "rad0"
    dropout = f"d{int(round(combination['dropout'] * 100)):03d}"
    smoothing = f"s{int(round(combination['label_smoothing'] * 100)):03d}"
    return (
        f"stage03_{selection['host']}_{arm}_"
        f"bg{int(selection['background_target']):05d}_{loss}_dih_"
        f"{rad}_{dropout}_{smoothing}_uniform_{DATE_TAG}"
    )


def validate_selection(selection: dict) -> dict:
    config_path = Path(selection["source_config"])
    checkpoint_path = Path(selection["source_checkpoint"])
    dataset_path = Path(selection["dataset"])
    expected = {
        config_path: selection["source_config_sha256"],
        checkpoint_path: selection["source_checkpoint_sha256"],
        dataset_path: selection["dataset_sha256"],
    }
    for path, digest in expected.items():
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Selected augmentation baseline changed: {path}")
    config = read_json(config_path)
    required_baseline = {
        "loss_strategy": selection["loss_strategy"],
        "spatial_augmentation": "dihedral",
        "radiometric_augmentation": False,
        "dropout": 0.25,
        "label_smoothing": 0.0,
    }
    mismatches = {
        key: {"expected": value, "found": config.get(key)}
        for key, value in required_baseline.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Selected loss baseline is not isolated: {mismatches}")
    if Path(config["csv_path"]).resolve() != dataset_path.resolve():
        raise ValueError("Selected augmentation dataset path changed")
    return config


def build_config(
    source: dict,
    selection: dict,
    combination: dict,
    radiometric_recipe: dict,
    stage_root: Path,
    selection_manifest: Path,
    selection_hash: str,
    calibration_hash: str,
) -> dict:
    config = deepcopy(source)
    name = experiment_name(selection, combination)
    radiometric = (
        deepcopy(radiometric_recipe) if combination["radiometric"] else False
    )
    spec = {
        "stage": "augmentation-screen",
        "host": selection["host"],
        "arm": selection["arm"],
        "background_target": int(selection["background_target"]),
        "epochs": 15,
        "loss_strategy": selection["loss_strategy"],
        "spatial_augmentation": "dihedral",
        "radiometric_augmentation": radiometric,
        "dropout": combination["dropout"],
        "label_smoothing": combination["label_smoothing"],
        "split_design": "uniform",
        "seed": int(source.get("seed", 42)),
    }
    config.update(
        {
            "experiment": name,
            "output_dir": str(stage_root),
            "epochs": 15,
            "loss_strategy": selection["loss_strategy"],
            "spatial_augmentation": "dihedral",
            "radiometric_augmentation": radiometric,
            "dropout": combination["dropout"],
            "label_smoothing": combination["label_smoothing"],
            "postfit_calibration": "none",
            "august_experiment_spec": spec,
            "selection_gate": {
                "selection_manifest": str(selection_manifest.resolve()),
                "selection_manifest_sha256": selection_hash,
                "selected_loss_baseline": selection,
                "reused_baseline_experiment": selection["source_experiment"],
                "reused_baseline_checkpoint_sha256": selection[
                    "source_checkpoint_sha256"
                ],
                "radiometric_calibration": str(RADIOMETRIC_CALIBRATION.resolve()),
                "radiometric_calibration_sha256": calibration_hash,
            },
            "scientific_note": (
                "August 2x2x2 radiometric-augmentation, dropout, and label-smoothing "
                "factorial on an evidence-selected dataset and loss. The unaugmented "
                "dropout-0.25 smoothing-0 baseline is reused rather than retrained. "
                "Validation remains unsmoothed and candidates require artifact replay "
                "and held-out BlockCV."
            ),
        }
    )
    return config


def write_design(selection_manifest: Path) -> tuple[Path, list[dict]]:
    selection_hash = sha256(selection_manifest)
    payload = read_json(selection_manifest)
    if (
        payload.get("status") != "complete"
        or payload.get("selection_type") != "loss_for_augmentation_screen"
    ):
        raise RuntimeError("Selection manifest is not approved for augmentation")
    selections = payload.get("selections") or []
    if not selections or len(selections) != payload.get("selection_count"):
        raise ValueError("Augmentation selection count is invalid")
    calibration_hash = sha256(RADIOMETRIC_CALIBRATION)
    if calibration_hash != RADIOMETRIC_CALIBRATION_SHA256:
        raise ValueError("Frozen radiometric calibration hash changed")
    calibration = read_json(RADIOMETRIC_CALIBRATION)
    if calibration.get("status") != "complete":
        raise RuntimeError("Radiometric calibration is incomplete")
    radiometric_recipe = calibration["radiometric_augmentation"]

    code_root = Path(__file__).resolve().parent
    code_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=code_root, check=True,
        capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=code_root, check=True,
        capture_output=True, text=True
    ).stdout.strip()
    if dirty:
        raise RuntimeError("Refusing to freeze augmentation design from a dirty tree")
    design_id = selection_hash[:10]
    label = f"stage03_augmentation_screen_{design_id}_{DATE_TAG}"
    stage_root = OUTPUT_BASE / label
    design_root = DESIGN_BASE / label
    config_root = design_root / "configs"
    rows = []
    for selection in selections:
        source = validate_selection(selection)
        for combination in augmentation_combinations():
            config = build_config(
                source, selection, combination, radiometric_recipe, stage_root,
                selection_manifest, selection_hash, calibration_hash
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
                    "reused_baseline_experiment": selection["source_experiment"],
                    "reused_baseline_checkpoint_sha256": selection[
                        "source_checkpoint_sha256"
                    ],
                    "status": "pending",
                    "note": "",
                }
            )
    registry = design_root / "experiment_registry.tsv"
    write_tsv_atomic(registry, rows, list(rows[0]))
    write_json_atomic(
        design_root / "design_metadata.json",
        {
            "date_tag": DATE_TAG,
            "stage": "augmentation-screen",
            "code_root": str(code_root),
            "code_commit": code_commit,
            "code_dirty": False,
            "selection_manifest": str(selection_manifest.resolve()),
            "selection_manifest_sha256": selection_hash,
            "radiometric_calibration": str(RADIOMETRIC_CALIBRATION.resolve()),
            "radiometric_calibration_sha256": calibration_hash,
            "stage_output_root": str(stage_root),
            "experiment_count": len(rows),
            "reused_baseline_count": len(selections),
            "registry": str(registry),
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
    print(f"Wrote gated augmentation-screen design: {registry}")
    if args.run:
        run_experiments(registry, rows, max_runs=args.max_runs)


if __name__ == "__main__":
    main()
