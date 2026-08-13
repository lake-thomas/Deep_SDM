#!/usr/bin/env python3
"""Generate and optionally run five held-out BlockCV folds for selected recipes."""

from __future__ import annotations

import argparse
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


FOLDS = [1, 2, 3, 4, 5]


def blockcv_name(selection: dict, fold: int) -> str:
    arm = "db" if selection["arm"] == "domain_balanced" else "rnd"
    loss = {
        "ordinary_bce": "bce",
        "pos_weight_bce": "posw",
        "balanced_batches": "balbatch",
    }[selection["loss_strategy"]]
    rad = "rad1" if isinstance(selection["radiometric_augmentation"], dict) else "rad0"
    return (
        f"stage04_{selection['host']}_{arm}_"
        f"bg{int(selection['background_target']):05d}_{loss}_{rad}_"
        f"d{int(round(selection['dropout'] * 100)):03d}_"
        f"s{int(round(selection['label_smoothing'] * 100)):03d}_"
        f"blockcv{fold}_{DATE_TAG}"
    )


def blockcv_paths(selection: dict, fold: int) -> tuple[Path, Path, Path]:
    root = (
        Path(selection["image_dir"])
        / "model_manifests"
        / selection["arm"]
        / f"background_{int(selection['background_target']):05d}"
    )
    return (
        root / f"blockcv_{fold}_dataset.csv",
        root / f"blockcv_{fold}_lat_lon_stats.json",
        root / f"qa_blockcv_{fold}" / "qa_status.json",
    )


def validate_selection(selection: dict) -> dict:
    config_path = Path(selection["source_config"])
    checkpoint_path = Path(selection["source_checkpoint"])
    uniform_path = Path(selection["uniform_dataset"])
    expected = {
        config_path: selection["source_config_sha256"],
        checkpoint_path: selection["source_checkpoint_sha256"],
        uniform_path: selection["uniform_dataset_sha256"],
    }
    for path, digest in expected.items():
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Selected BlockCV recipe artifact changed: {path}")
    config = read_json(config_path)
    recipe = {
        "loss_strategy": selection["loss_strategy"],
        "spatial_augmentation": selection["spatial_augmentation"],
        "radiometric_augmentation": selection["radiometric_augmentation"],
        "dropout": selection["dropout"],
        "label_smoothing": selection["label_smoothing"],
    }
    mismatches = {
        key: {"expected": value, "found": config.get(key)}
        for key, value in recipe.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Selected BlockCV recipe changed: {mismatches}")
    audit_path = (
        Path(selection["image_dir"])
        / "qa"
        / "materialization_attrition_audit"
        / "summary.json"
    )
    if not audit_path.is_file() or not read_json(audit_path).get("production_ready"):
        raise RuntimeError(f"Materialization attrition audit is not ready: {audit_path}")
    return config


def build_config(
    source: dict,
    selection: dict,
    fold: int,
    stage_root: Path,
    selection_manifest: Path,
    selection_hash: str,
) -> dict:
    manifest, lat_lon, qa_path = blockcv_paths(selection, fold)
    for path in [manifest, lat_lon, qa_path]:
        if not path.is_file():
            raise FileNotFoundError(path)
    qa = read_json(qa_path)
    if not qa.get("production_ready"):
        raise RuntimeError(f"BlockCV manifest QA failed: {qa_path}")
    config = deepcopy(source)
    name = blockcv_name(selection, fold)
    config.update(
        {
            "experiment": name,
            "output_dir": str(stage_root),
            "csv_path": str(manifest),
            "lat_lon_stats": read_json(lat_lon),
            "epochs": 100,
            "lr_patience": 5,
            "es_patience": 10,
            "postfit_calibration": "platt",
            "august_experiment_spec": {
                "stage": "blockcv",
                "host": selection["host"],
                "arm": selection["arm"],
                "background_target": int(selection["background_target"]),
                "epochs": 100,
                "loss_strategy": selection["loss_strategy"],
                "spatial_augmentation": selection["spatial_augmentation"],
                "radiometric_augmentation": selection[
                    "radiometric_augmentation"
                ],
                "dropout": selection["dropout"],
                "label_smoothing": selection["label_smoothing"],
                "split_design": f"blockcv_{fold}_heldout_fold",
                "cv_round": fold,
                "seed": selection["seed"],
            },
            "dataset_gate": {
                "manifest": str(manifest.resolve()),
                "manifest_sha256": sha256(manifest),
                "manifest_qa": str(qa_path.resolve()),
                "manifest_qa_sha256": sha256(qa_path),
                "lat_lon_stats": str(lat_lon.resolve()),
                "lat_lon_stats_sha256": sha256(lat_lon),
                "materialization_attrition_audit": str(
                    Path(selection["image_dir"])
                    / "qa"
                    / "materialization_attrition_audit"
                    / "summary.json"
                ),
            },
            "selection_gate": {
                "selection_manifest": str(selection_manifest.resolve()),
                "selection_manifest_sha256": selection_hash,
                "selected_recipe": selection,
            },
            "scientific_note": (
                "One of five held-out spatial BlockCV folds for an evidence-selected "
                "August recipe. Validation is one complete spatial fold, test is held "
                "out, and Platt calibration is fit only on spatially separate validation "
                "predictions. Fold distributions, not pooled metrics alone, govern the "
                "production gate."
            ),
        }
    )
    return config


def write_design(selection_manifest: Path) -> tuple[Path, list[dict]]:
    selection_hash = sha256(selection_manifest)
    payload = read_json(selection_manifest)
    if (
        payload.get("status") != "complete"
        or payload.get("selection_type") != "recipe_for_blockcv"
    ):
        raise RuntimeError("Selection manifest is not approved for BlockCV")
    selections = payload.get("selections") or []
    if not selections or len(selections) != payload.get("selection_count"):
        raise ValueError("BlockCV selection count is invalid")
    code_root = Path(__file__).resolve().parent
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=code_root, check=True,
        capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=code_root, check=True,
        capture_output=True, text=True
    ).stdout.strip()
    if dirty:
        raise RuntimeError("Refusing to freeze BlockCV design from a dirty tree")
    label = f"stage04_blockcv_{selection_hash[:10]}_{DATE_TAG}"
    stage_root = OUTPUT_BASE / label
    design_root = DESIGN_BASE / label
    config_root = design_root / "configs"
    rows = []
    for selection in selections:
        source = validate_selection(selection)
        for fold in FOLDS:
            config = build_config(
                source, selection, fold, stage_root, selection_manifest, selection_hash
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
            "date_tag": DATE_TAG,
            "stage": "blockcv",
            "code_root": str(code_root),
            "code_commit": commit,
            "code_dirty": False,
            "selection_manifest": str(selection_manifest.resolve()),
            "selection_manifest_sha256": selection_hash,
            "stage_output_root": str(stage_root),
            "folds_per_host": len(FOLDS),
            "experiment_count": len(rows),
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
    print(f"Wrote gated five-fold BlockCV design: {registry}")
    if args.run:
        run_experiments(registry, rows, max_runs=args.max_runs)


if __name__ == "__main__":
    main()
