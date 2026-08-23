#!/usr/bin/env python3
"""Generate and optionally run fresh BlockCV-approved August production fits."""

from __future__ import annotations

import argparse
import subprocess
from copy import deepcopy
from pathlib import Path

from august2026_combine_dataset_screen_evidence import HOSTS
from august2026_experiments import (
    DATE_TAG,
    DESIGN_BASE,
    read_json,
    run_experiments,
    sha256,
    write_json_atomic,
    write_tsv_atomic,
)


PRODUCTION_OUTPUT_ROOT = Path(
    "/mnt/rsstu_dsfas_gsv_naip/Host_NAIP_SDM/Outputs_Host_NAIP_SDMs/"
    "production_model_runs_august2026"
)


def production_name(selection: dict) -> str:
    arm = "db" if selection["arm"] == "domain_balanced" else "rnd"
    loss = {
        "ordinary_bce": "bce",
        "pos_weight_bce": "posw",
        "balanced_batches": "balbatch",
    }[selection["loss_strategy"]]
    rad = "rad1" if isinstance(selection["radiometric_augmentation"], dict) else "rad0"
    calibration = (
        "platt" if selection["production_postfit_calibration"] == "platt" else "raw"
    )
    return (
        f"production_{selection['host']}_{arm}_"
        f"bg{int(selection['background_target']):05d}_{loss}_{rad}_"
        f"d{int(round(selection['dropout'] * 100)):03d}_"
        f"s{int(round(selection['label_smoothing'] * 100)):03d}_"
        f"{calibration}_{DATE_TAG}"
    )


def validate_selection(selection: dict) -> dict:
    config_path = Path(selection["source_config"])
    dataset_path = Path(selection["uniform_dataset"])
    if not config_path.is_file() or sha256(config_path) != selection["source_config_sha256"]:
        raise ValueError(f"Production source config changed: {config_path}")
    if not dataset_path.is_file() or sha256(dataset_path) != selection["uniform_dataset_sha256"]:
        raise ValueError(f"Production uniform dataset changed: {dataset_path}")
    config = read_json(config_path)
    expected = {
        "loss_strategy": selection["loss_strategy"],
        "spatial_augmentation": selection["spatial_augmentation"],
        "radiometric_augmentation": selection["radiometric_augmentation"],
        "dropout": selection["dropout"],
        "label_smoothing": selection["label_smoothing"],
    }
    if "radiometric_consistency" in selection:
        expected["radiometric_consistency"] = selection["radiometric_consistency"]
    mismatches = {
        key: {"expected": value, "found": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Production recipe differs from approved source: {mismatches}")
    if Path(config["csv_path"]).resolve() != dataset_path.resolve():
        raise ValueError("Production source config does not use approved uniform dataset")
    return config


def build_config(
    source: dict,
    selection: dict,
    stage_root: Path,
    selection_manifest: Path,
    selection_hash: str,
) -> dict:
    config = deepcopy(source)
    name = production_name(selection)
    config.update(
        {
            "experiment": name,
            "output_dir": str(stage_root),
            "csv_path": selection["uniform_dataset"],
            "epochs": 100,
            "lr_patience": 5,
            "es_patience": 10,
            "postfit_calibration": selection["production_postfit_calibration"],
            "august_experiment_spec": {
                "stage": "production-fit",
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
                "split_design": "uniform_production",
                "seed": selection["seed"],
                "postfit_calibration": selection[
                    "production_postfit_calibration"
                ],
            },
            "selection_gate": {
                "selection_manifest": str(selection_manifest.resolve()),
                "selection_manifest_sha256": selection_hash,
                "approved_recipe": selection,
            },
            "scientific_note": (
                "Fresh August production fit after five-fold held-out BlockCV. The "
                "training/validation/test partitions come from the source-locked uniform "
                "manifest. Calibration, when approved by paired folds, is fit only from "
                "the separate validation subset. The model is not accepted until full "
                "artifact and map QA passes."
            ),
        }
    )
    if "radiometric_consistency" in selection:
        config["radiometric_consistency"] = selection["radiometric_consistency"]
        config["august_experiment_spec"]["radiometric_consistency"] = selection[
            "radiometric_consistency"
        ]
    return config


def write_design(selection_manifest: Path) -> tuple[Path, list[dict]]:
    selection_hash = sha256(selection_manifest)
    payload = read_json(selection_manifest)
    if (
        payload.get("status") != "complete"
        or payload.get("selection_type")
        != "blockcv_approved_recipe_for_production_fit"
    ):
        raise RuntimeError("Selection manifest is not approved for production fits")
    selections = payload.get("selections") or []
    if not selections or len(selections) != payload.get("selection_count"):
        raise ValueError("Production selection count is invalid")
    if {item.get("host") for item in selections} != HOSTS:
        raise ValueError("Production fitting requires exactly the five production hosts")
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
        raise RuntimeError("Refusing to freeze production configs from a dirty tree")
    label = f"production_fits_{selection_hash[:10]}_{DATE_TAG}"
    stage_root = PRODUCTION_OUTPUT_ROOT / label
    design_root = DESIGN_BASE / label
    config_root = design_root / "configs"
    rows = []
    for selection in selections:
        source = validate_selection(selection)
        config = build_config(
            source, selection, stage_root, selection_manifest, selection_hash
        )
        name = config["experiment"]
        config_path = config_root / f"{name}.json"
        if config_path.exists():
            if read_json(config_path) != config:
                raise FileExistsError(f"Refusing changed production config: {config_path}")
        else:
            write_json_atomic(config_path, config)
        rows.append(
            {
                **config["august_experiment_spec"],
                "experiment": name,
                "config_path": str(config_path),
                "config_sha256": sha256(config_path),
                "run_dir": str(stage_root / name),
                "approved_score_space": selection["production_score_space"],
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
            "stage": "production-fit",
            "code_root": str(code_root),
            "code_commit": commit,
            "code_dirty": False,
            "selection_manifest": str(selection_manifest.resolve()),
            "selection_manifest_sha256": selection_hash,
            "stage_output_root": str(stage_root),
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
    print(f"Wrote BlockCV-approved production fit design: {registry}")
    if args.run:
        run_experiments(registry, rows, max_runs=args.max_runs)


if __name__ == "__main__":
    main()
