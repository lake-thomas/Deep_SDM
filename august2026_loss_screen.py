#!/usr/bin/env python3
"""Generate and optionally run the evidence-gated August loss screen."""

from __future__ import annotations

import argparse
import json
import subprocess
from copy import deepcopy
from pathlib import Path

from august2026_combine_dataset_screen_evidence import HOSTS
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


LOSS_STRATEGIES = ["pos_weight_bce", "balanced_batches"]


def loss_experiment_name(selection: dict, strategy: str) -> str:
    strategy_label = {
        "pos_weight_bce": "posw",
        "balanced_batches": "balbatch",
    }[strategy]
    arm_label = "db" if selection["arm"] == "domain_balanced" else "rnd"
    return (
        f"stage02_{selection['host']}_{arm_label}_"
        f"bg{int(selection['background_target']):05d}_{strategy_label}_"
        f"dih_d025_s000_uniform_{DATE_TAG}"
    )


def validate_selection_entry(selection: dict) -> dict:
    required = {
        "host",
        "arm",
        "background_target",
        "source_experiment",
        "source_config",
        "source_config_sha256",
        "source_checkpoint",
        "source_checkpoint_sha256",
        "dataset",
        "dataset_sha256",
    }
    missing = required - set(selection)
    if missing:
        raise ValueError(f"Dataset selection is missing fields: {sorted(missing)}")
    paths = {
        "source_config": Path(selection["source_config"]),
        "source_checkpoint": Path(selection["source_checkpoint"]),
        "dataset": Path(selection["dataset"]),
    }
    missing_paths = [str(path) for path in paths.values() if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"Selected loss-screen inputs are missing: {missing_paths}")
    for key, path in paths.items():
        expected = selection[f"{key}_sha256"]
        if sha256(path) != expected:
            raise ValueError(f"Selected {key} hash changed: {path}")
    config = read_json(paths["source_config"])
    expected_source = {
        "loss_strategy": "ordinary_bce",
        "spatial_augmentation": "dihedral",
        "radiometric_augmentation": False,
        "dropout": 0.25,
        "label_smoothing": 0.0,
    }
    mismatches = {
        key: {"expected": expected, "found": config.get(key)}
        for key, expected in expected_source.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            f"Source ordinary-BCE baseline is not the frozen screen recipe: {mismatches}"
        )
    if Path(config["csv_path"]).resolve() != paths["dataset"].resolve():
        raise ValueError("Source config dataset path differs from selection")
    spec = config.get("august_experiment_spec") or {}
    if (
        spec.get("host") != selection["host"]
        or spec.get("arm") != selection["arm"]
        or int(spec.get("background_target", -1))
        != int(selection["background_target"])
    ):
        raise ValueError("Source config experiment identity differs from selection")
    return config


def build_loss_config(
    source_config: dict,
    selection: dict,
    strategy: str,
    stage_root: Path,
    selection_manifest: Path,
    selection_sha256: str,
) -> dict:
    if strategy not in LOSS_STRATEGIES:
        raise ValueError(f"Unsupported loss strategy: {strategy}")
    config = deepcopy(source_config)
    name = loss_experiment_name(selection, strategy)
    config.update(
        {
            "experiment": name,
            "output_dir": str(stage_root),
            "epochs": 15,
            "loss_strategy": strategy,
            "label_smoothing": 0.0,
            "spatial_augmentation": "dihedral",
            "radiometric_augmentation": False,
            "dropout": 0.25,
            "postfit_calibration": "none",
            "august_experiment_spec": {
                "stage": "loss-screen",
                "host": selection["host"],
                "arm": selection["arm"],
                "background_target": int(selection["background_target"]),
                "epochs": 15,
                "loss_strategy": strategy,
                "spatial_augmentation": "dihedral",
                "radiometric_augmentation": False,
                "dropout": 0.25,
                "label_smoothing": 0.0,
                "split_design": "uniform",
                "seed": int(source_config.get("seed", 42)),
            },
            "selection_gate": {
                "selection_manifest": str(selection_manifest.resolve()),
                "selection_manifest_sha256": selection_sha256,
                "selected_dataset": selection,
                "ordinary_bce_baseline_experiment": selection["source_experiment"],
                "ordinary_bce_baseline_checkpoint": selection["source_checkpoint"],
                "ordinary_bce_baseline_checkpoint_sha256": selection[
                    "source_checkpoint_sha256"
                ],
            },
            "scientific_note": (
                "August isolated loss-strategy screen on an evidence-selected dataset. "
                "The ordinary-BCE dataset-screen run is the identical baseline and is "
                "not retrained. Raw scores remain conditional on the background and "
                "optimization design; loss finalists still require artifact replay and "
                "held-out BlockCV."
            ),
        }
    )
    return config


def write_design(selection_manifest: Path) -> tuple[Path, list[dict]]:
    selection_hash = sha256(selection_manifest)
    selection_payload = read_json(selection_manifest)
    if (
        selection_payload.get("status") != "complete"
        or selection_payload.get("selection_type") != "dataset_for_loss_screen"
    ):
        raise RuntimeError("Selection manifest is not approved for the loss screen")
    selections = selection_payload.get("selections") or []
    if len(selections) != selection_payload.get("selection_count") or not selections:
        raise ValueError("Selection manifest count is invalid")
    if {item.get("host") for item in selections} != HOSTS:
        raise ValueError("Loss screen requires exactly the five production hosts")

    code_root = Path(__file__).resolve().parent
    code_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=code_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=code_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("Refusing to freeze a loss design from a dirty code tree")

    design_id = selection_hash[:10]
    stage_label = f"stage02_loss_screen_{design_id}_{DATE_TAG}"
    stage_root = OUTPUT_BASE / stage_label
    design_root = DESIGN_BASE / stage_label
    config_root = design_root / "configs"
    rows = []
    for selection in selections:
        source_config = validate_selection_entry(selection)
        for strategy in LOSS_STRATEGIES:
            config = build_loss_config(
                source_config,
                selection,
                strategy,
                stage_root,
                selection_manifest,
                selection_hash,
            )
            name = config["experiment"]
            config_path = config_root / f"{name}.json"
            if config_path.exists():
                if read_json(config_path) != config:
                    raise FileExistsError(f"Refusing to replace changed config: {config_path}")
            else:
                write_json_atomic(config_path, config)
            rows.append(
                {
                    **config["august_experiment_spec"],
                    "experiment": name,
                    "config_path": str(config_path),
                    "config_sha256": sha256(config_path),
                    "run_dir": str(stage_root / name),
                    "ordinary_bce_baseline_experiment": selection[
                        "source_experiment"
                    ],
                    "ordinary_bce_baseline_checkpoint_sha256": selection[
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
            "stage": "loss-screen",
            "code_root": str(code_root),
            "code_commit": code_commit,
            "code_dirty": False,
            "selection_manifest": str(selection_manifest.resolve()),
            "selection_manifest_sha256": selection_hash,
            "stage_output_root": str(stage_root),
            "experiment_count": len(rows),
            "ordinary_bce_baselines_reused": [
                selection["source_experiment"] for selection in selections
            ],
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
    print(f"Wrote gated loss-screen design: {registry}")
    if args.run:
        run_experiments(registry, rows, max_runs=args.max_runs)


if __name__ == "__main__":
    main()
