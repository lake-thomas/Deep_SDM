#!/usr/bin/env python3
"""Generate and optionally run one accepted replacement Ailanthus production fit."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from august2026_experiments import (
    DESIGN_BASE,
    read_json,
    run_experiments,
    sha256,
    write_json_atomic,
    write_tsv_atomic,
)
from august2026_production_train import (
    PRODUCTION_OUTPUT_ROOT,
    build_config,
    validate_selection,
)


DATE_TAG = "20260823"
HOST = "ailanthus_altissima"


def replacement_name(selection: dict) -> str:
    arm = "db" if selection["arm"] == "domain_balanced" else "rnd"
    rad = selection.get("radiometric_recipe")
    if rad not in {"rad1", "rad2"}:
        raise ValueError("Replacement production recipe lacks explicit radiometric label")
    calibration = (
        "platt" if selection["production_postfit_calibration"] == "platt" else "raw"
    )
    consistency = selection.get("radiometric_consistency", False)
    consistency_suffix = ""
    if isinstance(consistency, dict) and consistency.get("enabled", True):
        weight = float(consistency["weight"])
        consistency_suffix = f"_consw{int(round(weight * 100)):03d}"
    elif consistency:
        consistency_suffix = "_consw100"
    return (
        f"production_replacement_{HOST}_{arm}_"
        f"bg{int(selection['background_target']):05d}_posw_{rad}{consistency_suffix}_"
        f"d{int(round(selection['dropout'] * 100)):03d}_"
        f"s{int(round(selection['label_smoothing'] * 100)):03d}_"
        f"{calibration}_{DATE_TAG}"
    )


def write_design(selection_manifest: Path) -> tuple[Path, list[dict]]:
    payload = read_json(selection_manifest)
    if (
        payload.get("status") != "complete"
        or payload.get("selection_type")
        != "ailanthus_blockcv_approved_recipe_for_production_fit"
        or payload.get("selection_count") != 1
    ):
        raise RuntimeError("Selection is not approved for Ailanthus production")
    selection = payload["selections"][0]
    if selection.get("host") != HOST:
        raise ValueError("Replacement production selection is not Ailanthus")
    source = validate_selection(selection)
    selection_hash = sha256(selection_manifest)
    code_root = Path(__file__).resolve().parent
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=code_root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if subprocess.run(
        ["git", "status", "--porcelain"], cwd=code_root, check=True,
        capture_output=True, text=True,
    ).stdout.strip():
        raise RuntimeError("Refusing to freeze replacement production from a dirty tree")

    label = f"production_ailanthus_replacement_{selection_hash[:10]}_{DATE_TAG}"
    stage_root = PRODUCTION_OUTPUT_ROOT / label
    design_root = DESIGN_BASE / label
    config = build_config(
        source, selection, stage_root, selection_manifest, selection_hash
    )
    name = replacement_name(selection)
    config["experiment"] = name
    config["scientific_note"] = (
        "Fresh Ailanthus replacement production fit after the original August fit "
        "failed matched-boundary acceptance and this recipe passed the unchanged "
        "uniform-screen and five-fold spatial gates. Other hosts remain frozen."
    )
    config_path = design_root / "configs" / f"{name}.json"
    if config_path.exists():
        if read_json(config_path) != config:
            raise FileExistsError(f"Refusing changed config: {config_path}")
    else:
        write_json_atomic(config_path, config)
    row = {
        **config["august_experiment_spec"],
        "experiment": name,
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "run_dir": str(stage_root / name),
        "approved_score_space": selection["production_score_space"],
        "status": "pending",
        "note": "",
    }
    registry = design_root / "experiment_registry.tsv"
    write_tsv_atomic(registry, [row], list(row))
    write_json_atomic(design_root / "design_metadata.json", {
        "status": "complete",
        "date_tag": DATE_TAG,
        "stage": "ailanthus-replacement-production-fit",
        "code_root": str(code_root),
        "code_commit": commit,
        "code_dirty": False,
        "selection_manifest": str(selection_manifest.resolve()),
        "selection_manifest_sha256": selection_hash,
        "stage_output_root": str(stage_root),
        "experiment_count": 1,
        "registry": str(registry),
        "frozen_hosts_not_retrained": [
            "juglans_nigra", "notholithocarpus_densiflorus",
            "prunus_serotina", "vitis_labrusca",
        ],
    })
    return registry, [row]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.selection_manifest.is_file():
        raise FileNotFoundError(args.selection_manifest)
    registry, rows = write_design(args.selection_manifest)
    print(f"Wrote gated Ailanthus replacement production design: {registry}")
    if args.run:
        run_experiments(registry, rows)


if __name__ == "__main__":
    main()
