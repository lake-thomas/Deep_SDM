#!/usr/bin/env python3
"""Freeze and optionally run the predeclared 15-epoch Ailanthus retry."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

from august2026_ailanthus_production_train import HOST
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


DATE_TAG = "20260825"
HORIZON_EPOCHS = 15


def _history(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_horizon_evidence(selection: dict, failed_summary_path: Path) -> dict:
    failed = read_json(failed_summary_path)
    if (
        failed.get("status") != "complete"
        or failed.get("production_ready_for_release_assembly") is not False
        or set(failed.get("failures", []))
        != {"raw_matched_boundary_regression", "configured_matched_boundary_regression"}
    ):
        raise RuntimeError("Supplied long-horizon acceptance is not the expected narrow failure")
    training_metrics = Path(failed["training_metrics"])
    if sha256(training_metrics) != failed["training_metrics_sha256"]:
        raise ValueError("Failed long-horizon training metrics changed")
    registry = Path(failed["registry"])
    if sha256(registry) != failed["registry_sha256"]:
        raise ValueError("Failed long-horizon registry changed")
    with registry.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 1:
        raise ValueError("Failed long-horizon registry must contain one fit")
    long_config = Path(rows[0]["config_path"])
    long_run = Path(rows[0]["run_dir"])
    if not long_config.is_file() or not long_run.is_dir():
        raise FileNotFoundError("Failed long-horizon config or run is missing")
    long_payload = read_json(long_config)
    if (
        long_payload.get("selection_gate", {}).get("selection_manifest_sha256")
        != failed["production_selection_manifest_sha256"]
    ):
        raise ValueError("Failed long-horizon fit used a different production selection")

    screen_config = Path(selection["source_config"])
    screen_payload = read_json(screen_config)
    screen_run = Path(selection["source_checkpoint"]).parent.parent
    screen_history_path = screen_run / "training_history.csv"
    long_history_path = long_run / "training_history.csv"
    screen_history = _history(screen_history_path)
    long_history = _history(long_history_path)
    if int(screen_payload["epochs"]) != HORIZON_EPOCHS:
        raise ValueError("Approved screen did not use the declared retry horizon")
    if len(screen_history) != HORIZON_EPOCHS or len(long_history) <= HORIZON_EPOCHS:
        raise ValueError("Training histories do not span the required comparison")
    if screen_history != long_history[:HORIZON_EPOCHS]:
        raise ValueError("Screen and long fit are not identical through epoch 15")
    return {
        "failed_acceptance_summary": str(failed_summary_path.resolve()),
        "failed_acceptance_summary_sha256": sha256(failed_summary_path),
        "failed_long_config": str(long_config.resolve()),
        "failed_long_config_sha256": sha256(long_config),
        "failed_long_checkpoint_sha256": rows[0].get("best_checkpoint_sha256"),
        "screen_history": str(screen_history_path.resolve()),
        "screen_history_sha256": sha256(screen_history_path),
        "failed_long_history": str(long_history_path.resolve()),
        "failed_long_history_sha256": sha256(long_history_path),
        "identical_prefix_epochs": HORIZON_EPOCHS,
        "checkpoint_rule": (
            "lowest ordinary unsmoothed unweighted validation BCE among epochs 1-15"
        ),
        "gate_policy": "unchanged frozen matched-boundary and exact-overlap limits",
    }


def retry_name(selection: dict) -> str:
    rad = selection.get("radiometric_recipe")
    if rad not in {"rad1", "rad2"}:
        raise ValueError("Retry recipe lacks an explicit radiometric label")
    probability = float(selection.get("image_modality_dropout_probability", 0.0))
    calibration = (
        "platt" if selection["production_postfit_calibration"] == "platt" else "raw"
    )
    return (
        f"production_retry_{HOST}_db_bg{int(selection['background_target']):05d}_"
        f"posw_{rad}_moddrop{int(round(probability * 100)):02d}_"
        f"d{int(round(selection['dropout'] * 100)):03d}_"
        f"s{int(round(selection['label_smoothing'] * 100)):03d}_"
        f"h{HORIZON_EPOCHS:03d}_{calibration}_{DATE_TAG}"
    )


def write_design(
    selection_manifest: Path, failed_acceptance_summary: Path
) -> tuple[Path, list[dict]]:
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
        raise ValueError("Production selection is not Ailanthus")
    source = validate_selection(selection)
    horizon_evidence = validate_horizon_evidence(selection, failed_acceptance_summary)
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
        raise RuntimeError("Refusing to freeze horizon retry from a dirty tree")

    label = f"production_ailanthus_retry_h015_{selection_hash[:10]}_{DATE_TAG}"
    stage_root = PRODUCTION_OUTPUT_ROOT / label
    design_root = DESIGN_BASE / label
    config = build_config(source, selection, stage_root, selection_manifest, selection_hash)
    name = retry_name(selection)
    config.update({
        "experiment": name,
        "epochs": HORIZON_EPOCHS,
        "lr_patience": int(source["lr_patience"]),
        "es_patience": int(source["es_patience"]),
        "save_epoch_checkpoints": False,
        "horizon_selection_gate": horizon_evidence,
        "scientific_note": (
            "Predeclared fixed-horizon Ailanthus production retry. The deterministic "
            "long fit exactly reproduced the accepted screen for epochs 1-15, then its "
            "later validation-loss checkpoint narrowly failed the unchanged acquisition-"
            "boundary gate. This retry selects only by ordinary validation BCE within the "
            "original 15-epoch screen horizon; final artifact limits remain unchanged."
        ),
    })
    config["august_experiment_spec"].update({
        "stage": "production-fit",
        "epochs": HORIZON_EPOCHS,
        "checkpoint_horizon_epochs": HORIZON_EPOCHS,
        "checkpoint_rule": horizon_evidence["checkpoint_rule"],
    })
    config_path = design_root / "configs" / f"{name}.json"
    if config_path.exists():
        if read_json(config_path) != config:
            raise FileExistsError(f"Refusing changed retry config: {config_path}")
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
        "stage": "ailanthus-fixed-horizon-production-retry",
        "code_root": str(code_root),
        "code_commit": commit,
        "code_dirty": False,
        "selection_manifest": str(selection_manifest.resolve()),
        "selection_manifest_sha256": selection_hash,
        "horizon_evidence": horizon_evidence,
        "stage_output_root": str(stage_root),
        "experiment_count": 1,
        "registry": str(registry),
    })
    return registry, [row]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--failed-acceptance-summary", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    registry, rows = write_design(
        args.selection_manifest, args.failed_acceptance_summary
    )
    print(f"Wrote predeclared Ailanthus horizon retry: {registry}")
    if args.run:
        run_experiments(registry, rows)


if __name__ == "__main__":
    main()
