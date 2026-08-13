#!/usr/bin/env python3
"""Freeze BlockCV-approved recipes and calibration choices for production fits."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

from august2026_combine_dataset_screen_evidence import HOSTS
from august2026_summarize_experiments import read_json, sha256, write_json_atomic


def freeze_production_selection(
    blockcv_evidence_dir: Path,
    blockcv_selection_manifest: Path,
    output_path: Path,
    code_commit: str,
) -> dict:
    evidence_status_path = blockcv_evidence_dir / "summary.json"
    host_summary_path = blockcv_evidence_dir / "blockcv_host_summary.csv"
    fold_metrics_path = blockcv_evidence_dir / "blockcv_fold_metrics.csv"
    required = [
        evidence_status_path,
        host_summary_path,
        fold_metrics_path,
        blockcv_evidence_dir / "COMPLETED",
        blockcv_selection_manifest,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"BlockCV production evidence is incomplete: {missing}")
    evidence_status = read_json(evidence_status_path)
    if evidence_status.get("status") != "complete" or not evidence_status.get(
        "production_ready"
    ):
        raise RuntimeError("BlockCV evidence did not pass its production gate")
    selection = read_json(blockcv_selection_manifest)
    if (
        selection.get("status") != "complete"
        or selection.get("selection_type") != "recipe_for_blockcv"
    ):
        raise RuntimeError("BlockCV selection manifest is invalid")
    if (
        evidence_status.get("selection_manifest_sha256")
        != sha256(blockcv_selection_manifest)
    ):
        raise ValueError("BlockCV evidence does not reference the supplied selection")

    summaries = pd.read_csv(host_summary_path, low_memory=False)
    recipes = {item["host"]: item for item in selection["selections"]}
    if set(recipes) != HOSTS:
        raise ValueError("Production selection requires exactly the five production hosts")
    if set(summaries["host"]) != set(recipes):
        raise ValueError("BlockCV host summary differs from selected recipes")
    if not summaries["blockcv_production_gate_passed"].astype(bool).all():
        raise RuntimeError("One or more host BlockCV production gates failed")

    outputs = []
    for row in summaries.itertuples(index=False):
        recipe = recipes[row.host]
        uniform_dataset = Path(recipe["uniform_dataset"])
        source_config = Path(recipe["source_config"])
        if sha256(uniform_dataset) != recipe["uniform_dataset_sha256"]:
            raise ValueError(f"Uniform dataset changed for {row.host}")
        if sha256(source_config) != recipe["source_config_sha256"]:
            raise ValueError(f"Recipe config changed for {row.host}")
        calibration = row.calibration_recommendation
        if calibration not in {"raw", "platt"}:
            raise ValueError(f"Unknown calibration recommendation for {row.host}")
        outputs.append(
            {
                **recipe,
                "production_postfit_calibration": (
                    "platt" if calibration == "platt" else "none"
                ),
                "production_score_space": (
                    "platt_calibrated_probability"
                    if calibration == "platt"
                    else "raw_sigmoid_probability"
                ),
                "blockcv_roc_auc_min": row.roc_auc_min,
                "blockcv_roc_auc_median": row.roc_auc_median,
                "blockcv_roc_auc_max": row.roc_auc_max,
                "blockcv_calibration_recommendation": calibration,
                "blockcv_folds_platt_nonworse_both": int(
                    row.folds_platt_nonworse_both
                ),
            }
        )
    payload = {
        "status": "complete",
        "selection_type": "blockcv_approved_recipe_for_production_fit",
        "code_commit": code_commit,
        "blockcv_evidence_dir": str(blockcv_evidence_dir.resolve()),
        "blockcv_evidence_summary": str(evidence_status_path.resolve()),
        "blockcv_evidence_summary_sha256": sha256(evidence_status_path),
        "blockcv_host_summary": str(host_summary_path.resolve()),
        "blockcv_host_summary_sha256": sha256(host_summary_path),
        "blockcv_fold_metrics": str(fold_metrics_path.resolve()),
        "blockcv_fold_metrics_sha256": sha256(fold_metrics_path),
        "blockcv_selection_manifest": str(blockcv_selection_manifest.resolve()),
        "blockcv_selection_manifest_sha256": sha256(blockcv_selection_manifest),
        "selection_count": len(outputs),
        "selections": outputs,
        "guardrail": (
            "This manifest authorizes fresh uniform production fits only. Final model "
            "acceptance diagnostics, calibrated/raw score-space provenance, Hann "
            "inference, structural raster QA, and visual review remain required."
        ),
    }
    write_json_atomic(output_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blockcv-evidence-dir", type=Path, required=True)
    parser.add_argument("--blockcv-selection-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace production selection: {args.output}")
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
        raise RuntimeError("Refusing to freeze production selection from a dirty tree")
    payload = freeze_production_selection(
        args.blockcv_evidence_dir,
        args.blockcv_selection_manifest,
        args.output,
        commit,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
