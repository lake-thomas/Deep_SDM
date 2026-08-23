#!/usr/bin/env python3
"""Freeze a BlockCV-approved Ailanthus replacement production recipe."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

from august2026_summarize_experiments import read_json, sha256, write_json_atomic


HOST = "ailanthus_altissima"


def freeze_selection(
    blockcv_evidence_dir: Path,
    blockcv_selection_manifest: Path,
    output_path: Path,
    code_commit: str,
) -> dict:
    status_path = blockcv_evidence_dir / "summary.json"
    host_summary_path = blockcv_evidence_dir / "blockcv_host_summary.csv"
    fold_metrics_path = blockcv_evidence_dir / "blockcv_fold_metrics.csv"
    required = [
        status_path, host_summary_path, fold_metrics_path,
        blockcv_evidence_dir / "COMPLETED", blockcv_selection_manifest,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Ailanthus BlockCV evidence is incomplete: {missing}")
    status = read_json(status_path)
    selection = read_json(blockcv_selection_manifest)
    if (
        status.get("status") != "complete"
        or status.get("production_ready") is not True
        or status.get("host_count") != 1
        or status.get("selection_manifest_sha256") != sha256(blockcv_selection_manifest)
    ):
        raise RuntimeError("Ailanthus BlockCV evidence did not pass")
    if (
        selection.get("status") != "complete"
        or selection.get("selection_type") != "ailanthus_recipe_for_blockcv"
        or selection.get("selection_count") != 1
    ):
        raise RuntimeError("Ailanthus BlockCV selection identity is invalid")
    summary = pd.read_csv(host_summary_path, low_memory=False)
    if (
        len(summary) != 1
        or summary.iloc[0]["host"] != HOST
        or not bool(summary.iloc[0]["blockcv_production_gate_passed"])
    ):
        raise RuntimeError("Ailanthus fold summary failed its production gate")
    row = summary.iloc[0]
    recipe = selection["selections"][0]
    for path_key, hash_key in (
        ("uniform_dataset", "uniform_dataset_sha256"),
        ("source_config", "source_config_sha256"),
    ):
        path = Path(recipe[path_key])
        if not path.is_file() or sha256(path) != recipe[hash_key]:
            raise ValueError(f"Ailanthus selected artifact changed: {path}")
    calibration = row["calibration_recommendation"]
    if calibration not in {"raw", "platt"}:
        raise ValueError(f"Unknown calibration recommendation: {calibration}")
    approved = {
        **recipe,
        "production_postfit_calibration": "platt" if calibration == "platt" else "none",
        "production_score_space": (
            "platt_calibrated_probability"
            if calibration == "platt"
            else "raw_sigmoid_probability"
        ),
        "blockcv_roc_auc_min": float(row["roc_auc_min"]),
        "blockcv_roc_auc_median": float(row["roc_auc_median"]),
        "blockcv_roc_auc_max": float(row["roc_auc_max"]),
        "blockcv_calibration_recommendation": calibration,
        "blockcv_folds_platt_nonworse_both": int(row["folds_platt_nonworse_both"]),
    }
    payload = {
        "status": "complete",
        "selection_type": "ailanthus_blockcv_approved_recipe_for_production_fit",
        "code_commit": code_commit,
        "blockcv_evidence_dir": str(blockcv_evidence_dir.resolve()),
        "blockcv_evidence_summary": str(status_path.resolve()),
        "blockcv_evidence_summary_sha256": sha256(status_path),
        "blockcv_host_summary": str(host_summary_path.resolve()),
        "blockcv_host_summary_sha256": sha256(host_summary_path),
        "blockcv_fold_metrics": str(fold_metrics_path.resolve()),
        "blockcv_fold_metrics_sha256": sha256(fold_metrics_path),
        "blockcv_selection_manifest": str(blockcv_selection_manifest.resolve()),
        "blockcv_selection_manifest_sha256": sha256(blockcv_selection_manifest),
        "selection_count": 1,
        "selections": [approved],
        "guardrail": (
            "This manifest authorizes one fresh Ailanthus uniform production fit. "
            "Final raw/configured artifact acceptance is still required; the other "
            "four accepted production fits remain frozen."
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
        raise FileExistsError(f"Refusing to replace selection: {args.output}")
    code_root = Path(__file__).resolve().parent
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=code_root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if subprocess.run(
        ["git", "status", "--porcelain"], cwd=code_root, check=True,
        capture_output=True, text=True,
    ).stdout.strip():
        raise RuntimeError("Refusing to freeze production selection from a dirty tree")
    print(json.dumps(freeze_selection(
        args.blockcv_evidence_dir, args.blockcv_selection_manifest,
        args.output, commit,
    ), indent=2))


if __name__ == "__main__":
    main()
