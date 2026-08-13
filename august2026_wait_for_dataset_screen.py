#!/usr/bin/env python3
"""Run the August dataset screen after smoke and dataset QA gates pass."""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd


PYTHON = Path("/home/ncsucnr/miniforge3/envs/host-naip-sdm/bin/python")
CODE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(
    "/mnt/rsstu_dsfas_gsv_naip/Host_NAIP_SDM/Host_Datasets_USBg_August2026/"
    "materialized_source_locked_20260813"
)
DESIGN_ROOT = Path(
    "/mnt/rsstu_dsfas_gsv_naip/Host_NAIP_SDM/Host_Datasets_USBg_August2026/"
    "experiment_designs_20260813"
)
OUTPUT_ROOT = Path(
    "/mnt/rsstu_dsfas_gsv_naip/Host_NAIP_SDM/Outputs_Host_NAIP_SDMs/"
    "experimental_model_runs_august2026"
)
SMOKE_STATUS = DATA_ROOT / "stage00_vitis_smoke_watch_status.json"
SMOKE_REGISTRY = (
    DESIGN_ROOT / "stage00_smoke_20260813" / "experiment_registry.tsv"
)
DATASET_REGISTRY = (
    DESIGN_ROOT / "stage01_dataset_screen_20260813" / "experiment_registry.tsv"
)
ACCEPTANCE_ROOT = OUTPUT_ROOT / "stage01_dataset_screen_acceptance_20260813"
VITIS_MATCHED = ACCEPTANCE_ROOT / "matched" / "vitis_labrusca"
TANOAK_MATCHED = ACCEPTANCE_ROOT / "matched" / "notholithocarpus_densiflorus"
VITIS_EXACT = ACCEPTANCE_ROOT / "exact_overlap" / "vitis_labrusca"
SYNTHESIS_ROOT = (
    DESIGN_ROOT
    / "stage01_dataset_screen_20260813"
    / "analysis"
    / "combined_dataset_screen_evidence"
)
STATUS = DATA_ROOT / "stage01_dataset_screen_watch_status.json"
POLL_SECONDS = 60
TIMEOUT_HOURS = 18


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_status(payload: dict) -> None:
    partial = STATUS.with_name(f"{STATUS.name}.partial")
    with partial.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    os.replace(partial, STATUS)


def attrition_ready(host: str) -> bool:
    path = DATA_ROOT / host / "qa" / "materialization_attrition_audit" / "summary.json"
    if not path.is_file():
        return False
    state = read_json(path)
    return bool(state.get("status") == "complete" and state.get("production_ready"))


def smoke_evidence_ready(metrics: pd.DataFrame) -> bool:
    required = {
        "evidence_gate_passed",
        "epochs_completed",
        "test_roc_auc_all_thresholds",
        "test_brier_score",
        "test_calibration_binary_log_loss",
    }
    if len(metrics) != 1 or not required.issubset(metrics.columns):
        return False
    gate = str(metrics.iloc[0]["evidence_gate_passed"]).strip().lower() == "true"
    numeric = pd.to_numeric(
        metrics.iloc[0][
            [
                "epochs_completed",
                "test_roc_auc_all_thresholds",
                "test_brier_score",
                "test_calibration_binary_log_loss",
            ]
        ],
        errors="coerce",
    ).to_numpy(dtype=float)
    return bool(gate and np.isfinite(numeric).all() and numeric[0] >= 1)


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=CODE_ROOT, check=True)


def wait_for_gates() -> None:
    started = time.monotonic()
    started_at = now()
    while True:
        smoke = read_json(SMOKE_STATUS).get("status") if SMOKE_STATUS.is_file() else None
        attrition = {
            host: attrition_ready(host)
            for host in ["vitis_labrusca", "notholithocarpus_densiflorus"]
        }
        if smoke == "stage00_smoke_completed" and all(attrition.values()):
            return
        elapsed = time.monotonic() - started
        if elapsed > TIMEOUT_HOURS * 3600:
            raise TimeoutError("Dataset-screen smoke/attrition gates timed out")
        write_status(
            {
                "status": "waiting_for_smoke_and_attrition_gates",
                "started_at": started_at,
                "checked_at": now(),
                "smoke_status": smoke,
                "attrition_ready": attrition,
                "elapsed_seconds": elapsed,
            }
        )
        time.sleep(POLL_SECONDS)


def main_work() -> None:
    wait_for_gates()
    write_status({"status": "validating_stage00_smoke", "updated_at": now()})
    run(
        [
            str(PYTHON),
            str(CODE_ROOT / "august2026_summarize_experiments.py"),
            "--registry",
            str(SMOKE_REGISTRY),
        ]
    )
    smoke_metrics_path = SMOKE_REGISTRY.parent / "analysis" / "experiment_metrics.csv"
    if not smoke_metrics_path.is_file() or not smoke_evidence_ready(
        pd.read_csv(smoke_metrics_path, low_memory=False)
    ):
        raise RuntimeError("Stage-00 smoke did not pass its execution evidence gate")

    write_status(
        {
            "status": "running_stage01_dataset_screen",
            "updated_at": now(),
            "smoke_metrics": str(smoke_metrics_path),
        }
    )
    run(
        [
            str(PYTHON),
            str(CODE_ROOT / "august2026_experiments.py"),
            "--stage",
            "dataset-screen",
            "--run",
        ]
    )
    write_status(
        {
            "status": "validating_stage01_training_evidence",
            "updated_at": now(),
            "registry": str(DATASET_REGISTRY),
        }
    )
    run(
        [
            str(PYTHON),
            str(CODE_ROOT / "august2026_summarize_experiments.py"),
            "--registry",
            str(DATASET_REGISTRY),
        ]
    )
    training_metrics = DATASET_REGISTRY.parent / "analysis" / "experiment_metrics.csv"

    for host, output in [
        ("vitis_labrusca", VITIS_MATCHED),
        ("notholithocarpus_densiflorus", TANOAK_MATCHED),
    ]:
        write_status(
            {
                "status": "running_matched_artifact_replay",
                "updated_at": now(),
                "active_host": host,
            }
        )
        run(
            [
                str(PYTHON),
                str(CODE_ROOT / "august2026_batch_artifact_probe.py"),
                "--registry",
                str(DATASET_REGISTRY),
                "--host",
                host,
                "--output-root",
                str(output),
                "--device",
                "cuda",
                "--resume-completed",
            ]
        )

    write_status(
        {
            "status": "running_vitis_exact_overlap_replay",
            "updated_at": now(),
        }
    )
    run(
        [
            str(PYTHON),
            str(CODE_ROOT / "august2026_exact_overlap_probe.py"),
            "--registry",
            str(DATASET_REGISTRY),
            "--host",
            "vitis_labrusca",
            "--output-root",
            str(VITIS_EXACT),
            "--device",
            "cuda",
            "--resume-completed",
        ]
    )

    write_status(
        {
            "status": "synthesizing_dataset_screen_evidence",
            "updated_at": now(),
        }
    )
    run(
        [
            str(PYTHON),
            str(CODE_ROOT / "august2026_dataset_screen_evidence.py"),
            "--training-metrics",
            str(training_metrics),
            "--artifact-root",
            str(VITIS_MATCHED),
            "--artifact-root",
            str(TANOAK_MATCHED),
            "--exact-overlap-root",
            str(VITIS_EXACT),
            "--output-dir",
            str(SYNTHESIS_ROOT),
        ]
    )
    write_status(
        {
            "status": "stage01_dataset_screen_evidence_completed",
            "completed_at": now(),
            "registry": str(DATASET_REGISTRY),
            "training_metrics": str(training_metrics),
            "vitis_matched_artifacts": str(VITIS_MATCHED),
            "tanoak_matched_artifacts": str(TANOAK_MATCHED),
            "vitis_exact_overlap": str(VITIS_EXACT),
            "synthesis": str(SYNTHESIS_ROOT),
        }
    )


def main() -> None:
    try:
        main_work()
    except Exception as error:
        write_status(
            {
                "status": "failed",
                "failed_at": now(),
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        raise


if __name__ == "__main__":
    main()
