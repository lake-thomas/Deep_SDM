#!/usr/bin/env python3
"""Run the three-host control dataset screen after all prerequisite gates."""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import time
from pathlib import Path

from august2026_watch_io import read_json_retry, write_json_atomic_retry


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
CONTROL_HOSTS = ["ailanthus_altissima", "juglans_nigra", "prunus_serotina"]
CONTROL_MATERIALIZATION_STATUS = DATA_ROOT / "control_materialization_watch_status.json"
STAGE01_STATUS = DATA_ROOT / "stage01_dataset_screen_watch_status.json"
REGISTRY = (
    DESIGN_ROOT
    / "stage01_control_dataset_screen_20260813"
    / "experiment_registry.tsv"
)
ACCEPTANCE_ROOT = OUTPUT_ROOT / "stage01_control_dataset_screen_acceptance_20260813"
SYNTHESIS_ROOT = (
    REGISTRY.parent / "analysis" / "combined_control_dataset_screen_evidence"
)
INITIAL_SYNTHESIS_ROOT = (
    DESIGN_ROOT / "stage01_dataset_screen_20260813" / "analysis"
    / "combined_dataset_screen_evidence"
)
UNIFIED_SYNTHESIS_ROOT = (
    DESIGN_ROOT / "stage01_all_host_dataset_screen_20260813" / "analysis"
    / "combined_all_host_dataset_screen_evidence"
)
STATUS = DATA_ROOT / "stage01_control_dataset_screen_watch_status.json"
POLL_SECONDS = 60
TIMEOUT_HOURS = 60


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def read_json(path: Path) -> dict:
    return read_json_retry(path)


def write_status(payload: dict) -> None:
    write_json_atomic_retry(STATUS, payload)


def attrition_ready(host: str) -> bool:
    path = DATA_ROOT / host / "qa" / "materialization_attrition_audit" / "summary.json"
    provenance_path = DATA_ROOT / host / "qa" / "candidate_builder_provenance" / "summary.json"
    if not path.is_file() or not provenance_path.is_file():
        return False
    state = read_json(path)
    provenance = read_json(provenance_path)
    return bool(
        state.get("status") == "complete" and state.get("production_ready")
        and provenance.get("status") == "complete"
        and provenance.get("production_ready")
    )


def control_gates_ready(
    materialization_status: str | None,
    stage01_status: str | None,
    attrition: dict[str, bool],
) -> bool:
    return bool(
        materialization_status == "control_materializations_completed"
        and stage01_status == "stage01_dataset_screen_evidence_completed"
        and all(attrition.values())
    )


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=CODE_ROOT, check=True)


def wait_for_gates() -> None:
    started = time.monotonic()
    started_at = now()
    while True:
        materialization = (
            read_json(CONTROL_MATERIALIZATION_STATUS).get("status")
            if CONTROL_MATERIALIZATION_STATUS.is_file()
            else None
        )
        stage01 = read_json(STAGE01_STATUS).get("status") if STAGE01_STATUS.is_file() else None
        attrition = {host: attrition_ready(host) for host in CONTROL_HOSTS}
        if control_gates_ready(materialization, stage01, attrition):
            return
        elapsed = time.monotonic() - started
        if elapsed > TIMEOUT_HOURS * 3600:
            raise TimeoutError("Control dataset-screen prerequisites timed out")
        write_status(
            {
                "status": "waiting_for_control_screen_gates",
                "started_at": started_at,
                "checked_at": now(),
                "control_materialization_status": materialization,
                "stage01_dataset_screen_status": stage01,
                "attrition_ready": attrition,
                "elapsed_seconds": elapsed,
            }
        )
        time.sleep(POLL_SECONDS)


def main_work() -> None:
    wait_for_gates()
    write_status({"status": "running_control_dataset_screen", "updated_at": now()})
    run(
        [
            str(PYTHON),
            str(CODE_ROOT / "august2026_experiments.py"),
            "--stage",
            "control-dataset-screen",
            "--run",
        ]
    )
    write_status(
        {
            "status": "validating_control_training_evidence",
            "updated_at": now(),
            "registry": str(REGISTRY),
        }
    )
    run(
        [
            str(PYTHON),
            str(CODE_ROOT / "august2026_summarize_experiments.py"),
            "--registry",
            str(REGISTRY),
        ]
    )
    training_metrics = REGISTRY.parent / "analysis" / "experiment_metrics.csv"
    matched_roots, exact_roots = [], []
    for host in CONTROL_HOSTS:
        matched = ACCEPTANCE_ROOT / "matched" / host
        exact = ACCEPTANCE_ROOT / "exact_overlap" / host
        matched_roots.append(matched)
        exact_roots.append(exact)
        write_status(
            {
                "status": "running_control_artifact_replays",
                "updated_at": now(),
                "active_host": host,
                "active_diagnostic": "matched_ny",
            }
        )
        run(
            [
                str(PYTHON),
                str(CODE_ROOT / "august2026_batch_artifact_probe.py"),
                "--registry",
                str(REGISTRY),
                "--host",
                host,
                "--output-root",
                str(matched),
                "--device",
                "cuda",
                "--resume-completed",
            ]
        )
        write_status(
            {
                "status": "running_control_artifact_replays",
                "updated_at": now(),
                "active_host": host,
                "active_diagnostic": "exact_overlap",
            }
        )
        run(
            [
                str(PYTHON),
                str(CODE_ROOT / "august2026_exact_overlap_probe.py"),
                "--registry",
                str(REGISTRY),
                "--host",
                host,
                "--output-root",
                str(exact),
                "--device",
                "cuda",
                "--resume-completed",
            ]
        )

    command = [
        str(PYTHON),
        str(CODE_ROOT / "august2026_dataset_screen_evidence.py"),
        "--training-metrics",
        str(training_metrics),
    ]
    for path in matched_roots:
        command.extend(["--artifact-root", str(path)])
    for path in exact_roots:
        command.extend(["--exact-overlap-root", str(path)])
    command.extend(["--output-dir", str(SYNTHESIS_ROOT)])
    write_status(
        {"status": "synthesizing_control_dataset_evidence", "updated_at": now()}
    )
    run(command)
    write_status(
        {"status": "combining_five_host_dataset_evidence", "updated_at": now()}
    )
    run(
        [
            str(PYTHON),
            str(CODE_ROOT / "august2026_combine_dataset_screen_evidence.py"),
            "--source-dir", str(INITIAL_SYNTHESIS_ROOT),
            "--source-dir", str(SYNTHESIS_ROOT),
            "--output-dir", str(UNIFIED_SYNTHESIS_ROOT),
        ]
    )
    write_status(
        {
            "status": "five_host_dataset_screen_evidence_completed",
            "completed_at": now(),
            "registry": str(REGISTRY),
            "training_metrics": str(training_metrics),
            "matched_artifact_roots": [str(path) for path in matched_roots],
            "exact_overlap_roots": [str(path) for path in exact_roots],
            "synthesis": str(SYNTHESIS_ROOT),
            "unified_five_host_synthesis": str(UNIFIED_SYNTHESIS_ROOT),
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
