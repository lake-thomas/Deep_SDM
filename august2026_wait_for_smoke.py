#!/usr/bin/env python3
"""Wait for the Vitis source-locked build, then run only the stage-00 smoke."""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import time
from pathlib import Path

from august2026_watch_io import (
    count_lines_retry,
    read_json_retry,
    write_json_atomic_retry,
)


PYTHON = Path("/home/ncsucnr/miniforge3/envs/host-naip-sdm/bin/python")
CODE_ROOT = Path(__file__).resolve().parent
VITIS_ROOT = Path(
    "/mnt/rsstu_dsfas_gsv_naip/Host_NAIP_SDM/Host_Datasets_USBg_August2026/"
    "materialized_source_locked_20260813/vitis_labrusca"
)
MATERIALIZATION_STATE = VITIS_ROOT / "materialization_run.json"
PROGRESS = VITIS_ROOT / "materialization_progress.jsonl"
STATUS = VITIS_ROOT.parent / "stage00_vitis_smoke_watch_status.json"
ATTRITION_AUDIT_DIR = VITIS_ROOT / "qa" / "materialization_attrition_audit"
ATTRITION_AUDIT_SUMMARY = ATTRITION_AUDIT_DIR / "summary.json"
POLL_SECONDS = 60
TIMEOUT_HOURS = 12


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def write_status(payload: dict) -> None:
    write_json_atomic_retry(STATUS, payload)


def progress_lines() -> int:
    return count_lines_retry(PROGRESS)


def main() -> None:
    started = time.monotonic()
    started_at = now()
    while True:
        elapsed = time.monotonic() - started
        if elapsed > TIMEOUT_HOURS * 3600:
            payload = {
                "status": "failed_timeout",
                "started_at": started_at,
                "failed_at": now(),
                "elapsed_seconds": elapsed,
                "progress_jsonl_lines": progress_lines(),
            }
            write_status(payload)
            raise TimeoutError(f"Vitis materialization did not complete in {TIMEOUT_HOURS} hours")

        if MATERIALIZATION_STATE.is_file():
            state = read_json_retry(MATERIALIZATION_STATE)
            if state.get("status") == "complete":
                if not state.get("manifest_status", {}).get("production_ready_manifests"):
                    raise RuntimeError("Vitis materialization completed without production-ready manifests")
                break
            if state.get("status") not in {"running", "complete"}:
                raise RuntimeError(f"Unexpected Vitis materialization state: {state.get('status')}")

        write_status(
            {
                "status": "waiting_for_vitis_materialization",
                "started_at": started_at,
                "checked_at": now(),
                "elapsed_seconds": elapsed,
                "progress_jsonl_lines": progress_lines(),
            }
        )
        time.sleep(POLL_SECONDS)

    write_status(
        {
            "status": "running_vitis_materialization_attrition_audit",
            "started_at": started_at,
            "audit_started_at": now(),
            "materialization_state": str(MATERIALIZATION_STATE),
        }
    )
    audit_ready = False
    if ATTRITION_AUDIT_SUMMARY.is_file():
        audit = read_json_retry(ATTRITION_AUDIT_SUMMARY)
        audit_ready = bool(
            audit.get("status") == "complete" and audit.get("production_ready")
        )
    if not audit_ready:
        if ATTRITION_AUDIT_DIR.exists():
            raise RuntimeError(
                f"Refusing partial or failed attrition audit: {ATTRITION_AUDIT_DIR}"
            )
        subprocess.run(
            [
                str(PYTHON),
                str(CODE_ROOT / "august2026_materialization_attrition_audit.py"),
                "--host",
                "vitis_labrusca",
            ],
            cwd=CODE_ROOT,
            check=True,
        )

    write_status(
        {
            "status": "running_stage00_smoke",
            "started_at": started_at,
            "smoke_started_at": now(),
            "materialization_state": str(MATERIALIZATION_STATE),
            "materialization_attrition_audit": str(ATTRITION_AUDIT_SUMMARY),
        }
    )
    subprocess.run(
        [
            str(PYTHON),
            str(CODE_ROOT / "august2026_experiments.py"),
            "--stage",
            "smoke",
            "--run",
        ],
        cwd=CODE_ROOT,
        check=True,
    )
    write_status(
        {
            "status": "stage00_smoke_completed",
            "started_at": started_at,
            "completed_at": now(),
            "materialization_state": str(MATERIALIZATION_STATE),
            "materialization_attrition_audit": str(ATTRITION_AUDIT_SUMMARY),
        }
    )


if __name__ == "__main__":
    main()
