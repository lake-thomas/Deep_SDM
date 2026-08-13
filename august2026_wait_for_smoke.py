#!/usr/bin/env python3
"""Wait for the Vitis source-locked build, then run only the stage-00 smoke."""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import time
from pathlib import Path


PYTHON = Path("/home/ncsucnr/miniforge3/envs/host-naip-sdm/bin/python")
CODE_ROOT = Path(__file__).resolve().parent
VITIS_ROOT = Path(
    "/mnt/rsstu_dsfas_gsv_naip/Host_NAIP_SDM/Host_Datasets_USBg_August2026/"
    "materialized_source_locked_20260813/vitis_labrusca"
)
MATERIALIZATION_STATE = VITIS_ROOT / "materialization_run.json"
PROGRESS = VITIS_ROOT / "materialization_progress.jsonl"
STATUS = VITIS_ROOT.parent / "stage00_vitis_smoke_watch_status.json"
POLL_SECONDS = 60
TIMEOUT_HOURS = 12


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def write_status(payload: dict) -> None:
    partial = STATUS.with_name(f"{STATUS.name}.partial")
    with partial.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    os.replace(partial, STATUS)


def progress_lines() -> int:
    if not PROGRESS.is_file():
        return 0
    with PROGRESS.open("rb") as stream:
        return sum(1 for _ in stream)


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
            state = json.loads(MATERIALIZATION_STATE.read_text(encoding="utf-8"))
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
            "status": "running_stage00_smoke",
            "started_at": started_at,
            "smoke_started_at": now(),
            "materialization_state": str(MATERIALIZATION_STATE),
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
        }
    )


if __name__ == "__main__":
    main()
