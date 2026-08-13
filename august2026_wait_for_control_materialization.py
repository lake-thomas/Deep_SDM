#!/usr/bin/env python3
"""After the initial gates pass, materialize the three control hosts serially."""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import time
from pathlib import Path


PYTHON = Path("/home/ncsucnr/miniforge3/envs/host-naip-sdm/bin/python")
CODE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(
    "/mnt/rsstu_dsfas_gsv_naip/Host_NAIP_SDM/Host_Datasets_USBg_August2026/"
    "materialized_source_locked_20260813"
)
INITIAL_HOSTS = ["vitis_labrusca", "notholithocarpus_densiflorus"]
CONTROL_HOSTS = ["ailanthus_altissima", "juglans_nigra", "prunus_serotina"]
SMOKE_STATUS = DATA_ROOT / "stage00_vitis_smoke_watch_status.json"
STATUS = DATA_ROOT / "control_materialization_watch_status.json"
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


def materialization_passed(host: str) -> bool:
    path = DATA_ROOT / host / "materialization_run.json"
    if not path.is_file():
        return False
    state = read_json(path)
    return bool(
        state.get("status") == "complete"
        and state.get("manifest_status", {}).get("production_ready_manifests")
    )


def attrition_audit_passed(host: str) -> bool:
    path = DATA_ROOT / host / "qa" / "materialization_attrition_audit" / "summary.json"
    if not path.is_file():
        return False
    state = read_json(path)
    return bool(state.get("status") == "complete" and state.get("production_ready"))


def ensure_attrition_audit(host: str) -> None:
    if attrition_audit_passed(host):
        return
    output = DATA_ROOT / host / "qa" / "materialization_attrition_audit"
    if output.exists():
        raise RuntimeError(f"Refusing partial or failed attrition audit: {output}")
    subprocess.run(
        [
            str(PYTHON),
            str(CODE_ROOT / "august2026_materialization_attrition_audit.py"),
            "--host",
            host,
        ],
        cwd=CODE_ROOT,
        check=True,
    )
    if not attrition_audit_passed(host):
        raise RuntimeError(f"Attrition audit completed without passing: {host}")


def main() -> None:
    started_at = now()
    started = time.monotonic()
    while True:
        elapsed = time.monotonic() - started
        initial = {host: materialization_passed(host) for host in INITIAL_HOSTS}
        smoke = read_json(SMOKE_STATUS).get("status") if SMOKE_STATUS.is_file() else None
        if all(initial.values()) and smoke == "stage00_smoke_completed":
            break
        if elapsed > TIMEOUT_HOURS * 3600:
            write_status(
                {
                    "status": "failed_timeout",
                    "started_at": started_at,
                    "failed_at": now(),
                    "initial_materialization_passed": initial,
                    "stage00_smoke_status": smoke,
                }
            )
            raise TimeoutError("Initial dataset/smoke gates did not pass before timeout")
        write_status(
            {
                "status": "waiting_for_initial_dataset_and_smoke_gates",
                "started_at": started_at,
                "checked_at": now(),
                "initial_materialization_passed": initial,
                "stage00_smoke_status": smoke,
            }
        )
        time.sleep(POLL_SECONDS)

    for host in INITIAL_HOSTS:
        ensure_attrition_audit(host)

    completed = []
    for host in CONTROL_HOSTS:
        if materialization_passed(host):
            ensure_attrition_audit(host)
            completed.append(host)
            continue
        output = DATA_ROOT / host
        resume = output.exists() and any(output.iterdir())
        write_status(
            {
                "status": "materializing_control_host",
                "started_at": started_at,
                "updated_at": now(),
                "active_host": host,
                "completed_hosts": completed,
                "resume": resume,
            }
        )
        command = [
            str(PYTHON),
            str(CODE_ROOT / "august2026_materialize_controls.py"),
            "--host",
            host,
            "--workers",
            "16",
        ]
        if resume:
            command.append("--resume")
        subprocess.run(command, cwd=CODE_ROOT, check=True)
        if not materialization_passed(host):
            raise RuntimeError(f"Control host completed without passing its gate: {host}")
        ensure_attrition_audit(host)
        completed.append(host)

    write_status(
        {
            "status": "control_materializations_completed",
            "started_at": started_at,
            "completed_at": now(),
            "completed_hosts": completed,
        }
    )


if __name__ == "__main__":
    main()
