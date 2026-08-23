#!/usr/bin/env python3
"""Run the complete raw/configured acceptance gate for one Ailanthus replacement."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from august2026_ailanthus_replacement_acceptance import synthesize_acceptance
from august2026_run_augmentation_acceptance import current_clean_commit
from august2026_run_final_model_acceptance import run_command_step
from august2026_summarize_experiments import (
    read_json,
    sha256,
    summarize_registry,
    write_json_atomic,
)


HOST = "ailanthus_altissima"


def probe_commands(
    registry: Path, output_root: Path, device: str, batch_size: int,
    resume_completed: bool,
) -> list[tuple[str, Path, list[str]]]:
    code_root = Path(__file__).resolve().parent
    commands = []
    for score_space in ("raw", "configured"):
        for diagnostic, script in (
            ("matched", "august2026_batch_artifact_probe.py"),
            ("exact_overlap", "august2026_exact_overlap_probe.py"),
        ):
            output = output_root / "artifacts" / score_space / diagnostic / HOST
            command = [
                sys.executable, str(code_root / script),
                "--registry", str(registry), "--host", HOST,
                "--output-root", str(output), "--device", device,
                "--batch-size", str(batch_size), "--score-space", score_space,
            ]
            if resume_completed:
                command.append("--resume-completed")
            commands.append((f"{score_space}:{diagnostic}", output, command))
    return commands


def run_acceptance(
    registry: Path,
    blockcv_evidence_dir: Path,
    output_root: Path,
    device: str,
    batch_size: int,
    resume_completed: bool,
) -> dict:
    training_status = summarize_registry(registry, allow_incomplete=False)
    training = pd.read_csv(training_status["metrics_csv"], low_memory=False)
    if (
        len(training) != 1
        or set(training["host"]) != {HOST}
        or set(training["stage"]) != {"production-fit"}
        or not training["evidence_gate_passed"].astype(bool).all()
    ):
        raise RuntimeError("Replacement Ailanthus training evidence is incomplete")
    blockcv_summary = blockcv_evidence_dir / "summary.json"
    blockcv_marker = blockcv_evidence_dir / "COMPLETED"
    if not blockcv_summary.is_file() or not blockcv_marker.is_file():
        raise FileNotFoundError("Replacement BlockCV evidence is incomplete")
    blockcv = read_json(blockcv_summary)
    if (
        blockcv.get("status") != "complete"
        or blockcv.get("production_ready") is not True
        or blockcv.get("host_count") != 1
    ):
        raise RuntimeError("Replacement BlockCV gate failed")

    code_root = Path(__file__).resolve().parent
    commit = current_clean_commit(code_root)
    status_path = output_root / "acceptance_status.json"
    identity = {
        "registry_sha256": sha256(registry),
        "training_metrics_sha256": sha256(Path(training_status["metrics_csv"])),
        "blockcv_summary_sha256": sha256(blockcv_summary),
        "code_commit": commit,
        "device": device,
        "batch_size": batch_size,
    }
    if output_root.exists() and any(output_root.iterdir()):
        if not resume_completed or not status_path.is_file():
            raise FileExistsError(f"Refusing nonempty replacement gate: {output_root}")
        status = read_json(status_path)
        observed = {key: status.get(key) for key in identity}
        if observed != identity:
            raise ValueError(
                f"Replacement acceptance resume identity changed: {observed} != {identity}"
            )
    else:
        output_root.mkdir(parents=True, exist_ok=True)
        status = {
            "status": "running",
            "stage": "ailanthus-replacement-final-acceptance-runner",
            "registry": str(registry.resolve()),
            "blockcv_evidence_dir": str(blockcv_evidence_dir.resolve()),
            **identity,
            "completed_steps": [],
        }
        write_json_atomic(status_path, status)

    for step, output, command in probe_commands(
        registry, output_root, device, batch_size, resume_completed
    ):
        run_command_step(
            status, status_path, step, output / "COMPLETED", command, code_root
        )

    acceptance_dir = output_root / "acceptance"
    acceptance_summary = acceptance_dir / "summary.json"
    if acceptance_dir.exists():
        if not resume_completed or not acceptance_summary.is_file():
            raise FileExistsError(f"Refusing existing replacement synthesis: {acceptance_dir}")
        final = read_json(acceptance_summary)
    else:
        try:
            final = synthesize_acceptance(
                registry, blockcv_evidence_dir,
                output_root / "artifacts/raw/matched" / HOST,
                output_root / "artifacts/configured/matched" / HOST,
                output_root / "artifacts/raw/exact_overlap" / HOST,
                output_root / "artifacts/configured/exact_overlap" / HOST,
                acceptance_dir,
            )
        except RuntimeError:
            if not acceptance_summary.is_file():
                raise
            final = read_json(acceptance_summary)
    ready = final.get("production_ready_for_release_assembly") is True
    status.pop("active_step", None)
    status.update({
        "status": "complete" if ready else "complete_with_failures",
        "completed_step_count": len(status.get("completed_steps", [])),
        "production_ready_for_release_assembly": ready,
        "replacement_acceptance": str(acceptance_summary.resolve()),
        "replacement_acceptance_sha256": sha256(acceptance_summary),
        "failures": final.get("failures", []),
    })
    write_json_atomic(status_path, status)
    (output_root / "COMPLETED").write_text(
        "complete\n" if ready else "complete_with_failures\n", encoding="utf-8"
    )
    if not ready:
        raise RuntimeError(f"Replacement Ailanthus failed acceptance: {status['failures']}")
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--blockcv-evidence-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--resume-completed", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    print(json.dumps(run_acceptance(
        args.registry, args.blockcv_evidence_dir, args.output_root,
        args.device, args.batch_size, args.resume_completed,
    ), indent=2))


if __name__ == "__main__":
    main()
