#!/usr/bin/env python3
"""Run and synthesize the frozen Stage 03 artifact-acceptance package."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from august2026_augmentation_screen_evidence import (
    synthesize_augmentation_evidence,
)
from august2026_combine_dataset_screen_evidence import HOSTS
from august2026_exact_overlap_probe import EASTERN_HOSTS
from august2026_summarize_experiments import (
    read_json,
    sha256,
    summarize_registry,
    write_json_atomic,
)


def probe_commands(
    registry: Path,
    output_root: Path,
    device: str,
    batch_size: int,
    resume_completed: bool,
) -> list[tuple[str, str, Path, list[str]]]:
    commands = []
    code_root = Path(__file__).resolve().parent
    for host in sorted(HOSTS):
        output = output_root / "matched" / host
        command = [
            sys.executable,
            str(code_root / "august2026_batch_artifact_probe.py"),
            "--registry", str(registry),
            "--host", host,
            "--output-root", str(output),
            "--device", device,
            "--batch-size", str(batch_size),
            "--score-space", "raw",
        ]
        if resume_completed:
            command.append("--resume-completed")
        commands.append(("matched", host, output, command))
    for host in sorted(EASTERN_HOSTS):
        output = output_root / "exact_overlap" / host
        command = [
            sys.executable,
            str(code_root / "august2026_exact_overlap_probe.py"),
            "--registry", str(registry),
            "--host", host,
            "--output-root", str(output),
            "--device", device,
            "--batch-size", str(batch_size),
            "--score-space", "raw",
        ]
        if resume_completed:
            command.append("--resume-completed")
        commands.append(("exact_overlap", host, output, command))
    return commands


def _validate_resume_identity(
    status_path: Path,
    registry: Path,
    selection_manifest: Path,
    training_metrics: Path,
    code_commit: str,
    device: str,
    batch_size: int,
) -> dict:
    status = read_json(status_path)
    expected = {
        "registry_sha256": sha256(registry),
        "selection_manifest_sha256": sha256(selection_manifest),
        "training_metrics_sha256": sha256(training_metrics),
        "code_commit": code_commit,
        "device": device,
        "batch_size": batch_size,
    }
    observed = {key: status.get(key) for key in expected}
    if observed != expected:
        raise ValueError(
            f"Acceptance resume identity changed: expected={expected}, "
            f"observed={observed}"
        )
    return status


def current_clean_commit(code_root: Path) -> str:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=code_root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=code_root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("Refusing Stage 03 acceptance from a dirty code tree")
    return commit


def run_acceptance(
    registry: Path,
    selection_manifest: Path,
    output_root: Path,
    device: str,
    batch_size: int,
    resume_completed: bool = False,
) -> dict:
    selection = read_json(selection_manifest)
    if (
        selection.get("status") != "complete"
        or selection.get("selection_type") != "loss_for_augmentation_screen"
    ):
        raise RuntimeError("Selection manifest is not approved for Stage 03")

    # This fails before creating an acceptance root unless all 35 training runs
    # and their traceable evidence pass the frozen registry gate.
    training_status = summarize_registry(registry, allow_incomplete=False)
    if (
        training_status["experiment_count"] != 35
        or training_status["complete_gate_count"] != 35
        or not training_status["all_experiments_evidence_complete"]
    ):
        raise RuntimeError(f"Stage 03 training gate is incomplete: {training_status}")
    training_metrics = Path(training_status["metrics_csv"])
    code_root = Path(__file__).resolve().parent
    code_commit = current_clean_commit(code_root)

    status_path = output_root / "acceptance_status.json"
    if output_root.exists() and any(output_root.iterdir()):
        if not resume_completed or not status_path.is_file():
            raise FileExistsError(
                f"Refusing nonempty Stage 03 acceptance root: {output_root}"
            )
        status = _validate_resume_identity(
            status_path, registry, selection_manifest, training_metrics,
            code_commit, device, batch_size,
        )
    else:
        output_root.mkdir(parents=True, exist_ok=True)
        status = {
            "status": "running",
            "stage": "augmentation-screen-acceptance",
            "registry": str(registry.resolve()),
            "registry_sha256": sha256(registry),
            "selection_manifest": str(selection_manifest.resolve()),
            "selection_manifest_sha256": sha256(selection_manifest),
            "training_metrics": str(training_metrics.resolve()),
            "training_metrics_sha256": sha256(training_metrics),
            "code_commit": code_commit,
            "device": device,
            "batch_size": batch_size,
            "completed_steps": [],
        }
        write_json_atomic(status_path, status)

    commands = probe_commands(
        registry, output_root, device, batch_size, resume_completed
    )
    for diagnostic, host, child_root, command in commands:
        step = f"{diagnostic}:{host}"
        if step in status.get("completed_steps", []):
            marker = child_root / "COMPLETED"
            if not marker.is_file():
                raise FileNotFoundError(
                    f"Recorded completed step lacks marker: {step}"
                )
            continue
        status["active_step"] = step
        write_json_atomic(status_path, status)
        subprocess.run(command, cwd=code_root, check=True)
        if not (child_root / "COMPLETED").is_file():
            raise RuntimeError(f"Artifact probe did not complete: {step}")
        status.setdefault("completed_steps", []).append(step)
        write_json_atomic(status_path, status)

    evidence_root = output_root / "analysis" / "combined_augmentation_evidence"
    if evidence_root.exists():
        required = [evidence_root / "COMPLETED", evidence_root / "summary.json"]
        if not resume_completed or not all(path.is_file() for path in required):
            raise FileExistsError(
                f"Refusing existing incomplete augmentation evidence: {evidence_root}"
            )
        evidence = read_json(evidence_root / "summary.json")
        if (
            evidence.get("status") != "complete"
            or evidence.get("selection_manifest_sha256")
            != sha256(selection_manifest)
            or evidence.get("augmentation_training_metrics_sha256")
            != sha256(training_metrics)
        ):
            raise ValueError("Existing augmentation evidence identity changed")
    else:
        evidence = synthesize_augmentation_evidence(
            selection_manifest,
            training_metrics,
            [output_root / "matched" / host for host in sorted(HOSTS)],
            evidence_root,
            exact_overlap_roots=[
                output_root / "exact_overlap" / host
                for host in sorted(EASTERN_HOSTS)
            ],
        )

    status.pop("active_step", None)
    status.update(
        {
            "status": "complete",
            "completed_step_count": len(status.get("completed_steps", [])),
            "combined_evidence": str(
                (evidence_root / "summary.json").resolve()
            ),
            "combined_evidence_sha256": sha256(evidence_root / "summary.json"),
            "combined_candidate_count": evidence["candidate_count"],
        }
    )
    write_json_atomic(status_path, status)
    (output_root / "COMPLETED").write_text("completed\n", encoding="utf-8")
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--resume-completed", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    for path in (args.registry, args.selection_manifest):
        if not path.is_file():
            parser.error(f"Missing required input: {path}")
    return args


def main() -> None:
    args = parse_args()
    status = run_acceptance(
        args.registry,
        args.selection_manifest,
        args.output_root,
        args.device,
        args.batch_size,
        resume_completed=args.resume_completed,
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
