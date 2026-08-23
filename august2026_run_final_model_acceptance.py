#!/usr/bin/env python3
"""Run the complete post-fit acceptance suite for five August production models."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from august2026_final_model_acceptance import (
    EASTERN_HOSTS,
    HOSTS,
    synthesize_acceptance,
)
from august2026_run_augmentation_acceptance import current_clean_commit
from august2026_summarize_experiments import (
    read_json,
    sha256,
    summarize_registry,
    write_json_atomic,
)


def final_probe_commands(
    registry: Path,
    output_root: Path,
    device: str,
    batch_size: int,
    resume_completed: bool,
) -> list[tuple[str, str, str, Path, list[str]]]:
    code_root = Path(__file__).resolve().parent
    commands = []
    for score_space in ("raw", "configured"):
        for host in sorted(HOSTS):
            output = output_root / "artifacts" / score_space / "matched" / host
            command = [
                sys.executable,
                str(code_root / "august2026_batch_artifact_probe.py"),
                "--registry", str(registry),
                "--host", host,
                "--output-root", str(output),
                "--device", device,
                "--batch-size", str(batch_size),
                "--score-space", score_space,
            ]
            if resume_completed:
                command.append("--resume-completed")
            commands.append((score_space, "matched", host, output, command))
        for host in sorted(EASTERN_HOSTS):
            output = (
                output_root / "artifacts" / score_space / "exact_overlap" / host
            )
            command = [
                sys.executable,
                str(code_root / "august2026_exact_overlap_probe.py"),
                "--registry", str(registry),
                "--host", host,
                "--output-root", str(output),
                "--device", device,
                "--batch-size", str(batch_size),
                "--score-space", score_space,
            ]
            if resume_completed:
                command.append("--resume-completed")
            commands.append((score_space, "exact_overlap", host, output, command))
    return commands


def production_datasets(training_metrics: Path) -> dict[str, Path]:
    training = pd.read_csv(training_metrics, low_memory=False)
    if (
        len(training) != 5
        or set(training["host"]) != HOSTS
        or set(training["stage"]) != {"production-fit"}
        or training["experiment"].duplicated().any()
    ):
        raise ValueError("Production evidence must contain exactly five unique hosts")
    datasets = {}
    for row in training.itertuples(index=False):
        config = read_json(Path(row.config_path))
        dataset = Path(config["csv_path"])
        if not dataset.is_file() or sha256(dataset) != row.training_dataset_csv_sha256:
            raise ValueError(f"Production dataset changed for {row.host}")
        datasets[row.host] = dataset
    return datasets


def _resume_status(
    status_path: Path,
    registry: Path,
    training_metrics: Path,
    blockcv_summary: Path,
    code_commit: str,
    device: str,
    batch_size: int,
    samples_per_origin_class: int,
    allow_code_commit_upgrade: bool = False,
    code_root: Path | None = None,
) -> dict:
    status = read_json(status_path)
    expected = {
        "registry_sha256": sha256(registry),
        "training_metrics_sha256": sha256(training_metrics),
        "blockcv_summary_sha256": sha256(blockcv_summary),
        "code_commit": code_commit,
        "device": device,
        "batch_size": batch_size,
        "samples_per_origin_class": samples_per_origin_class,
    }
    observed = {key: status.get(key) for key in expected}
    if observed == expected:
        return status
    changed = {key for key in expected if observed.get(key) != expected[key]}
    if allow_code_commit_upgrade and changed == {"code_commit"}:
        if code_root is None:
            raise ValueError("Code root is required for a resume commit upgrade")
        previous_commit = str(observed["code_commit"])
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", previous_commit, code_commit],
            cwd=code_root,
        )
        if ancestor.returncode != 0:
            raise ValueError(
                "Refusing final acceptance resume from a non-ancestor code commit"
            )
        status.setdefault("code_commit_history", []).append(previous_commit)
        status.setdefault("resume_code_commit_upgrades", []).append(
            {
                "previous_commit": previous_commit,
                "new_commit": code_commit,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "reason": (
                    "Resume after a configured-score probe code-path fix; completed "
                    "probe outputs are retained and identity-checked."
                ),
            }
        )
        status["code_commit"] = code_commit
        write_json_atomic(status_path, status)
        return status
    raise ValueError(
        f"Final acceptance resume identity changed: expected={expected}, "
        f"observed={observed}"
    )


def run_command_step(
    status: dict,
    status_path: Path,
    step: str,
    marker: Path,
    command: list[str],
    code_root: Path,
) -> None:
    if step in status.get("completed_steps", []):
        if not marker.is_file():
            raise FileNotFoundError(f"Recorded completed step lacks marker: {step}")
        return
    status["active_step"] = step
    write_json_atomic(status_path, status)
    subprocess.run(command, cwd=code_root, check=True)
    if not marker.is_file():
        raise RuntimeError(f"Acceptance command did not create marker: {step}")
    status.setdefault("completed_steps", []).append(step)
    write_json_atomic(status_path, status)


def adopt_completed_support_step(
    status: dict,
    status_path: Path,
    step: str,
    marker: Path,
    summary_path: Path,
    expected_hashes: dict[str, str],
) -> bool:
    if step in status.get("completed_steps", []):
        if not marker.is_file() or not summary_path.is_file():
            raise FileNotFoundError(f"Recorded support step is incomplete: {step}")
    elif not marker.is_file() or not summary_path.is_file():
        return False
    summary = read_json(summary_path)
    if summary.get("status") != "complete" or any(
        summary.get(key) != value for key, value in expected_hashes.items()
    ):
        raise ValueError(f"Completed support step identity changed: {step}")
    if step not in status.get("completed_steps", []):
        status.setdefault("completed_steps", []).append(step)
        write_json_atomic(status_path, status)
    return True


def run_final_acceptance(
    registry: Path,
    blockcv_evidence_dir: Path,
    output_root: Path,
    device: str,
    batch_size: int,
    samples_per_origin_class: int,
    resume_completed: bool = False,
    allow_code_commit_upgrade: bool = False,
) -> dict:
    training_status = summarize_registry(registry, allow_incomplete=False)
    if (
        training_status["experiment_count"] != 5
        or training_status["complete_gate_count"] != 5
        or not training_status["all_experiments_evidence_complete"]
    ):
        raise RuntimeError(f"Production training gate is incomplete: {training_status}")
    training_metrics = Path(training_status["metrics_csv"])
    datasets = production_datasets(training_metrics)
    blockcv_summary = blockcv_evidence_dir / "summary.json"
    blockcv_marker = blockcv_evidence_dir / "COMPLETED"
    if not blockcv_summary.is_file() or not blockcv_marker.is_file():
        raise FileNotFoundError("BlockCV evidence package is incomplete")
    blockcv = read_json(blockcv_summary)
    if blockcv.get("status") != "complete" or not blockcv.get("production_ready"):
        raise RuntimeError("BlockCV evidence is not production-ready")

    code_root = Path(__file__).resolve().parent
    code_commit = current_clean_commit(code_root)
    status_path = output_root / "acceptance_status.json"
    if output_root.exists() and any(output_root.iterdir()):
        if not resume_completed or not status_path.is_file():
            raise FileExistsError(
                f"Refusing nonempty final acceptance root: {output_root}"
            )
        status = _resume_status(
            status_path, registry, training_metrics, blockcv_summary,
            code_commit, device, batch_size, samples_per_origin_class,
            allow_code_commit_upgrade=allow_code_commit_upgrade,
            code_root=code_root,
        )
    else:
        output_root.mkdir(parents=True, exist_ok=True)
        status = {
            "status": "running",
            "stage": "final-model-acceptance",
            "registry": str(registry.resolve()),
            "registry_sha256": sha256(registry),
            "training_metrics": str(training_metrics.resolve()),
            "training_metrics_sha256": sha256(training_metrics),
            "blockcv_evidence_dir": str(blockcv_evidence_dir.resolve()),
            "blockcv_summary_sha256": sha256(blockcv_summary),
            "code_commit": code_commit,
            "device": device,
            "batch_size": batch_size,
            "samples_per_origin_class": samples_per_origin_class,
            "completed_steps": [],
        }
        write_json_atomic(status_path, status)

    for score, diagnostic, host, child_root, command in final_probe_commands(
        registry, output_root, device, batch_size, resume_completed
    ):
        run_command_step(
            status,
            status_path,
            f"{score}:{diagnostic}:{host}",
            child_root / "COMPLETED",
            command,
            code_root,
        )

    support_root = output_root / "support_audits"
    overlap_root = support_root / "ailanthus_juglans_dataset_overlap"
    overlap_command = [
        sys.executable,
        str(code_root / "august2026_dataset_overlap_audit.py"),
        "--first-dataset", str(datasets["ailanthus_altissima"]),
        "--second-dataset", str(datasets["juglans_nigra"]),
        "--first-label", "ailanthus_altissima",
        "--second-label", "juglans_nigra",
        "--output-dir", str(overlap_root),
    ]
    overlap_step = "support:dataset_overlap"
    if not adopt_completed_support_step(
        status,
        status_path,
        overlap_step,
        overlap_root / "COMPLETED",
        overlap_root / "summary.json",
        {
            "first_dataset_sha256": sha256(datasets["ailanthus_altissima"]),
            "second_dataset_sha256": sha256(datasets["juglans_nigra"]),
        },
    ):
        run_command_step(
            status, status_path, overlap_step,
            overlap_root / "COMPLETED", overlap_command, code_root,
        )

    pathway_root = support_root / "ailanthus_juglans_pathway_audit"
    pathway_command = [
        sys.executable,
        str(code_root / "august2026_cross_host_pathway_audit.py"),
        "--registry", str(registry),
        "--output-dir", str(pathway_root),
        "--samples-per-origin-class", str(samples_per_origin_class),
        "--batch-size", str(batch_size),
        "--device", device,
    ]
    pathway_step = "support:pathway_audit"
    if not adopt_completed_support_step(
        status,
        status_path,
        pathway_step,
        pathway_root / "COMPLETED",
        pathway_root / "summary.json",
        {"registry_sha256": sha256(registry)},
    ):
        run_command_step(
            status, status_path, pathway_step,
            pathway_root / "COMPLETED", pathway_command, code_root,
        )

    final_root = output_root / "final_model_acceptance"
    if final_root.exists():
        summary_path = final_root / "summary.json"
        if not resume_completed or not summary_path.is_file():
            raise FileExistsError(f"Refusing existing final gate: {final_root}")
        final = read_json(summary_path)
        if (
            final.get("status") != "complete"
            or not final.get("production_ready_for_inference")
            or final.get("training_metrics_sha256") != sha256(training_metrics)
            or final.get("blockcv_summary_sha256") != sha256(blockcv_summary)
        ):
            raise RuntimeError("Existing final model acceptance is not reusable")
    else:
        final = synthesize_acceptance(
            training_metrics,
            blockcv_evidence_dir,
            [
                output_root / "artifacts" / "raw" / "matched" / host
                for host in sorted(HOSTS)
            ],
            [
                output_root / "artifacts" / "configured" / "matched" / host
                for host in sorted(HOSTS)
            ],
            [
                output_root / "artifacts" / "raw" / "exact_overlap" / host
                for host in sorted(EASTERN_HOSTS)
            ],
            [
                output_root / "artifacts" / "configured" / "exact_overlap" / host
                for host in sorted(EASTERN_HOSTS)
            ],
            overlap_root / "summary.json",
            pathway_root / "summary.json",
            final_root,
        )

    status.pop("active_step", None)
    status.update(
        {
            "status": "complete",
            "completed_step_count": len(status.get("completed_steps", [])),
            "production_ready_for_inference": final[
                "production_ready_for_inference"
            ],
            "final_model_acceptance": str((final_root / "summary.json").resolve()),
            "final_model_acceptance_sha256": sha256(final_root / "summary.json"),
        }
    )
    write_json_atomic(status_path, status)
    (output_root / "COMPLETED").write_text("completed\n", encoding="utf-8")
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--blockcv-evidence-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--samples-per-origin-class", type=int, default=512)
    parser.add_argument("--resume-completed", action="store_true")
    parser.add_argument(
        "--allow-code-commit-upgrade",
        action="store_true",
        help=(
            "On resume, permit only an ancestor-to-descendant clean code-commit "
            "upgrade while retaining hash-validated completed probes."
        ),
    )
    args = parser.parse_args()
    if args.batch_size <= 0 or args.samples_per_origin_class <= 0:
        parser.error("Batch and common-support sample sizes must be positive")
    if not args.registry.is_file():
        parser.error(f"Missing registry: {args.registry}")
    return args


def main() -> None:
    args = parse_args()
    status = run_final_acceptance(
        args.registry,
        args.blockcv_evidence_dir,
        args.output_root,
        args.device,
        args.batch_size,
        args.samples_per_origin_class,
        resume_completed=args.resume_completed,
        allow_code_commit_upgrade=args.allow_code_commit_upgrade,
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
