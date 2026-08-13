#!/usr/bin/env python3
"""Freeze accepted production artifacts and exact code for Hazel inference."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from august2026_summarize_experiments import read_json, sha256, write_json_atomic


CODE_FILES = ["inference_multimodal_production.py", "model.py"]
SUPPORT_FILES = [
    "build_archive_manifests.py",
    "make_tile_shards.py",
    "resolve_production_candidate.py",
    "submit_inference.sbatch",
    "submit_shards.sh",
]


def prepare_bundle(
    registry: Path,
    acceptance_summary: Path,
    output_dir: Path,
) -> dict:
    acceptance = read_json(acceptance_summary)
    if acceptance.get("status") != "complete" or not acceptance.get(
        "production_ready_for_inference"
    ):
        raise RuntimeError("Production candidates have not passed final model acceptance")
    registry_frame = pd.read_csv(registry, sep="\t")
    if len(registry_frame) != 5 or set(registry_frame["host"]) != {
        "ailanthus_altissima", "juglans_nigra", "prunus_serotina",
        "vitis_labrusca", "notholithocarpus_densiflorus",
    }:
        raise ValueError("Hazel bundle requires the five-host production registry")
    host_acceptance_path = Path(acceptance["host_acceptance"])
    if (
        not host_acceptance_path.is_file()
        or sha256(host_acceptance_path) != acceptance["host_acceptance_sha256"]
    ):
        raise ValueError("Final host-acceptance table is missing or changed")
    host_acceptance = pd.read_csv(host_acceptance_path, low_memory=False)
    if set(host_acceptance["experiment"]) != set(registry_frame["experiment"]):
        raise ValueError("Hazel registry differs from the accepted production experiments")
    if not host_acceptance["acceptance_passed"].astype(bool).all():
        raise RuntimeError("One or more production hosts failed final model acceptance")
    code_root = Path(__file__).resolve().parent
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=code_root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("Refusing to freeze a Hazel bundle from a dirty tree")
    output_dir.mkdir(parents=True, exist_ok=False)
    code_dir = output_dir / "code_snapshot"
    code_dir.mkdir()
    support_root = code_root / "hazel_august2026"
    hashes = {}
    for name in CODE_FILES:
        target = code_dir / name
        shutil.copy2(code_root / name, target)
        hashes[f"code_snapshot/{name}"] = sha256(target)
    for name in SUPPORT_FILES:
        target = output_dir / name
        shutil.copy2(support_root / name, target)
        target.chmod(0o755)
        hashes[name] = sha256(target)
    registry_copy = output_dir / "production_experiment_registry.tsv"
    shutil.copy2(registry, registry_copy)
    hashes[registry_copy.name] = sha256(registry_copy)
    acceptance_copy = output_dir / "final_model_acceptance_summary.json"
    shutil.copy2(acceptance_summary, acceptance_copy)
    hashes[acceptance_copy.name] = sha256(acceptance_copy)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=code_root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    payload = {
        "status": "complete",
        "bundle_type": "accepted_august2026_hazel_hann_inference",
        "code_commit": commit,
        "source_registry": str(registry.resolve()),
        "source_registry_sha256": sha256(registry),
        "acceptance_summary": str(acceptance_summary.resolve()),
        "acceptance_summary_sha256": sha256(acceptance_summary),
        "file_sha256": hashes,
        "inference_defaults": {
            "chip_size": 256,
            "stride": 64,
            "blend_window": "hann",
            "blend_edge_floor": 0.05,
            "deterministic_mc_samples": 1,
            "uncertainty_mc_samples": 10,
            "calibration": "candidate-specific configured release score space",
        },
        "guardrail": (
            "Run smoke shards and structural/tag QA before submitting full shards. "
            "Use Slurm on Hazel; do not SSH directly to compute nodes."
        ),
    }
    write_json_atomic(output_dir / "bundle_metadata.json", payload)
    (output_dir / "COMPLETED").write_text("completed\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--acceptance-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.registry, args.acceptance_summary):
        if not path.is_file():
            raise FileNotFoundError(path)
    payload = prepare_bundle(args.registry, args.acceptance_summary, args.output_dir)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
