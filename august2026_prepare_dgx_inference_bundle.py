#!/usr/bin/env python3
"""Freeze accepted models, MC evidence, and exact code for local-DGX inference."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from august2026_combine_dataset_screen_evidence import HOSTS
from august2026_summarize_experiments import read_json, sha256, write_json_atomic

CODE_FILES = ["inference_multimodal_production.py", "august2026_inference_provenance.py", "model.py"]
TOP_LEVEL_SUPPORT = [
    "august2026_inference_output_qa.py", "august2026_native_mosaic_parity.py",
    "august2026_prepare_mosaic_inputs.py", "august2026_render_fixed_quickviews.py",
    "august2026_record_visual_review.py", "august2026_release_evidence.py",
    "august2026_validate_mosaics.py", "august2026_summarize_experiments.py",
]
DGX_SUPPORT = [
    "prepare_inference_design.sh", "run_inference_queue.sh",
    "run_inference_shard.sh", "run_mosaic_host.sh", "validate_dgx_bundle.py",
]
GENERIC_SUPPORT = ["build_archive_manifests.py", "make_tile_shards.py", "resolve_production_candidate.py"]


def prepare_bundle(registry: Path, acceptance_summary: Path, mc_stability_summary: Path, output_dir: Path) -> dict:
    acceptance = read_json(acceptance_summary)
    if acceptance.get("status") != "complete" or not acceptance.get("production_ready_for_inference"):
        raise RuntimeError("Five-host models have not passed final acceptance")
    frame = pd.read_csv(registry, sep="\t", low_memory=False)
    if len(frame) != 5 or set(frame["host"]) != HOSTS:
        raise ValueError("DGX bundle requires the accepted five-host registry")
    host_acceptance_path = Path(acceptance["host_acceptance"])
    if not host_acceptance_path.is_file() or sha256(host_acceptance_path) != acceptance["host_acceptance_sha256"]:
        raise ValueError("Final host acceptance is missing or changed")
    host_acceptance = pd.read_csv(host_acceptance_path, low_memory=False)
    if set(host_acceptance["experiment"]) != set(frame["experiment"]) or not host_acceptance["acceptance_passed"].astype(bool).all():
        raise RuntimeError("Production registry differs from final acceptance")
    mc = read_json(mc_stability_summary)
    recommended = mc.get("recommended_mc_samples")
    if mc.get("status") != "complete" or mc.get("stage") != "mc-dropout-pass-stability" or mc.get("host_count") != 5 or recommended not in {2, 3, 5, 10}:
        raise RuntimeError("MC stability evidence is incomplete or invalid")
    mc_evidence = Path(mc["evidence"])
    if not mc_evidence.is_file() or sha256(mc_evidence) != mc["evidence_sha256"]:
        raise ValueError("MC stability evidence table changed")

    code_root = Path(__file__).resolve().parent
    if subprocess.run(["git", "status", "--porcelain"], cwd=code_root, check=True, capture_output=True, text=True).stdout.strip():
        raise RuntimeError("Refusing to freeze DGX bundle from a dirty tree")
    output_dir.mkdir(parents=True, exist_ok=False)
    code_dir = output_dir / "code_snapshot"; code_dir.mkdir()
    hashes = {}
    for name in CODE_FILES:
        target = code_dir / name; shutil.copy2(code_root / name, target)
        hashes[f"code_snapshot/{name}"] = sha256(target)
    sources = {
        **{name: code_root / name for name in TOP_LEVEL_SUPPORT},
        **{name: code_root / "dgx_august2026" / name for name in DGX_SUPPORT},
        **{name: code_root / "hazel_august2026" / name for name in GENERIC_SUPPORT},
    }
    for name, source in sources.items():
        target = output_dir / name; shutil.copy2(source, target); target.chmod(0o755)
        hashes[name] = sha256(target)
    copies = {
        "production_experiment_registry.tsv": registry,
        "final_model_acceptance_summary.json": acceptance_summary,
        "mc_stability_summary.json": mc_stability_summary,
        "mc_stability_by_host.csv": mc_evidence,
    }
    for name, source in copies.items():
        target = output_dir / name; shutil.copy2(source, target); hashes[name] = sha256(target)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=code_root, check=True, capture_output=True, text=True).stdout.strip()
    payload = {
        "status": "complete", "bundle_type": "accepted_august2026_dgx_hann_inference",
        "code_commit": commit, "source_registry": str(registry.resolve()),
        "source_registry_sha256": sha256(registry),
        "acceptance_summary": str(acceptance_summary.resolve()),
        "acceptance_summary_sha256": sha256(acceptance_summary),
        "mc_stability_summary": str(mc_stability_summary.resolve()),
        "mc_stability_summary_sha256": sha256(mc_stability_summary),
        "file_sha256": hashes,
        "inference_defaults": {
            "chip_size": 256, "stride": 64, "blend_window": "hann",
            "blend_edge_floor": 0.05, "base_seed": 42,
            "deterministic_mc_samples": 1, "uncertainty_mc_samples": recommended,
            "calibration": "candidate-specific configured release score space",
            "active_tiled_output_root": "/mnt/rsstu_dsfas_gsv_naip/Host_NAIP_SDM/Outputs_Host_NAIP_SDMs/production_model_inference_august2026",
        },
        "guardrail": (
            "Run two-tile smoke shards and structural/tag QA before the full queue. "
            "Use the local DGX only; this bundle does not authorize Hazel submission."
        ),
    }
    write_json_atomic(output_dir / "bundle_metadata.json", payload)
    (output_dir / "COMPLETED").write_text("completed\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--acceptance-summary", type=Path, required=True)
    parser.add_argument("--mc-stability-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.registry, args.acceptance_summary, args.mc_stability_summary):
        if not path.is_file(): raise FileNotFoundError(path)
    print(json.dumps(prepare_bundle(args.registry, args.acceptance_summary, args.mc_stability_summary, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
