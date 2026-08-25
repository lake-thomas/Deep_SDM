#!/usr/bin/env python3
"""Run, QA, compact, and verify one bounded local-DGX inference shard group."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from august2026_compact_inference_shard import compact_shard
from august2026_summarize_experiments import sha256, write_json_atomic


EAST_HOSTS = (
    "ailanthus_altissima",
    "juglans_nigra",
    "prunus_serotina",
    "vitis_labrusca",
)
TANOAK_HOST = "notholithocarpus_densiflorus"
PRODUCTS = ("deterministic", "uncertainty")


def schedule_groups(region: str, hosts: tuple[str, ...], uncertainty_mc_samples: int) -> list[list[tuple[str, str, int]]]:
    if region == "east_of_mississippi":
        if tuple(hosts) != EAST_HOSTS:
            raise ValueError(f"Eastern schedule requires hosts in frozen order: {EAST_HOSTS}")
        return [
            [(host, "deterministic", 1) for host in hosts],
            [(host, "uncertainty", uncertainty_mc_samples) for host in hosts],
        ]
    if region == "tanoak_ca_or":
        if tuple(hosts) != (TANOAK_HOST,):
            raise ValueError("Tanoak schedule requires only notholithocarpus_densiflorus")
        return [[
            (TANOAK_HOST, "deterministic", 1),
            (TANOAK_HOST, "uncertainty", uncertainty_mc_samples),
        ]]
    raise ValueError(f"Unsupported compacted schedule region: {region}")


def validate_staging_root(staging_base: Path, shard_stage: Path, output_root: Path) -> None:
    base = staging_base.resolve()
    stage = shard_stage.resolve()
    output = output_root.resolve()
    if base in {Path("/"), Path.home().resolve()} or len(base.parts) < 4:
        raise ValueError(f"Staging base is too broad: {base}")
    if not stage.is_relative_to(base) or stage == base:
        raise ValueError(f"Shard staging directory escapes its base: {stage}")
    if stage == output or output.is_relative_to(stage) or stage.is_relative_to(output):
        raise ValueError("Staging and durable output roots must be disjoint")


def read_registry(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    index = {}
    for row in rows:
        host = row["host"]
        if host in index:
            raise ValueError(f"Duplicate production registry host: {host}")
        index[host] = row
    return index


def resolve_candidate(row: dict[str, str]) -> dict:
    run_dir = Path(row["run_dir"])
    config = run_dir / "config.input.json"
    checkpoint = run_dir / "checkpoints/best.pt"
    for path in (config, checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    payload = json.loads(config.read_text(encoding="utf-8"))
    calibration = str(payload.get("postfit_calibration", "none")).lower()
    calibrator = run_dir / "platt_calibrator.json" if calibration == "platt" else None
    if calibrator is not None and not calibrator.is_file():
        raise FileNotFoundError(calibrator)
    return {
        "experiment": row["experiment"],
        "score_space": row["approved_score_space"],
        "config": config,
        "checkpoint": checkpoint,
        "calibrator": calibrator,
        "calibrator_sha256": sha256(calibrator) if calibrator is not None else None,
    }


def inference_command(
    python_bin: Path,
    bundle_root: Path,
    project_root: Path,
    candidate: dict,
    tile_list: Path,
    output_dir: Path,
    mc_samples: int,
    base_seed: int,
    batch_size: int,
) -> list[str]:
    command = [
        str(python_bin), str(bundle_root / "code_snapshot/inference_multimodal_production.py"),
        "--config", str(candidate["config"]),
        "--checkpoint", str(candidate["checkpoint"]), "--strict-checkpoint",
        "--tile-list", str(tile_list), "--output-dir", str(output_dir),
        "--worldclim-dir", str(project_root / "Env_Data/Worldclim"),
        "--ghm-raster", str(project_root / "Env_Data/Global_Human_Modification/gHM_WGS84.tif"),
        "--dem-raster", str(project_root / "Env_Data/Topography_3DEP/mosaic/dem_3dep_13_epsg5070_30m.tif"),
        "--slope-raster", str(project_root / "Env_Data/Topography_3DEP/mosaic/slope_degrees_3dep_13_epsg5070_30m.tif"),
        "--northness-raster", str(project_root / "Env_Data/Topography_3DEP/mosaic/northness_3dep_13_epsg5070_30m.tif"),
        "--eastness-raster", str(project_root / "Env_Data/Topography_3DEP/mosaic/eastness_3dep_13_epsg5070_30m.tif"),
        "--topo-normalization-stats", str(project_root / "Env_Data/Topography_3DEP/mosaic/topo_norm_stats_3dep_30m.json"),
        "--chip-size", "256", "--stride", "64", "--batch-size", str(batch_size),
        "--mc-samples", str(mc_samples), "--seed", str(base_seed),
        "--env-clip-conus", "--fast-topo-scalars",
        "--blend-window", "hann", "--blend-edge-floor", "0.05", "--skip-existing",
    ]
    if candidate["calibrator"] is not None:
        command.extend(["--platt-calibrator", str(candidate["calibrator"])])
    return command


def verify_inference_summary(path: Path, expected_count: int) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = json.loads(path.read_text(encoding="utf-8"))
    allowed = {"ok", "skipped_existing"}
    if len(rows) != expected_count or any(row.get("status") not in allowed for row in rows):
        raise RuntimeError(f"Incomplete tiled inference summary: {path}")


def run_inference_task(
    task: tuple[str, str, int],
    candidate: dict,
    tile_list: Path,
    expected_count: int,
    native_dir: Path,
    log_path: Path,
    python_bin: Path,
    bundle_root: Path,
    project_root: Path,
    base_seed: int,
    batch_size: int,
) -> dict:
    host, product, mc_samples = task
    native_dir.mkdir(parents=True, exist_ok=True)
    command = inference_command(
        python_bin, bundle_root, project_root, candidate, tile_list,
        native_dir, mc_samples, base_seed, batch_size,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write("command=" + " ".join(command) + "\n")
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
    elapsed = time.monotonic() - started
    if result.returncode != 0:
        raise RuntimeError(f"Inference failed for {host}:{product}; see {log_path}")
    verify_inference_summary(native_dir / "tiled_inference_summary.json", expected_count)
    return {
        "host": host, "product": product, "mc_samples": mc_samples,
        "batch_size": batch_size, "elapsed_seconds": elapsed,
        "native_dir": str(native_dir.resolve()), "log": str(log_path.resolve()),
        "experiment": candidate["experiment"],
        "config_sha256": sha256(candidate["config"]),
        "checkpoint_sha256": sha256(candidate["checkpoint"]),
    }


def run_output_qa(
    python_bin: Path,
    bundle_root: Path,
    tile_list: Path,
    output_root: Path,
    output_dir: Path,
    host: str,
    region: str,
    product: str,
    candidate: dict,
    base_seed: int,
    mc_samples: int,
) -> dict:
    summary = output_dir / "summary.json"
    if summary.is_file():
        payload = json.loads(summary.read_text(encoding="utf-8"))
        if payload.get("production_ready") is True:
            return payload
        raise RuntimeError(f"Pre-existing QA is not ready: {summary}")
    if output_dir.exists():
        raise FileExistsError(f"Refusing incomplete QA directory: {output_dir}")
    command = [
        str(python_bin), str(bundle_root / "august2026_inference_output_qa.py"),
        "--tile-manifest", str(tile_list), "--output-root", str(output_root),
        "--host", host, "--region", region, "--product", product,
        "--expected-score-space", candidate["score_space"],
        "--base-seed", str(base_seed), "--expected-mc-samples", str(mc_samples),
        "--output-dir", str(output_dir),
    ]
    if candidate["calibrator_sha256"] is not None:
        command.extend([
            "--expected-calibrator-sha256", candidate["calibrator_sha256"]
        ])
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not summary.is_file():
        raise RuntimeError(
            f"Output QA failed for {host}:{product}: {result.stdout}\n{result.stderr}"
        )
    payload = json.loads(summary.read_text(encoding="utf-8"))
    if payload.get("production_ready") is not True:
        raise RuntimeError(f"Output QA did not approve {host}:{product}")
    return payload


def execute(args: argparse.Namespace) -> dict:
    bundle_metadata = args.bundle_root / "bundle_metadata.json"
    metadata = json.loads(bundle_metadata.read_text(encoding="utf-8"))
    defaults = metadata["inference_defaults"]
    durable_root = Path(defaults["active_tiled_output_root"]).resolve()
    if args.output_root is not None and args.output_root.resolve() != durable_root:
        raise ValueError("Requested output root differs from the frozen bundle root")
    batch_size = int(defaults.get("batch_size", 128))
    if batch_size != 128:
        raise ValueError("Only the artifact-qualified batch size 128 is allowed")
    storage = defaults.get("storage_compaction", {})
    if storage.get("enabled") is not True or storage.get("float_compression") != "LERC_ZSTD":
        raise ValueError("Bundle does not authorize LERC_ZSTD shard compaction")
    max_z_error = float(storage.get("max_z_error", -1))
    if max_z_error != 1e-5:
        raise ValueError("Bundle max_z_error differs from the qualified 1e-5")
    base_seed = int(defaults["base_seed"])
    uncertainty_mc = int(defaults["uncertainty_mc_samples"])
    groups = schedule_groups(args.region, tuple(args.host), uncertainty_mc)
    groups_json = [
        [[host, product, mc_samples] for host, product, mc_samples in group]
        for group in groups
    ]
    expected_count = len([
        line for line in args.tile_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ])
    if expected_count < 1 or expected_count > 250:
        raise ValueError("A compacted shard group must contain between 1 and 250 tiles")
    shard_id = args.shard_id
    shard_stage = (
        args.staging_base / durable_root.name / args.region / f"shard_{shard_id}"
    )
    validate_staging_root(args.staging_base, shard_stage, durable_root)
    if shutil.disk_usage(args.staging_base).free < 100 * 1024**3:
        raise RuntimeError("Local staging has less than 100 GiB free")
    if shutil.disk_usage(durable_root.parent).free < 100 * 1024**3:
        raise RuntimeError("Durable output volume has less than 100 GiB free")
    registry = read_registry(args.registry)
    candidates = {host: resolve_candidate(registry[host]) for host in args.host}
    metadata_root = (
        durable_root / "_run_metadata/compacted_shards" / args.region / f"shard_{shard_id}"
    )
    metadata_root.mkdir(parents=True, exist_ok=True)
    design = {
        "status": "frozen", "region": args.region, "shard_id": shard_id,
        "tile_list": str(args.tile_list.resolve()),
        "tile_list_sha256": sha256(args.tile_list), "tile_count": expected_count,
        "hosts": list(args.host), "groups": groups_json, "batch_size": batch_size,
        "base_seed": base_seed, "uncertainty_mc_samples": uncertainty_mc,
        "bundle_metadata_sha256": sha256(bundle_metadata),
        "staging_base": str(args.staging_base.resolve()),
        "durable_root": str(durable_root), "max_z_error": max_z_error,
    }
    design_path = metadata_root / "design.json"
    if design_path.is_file():
        if json.loads(design_path.read_text(encoding="utf-8")) != design:
            raise RuntimeError("Pre-existing shard design differs")
    else:
        write_json_atomic(design_path, design)
    completed = metadata_root / "COMPLETED"
    if completed.is_file():
        return json.loads((metadata_root / "summary.json").read_text(encoding="utf-8"))

    runs = []
    compactions = []
    compact_qas = []
    started = time.monotonic()
    for group_index, group in enumerate(groups):
        pending = []
        for host, product, mc_samples in group:
            compact_dir = durable_root / args.region / host / product / f"shard_{shard_id}"
            compact_summary = compact_dir / "summary.json"
            compact_qa_dir = metadata_root / "compact_qa" / host / product
            if compact_summary.is_file():
                compact_payload = json.loads(compact_summary.read_text(encoding="utf-8"))
                if compact_payload.get("status") != "complete":
                    raise RuntimeError(f"Pre-existing compaction is not complete: {compact_summary}")
                qa_payload = run_output_qa(
                    args.python_bin, args.bundle_root, args.tile_list, compact_dir,
                    compact_qa_dir, host, args.region, product, candidates[host],
                    base_seed, mc_samples,
                )
                compactions.append(compact_payload)
                compact_qas.append(qa_payload)
                continue
            if compact_dir.exists():
                raise FileExistsError(f"Refusing incomplete compact shard: {compact_dir}")
            native_dir = shard_stage / host / product
            log = metadata_root / "logs" / f"g{group_index}_{host}_{product}.log"
            pending.append((
                (host, product, mc_samples), candidates[host], native_dir, log
            ))
        if pending:
            with ThreadPoolExecutor(max_workers=len(pending)) as executor:
                futures = [
                    executor.submit(
                        run_inference_task, task, candidate, args.tile_list,
                        expected_count, native_dir, log, args.python_bin,
                        args.bundle_root, args.project_root, base_seed, batch_size,
                    )
                    for task, candidate, native_dir, log in pending
                ]
                group_runs = [future.result() for future in as_completed(futures)]
            runs.extend(group_runs)
            for run in sorted(group_runs, key=lambda row: (row["host"], row["product"])):
                host, product = run["host"], run["product"]
                candidate = candidates[host]
                native_dir = Path(run["native_dir"])
                native_qa_dir = metadata_root / "native_qa" / host / product
                native_qa = run_output_qa(
                    args.python_bin, args.bundle_root, args.tile_list, native_dir,
                    native_qa_dir, host, args.region, product, candidate,
                    base_seed, int(run["mc_samples"]),
                )
                if native_qa.get("production_ready") is not True:
                    raise RuntimeError(f"Native QA failed for {host}:{product}")
                compact_dir = durable_root / args.region / host / product / f"shard_{shard_id}"
                compact_payload = compact_shard(
                    args.tile_list, native_dir, compact_dir,
                    max_z_error=max_z_error, workers=args.compaction_workers,
                )
                compact_qa_dir = metadata_root / "compact_qa" / host / product
                compact_qa = run_output_qa(
                    args.python_bin, args.bundle_root, args.tile_list, compact_dir,
                    compact_qa_dir, host, args.region, product, candidate,
                    base_seed, int(run["mc_samples"]),
                )
                compactions.append(compact_payload)
                compact_qas.append(compact_qa)

    if len(compactions) != sum(len(group) for group in groups):
        raise RuntimeError("Not every scheduled host/product produced compaction evidence")
    if not all(payload.get("production_ready") is True for payload in compact_qas):
        raise RuntimeError("Not every compact shard passed independent output QA")
    detail = metadata_root / "inference_runs.json"
    write_json_atomic(detail, runs)
    compaction_detail = metadata_root / "compactions.json"
    write_json_atomic(compaction_detail, compactions)
    summary = {
        "status": "complete", "stage": "local-dgx-compacted-inference-shard",
        "region": args.region, "shard_id": shard_id,
        "tile_count": expected_count, "scheduled_output_count": sum(len(g) for g in groups),
        "inference_run_count": len(runs), "compaction_count": len(compactions),
        "compact_qa_count": len(compact_qas), "all_compact_qa_passed": True,
        "source_bytes": sum(int(row["source_bytes"]) for row in compactions),
        "compact_bytes": sum(int(row["compact_bytes"]) for row in compactions),
        "elapsed_seconds": time.monotonic() - started,
        "design": str(design_path.resolve()), "design_sha256": sha256(design_path),
        "inference_runs": str(detail.resolve()), "inference_runs_sha256": sha256(detail),
        "compactions": str(compaction_detail.resolve()),
        "compactions_sha256": sha256(compaction_detail),
        "staging_cleanup_requested": bool(args.cleanup_staging),
        "guardrail": "Only this explicit shard was processed; no queue continuation occurs here.",
    }
    write_json_atomic(metadata_root / "summary.json", summary)
    if args.cleanup_staging and shard_stage.exists():
        validate_staging_root(args.staging_base, shard_stage, durable_root)
        shutil.rmtree(shard_stage)
        summary["staging_cleaned"] = True
        write_json_atomic(metadata_root / "summary.json", summary)
    else:
        summary["staging_cleaned"] = False
        write_json_atomic(metadata_root / "summary.json", summary)
    completed.write_text("complete\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", choices=("east_of_mississippi", "tanoak_ca_or"), required=True)
    parser.add_argument("--shard-id", required=True)
    parser.add_argument("--tile-list", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--host", action="append", required=True)
    parser.add_argument("--staging-base", type=Path, default=Path("/home/ncsucnr/host_sdm_august2026_staging"))
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--python-bin", type=Path, default=Path("/home/ncsucnr/miniforge3/envs/host-naip-sdm/bin/python"))
    parser.add_argument("--compaction-workers", type=int, default=4)
    parser.add_argument("--cleanup-staging", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for required in (args.tile_list, args.registry, args.bundle_root / "COMPLETED"):
        if not required.is_file():
            raise FileNotFoundError(required)
    if not args.shard_id.isdigit() or len(args.shard_id) != 5:
        raise ValueError("shard-id must be five digits")
    args.staging_base.mkdir(parents=True, exist_ok=True)
    payload = execute(args)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
