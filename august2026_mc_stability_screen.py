#!/usr/bin/env python3
"""Freeze and optionally run the five-host configured-score MC stability screen."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import pandas as pd

from august2026_combine_dataset_screen_evidence import HOSTS
from august2026_experiments import (
    assert_frozen_runtime_identity,
    read_json,
    sha256,
    write_json_atomic,
    write_tsv_atomic,
)
from august2026_mc_stability_evidence import PASS_LEVELS, REPLICATE_SEEDS


DATE_TAG = "20260823"
EASTERN_HOSTS = HOSTS - {"notholithocarpus_densiflorus"}
EASTERN_TILES = [
    Path(
        "/mnt/rsstu_dsfas_gsv_naip/Host_NAIP_SDM/NAIP_Archive/states/NY/2021/raw/"
        "m_4107701_nw_18_060_20211019.tif"
    ),
    Path(
        "/mnt/rsstu_dsfas_gsv_naip/Host_NAIP_SDM/NAIP_Archive/states/PA/2022/raw/"
        "m_4107701_nw_18_060_20221110.tif"
    ),
]
TANOAK_TILES = [
    Path(
        "/mnt/rsstu_dsfas_gsv_naip/Host_NAIP_SDM/NAIP_Archive/states/CA/2022/raw/"
        "m_4112407_nw_10_030_20220622.tif"
    ),
    Path(
        "/mnt/rsstu_dsfas_gsv_naip/Host_NAIP_SDM/NAIP_Archive/states/OR/2020/raw/"
        "m_4212462_se_10_060_20200618.tif"
    ),
]
PROJECT_ROOT = Path(
    "/mnt/rsstu_dsfas_gsv_naip/Host_NAIP_SDM"
)
OUTPUT_ROOT = (
    PROJECT_ROOT / "Outputs_Host_NAIP_SDMs" /
    "experimental_model_runs_august2026" / "mc_stability_screen_20260823"
)
DESIGN_ROOT = (
    PROJECT_ROOT / "Host_Datasets_USBg_August2026" /
    "experiment_designs_20260813" / "mc_stability_screen_20260823"
)
PYTHON = Path("/home/ncsucnr/miniforge3/envs/host-naip-sdm/bin/python")


def experiment_name(host: str, mc_samples: int, seed: int) -> str:
    return f"mc_stability_{host}_mc{mc_samples:02d}_seed{seed}_{DATE_TAG}"


def host_tiles(host: str) -> list[Path]:
    return TANOAK_TILES if host == "notholithocarpus_densiflorus" else EASTERN_TILES


def _candidate(registry: pd.DataFrame, host: str) -> dict:
    matched = registry[registry["host"] == host]
    if len(matched) != 1:
        raise ValueError(f"Expected one accepted production row for {host}")
    row = matched.iloc[0].to_dict()
    run_dir = Path(row["run_dir"])
    checkpoint = run_dir / "checkpoints" / "best.pt"
    config = run_dir / "config.input.json"
    for path in (checkpoint, config):
        if not path.is_file():
            raise FileNotFoundError(path)
    config_payload = read_json(config)
    calibration = str(config_payload.get("postfit_calibration", "none")).lower()
    calibrator = run_dir / "platt_calibrator.json" if calibration == "platt" else None
    if calibrator is not None and not calibrator.is_file():
        raise FileNotFoundError(calibrator)
    expected_space = (
        "platt_calibrated_probability" if calibrator else "raw_sigmoid_probability"
    )
    if row["approved_score_space"] != expected_space:
        raise ValueError(f"Registry/configured score space differs for {host}")
    return {
        "experiment": row["experiment"],
        "checkpoint": checkpoint,
        "checkpoint_sha256": sha256(checkpoint),
        "config": config,
        "config_sha256": sha256(config),
        "calibrator": calibrator,
        "calibrator_sha256": sha256(calibrator) if calibrator else None,
        "score_space": expected_space,
    }


def write_design(registry_path: Path, acceptance_path: Path) -> tuple[Path, list[dict]]:
    registry_out = DESIGN_ROOT / "experiment_registry.tsv"
    if registry_out.exists():
        raise FileExistsError(f"Refusing to replace MC stability design: {registry_out}")
    acceptance = read_json(acceptance_path)
    if (
        acceptance.get("status") != "complete"
        or acceptance.get("production_ready_for_inference") is not True
    ):
        raise RuntimeError("MC stability screen requires final five-host acceptance")
    registry = pd.read_csv(registry_path, sep="\t", low_memory=False)
    if len(registry) != 5 or set(registry["host"]) != HOSTS:
        raise ValueError("MC stability screen requires the accepted five-host registry")
    host_acceptance_path = Path(acceptance["host_acceptance"])
    if (
        not host_acceptance_path.is_file()
        or sha256(host_acceptance_path) != acceptance["host_acceptance_sha256"]
    ):
        raise ValueError("Final host acceptance changed")
    accepted = pd.read_csv(host_acceptance_path, low_memory=False)
    if (
        set(accepted["experiment"]) != set(registry["experiment"])
        or not accepted["acceptance_passed"].astype(bool).all()
    ):
        raise ValueError("Production registry differs from final acceptance")
    for path in [*EASTERN_TILES, *TANOAK_TILES]:
        if not path.is_file():
            raise FileNotFoundError(path)
    code_root = Path(__file__).resolve().parent
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=code_root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if subprocess.run(
        ["git", "status", "--porcelain"], cwd=code_root, check=True,
        capture_output=True, text=True,
    ).stdout.strip():
        raise RuntimeError("Refusing to freeze MC screen from a dirty tree")

    rows = []
    tile_list_root = DESIGN_ROOT / "tile_lists"
    for host in sorted(HOSTS):
        candidate = _candidate(registry, host)
        tiles = host_tiles(host)
        tile_list = tile_list_root / f"{host}.txt"
        content = "\n".join(str(path.resolve()) for path in tiles) + "\n"
        if tile_list.exists() and tile_list.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Refusing changed MC tile list: {tile_list}")
        tile_list.parent.mkdir(parents=True, exist_ok=True)
        if not tile_list.exists():
            tile_list.write_text(content, encoding="utf-8")
        for mc_samples in PASS_LEVELS:
            for seed in REPLICATE_SEEDS:
                name = experiment_name(host, mc_samples, seed)
                rows.append({
                    "stage": "mc-stability-screen",
                    "host": host,
                    "production_experiment": candidate["experiment"],
                    "mc_samples": mc_samples,
                    "replicate_seed": seed,
                    "tile_count": len(tiles),
                    "tile_list": str(tile_list),
                    "tile_list_sha256": sha256(tile_list),
                    "checkpoint": str(candidate["checkpoint"]),
                    "checkpoint_sha256": candidate["checkpoint_sha256"],
                    "config": str(candidate["config"]),
                    "config_sha256": candidate["config_sha256"],
                    "calibrator": str(candidate["calibrator"] or ""),
                    "calibrator_sha256": candidate["calibrator_sha256"] or "",
                    "score_space": candidate["score_space"],
                    "experiment": name,
                    "run_dir": str(OUTPUT_ROOT / name),
                    "status": "pending",
                    "note": "",
                })
    write_tsv_atomic(registry_out, rows, list(rows[0]))
    write_json_atomic(DESIGN_ROOT / "design_metadata.json", {
        "status": "complete",
        "date_tag": DATE_TAG,
        "stage": "mc-stability-screen",
        "code_root": str(code_root),
        "code_commit": commit,
        "code_dirty": False,
        "production_registry": str(registry_path.resolve()),
        "production_registry_sha256": sha256(registry_path),
        "final_acceptance": str(acceptance_path.resolve()),
        "final_acceptance_sha256": sha256(acceptance_path),
        "experiment_count": len(rows),
        "registry": str(registry_out),
        "domains": {
            "eastern": [str(path) for path in EASTERN_TILES],
            "tanoak": [str(path) for path in TANOAK_TILES],
        },
        "guardrail": (
            "Configured release score space, Hann edge floor 0.05, two acquisition-"
            "sensitive tiles per host, four MC levels, and two base seeds are frozen."
        ),
    })
    return registry_out, rows


def _command(row: dict) -> list[str]:
    command = [
        str(PYTHON),
        str(Path(__file__).resolve().parent / "inference_multimodal_production.py"),
        "--config", row["config"],
        "--checkpoint", row["checkpoint"],
        "--strict-checkpoint",
        "--tile-list", row["tile_list"],
        "--output-dir", row["run_dir"],
        "--worldclim-dir", str(PROJECT_ROOT / "Env_Data/Worldclim"),
        "--ghm-raster", str(
            PROJECT_ROOT / "Env_Data/Global_Human_Modification/gHM_WGS84.tif"
        ),
        "--dem-raster", str(
            PROJECT_ROOT / "Env_Data/Topography_3DEP/mosaic/dem_3dep_13_epsg5070_30m.tif"
        ),
        "--slope-raster", str(
            PROJECT_ROOT / "Env_Data/Topography_3DEP/mosaic/slope_degrees_3dep_13_epsg5070_30m.tif"
        ),
        "--northness-raster", str(
            PROJECT_ROOT / "Env_Data/Topography_3DEP/mosaic/northness_3dep_13_epsg5070_30m.tif"
        ),
        "--eastness-raster", str(
            PROJECT_ROOT / "Env_Data/Topography_3DEP/mosaic/eastness_3dep_13_epsg5070_30m.tif"
        ),
        "--topo-normalization-stats", str(
            PROJECT_ROOT / "Env_Data/Topography_3DEP/mosaic/topo_norm_stats_3dep_30m.json"
        ),
        "--chip-size", "256", "--stride", "64", "--batch-size", "128",
        "--mc-samples", str(row["mc_samples"]), "--seed", str(row["replicate_seed"]),
        "--env-clip-conus", "--fast-topo-scalars",
        "--blend-window", "hann", "--blend-edge-floor", "0.05",
        "--skip-existing",
    ]
    if row["calibrator"]:
        command.extend(["--platt-calibrator", row["calibrator"]])
    return command


def _write_registry(path: Path, rows: list[dict]) -> None:
    write_tsv_atomic(path, rows, list(rows[0]))


def run_design(registry_path: Path, rows: list[dict], resume_completed: bool) -> None:
    logs = registry_path.parent / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows):
        output = Path(row["run_dir"])
        summary_path = output / "tiled_inference_summary.json"
        if row["status"] == "completed":
            if not resume_completed or not summary_path.is_file():
                raise RuntimeError(f"Recorded completed MC run is not reusable: {output}")
            continue
        if output.exists() and any(output.iterdir()) and not resume_completed:
            raise FileExistsError(f"Refusing partial MC output: {output}")
        # This matrix contains 40 sequential inference jobs and can run long
        # enough for the active worktree to change between rows. Keep every
        # result on the exact code identity frozen in design metadata.
        assert_frozen_runtime_identity(registry_path)
        row["status"] = "running"
        _write_registry(registry_path, rows)
        log_path = logs / f"{row['experiment']}.log"
        started = time.time()
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                _command(row), cwd=Path(__file__).resolve().parent,
                stdout=log, stderr=subprocess.STDOUT,
            )
        if result.returncode != 0 or not summary_path.is_file():
            row["status"] = "failed"
            row["note"] = f"returncode={result.returncode}; log={log_path}"
            _write_registry(registry_path, rows)
            raise RuntimeError(f"MC stability run failed: {row['experiment']}")
        summaries = read_json(summary_path)
        if len(summaries) != int(row["tile_count"]) or any(
            item.get("status") != "ok" for item in summaries
        ):
            row["status"] = "failed"
            row["note"] = f"invalid_tile_summaries; log={log_path}"
            _write_registry(registry_path, rows)
            raise RuntimeError(f"MC stability tile evidence failed: {row['experiment']}")
        row["status"] = "completed"
        row["note"] = f"elapsed_seconds={time.time() - started:.3f}; log={log_path}"
        _write_registry(registry_path, rows)


def materialize_analysis_manifest(registry_path: Path) -> Path:
    registry = pd.read_csv(registry_path, sep="\t", low_memory=False)
    if len(registry) != len(HOSTS) * len(PASS_LEVELS) * len(REPLICATE_SEEDS):
        raise ValueError("MC stability execution matrix is incomplete")
    if set(registry["status"]) != {"completed"}:
        raise RuntimeError("MC stability runs are not all complete")
    records = []
    for row in registry.to_dict("records"):
        summaries = read_json(Path(row["run_dir"]) / "tiled_inference_summary.json")
        for item in summaries:
            tile = Path(item["tile"])
            records.append({
                "host": row["host"],
                "tile_id": tile.stem,
                "tile_path": str(tile.resolve()),
                "mc_samples": int(row["mc_samples"]),
                "replicate_seed": int(row["replicate_seed"]),
                "prediction": item["prediction_fp"],
                "uncertainty": item["uncertainty_fp"],
                "elapsed_seconds": item["elapsed_seconds"],
            })
    output = registry_path.parent / "analysis" / "mc_stability_manifest.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f"{output.name}.partial")
    pd.DataFrame(records).to_csv(partial, index=False)
    os.replace(partial, output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-registry", type=Path, required=True)
    parser.add_argument("--final-acceptance", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume-completed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    existing_registry = DESIGN_ROOT / "experiment_registry.tsv"
    if args.resume_completed and existing_registry.is_file():
        metadata = read_json(DESIGN_ROOT / "design_metadata.json")
        if (
            metadata.get("production_registry_sha256")
            != sha256(args.production_registry)
            or metadata.get("final_acceptance_sha256")
            != sha256(args.final_acceptance)
        ):
            raise ValueError("MC stability resume inputs changed")
        current_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        if metadata.get("code_commit") != current_commit:
            raise ValueError("MC stability resume code commit changed")
        registry = existing_registry
        rows = pd.read_csv(
            registry, sep="\t", keep_default_na=False
        ).to_dict("records")
        print(f"Resuming gated MC stability design: {registry}")
    else:
        registry, rows = write_design(args.production_registry, args.final_acceptance)
        print(f"Wrote gated MC stability design: {registry}")
    if args.run:
        run_design(registry, rows, args.resume_completed)
        print(f"Wrote MC analysis manifest: {materialize_analysis_manifest(registry)}")


if __name__ == "__main__":
    main()
