#!/usr/bin/env python3
"""Gated source-locked materialization launcher for the three control hosts."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


PYTHON = Path("/home/ncsucnr/miniforge3/envs/host-naip-sdm/bin/python")
CODE_ROOT = Path(__file__).resolve().parent
CANDIDATE_ROOT = Path(
    "/mnt/rsstu_dsfas_gsv_naip/Host_NAIP_SDM/Host_Datasets_USBg_August2026/"
    "candidate_manifests_20260813"
)
OUTPUT_ROOT = Path(
    "/mnt/rsstu_dsfas_gsv_naip/Host_NAIP_SDM/Host_Datasets_USBg_August2026/"
    "materialized_source_locked_20260813"
)
TILE_INDEX = Path(
    "/mnt/rsstu_dsfas_gsv_naip/Host_NAIP_SDM/NAIP_Imagery_Tile_Indices/"
    "NAIP_US_Local_Archive_Tile_Index_ByTileKey.gpkg"
)
WORLDCLIM_STATS = Path(
    "/mnt/rsstu_dsfas_gsv_naip/Host_NAIP_SDM/Outputs_Host_NAIP_SDMs/"
    "production_model_runs_june222026/pre_fit_qa_june222026/"
    "worldclim_stats_inferred_from_usbg_csv_june222026.json"
)
TOPO_ROOT = Path(
    "/mnt/rsstu_dsfas_gsv_naip/Host_NAIP_SDM/Env_Data/Topography_3DEP/mosaic"
)
TOPO_STATS = TOPO_ROOT / "topo_norm_stats_3dep_30m.json"
TOPO_RASTERS = {
    "--dem-raster": TOPO_ROOT / "dem_3dep_13_epsg5070_30m.tif",
    "--slope-raster": TOPO_ROOT / "slope_degrees_3dep_13_epsg5070_30m.tif",
    "--northness-raster": TOPO_ROOT / "northness_3dep_13_epsg5070_30m.tif",
    "--eastness-raster": TOPO_ROOT / "eastness_3dep_13_epsg5070_30m.tif",
}
CONTROL_HOSTS = ["ailanthus_altissima", "juglans_nigra", "prunus_serotina"]


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def validate_candidate(host: str) -> Path:
    package = CANDIDATE_ROOT / host
    structural = package / "qa" / "status.json"
    environmental = package / "qa" / "environment_status.json"
    required = [
        package / "manifest_metadata.json",
        package / "extraction_candidates.csv",
        structural,
        environmental,
        TILE_INDEX,
        WORLDCLIM_STATS,
        TOPO_STATS,
        *TOPO_RASTERS.values(),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Control materialization gate is incomplete: {missing}")
    if not read_json(structural).get("production_ready_for_extraction_review"):
        raise RuntimeError(f"Structural candidate gate failed: {structural}")
    if not read_json(environmental).get("production_ready_environment"):
        raise RuntimeError(f"Environmental candidate gate failed: {environmental}")
    metadata = read_json(package / "manifest_metadata.json")
    if metadata.get("targets") != [20000]:
        raise RuntimeError(f"Unexpected control-host target design: {metadata.get('targets')}")
    return package


def build_command(host: str, workers: int, resume: bool) -> list[str]:
    package = validate_candidate(host)
    output = OUTPUT_ROOT / host
    command = [
        str(PYTHON),
        "-m",
        "host_dataset_builder.materialize_candidates",
        "--candidate-package",
        str(package),
        "--output-root",
        str(output),
        "--tileindex",
        str(TILE_INDEX),
        "--worldclim-stats-json",
        str(WORLDCLIM_STATS),
        "--topo-normalization-stats",
        str(TOPO_STATS),
    ]
    for option, path in TOPO_RASTERS.items():
        command.extend([option, str(path)])
    command.extend(
        [
            "--chip-size",
            "256",
            "--topo-chip-size",
            "64",
            "--naip-min-valid-frac",
            "1.0",
            "--topo-min-valid-frac",
            "0.50",
            "--workers",
            str(workers),
            "--seed",
            "42",
            "--train-frac",
            "0.70",
            "--val-frac",
            "0.15",
            "--test-frac",
            "0.15",
            "--block-size-m",
            "200000",
            "--n-folds",
            "5",
        ]
    )
    if resume:
        command.append("--resume")
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=CONTROL_HOSTS, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--print-command", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    return args


def main() -> None:
    args = parse_args()
    output = OUTPUT_ROOT / args.host
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(
            f"Refusing to reuse nonempty output without --resume: {output}"
        )
    command = build_command(args.host, args.workers, args.resume)
    if args.print_command:
        print(" ".join(command))
        return
    subprocess.run(command, cwd=CODE_ROOT, check=True)


if __name__ == "__main__":
    main()
