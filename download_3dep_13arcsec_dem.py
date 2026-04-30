#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Download USGS 3DEP 1/3 arc-second DEM tiles and build a mosaic.

Example:
    python download_3dep_dem.py ^
        --bounds -125 32 -114 43 ^
        --out-dir "Y:/Host_NAIP_SDM/Env_Data/Topography_3DEP" ^
        --mosaic-name "USGS_3DEP_13_CA_OR_DEM"

Notes:
    - Downloads the .tif DEM from each 1-degree tile folder.
    - Optionally downloads .xml metadata.
    - Builds a GDAL VRT mosaic.
    - Optionally translates the VRT into a compressed tiled BigTIFF.

3DEP Source: https://prd-tnm.s3.amazonaws.com/index.html?prefix=StagedProducts/Elevation/13/
"""

import os

conda_env = r"C:\Users\talake2\AppData\Local\anaconda3\envs\naip_ailanthus_env"
os.environ["GDAL_DATA"] = os.path.join(conda_env, "Library", "share", "gdal")
os.environ["PROJ_LIB"] = os.path.join(conda_env, "Library", "share", "proj")
os.environ["PATH"] += os.pathsep + os.path.join(conda_env, "Library", "bin")

import argparse
import math
import subprocess
from pathlib import Path

import requests
from tqdm import tqdm


BASE_URL = (
    "https://prd-tnm.s3.amazonaws.com/"
    "StagedProducts/Elevation/13/TIFF/current"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download USGS 3DEP 1/3 arc-second DEM tiles and build a mosaic."
    )

    parser.add_argument(
        "--bounds",
        type=float,
        nargs=4,
        required=True,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help="Bounding box in EPSG:4326 longitude/latitude."
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory for downloaded DEM tiles and mosaics."
    )

    parser.add_argument(
        "--mosaic-name",
        type=str,
        default="USGS_3DEP_13_DEM_Mosaic",
        help="Base name for output mosaic files."
    )

    parser.add_argument(
        "--download-xml",
        action="store_true",
        help="Also download .xml metadata files."
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download existing files."
    )

    parser.add_argument(
        "--make-tif",
        action="store_true",
        help="Also convert the VRT to a compressed BigTIFF GeoTIFF."
    )

    parser.add_argument(
        "--gdalbuildvrt",
        type=str,
        default="gdalbuildvrt",
        help="Path to gdalbuildvrt executable if not on PATH."
    )

    parser.add_argument(
        "--gdal_translate",
        type=str,
        default="gdal_translate",
        help="Path to gdal_translate executable if not on PATH."
    )

    return parser.parse_args()


def format_tile_name(lat_deg: int, lon_deg: int) -> str:
    """
    Convert integer lower-left tile coordinates to USGS tile name.

    Examples:
        lat=37, lon=-122 -> n37w122
        lat=7, lon=158   -> n07e158
        lat=-14, lon=-170 -> s14w170
    """
    ns = "n" if lat_deg >= 0 else "s"
    ew = "e" if lon_deg >= 0 else "w"

    lat_abs = abs(lat_deg)
    lon_abs = abs(lon_deg)

    return f"{ns}{lat_abs:02d}{ew}{lon_abs:03d}"


def tiles_from_bounds(min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> list[str]:
    """
    Generate 1-degree USGS tile names intersecting a lon/lat bounding box.

    The tile n37w122 covers approximately:
        lon -122 to -121
        lat  37 to 38
    """
    lon_start = math.floor(min_lon)
    lon_end = math.ceil(max_lon) - 1

    lat_start = math.floor(min_lat)
    lat_end = math.ceil(max_lat) - 1

    tiles = []
    for lat in range(lat_start, lat_end + 1):
        for lon in range(lon_start, lon_end + 1):
            tiles.append(format_tile_name(lat, lon))

    return tiles


def tile_url(tile: str, suffix: str = ".tif") -> str:
    return f"{BASE_URL}/{tile}/USGS_13_{tile}{suffix}"


def download_file(url: str, out_fp: Path, overwrite: bool = False) -> bool:
    """
    Download a URL to disk.

    Returns True if downloaded or already exists.
    Returns False if the URL does not exist.
    """
    out_fp.parent.mkdir(parents=True, exist_ok=True)

    if out_fp.exists() and out_fp.stat().st_size > 0 and not overwrite:
        print(f"Exists, skipping: {out_fp.name}")
        return True

    tmp_fp = out_fp.with_suffix(out_fp.suffix + ".part")

    try:
        with requests.get(url, stream=True, timeout=60) as r:
            if r.status_code == 404:
                print(f"Missing on server: {url}")
                return False

            r.raise_for_status()

            total = int(r.headers.get("content-length", 0))

            with open(tmp_fp, "wb") as f, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=out_fp.name,
                leave=False
            ) as pbar:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        tmp_fp.replace(out_fp)
        print(f"Downloaded: {out_fp.name}")
        return True

    except Exception as e:
        if tmp_fp.exists():
            tmp_fp.unlink()
        print(f"Failed: {url}")
        print(f"  Reason: {e}")
        return False


def build_vrt(tile_files: list[Path], vrt_fp: Path, gdalbuildvrt: str = "gdalbuildvrt"):
    """
    Build a GDAL VRT mosaic from downloaded tiles.
    """
    if not tile_files:
        raise RuntimeError("No tile files available to build VRT.")

    tile_list_fp = vrt_fp.with_suffix(".txt")
    with open(tile_list_fp, "w") as f:
        for fp in tile_files:
            f.write(str(fp.resolve()) + "\n")

    cmd = [
        gdalbuildvrt,
        "-overwrite",
        "-input_file_list",
        str(tile_list_fp),
        str(vrt_fp)
    ]

    print("\nBuilding VRT:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    print(f"Saved VRT: {vrt_fp}")


def translate_vrt_to_tif(vrt_fp: Path, tif_fp: Path, gdal_translate: str = "gdal_translate"):
    """
    Convert VRT to compressed tiled BigTIFF.

    This can be large. The VRT is often enough for local processing.
    """
    cmd = [
        gdal_translate,
        str(vrt_fp),
        str(tif_fp),
        "-of", "GTiff",
        "-co", "TILED=YES",
        "-co", "COMPRESS=DEFLATE",
        "-co", "PREDICTOR=2",
        "-co", "BIGTIFF=YES",
        "-co", "NUM_THREADS=ALL_CPUS"
    ]

    print("\nTranslating VRT to compressed BigTIFF:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    print(f"Saved GeoTIFF mosaic: {tif_fp}")


def main():
    args = parse_args()

    min_lon, min_lat, max_lon, max_lat = args.bounds

    out_dir = Path(args.out_dir)
    tiles_dir = out_dir / "tiles"
    mosaic_dir = out_dir / "mosaic"

    tiles_dir.mkdir(parents=True, exist_ok=True)
    mosaic_dir.mkdir(parents=True, exist_ok=True)

    tiles = tiles_from_bounds(min_lon, min_lat, max_lon, max_lat)

    print(f"Requested bounds: {args.bounds}")
    print(f"Candidate 1-degree tiles: {len(tiles):,}")

    downloaded_tifs = []

    for tile in tiles:
        tif_url = tile_url(tile, ".tif")
        tif_fp = tiles_dir / f"USGS_13_{tile}.tif"

        ok = download_file(tif_url, tif_fp, overwrite=args.overwrite)
        if ok and tif_fp.exists() and tif_fp.stat().st_size > 0:
            downloaded_tifs.append(tif_fp)

        if args.download_xml:
            xml_url = tile_url(tile, ".xml")
            xml_fp = tiles_dir / f"USGS_13_{tile}.xml"
            download_file(xml_url, xml_fp, overwrite=args.overwrite)

    print(f"\nDownloaded/found DEM tiles: {len(downloaded_tifs):,}")

    vrt_fp = mosaic_dir / f"{args.mosaic_name}.vrt"
    build_vrt(downloaded_tifs, vrt_fp, gdalbuildvrt=args.gdalbuildvrt)

    if args.make_tif:
        tif_fp = mosaic_dir / f"{args.mosaic_name}.tif"
        translate_vrt_to_tif(vrt_fp, tif_fp, gdal_translate=args.gdal_translate)

    print("\nDone.")


if __name__ == "__main__":
    main()
