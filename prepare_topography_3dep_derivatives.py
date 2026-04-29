#!/usr/bin/env python
"""Prepare 3DEP topographic derivatives in a projected CRS.
"""

import os

conda_env = r"C:\Users\talake2\AppData\Local\anaconda3\envs\naip_ailanthus_env"
os.environ["GDAL_DATA"] = os.path.join(conda_env, "Library", "share", "gdal")
os.environ["PROJ_LIB"] = os.path.join(conda_env, "Library", "share", "proj")
os.environ["PATH"] += os.pathsep + os.path.join(conda_env, "Library", "bin")

import argparse
from pathlib import Path
import subprocess
import numpy as np
import rasterio


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dem-input", required=True, help="DEM raster/VRT in geographic CRS")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--target-crs", default="EPSG:5070") # Note: Keep 5070 as the slope output si incorrect with gdaldem in EPGS:4326? 
    p.add_argument("--gdaldem", default="gdaldem")
    p.add_argument("--gdalwarp", default="gdalwarp")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def run(cmd):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def maybe_skip(fp: Path, overwrite: bool):
    return fp.exists() and fp.stat().st_size > 0 and not overwrite


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dem_proj = out_dir / "dem_3dep_13_epsg5070.tif"
    slope = out_dir / "slope_degrees_3dep_13_epsg5070.tif"
    aspect = out_dir / "aspect_degrees_3dep_13_epsg5070.tif"
    northness = out_dir / "northness_3dep_13_epsg5070.tif"
    eastness = out_dir / "eastness_3dep_13_epsg5070.tif"

    # Reproject DEM to target CRS with bilinear resampling, then compute slope and aspect with gdaldem.
    if not maybe_skip(dem_proj, args.overwrite):
        run([args.gdalwarp, "-overwrite", "-t_srs", args.target_crs, "-r", "bilinear", "-multi", "-wo", "NUM_THREADS=ALL_CPUS", "-co", "TILED=YES", "-co", "COMPRESS=DEFLATE", str(args.dem_input), str(dem_proj)])

    # Compute slope using gdaldem.
    # Important: without -p, gdaldem slope outputs degrees.
    # With -p, it outputs percent slope.
    if not maybe_skip(slope, args.overwrite):
        run([
            args.gdaldem, "slope",
            str(dem_proj),
            str(slope),
            "-compute_edges",
            "-of", "GTiff",
            "-co", "TILED=YES",
            "-co", "BLOCKXSIZE=256",
            "-co", "BLOCKYSIZE=256",
            "-co", "COMPRESS=DEFLATE",
            "-co", "PREDICTOR=2",
            "-co", "BIGTIFF=IF_SAFER"
        ])

    # Compute aspect using gdaldem.
    if not maybe_skip(aspect, args.overwrite):
        run([
            args.gdaldem, "aspect",
            str(dem_proj),
            str(aspect),
            "-compute_edges",
            "-of", "GTiff",
            "-co", "TILED=YES",
            "-co", "BLOCKXSIZE=256",
            "-co", "BLOCKYSIZE=256",
            "-co", "COMPRESS=DEFLATE",
            "-co", "PREDICTOR=2",
            "-co", "BIGTIFF=IF_SAFER"
        ])

    if maybe_skip(northness, args.overwrite) and maybe_skip(eastness, args.overwrite):
        return

    # Remove corrupt/incomplete outputs before writing.
    # This prevents "MissingRequired: TIFF directory is missing required StripOffsets" errors.
    for fp in [northness, eastness]:
        if fp.exists() and not maybe_skip(fp, args.overwrite):
            fp.unlink()

    with rasterio.open(aspect) as src:
        arr = src.read(1).astype(np.float32)
        prof = src.profile.copy()
        nodata = src.nodata

    mask = np.isfinite(arr)

    if nodata is not None:
        mask &= arr != nodata

    # Aspect should be 0-360 degrees.
    # Invalid values become nodata.
    mask &= (arr >= 0.0) & (arr <= 360.0)

    rad = np.deg2rad(arr) # Convert aspect to radians for trigonometric functions.

    out_nodata = -9999.0
    n_arr = np.full(arr.shape, out_nodata, dtype=np.float32)
    e_arr = np.full(arr.shape, out_nodata, dtype=np.float32)

    n_arr[mask] = np.cos(rad[mask]).astype(np.float32) # northness = cos(aspect)
    e_arr[mask] = np.sin(rad[mask]).astype(np.float32) # eastness = sin(aspect)

    # Clean GeoTIFF profile. Do not inherit bad block sizes from the aspect raster.
    prof.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=out_nodata,
        compress="deflate",
        predictor=2,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        BIGTIFF="IF_SAFER"
    )

    with rasterio.open(northness, "w", **prof) as dst:
        dst.write(n_arr, 1)

    with rasterio.open(eastness, "w", **prof) as dst:
        dst.write(e_arr, 1)


if __name__ == "__main__":
    main()
