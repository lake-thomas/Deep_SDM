#!/usr/bin/env python
"""Prepare 3DEP topographic derivatives in a projected CRS.
"""
import argparse
from pathlib import Path
import subprocess
import numpy as np
import rasterio


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dem-input", required=True, help="DEM raster/VRT in geographic CRS")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--target-crs", default="EPSG:5070")
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

    if not maybe_skip(dem_proj, args.overwrite):
        run([args.gdalwarp, "-overwrite", "-t_srs", args.target_crs, "-r", "bilinear", "-multi", "-wo", "NUM_THREADS=ALL_CPUS", "-co", "TILED=YES", "-co", "COMPRESS=DEFLATE", str(args.dem_input), str(dem_proj)])

    if not maybe_skip(slope, args.overwrite):
        run([args.gdaldem, "slope", str(dem_proj), str(slope), "-compute_edges"])

    if not maybe_skip(aspect, args.overwrite):
        run([args.gdaldem, "aspect", str(dem_proj), str(aspect), "-compute_edges"])

    if maybe_skip(northness, args.overwrite) and maybe_skip(eastness, args.overwrite):
        return

    with rasterio.open(aspect) as src:
        arr = src.read(1).astype(np.float32)
        prof = src.profile.copy()
        nodata = src.nodata

    mask = np.isfinite(arr)
    if nodata is not None:
        mask &= arr != nodata
    rad = np.deg2rad(arr)
    n_arr = np.full(arr.shape, np.nan, dtype=np.float32)
    e_arr = np.full(arr.shape, np.nan, dtype=np.float32)
    n_arr[mask] = np.cos(rad[mask])
    e_arr[mask] = np.sin(rad[mask])

    prof.update(dtype="float32", count=1, compress="deflate", tiled=True)
    with rasterio.open(northness, "w", **prof) as dst:
        dst.write(n_arr, 1)
    with rasterio.open(eastness, "w", **prof) as dst:
        dst.write(e_arr, 1)


if __name__ == "__main__":
    main()
