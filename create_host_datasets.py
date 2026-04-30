#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Occurrence and Background Point Generation for Species Distribution Modeling with NAIP Images
# Uniform Random Train/ Val/ Test Split and Block Cross-validation Split
# Thomas Lake, April 2026

"""
Generalized species SDM dataset builder
---------------------------------------
Builds:
  1) Presence/background points (with optional doughnut sampling)
  2) Uniform train/val/test dataset
  3) Spatial block cross-validation datasets
  4) NAIP image chips + climate covariates

Designed to generalize to additional host species CSVs.

Note: Promit Saha (ppsaha2@ncsu) created the occurrence and thinning pipeline below:
Expected species occurrence CSV columns:
    species
    decimalLatitude
    decimalLongitude
    dateIdentified   (optional)
    Source           (optional)

Example usage:
    python Host_NAIP_SDM_Dataset_Prep_April2026.py ^
        --occurrence-dir "Y:/Promit_Host_Occurrences/Fully_thinned_data" ^
        --background-buffer-km 50 ^
        --output-root "Y:/Host_NAIP_SDM" ^
        --tileindex "Y:/Host_NAIP_SDM/NAIP_Imagery_Tile_Indices/NAIP_US_State_Tile_Indices_URL_Paths_jan26.shp" ^
        --naip-folder "Y:/Host_NAIP_SDM/NAIP_Archive" ^
        --worldclim-folder "Y:/Host_NAIP_SDM/Env_Data/Worldclim" ^
        --ghm-raster "Y:/Host_NAIP_SDM/Env_Data/Global_Human_modification/gHM_WGS84.tif"

Or for a single species:
    python Host_NAIP_SDM_Dataset_Prep_April2026.py ^
        --occurrence-file "Y:/Promit_Host_Occurrences/Fully_thinned_data/juglans_nigra_thinned.csv" ^
        --background-buffer-km 50 ^
        --output-root "Y:/Host_NAIP_SDM" ^
        --tileindex "Y:/Host_NAIP_SDM/NAIP_Imagery_Tile_Indices/NAIP_US_State_Tile_Indices_URL_Paths_jan26.shp" ^
        --naip-folder "Y:/Host_NAIP_SDM/NAIP_Archive" ^
        --worldclim-folder "Y:/Host_NAIP_SDM/Env_Data/Worldclim" ^
        --ghm-raster "Y:/Host_NAIP_SDM/Env_Data/Global_Human_modification/gHM_WGS84.tif"
"""

import os

conda_env = r"C:\Users\talake2\AppData\Local\anaconda3\envs\naip_ailanthus_env"
os.environ["GDAL_DATA"] = os.path.join(conda_env, "Library", "share", "gdal")
os.environ["PROJ_LIB"] = os.path.join(conda_env, "Library", "share", "proj")
os.environ["PATH"] += os.pathsep + os.path.join(conda_env, "Library", "bin")

import argparse
import math
import random
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.transform import from_origin
from rasterio.windows import Window
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.warp import reproject
from shapely.geometry import Point, box
from tqdm import tqdm


# ---------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------

WC_VARS = [f"wc2.1_30s_bio_{i}" for i in range(1, 20)] # 19 Bioclimatic variables from WorldClim 2.1 at 30 arc-second resolution
ALL_ENV_VARS = WC_VARS + ["ghm"] # All Env vars including GHM
TOPO_IMAGE_BANDS = ["elevation", "slope", "northness", "eastness"]
TOPO_SCALAR_COLUMNS = ["elev_mean","elev_sd","elev_min","elev_max","slope_mean","slope_sd","slope_min","slope_max","northness_mean","eastness_mean","topo_valid_frac"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build uniform and spatial block CV SDM datasets for one or more species."
    )

    # Input mode: one file or a directory of files
    parser.add_argument("--occurrence-file", type=str, default=None,
                        help="Path to a single species occurrence CSV.")
    parser.add_argument("--occurrence-dir", type=str, default=None,
                        help="Directory containing multiple species occurrence CSVs.")

    # Shared resources
    parser.add_argument("--output-root", type=str, required=True,
                        help="Root output directory.")
    parser.add_argument("--background-inner-buffer-km", type=float, default=0.0,
                        help = "Optinal inner exclusion buffer radius in kilometers around presences when sampling background points." \
                        "Background points will be sampled further than this distance, but within --background-buffer-km." \
                        "Use 5 for a 5-50 km doughnut.")
    parser.add_argument("--background-max-sampling-rounds", type=int, default=25,
                        help = "Max number of candidate-sampling rounds before failing. Increase if the doughnut sampling area is fragmented or very small.")
    parser.add_argument("--background-buffer-km", type=float, default=50.0,
                        help = "Buffer radius in kilometers for sampling background points around presences. Default = 50 km.")
    parser.add_argument("--tileindex", type=str, required=True,
                        help="NAIP tile index shapefile.")
    parser.add_argument("--naip-folder", type=str, required=True,
                        help="Folder containing downloaded NAIP .tif files.")
    parser.add_argument("--worldclim-folder", type=str, required=True,
                        help="Folder containing wc2.1_30s_bio_1.tif ... wc2.1_30s_bio_19.tif")
    parser.add_argument("--ghm-raster", type=str, required=True,
                        help="Path to global human modification raster.")

    # Output / processing settings
    parser.add_argument("--chip-size", type=int, default=256,
                        help="NAIP chip size in pixels.") # 256 pixels at 2m-resolution = 512m x 512m chips
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--train-frac", type=float, default=0.70,
                        help="Fraction of data for train split.")
    parser.add_argument("--val-frac", type=float, default=0.15,
                        help="Fraction of data for validation split.")
    parser.add_argument("--test-frac", type=float, default=0.15,
                        help="Fraction of data for test split.")
    parser.add_argument("--n-folds", type=int, default=5,
                        help="Number of spatial CV folds.")
    parser.add_argument("--block-size-m", type=float, default=200000,
                        help="Spatial block size in meters for block CV. Default = 200 km.")
    parser.add_argument("--background-multiplier", type=float, default=3.0,
                        help="Oversampling multiplier when drawing candidate background points.") # Oversample candidates, then filter/deduplicate
    parser.add_argument("--species-name-col", type=str, default="species")
    parser.add_argument("--lat-col", type=str, default="decimalLatitude")
    parser.add_argument("--lon-col", type=str, default="decimalLongitude")
    parser.add_argument("--source-col", type=str, default="Source")

    parser.add_argument("--topo-mode", choices=["none", "scalar", "chip", "both"], default="none")
    parser.add_argument("--dem-raster", type=str, default=None)
    parser.add_argument("--slope-raster", type=str, default=None)
    parser.add_argument("--aspect-raster", type=str, default=None)
    parser.add_argument("--northness-raster", type=str, default=None)
    parser.add_argument("--eastness-raster", type=str, default=None)
    parser.add_argument("--topo-chip-size", type=int, default=128)
    parser.add_argument("--topo-min-valid-frac", type=float, default=0.90)

    args = parser.parse_args()

    if not args.occurrence_file and not args.occurrence_dir:
        parser.error("Provide either --occurrence-file or --occurrence-dir")

    if args.occurrence_file and args.occurrence_dir:
        parser.error("Provide only one of --occurrence-file or --occurrence-dir")

    total = args.train_frac + args.val_frac + args.test_frac
    if not np.isclose(total, 1.0):
        parser.error("train-frac + val-frac + test-frac must sum to 1.0")

    return args


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def slugify_species_name(name: str) -> str:
    """Convert species name to a consistent filesystem-safe slug."""
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def display_name_from_slug(slug: str) -> str:
    """Convert a slug to title-style label."""
    return "_".join([x.capitalize() for x in slug.split("_")])


def infer_species_slug_from_filename(csv_path: Path) -> str:
    stem = csv_path.stem
    stem = re.sub(r"_thinned$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"_thin.*$", "", stem, flags=re.IGNORECASE)
    return slugify_species_name(stem)


# ---------------------------------------------------------------------
# I/O and geometry helpers
# ---------------------------------------------------------------------

def build_naip_file_index(naip_folder: str) -> dict:
    """
    Recursively index all NAIP .tif/.TIF files under naip_folder.

    Returns
    -------
    dict
        Mapping from basename filename -> full file path

    Notes
    -----
    - Uses lowercase basenames for robust matching.
    - If duplicate basenames exist, later matches will overwrite earlier ones.
      You can change this behavior if needed.
    """
    naip_index = {}

    for root, _, files in os.walk(naip_folder):
        for f in files:
            if f.lower().endswith(".tif"):
                naip_index[f.lower()] = os.path.join(root, f)

    if not naip_index:
        raise RuntimeError(f"No .tif files found recursively under: {naip_folder}")

    print(f"Indexed {len(naip_index):,} NAIP TIFF files under {naip_folder}")
    return naip_index


def load_tileindex(tileindex_fp: str, naip_folder: str):
    """
    Load tile index, standardize filename column, and keep only downloaded tiles.

    Returns
    -------
    tileindex : GeoDataFrame
    downloaded_tiles : GeoDataFrame
    naip_file_index : dict
        Mapping from lowercase basename -> full path on disk
    """
    tileindex = gpd.read_file(tileindex_fp).to_crs("EPSG:4326")

    if "filename" not in tileindex.columns:
        raise ValueError("Tile index must contain a 'filename' column.")

    # Normalize filename column to basename only
    tileindex["filename"] = tileindex["filename"].apply(os.path.basename)
    tileindex["filename_lower"] = tileindex["filename"].str.lower()

    # Recursively index all downloaded TIFFs
    naip_file_index = build_naip_file_index(naip_folder)

    # Mark whether each tile exists somewhere in the archive
    tileindex["downloaded"] = tileindex["filename_lower"].isin(naip_file_index)
    downloaded_tiles = tileindex[tileindex["downloaded"]].copy()

    print(f"Tile index rows: {len(tileindex):,}")
    print(f"Downloaded tiles matched: {len(downloaded_tiles):,}")

    if downloaded_tiles.empty:
        sample_index_names = tileindex["filename"].head(5).tolist()
        sample_disk_names = list(naip_file_index.keys())[:5]
        raise RuntimeError(
            "No downloaded NAIP tiles matched the tile index filenames.\n"
            f"Example tileindex names: {sample_index_names}\n"
            f"Example disk basenames: {sample_disk_names}"
        )

    return tileindex, downloaded_tiles, naip_file_index


def load_occurrence_csv(csv_path: Path,
                        species_name_col: str,
                        lat_col: str,
                        lon_col: str,
                        source_col: str) -> gpd.GeoDataFrame:
    """
    Load a species occurrence CSV and standardize the required columns.
    """
    df = pd.read_csv(csv_path)

    missing = [c for c in [lat_col, lon_col] if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name} is missing required columns: {missing}")

    df = df.copy()
    df = df.rename(columns={
        lat_col: "lat",
        lon_col: "lon"
    })

    if species_name_col in df.columns:
        df["species"] = df[species_name_col]
    else:
        df["species"] = infer_species_slug_from_filename(csv_path).replace("_", " ").title()

    if source_col in df.columns:
        df["source"] = df[source_col].fillna("unknown")
    else:
        df["source"] = "unknown"

    df = df.dropna(subset=["lat", "lon"]).copy()
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"]).copy()

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326"
    )
    return gdf


# ---------------------------------------------------------------------
# Presence thinning and background sampling
# ---------------------------------------------------------------------

def deduplicate_presences_to_worldclim_cells(occurrences_gdf: gpd.GeoDataFrame,
                                             wc_reference_raster_fp: str,
                                             seed: int) -> gpd.GeoDataFrame:
    """
    Deduplicate occurrence points to WorldClim raster cells.
    One random presence is retained per raster cell.
    """
    with rasterio.open(wc_reference_raster_fp) as src:
        affine = src.transform
        raster_crs = src.crs

    occ = occurrences_gdf.copy()
    if occ.crs != raster_crs:
        occ = occ.to_crs(raster_crs)

    rows, cols = rasterio.transform.rowcol(
        affine,
        occ.geometry.x.values,
        occ.geometry.y.values
    )
    occ["cell_id"] = [f"{r}_{c}" for r, c in zip(rows, cols)]

    dedup = (
        occ.groupby("cell_id", group_keys=False)
        .apply(lambda x: x.sample(n=1, random_state=seed))
        .reset_index(drop=True)
    )

    return dedup.to_crs("EPSG:4326")


def random_points_in_polygon(polygon, n: int, seed: int, max_batches: int = 1000) -> list[Point]:
    """
    Uniform random point sampling within a polygon or multipolygon.

    This version includes a maximum batch limit so that tiny or fragmented
    polygons do not cause an infinite loop.
    """
    if polygon is None or polygon.is_empty:
        raise RuntimeError("Cannot sample random points from an empty polygon.")

    # Attempt to repair minor geometry issues
    polygon = polygon.buffer(0)

    if polygon.is_empty:
        raise RuntimeError("Sampling polygon became empty after geometry repair.")

    rng = np.random.default_rng(seed)
    points = []
    minx, miny, maxx, maxy = polygon.bounds

    n_batches = 0
    while len(points) < n and n_batches < max_batches:
        needed = n - len(points)

        # Larger batches help when the polygon occupies a small fraction of its bbox
        batch_n = max(int(needed * 5), 2000)

        xs = rng.uniform(minx, maxx, size=batch_n)
        ys = rng.uniform(miny, maxy, size=batch_n)

        for x, y in zip(xs, ys):
            pt = Point(x, y)
            if polygon.contains(pt):
                points.append(pt)
                if len(points) >= n:
                    break

        n_batches += 1

    if len(points) < n:
        raise RuntimeError(
            f"Only sampled {len(points)} points from polygon after {max_batches} batches; "
            f"requested {n}. The sampling area may be too small or too fragmented."
        )

    return points[:n]


def sample_background_points(presence_gdf: gpd.GeoDataFrame,
                             downloaded_tiles_gdf: gpd.GeoDataFrame,
                             wc_reference_raster_fp: str,
                             background_multiplier: float,
                             seed: int,
                             buffer_km: float,
                             inner_buffer_km: float = 0.0,
                             max_sampling_rounds: int = 25) -> gpd.GeoDataFrame:
    """
    Sample one background point per presence using an optional doughnut method.

    Background points are constrained to:
      - downloaded NAIP footprint
      - farther than inner_buffer_km from any presence
      - within buffer_km of any presence
      - unique WorldClim cells
      - non-overlapping with presence WorldClim cells

    For example:
      inner_buffer_km = 5
      buffer_km = 50

    gives a 5-50 km annulus around the known presences.
    """
    n_background = len(presence_gdf)
    if n_background == 0:
        raise ValueError("No presence points available after thinning.")

    if inner_buffer_km < 0:
        raise ValueError("--background-inner-buffer-km must be >= 0.")

    if inner_buffer_km >= buffer_km:
        raise ValueError(
            f"Inner buffer ({inner_buffer_km} km) must be smaller than "
            f"outer buffer ({buffer_km} km)."
        )

    with rasterio.open(wc_reference_raster_fp) as src:
        affine = src.transform
        raster_crs = src.crs

    # -----------------------------------------------------------------
    # 1. Build the doughnut sampling area in a projected CRS
    # -----------------------------------------------------------------
    presence_proj = presence_gdf.to_crs("EPSG:5070").copy()

    outer_buffer_m = buffer_km * 1000.0
    inner_buffer_m = inner_buffer_km * 1000.0

    outer_buffer_proj = presence_proj.buffer(outer_buffer_m).union_all()

    if inner_buffer_km > 0:
        inner_buffer_proj = presence_proj.buffer(inner_buffer_m).union_all()
        sampling_zone_proj = outer_buffer_proj.difference(inner_buffer_proj)
    else:
        sampling_zone_proj = outer_buffer_proj

    if sampling_zone_proj.is_empty:
        raise RuntimeError(
            "Sampling zone is empty after constructing the doughnut buffer. "
            "Try reducing --background-inner-buffer-km or increasing --background-buffer-km."
        )

    sampling_zone_wgs84 = (
        gpd.GeoSeries([sampling_zone_proj], crs="EPSG:5070")
        .to_crs("EPSG:4326")
        .iloc[0]
    )

    # -----------------------------------------------------------------
    # 2. Restrict to downloaded NAIP footprint
    # -----------------------------------------------------------------
    footprint = downloaded_tiles_gdf.to_crs("EPSG:4326").geometry.union_all()
    sampling_area = footprint.intersection(sampling_zone_wgs84)

    # Repair possible geometry artifacts from intersection/difference
    sampling_area = sampling_area.buffer(0)

    if sampling_area.is_empty:
        raise RuntimeError(
            f"Sampling area is empty after intersecting NAIP footprint with "
            f"{inner_buffer_km}-{buffer_km} km presence doughnut."
        )

    sampling_area_km2 = (
        gpd.GeoSeries([sampling_area], crs="EPSG:4326")
        .to_crs("EPSG:5070")
        .area.iloc[0] / 1_000_000.0
    )

    print(
        f"Background sampling rule: {inner_buffer_km:g}-{buffer_km:g} km from presences"
    )
    print(f"Background sampling area: {sampling_area_km2:,.1f} km²")

    # -----------------------------------------------------------------
    # 3. Identify WorldClim cells already occupied by presences
    # -----------------------------------------------------------------
    pres_for_cells = presence_gdf.to_crs(raster_crs).copy()

    rows_pr, cols_pr = rasterio.transform.rowcol(
        affine,
        pres_for_cells.geometry.x.values,
        pres_for_cells.geometry.y.values
    )
    pres_for_cells["cell_id"] = [f"{r}_{c}" for r, c in zip(rows_pr, cols_pr)]
    presence_cells = set(pres_for_cells["cell_id"])

    # -----------------------------------------------------------------
    # 4. Iteratively sample candidates until enough unique valid cells exist
    # -----------------------------------------------------------------
    valid_candidate_chunks = []
    n_candidates_per_round = int(math.ceil(n_background * background_multiplier))

    for round_i in range(max_sampling_rounds):
        round_seed = seed + round_i

        candidate_points = random_points_in_polygon(
            sampling_area,
            n=n_candidates_per_round,
            seed=round_seed
        )

        candidates = gpd.GeoDataFrame(
            geometry=candidate_points,
            crs="EPSG:4326"
        ).to_crs(raster_crs)

        rows_bg, cols_bg = rasterio.transform.rowcol(
            affine,
            candidates.geometry.x.values,
            candidates.geometry.y.values
        )
        candidates["cell_id"] = [f"{r}_{c}" for r, c in zip(rows_bg, cols_bg)]

        # Remove candidate cells that overlap presence cells
        candidates = candidates[~candidates["cell_id"].isin(presence_cells)].copy()

        if not candidates.empty:
            valid_candidate_chunks.append(candidates)

        if valid_candidate_chunks:
            pooled = pd.concat(valid_candidate_chunks, ignore_index=True)
            pooled = gpd.GeoDataFrame(pooled, geometry="geometry", crs=raster_crs)

            # Keep one candidate per WorldClim cell
            pooled = (
                pooled.groupby("cell_id", group_keys=False)
                .apply(lambda x: x.sample(n=1, random_state=seed))
                .reset_index(drop=True)
            )

            print(
                f"Sampling round {round_i + 1}: "
                f"{len(pooled):,} unique valid background cells"
            )

            if len(pooled) >= n_background:
                break
    else:
        raise RuntimeError(
            f"Only {len(pooled) if valid_candidate_chunks else 0} valid unique background cells "
            f"available after {max_sampling_rounds} rounds, but {n_background} are needed. "
            f"Try increasing --background-multiplier, increasing --background-buffer-km, "
            f"or decreasing --background-inner-buffer-km."
        )

    # -----------------------------------------------------------------
    # 5. Final sample: one background point per presence
    # -----------------------------------------------------------------
    background_sampled = pooled.sample(n=n_background, random_state=seed).copy()

    # Add nearest-presence distance for QC
    bg_for_dist = background_sampled.to_crs("EPSG:5070").copy()
    presence_union_proj = presence_proj.geometry.union_all()

    background_sampled["nearest_presence_km"] = (
        bg_for_dist.geometry.distance(presence_union_proj) / 1000.0
    )

    background_sampled = background_sampled.to_crs("EPSG:4326")
    background_sampled["lat"] = background_sampled.geometry.y
    background_sampled["lon"] = background_sampled.geometry.x
    background_sampled["source"] = "background"
    background_sampled["background_sampling_rule"] = (
        f"doughnut_{inner_buffer_km:g}_{buffer_km:g}_km"
    )

    print(
        "Final background nearest-presence distance summary, km: "
        f"min={background_sampled['nearest_presence_km'].min():.2f}, "
        f"median={background_sampled['nearest_presence_km'].median():.2f}, "
        f"max={background_sampled['nearest_presence_km'].max():.2f}"
    )

    return background_sampled



def attach_naip_filename_or_url(points_gdf: gpd.GeoDataFrame,
                                tileindex_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Spatially join points to the tileindex to optionally preserve filename/url metadata.
    """
    cols = ["geometry"]
    for c in ["filename", "url"]:
        if c in tileindex_gdf.columns:
            cols.append(c)

    joined = gpd.sjoin(
        points_gdf,
        tileindex_gdf[cols],
        how="left",
        predicate="within"
    ).drop(columns=["index_right"], errors="ignore")

    return joined


def build_presence_background_points(presence_gdf: gpd.GeoDataFrame,
                                     background_gdf: gpd.GeoDataFrame,
                                     tileindex_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Combine presences and background into a single GeoDataFrame.
    """
    pres = presence_gdf.copy()
    pres = pres.to_crs("EPSG:4326")
    pres["lat"] = pres.geometry.y
    pres["lon"] = pres.geometry.x
    pres["presence"] = 1

    bg = background_gdf.copy()
    bg["presence"] = 0

    pres = attach_naip_filename_or_url(pres, tileindex_gdf)
    bg = attach_naip_filename_or_url(bg, tileindex_gdf)

    keep_cols = ["species", "source", "lat", "lon", "presence", "geometry"]
    for extra in [
    "filename",
    "url",
    "cell_id",
    "nearest_presence_km",
    "background_sampling_rule"]:
        if extra in pres.columns or extra in bg.columns:
            keep_cols.append(extra)

    pres = pres[[c for c in keep_cols if c in pres.columns]].copy()
    bg = bg[[c for c in keep_cols if c in bg.columns]].copy()

    combined = pd.concat([pres, bg], ignore_index=True)
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs="EPSG:4326")
    combined = combined.drop_duplicates(subset=["lat", "lon", "presence"])

    return combined


# ---------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------

def stratified_train_val_test_split(df: pd.DataFrame,
                                    train_frac: float,
                                    val_frac: float,
                                    test_frac: float,
                                    seed: int) -> pd.DataFrame:
    """
    Stratified split by presence class so class balance is maintained.
    """
    rng = np.random.default_rng(seed)
    chunks = []

    for presence_value, sub in df.groupby("presence"):
        idx = np.array(sub.index)
        rng.shuffle(idx)

        n = len(idx)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        n_test = n - n_train - n_val

        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:n_train + n_val + n_test]

        tmp = df.loc[train_idx].copy()
        tmp["split"] = "train"
        chunks.append(tmp)

        tmp = df.loc[val_idx].copy()
        tmp["split"] = "val"
        chunks.append(tmp)

        tmp = df.loc[test_idx].copy()
        tmp["split"] = "test"
        chunks.append(tmp)

    out = pd.concat(chunks, ignore_index=False).sort_index().reset_index(drop=True)
    return out


def assign_spatial_blocks(points_gdf: gpd.GeoDataFrame,
                          block_size_m: float,
                          n_folds: int,
                          seed: int) -> gpd.GeoDataFrame:
    """
    Assign spatial blocks and folds from projected coordinates.
    Uses EPSG:5070 (CONUS Albers Equal Area) for U.S.-scale blocking.
    """
    gdf = points_gdf.copy().to_crs("EPSG:5070")
    minx, miny, _, _ = gdf.total_bounds

    gdf["block_x"] = ((gdf.geometry.x - minx) // block_size_m).astype(int)
    gdf["block_y"] = ((gdf.geometry.y - miny) // block_size_m).astype(int)
    gdf["block_id"] = gdf["block_x"].astype(str) + "_" + gdf["block_y"].astype(str)

    unique_blocks = sorted(gdf["block_id"].unique())
    rng = np.random.default_rng(seed)
    shuffled = np.array(unique_blocks, dtype=object)
    rng.shuffle(shuffled)

    block_to_fold = {}
    for i, block_id in enumerate(shuffled):
        block_to_fold[block_id] = (i % n_folds) + 1

    gdf["fold"] = gdf["block_id"].map(block_to_fold).astype(int)
    return gdf.to_crs("EPSG:4326")


def make_spatial_cv_rounds(points_gdf: gpd.GeoDataFrame,
                           n_folds: int,
                           seed: int) -> pd.DataFrame:
    """
    For each held-out fold:
      - test = held-out fold
      - train/val = remaining folds, split stratified by presence
    """
    rounds = []

    for heldout in range(1, n_folds + 1):
        train_val = points_gdf[points_gdf["fold"] != heldout].copy()
        test = points_gdf[points_gdf["fold"] == heldout].copy()

        split_train_val = stratified_train_val_test_split(
            train_val.drop(columns=["fold"], errors="ignore"),
            train_frac=0.70,
            val_frac=0.30,
            test_frac=0.0,
            seed=seed
        )

        split_train_val.loc[split_train_val["split"] == "test", "split"] = "val"

        test = test.copy()
        test["split"] = "test"

        # restore fold/block metadata
        split_train_val = split_train_val.merge(
            train_val[["lat", "lon", "fold", "block_id"]].drop_duplicates(),
            on=["lat", "lon"],
            how="left"
        )

        test = test.copy()

        round_df = pd.concat([split_train_val, test], ignore_index=True)
        round_df["cv_round"] = heldout
        rounds.append(round_df)

    return pd.concat(rounds, ignore_index=True)


# ---------------------------------------------------------------------
# Climate helpers
# ---------------------------------------------------------------------

def compute_worldclim_stats(worldclim_folder: str, study_geom, buffer_deg: float = 0.01) -> dict:
    """
    Compute mean/std for each WorldClim variable inside the study footprint.
    """
    stats = {}

    buffered_geom = (
        gpd.GeoSeries([study_geom], crs="EPSG:4326")
        .buffer(buffer_deg)
        .iloc[0]
    )

    geom_json = [buffered_geom.__geo_interface__]

    for varname in WC_VARS:
        raster_fp = os.path.join(worldclim_folder, f"{varname}.tif")
        with rasterio.open(raster_fp) as src:
            out_image, _ = mask(src, geom_json, crop=True)
            data = out_image[0]

            if src.nodata is not None:
                data = np.ma.masked_equal(data, src.nodata)
            else:
                data = np.ma.masked_where(~np.isfinite(data), data)

            mean = float(data.mean())
            std = float(data.std())

            if std == 0 or not np.isfinite(std):
                raise RuntimeError(f"Invalid std for {varname}: {std}")

            stats[varname] = {"mean": mean, "std": std}

    return stats


def open_worldclim_datasets(worldclim_folder: str) -> dict:
    return {
        var: rasterio.open(os.path.join(worldclim_folder, f"{var}.tif"))
        for var in WC_VARS
    }


def extract_worldclim_vars_for_point(lon: float,
                                     lat: float,
                                     wc_datasets: dict,
                                     normalization_stats: dict) -> dict:
    """
    Extract normalized WorldClim values for one point.
    """
    vals = {}
    for varname, ds in wc_datasets.items():
        raw_value = list(ds.sample([(lon, lat)]))[0][0]
        mean = normalization_stats[varname]["mean"]
        std = normalization_stats[varname]["std"]
        vals[varname] = float((raw_value - mean) / std)
    return vals


def extract_ghm_for_point(lon: float, lat: float, ghm_ds) -> float:
    """
    Extract a clamped GHM value for one point.
    """
    raw_value = list(ghm_ds.sample([(lon, lat)]))[0][0]
    if not np.isfinite(raw_value):
        return np.nan
    return float(np.clip(raw_value, 0.0, 1.0))


def is_valid_env_record(env_vars: dict, min_val: float = -100, max_val: float = 100) -> bool:
    for v in ALL_ENV_VARS:
        val = env_vars.get(v, np.nan)
        if not np.isfinite(val):
            return False
        if val < min_val or val > max_val:
            return False
    return True


# ---------------------------------------------------------------------
# NAIP chip extraction
# ---------------------------------------------------------------------

def extract_naip_chip_for_point(lon: float,
                                lat: float,
                                out_fp: str,
                                chip_size: int,
                                tileindex_gdf: gpd.GeoDataFrame,
                                naip_file_index: dict) -> bool:
    """
    Extract a chip around a point from one or more overlapping NAIP tiles.
    """
    point_wgs84 = Point(lon, lat)
    matches = tileindex_gdf[tileindex_gdf.geometry.intersects(point_wgs84)]

    if matches.empty:
        return False

    naip_fp = None
    for _, row in matches.iterrows():
        candidate = naip_file_index.get(str(row["filename"]).lower())
        if candidate and os.path.exists(candidate):
            naip_fp = candidate
            break

    if naip_fp is None:
        return False

    with rasterio.open(naip_fp) as src:
        dst_crs = src.crs
        point_proj = gpd.GeoSeries([point_wgs84], crs="EPSG:4326").to_crs(dst_crs).iloc[0]

        # rasterio.index returns (row, col), not (x, y)
        row, col = src.index(point_proj.x, point_proj.y)
        half = chip_size // 2

        row_off = row - half
        col_off = col - half
        chip_win = Window(col_off=col_off, row_off=row_off, width=chip_size, height=chip_size)

        chip_transform = src.window_transform(chip_win)
        chip_left, chip_top = chip_transform * (0, 0)
        chip_right, chip_bottom = chip_transform * (chip_size, chip_size)

        bbox_proj = box(
            min(chip_left, chip_right),
            min(chip_top, chip_bottom),
            max(chip_left, chip_right),
            max(chip_top, chip_bottom)
        )
        bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_proj], crs=dst_crs).to_crs("EPSG:4326")

    overlapping = tileindex_gdf[tileindex_gdf.geometry.intersects(bbox_gdf.geometry.iloc[0])]
    if overlapping.empty:
        return False

    naip_fps = []
    for _, r in overlapping.iterrows():
        fp = naip_file_index.get(str(r["filename"]).lower())
        if fp and os.path.exists(fp):
            naip_fps.append(fp)

    if not naip_fps:
        return False

    datasets = [rasterio.open(fp) for fp in naip_fps]
    try:
        crs_set = {ds.crs.to_string() for ds in datasets}
        if len(crs_set) > 1:
            # Skip rare cross-CRS chips
            return False

        xres, yres = datasets[0].res

        mosaic, _ = merge(
            datasets,
            bounds=(
                min(chip_left, chip_right),
                min(chip_top, chip_bottom),
                max(chip_left, chip_right),
                max(chip_top, chip_bottom)
            ),
            res=(xres, yres),
            nodata=0
        )

        chip = mosaic[:, :chip_size, :chip_size]

        if chip.shape[1] < chip_size or chip.shape[2] < chip_size:
            padded = np.zeros((chip.shape[0], chip_size, chip_size), dtype=chip.dtype)
            padded[:, :chip.shape[1], :chip.shape[2]] = chip
            chip = padded

        out_transform = from_origin(
            min(chip_left, chip_right),
            max(chip_top, chip_bottom),
            xres,
            yres
        )

        Path(out_fp).parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(
            out_fp,
            "w",
            driver="GTiff",
            height=chip.shape[1],
            width=chip.shape[2],
            count=chip.shape[0],
            dtype=chip.dtype,
            crs=datasets[0].crs,
            transform=out_transform
        ) as dst:
            dst.write(chip)

        return True

    finally:
        for ds in datasets:
            ds.close()


# ---------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------

def extract_topo_for_naip_chip(
    naip_chip_fp: str,
    topo_sources: dict,
    out_fp: str | None,
    topo_chip_size: int
):
    """
    Extract a topographic chip over the same spatial footprint as a NAIP chip.

    Output band order:
      1 elevation
      2 slope
      3 northness
      4 eastness
    """
    out_nodata = -9999.0

    with rasterio.open(naip_chip_fp) as naip:
        dst_crs = naip.crs
        dst_transform = naip.transform * rasterio.Affine.scale(
            naip.width / topo_chip_size,
            naip.height / topo_chip_size
        )

    layers = {}

    for name in ["elevation", "slope", "northness", "eastness"]:
        src = topo_sources.get(name)
        if src is None:
            continue

        dst = np.full(
            (topo_chip_size, topo_chip_size),
            np.nan,
            dtype=np.float32
        )

        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear
        )

        layers[name] = dst

    # Fallback only if northness/eastness rasters were not supplied.
    # Prefer precomputed northness/eastness rasters when possible.
    if ("northness" not in layers or "eastness" not in layers) and topo_sources.get("aspect") is not None:
        aspect = topo_sources["aspect"]

        a = np.full(
            (topo_chip_size, topo_chip_size),
            np.nan,
            dtype=np.float32
        )

        reproject(
            source=rasterio.band(aspect, 1),
            destination=a,
            src_transform=aspect.transform,
            src_crs=aspect.crs,
            src_nodata=aspect.nodata,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=Resampling.nearest
        )

        valid_aspect = np.isfinite(a) & (a >= 0.0) & (a <= 360.0)
        rad = np.deg2rad(a)

        north = np.full_like(a, np.nan, dtype=np.float32)
        east = np.full_like(a, np.nan, dtype=np.float32)

        north[valid_aspect] = np.cos(rad[valid_aspect])
        east[valid_aspect] = np.sin(rad[valid_aspect])

        layers.setdefault("northness", north)
        layers.setdefault("eastness", east)

    missing = [k for k in TOPO_IMAGE_BANDS if k not in layers]
    if missing:
        raise RuntimeError(f"Missing required topo layers: {missing}")

    stack = np.stack(
        [layers[k].astype(np.float32) for k in TOPO_IMAGE_BANDS],
        axis=0
    )

    valid = np.isfinite(stack).all(axis=0)
    valid_frac = float(valid.mean())

    stats = {"topo_valid_frac": valid_frac}

    if valid.any():
        elev = stack[0][valid]
        slope = stack[1][valid]
        north = stack[2][valid]
        east = stack[3][valid]

        stats.update({
            "elev_mean": float(np.mean(elev)),
            "elev_sd": float(np.std(elev)),
            "elev_min": float(np.min(elev)),
            "elev_max": float(np.max(elev)),
            "slope_mean": float(np.mean(slope)),
            "slope_sd": float(np.std(slope)),
            "slope_min": float(np.min(slope)),
            "slope_max": float(np.max(slope)),
            "northness_mean": float(np.mean(north)),
            "eastness_mean": float(np.mean(east)),
        })
    else:
        for c in TOPO_SCALAR_COLUMNS:
            stats[c] = np.nan

    if out_fp is not None:
        out_stack = np.where(np.isfinite(stack), stack, out_nodata).astype(np.float32)

        profile = {
            "driver": "GTiff",
            "height": topo_chip_size,
            "width": topo_chip_size,
            "count": 4,
            "dtype": "float32",
            "crs": dst_crs,
            "transform": dst_transform,
            "nodata": out_nodata,
            "compress": "deflate",
            "predictor": 2,
            "tiled": True,
            "blockxsize": min(128, topo_chip_size),
            "blockysize": min(128, topo_chip_size),
            "BIGTIFF": "IF_SAFER",
        }

        # block sizes must be multiples of 16
        if profile["blockxsize"] % 16 != 0 or profile["blockysize"] % 16 != 0:
            profile.pop("tiled")
            profile.pop("blockxsize")
            profile.pop("blockysize")

        Path(out_fp).parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(out_fp, "w", **profile) as dst:
            dst.write(out_stack)

    return stats


def add_normalized_lat_lon(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lat_std = df["lat"].std()
    lon_std = df["lon"].std()

    if lat_std == 0:
        df["lat_norm"] = 0.0
    else:
        df["lat_norm"] = (df["lat"] - df["lat"].mean()) / lat_std

    if lon_std == 0:
        df["lon_norm"] = 0.0
    else:
        df["lon_norm"] = (df["lon"] - df["lon"].mean()) / lon_std

    return df


def build_dataset_csv(points_df: pd.DataFrame,
                      topo_sources: dict | None,
                      topo_mode: str,
                      topo_chip_size: int,
                      topo_min_valid_frac: float,
                      dataset_root: Path,
                      tileindex_gdf: gpd.GeoDataFrame,
                      naip_file_index: dict,
                      wc_datasets: dict,
                      wc_stats: dict,
                      ghm_ds,
                      chip_size: int,
                      species_label: str,
                      suffix: str = "",
                      cv_round: int | None = None) -> pd.DataFrame:
    """
    Extract NAIP chips + env vars and save the final modeling dataset CSV.
    """
    chip_dir = dataset_root / "images"
    chip_dir.mkdir(parents=True, exist_ok=True)

    records = []

    for idx, row in tqdm(points_df.iterrows(), total=len(points_df), desc=f"{species_label}{suffix}"):
        lat = row["lat"]
        lon = row["lon"]
        presence = int(row["presence"])
        split = row["split"]
        source = row.get("source", "unknown")

        if cv_round is None:
            sample_id = f"{idx}_{presence}_{split}"
            chip_fn = f"chip_{idx}_{'pres' if presence else 'abs'}_{split}.tif"
        else:
            sample_id = f"{idx}_{presence}_{split}_CV{cv_round}"
            chip_fn = f"chip_{idx}_{'pres' if presence else 'abs'}_{split}_CV{cv_round}.tif"

        chip_fp = chip_dir / chip_fn

        ok = extract_naip_chip_for_point(
            lon=lon,
            lat=lat,
            out_fp=str(chip_fp),
            chip_size=chip_size,
            tileindex_gdf=tileindex_gdf,
            naip_file_index=naip_file_index
        )
        if not ok:
            continue

        env = extract_worldclim_vars_for_point(lon, lat, wc_datasets, wc_stats)
        env["ghm"] = extract_ghm_for_point(lon, lat, ghm_ds)

        if not is_valid_env_record(env):
            continue

        topo_chip_rel = None
        topo_stats = {}
        if topo_mode != "none":
            topo_chip_fp = None
            if topo_mode in {"chip","both"}:
                topo_chip_fp = chip_dir / chip_fn.replace("chip_", "topo_chip_")
                topo_chip_rel = os.path.relpath(topo_chip_fp, dataset_root)
            topo_stats = extract_topo_for_naip_chip(str(chip_fp), topo_sources, str(topo_chip_fp) if topo_chip_fp else None, topo_chip_size)
            if topo_stats.get("topo_valid_frac", 0.0) < topo_min_valid_frac:
                continue

        rec = {
            "sample_id": sample_id,
            "chip_path": os.path.relpath(chip_fp, dataset_root),
            "split": split,
            "presence": presence,
            "lat": lat,
            "lon": lon,
            "source": source,
            "topo_chip_path": topo_chip_rel
        }

        for optional_col in ["nearest_presence_km", "background_sampling_rule"]:
            if optional_col in row:
                rec[optional_col] = row.get(optional_col)

        if cv_round is not None:
            rec["cv_round"] = cv_round

        rec.update(env)
        if topo_mode in {"scalar","both","chip"}:
            rec.update(topo_stats)
        records.append(rec)

    df = pd.DataFrame(records)

    if df.empty:
        return df

    # Keep only rows whose chip actually exists
    df["chip_path_abs"] = df["chip_path"].apply(lambda p: os.path.join(dataset_root, p))
    df = df[df["chip_path_abs"].apply(os.path.exists)].drop(columns=["chip_path_abs"])

    df = add_normalized_lat_lon(df)
    return df


# ---------------------------------------------------------------------
# Species-level pipeline
# ---------------------------------------------------------------------

def process_species(csv_path: Path, args, tileindex_gdf, downloaded_tiles_gdf, wc_stats, naip_file_index):
    species_slug = infer_species_slug_from_filename(csv_path)
    species_label = display_name_from_slug(species_slug)

    print("\n" + "=" * 80)
    print(f"Processing species: {species_slug}")
    print("=" * 80)

    # Output roots
    species_root = Path(args.output_root)
    datasets_root = species_root / f"{species_label}_Datasets"
    occurrences_root = species_root / f"{species_label}_US_Occurrences"

    datasets_root.mkdir(parents=True, exist_ok=True)
    occurrences_root.mkdir(parents=True, exist_ok=True)

    uniform_name = f"{species_label}_US_Uniform_PA_NAIP_{args.chip_size}_april2026"
    blockcv_name = f"{species_label}_US_BlockCV_PA_NAIP_{args.chip_size}_april2026"

    uniform_root = datasets_root / uniform_name
    blockcv_root = datasets_root / blockcv_name
    uniform_root.mkdir(parents=True, exist_ok=True)
    blockcv_root.mkdir(parents=True, exist_ok=True)

    # WorldClim reference raster
    wc_reference = os.path.join(args.worldclim_folder, "wc2.1_30s_bio_1.tif")

    # Load and clean occurrences
    occurrences = load_occurrence_csv(
        csv_path,
        species_name_col=args.species_name_col,
        lat_col=args.lat_col,
        lon_col=args.lon_col,
        source_col=args.source_col
    )

    print(f"Loaded {len(occurrences):,} raw occurrences from {csv_path.name}")

    # Deduplicate by exact coordinate first
    occurrences = occurrences.drop_duplicates(subset=["lat", "lon"]).copy()
    print(f"After exact lat/lon deduplication: {len(occurrences):,}")

    # Thin to WorldClim cells
    presences = deduplicate_presences_to_worldclim_cells(
        occurrences_gdf=occurrences,
        wc_reference_raster_fp=wc_reference,
        seed=args.seed
    )
    print(f"After WorldClim-cell thinning: {len(presences):,}")

    # Sample backgrounds
    backgrounds = sample_background_points(
        presence_gdf=presences,
        downloaded_tiles_gdf=downloaded_tiles_gdf,
        wc_reference_raster_fp=wc_reference,
        background_multiplier=args.background_multiplier,
        buffer_km=args.background_buffer_km,
        inner_buffer_km=args.background_inner_buffer_km,
        max_sampling_rounds=args.background_max_sampling_rounds,
        seed=args.seed
    )
    print(f"Sampled {len(backgrounds):,} background points")

    # Combine P/B dataset
    pa_points = build_presence_background_points(
        presence_gdf=presences,
        background_gdf=backgrounds,
        tileindex_gdf=tileindex_gdf
    )
    print(f"Combined presence/background dataset: {len(pa_points):,}")

    # Save combined points before split
    pb_points_csv = occurrences_root / f"{species_label}_US_Presence_Background_Points.csv"
    pa_points.drop(columns="geometry").to_csv(pb_points_csv, index=False)
    print(f"Saved: {pb_points_csv}")

    # -----------------------------------------------------------------
    # Uniform train/val/test
    # -----------------------------------------------------------------
    uniform_points = stratified_train_val_test_split(
        pa_points.drop(columns="geometry"),
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed
    )

    uniform_points_gdf = gpd.GeoDataFrame(
        uniform_points,
        geometry=gpd.points_from_xy(uniform_points["lon"], uniform_points["lat"]),
        crs="EPSG:4326"
    )

    uniform_points_csv = uniform_root / f"{species_label}_Pres_Bg_US_Uniform_Train_Val_Test_Points.csv"
    uniform_points_gdf.drop(columns="geometry").to_csv(uniform_points_csv, index=False)
    print(f"Saved: {uniform_points_csv}")

    # -----------------------------------------------------------------
    # Spatial block CV
    # -----------------------------------------------------------------
    pa_points_blocks = assign_spatial_blocks(
        gpd.GeoDataFrame(pa_points.copy(), geometry="geometry", crs="EPSG:4326"),
        block_size_m=args.block_size_m,
        n_folds=args.n_folds,
        seed=args.seed
    )

    block_points_csv = blockcv_root / f"{species_label}_Pres_Bg_US_SpatialCV_Train_Val_Test_Points.csv"

    cv_rounds = make_spatial_cv_rounds(
        points_gdf=pa_points_blocks.drop(columns="geometry"),
        n_folds=args.n_folds,
        seed=args.seed
    )
    cv_rounds.to_csv(block_points_csv, index=False)
    print(f"Saved: {block_points_csv}")

    for round_num in range(1, args.n_folds + 1):
        fold_df = cv_rounds[cv_rounds["cv_round"] == round_num].copy()
        fold_csv = blockcv_root / f"{species_label}_Pres_Bg_US_SpatialCV_Blocks_Fold_{round_num}.csv"
        fold_df.to_csv(fold_csv, index=False)
        print(f"Saved: {fold_csv}")

    # -----------------------------------------------------------------
    # Open shared rasters once for final datasets
    # -----------------------------------------------------------------
    wc_datasets = open_worldclim_datasets(args.worldclim_folder)
    ghm_ds = rasterio.open(args.ghm_raster)
    topo_sources = None
    if args.topo_mode != "none":
        topo_sources = {"elevation": rasterio.open(args.dem_raster), "slope": rasterio.open(args.slope_raster)}
        if args.northness_raster and args.eastness_raster:
            topo_sources["northness"] = rasterio.open(args.northness_raster)
            topo_sources["eastness"] = rasterio.open(args.eastness_raster)
        elif args.aspect_raster:
            topo_sources["aspect"] = rasterio.open(args.aspect_raster)

    try:
        # ----------------------- Uniform final dataset -----------------------
        uniform_final = build_dataset_csv(
            points_df=uniform_points,
            topo_sources=topo_sources,
            topo_mode=args.topo_mode,
            topo_chip_size=args.topo_chip_size,
            topo_min_valid_frac=args.topo_min_valid_frac,
            dataset_root=uniform_root,
            tileindex_gdf=tileindex_gdf,
            naip_file_index=naip_file_index,
            wc_datasets=wc_datasets,
            wc_stats=wc_stats,
            ghm_ds=ghm_ds,
            chip_size=args.chip_size,
            species_label=species_label,
            suffix="_uniform",
            cv_round=None
        )

        uniform_dataset_csv = uniform_root / f"{species_label}_Pres_Bg_US_Uniform_Train_Val_Test_Dataset.csv"
        uniform_final.to_csv(uniform_dataset_csv, index=False)
        print(f"Saved: {uniform_dataset_csv}")

        # ----------------------- Block CV final datasets -----------------------
        for round_num in range(1, args.n_folds + 1):
            round_df = cv_rounds[cv_rounds["cv_round"] == round_num].copy()
            cv_dir = blockcv_root / f"CV_{round_num}"
            cv_dir.mkdir(parents=True, exist_ok=True)

            cv_final = build_dataset_csv(
                points_df=round_df,
                topo_sources=topo_sources,
                topo_mode=args.topo_mode,
                topo_chip_size=args.topo_chip_size,
                topo_min_valid_frac=args.topo_min_valid_frac,
                dataset_root=cv_dir,
                tileindex_gdf=tileindex_gdf,
                naip_file_index=naip_file_index,
                wc_datasets=wc_datasets,
                wc_stats=wc_stats,
                ghm_ds=ghm_ds,
                chip_size=args.chip_size,
                species_label=species_label,
                suffix=f"_CV{round_num}",
                cv_round=round_num
            )

            cv_dataset_csv = cv_dir / f"{species_label}_Train_Val_Test_US_BlockCV_{round_num}.csv"
            cv_final.to_csv(cv_dataset_csv, index=False)
            print(f"Saved: {cv_dataset_csv}")

    finally:
        for ds in wc_datasets.values():
            ds.close()
        ghm_ds.close()
        if topo_sources:
            for ds in topo_sources.values():
                ds.close()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    # Optional GDAL/PROJ setup for Windows conda environments
    # Uncomment and edit if needed:
    # Configure GDAL and PROJ environment variables
    conda_env = r"C:\Users\talake2\AppData\Local\anaconda3\envs\naip_ailanthus_env"
    os.environ["GDAL_DATA"] = os.path.join(conda_env, "Library", "share", "gdal")
    os.environ["PROJ_LIB"] = os.path.join(conda_env, "Library", "share", "proj")
    os.environ["PATH"] += os.pathsep + os.path.join(conda_env, "Library", "bin")

    tileindex_gdf, downloaded_tiles_gdf, naip_file_index = load_tileindex(args.tileindex, args.naip_folder)
    downloaded_union = downloaded_tiles_gdf.geometry.union_all()

    print("Computing WorldClim normalization statistics once from downloaded NAIP footprint...")
    wc_stats = compute_worldclim_stats(args.worldclim_folder, downloaded_union) # NAIP tile footprint is the area to consider for stats 

    if args.occurrence_file:
        csv_files = [Path(args.occurrence_file)]
    else:
        csv_files = sorted(Path(args.occurrence_dir).glob("*.csv"))

    if not csv_files:
        raise RuntimeError("No occurrence CSV files found.")

    print(f"Found {len(csv_files)} occurrence CSV file(s).")

    for csv_path in csv_files:
        try:
            process_species(
                csv_path=csv_path,
                args=args,
                tileindex_gdf=tileindex_gdf,
                downloaded_tiles_gdf=downloaded_tiles_gdf,
                wc_stats=wc_stats,
                naip_file_index=naip_file_index
            )
        except Exception as e:
            print(f"\n[ERROR] Failed processing {csv_path.name}: {e}\n")

    print("\nDone.")


if __name__ == "__main__":
    main()



# EOF
