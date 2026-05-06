#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Host-NAIP-SDM dataset builder
=============================================================

Builds presence/background datasets for deep learning species distribution
models that combine:

  1. High-resolution NAIP image chips
  2. WorldClim bioclimatic variables
  3. Global Human Modification (GHM)
  4. USGS 3DEP-derived topographic covariates

The script creates:

  1. Presence/background points

  2. Uniform train/validation/test datasets

  3. Spatial block cross-validation datasets

  4. NAIP image chips

  5. Environmental/tabular covariates

  6. Topographic covariates

Topography mode notes:
----------------

--topo-mode none
    Do not use topography.

--topo-mode scalar
    Extract topographic summary statistics only. No topo image chip is written.

--topo-mode chip
    Write a 4-band topographic image chip. Topographic summary statistics are
    also written for QA/QC and possible tabular-model use.

--topo-mode both
    Write both the topographic chip and scalar topographic summaries for model
    training.

Topographic image band order
----------------------------

All topographic chips are written with this fixed band order:

    band 1 = elevation
    band 2 = slope
    band 3 = northness
    band 4 = eastness

Expected occurrence CSV columns
-------------------------------

Required:
    decimalLatitude
    decimalLongitude

Optional:
    species
    Source
    dateIdentified

Example: single species with normalized topography
--------------------------------------------------

python Host_NAIP_SDM_Dataset_Builder_TopoNorm.py ^
    --occurrence-file "Y:/Promit_Host_Occurrences/Fully_thinned_data/notholithocarpus_densiflorus_thinned.csv" ^
    --output-root "Y:/Host_NAIP_SDM" ^
    --tileindex "Y:/Host_NAIP_SDM/NAIP_Imagery_Tile_Indices/NAIP_US_State_Tile_Indices_URL_Paths_jan26.shp" ^
    --naip-folder "Y:/Host_NAIP_SDM/NAIP_Archive" ^
    --worldclim-folder "Y:/Host_NAIP_SDM/Env_Data/Worldclim" ^
    --ghm-raster "Y:/Host_NAIP_SDM/Env_Data/Global_Human_Modification/gHM_WGS84.tif" ^
    --background-inner-buffer-km 5 ^
    --background-buffer-km 50 ^
    --background-multiplier 10 ^
    --topo-mode both ^
    --dem-raster "Y:/Host_NAIP_SDM/Env_Data/Topography_3DEP/derived_30m/dem_3dep_13_epsg5070_30m.tif" ^
    --slope-raster "Y:/Host_NAIP_SDM/Env_Data/Topography_3DEP/derived_30m/slope_degrees_3dep_13_epsg5070_30m.tif" ^
    --northness-raster "Y:/Host_NAIP_SDM/Env_Data/Topography_3DEP/derived_30m/northness_3dep_13_epsg5070_30m.tif" ^
    --eastness-raster "Y:/Host_NAIP_SDM/Env_Data/Topography_3DEP/derived_30m/eastness_3dep_13_epsg5070_30m.tif" ^
    --topo-chip-size 64 ^
    --topo-min-valid-frac 0.90

"""

from __future__ import annotations
import os

conda_env = r"C:\Users\talake2\AppData\Local\anaconda3\envs\naip_ailanthus_env"
os.environ["GDAL_DATA"] = os.path.join(conda_env, "Library", "share", "gdal")
os.environ["PROJ_LIB"] = os.path.join(conda_env, "Library", "share", "proj")
os.environ["PATH"] += os.pathsep + os.path.join(conda_env, "Library", "bin")


import argparse
import json
import math
import random
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.transform import from_origin
from rasterio.warp import reproject
from rasterio.windows import Window
from shapely.geometry import Point, box
from tqdm import tqdm


# ---------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------

WC_VARS = [f"wc2.1_30s_bio_{i}" for i in range(1, 20)]

# Variables required for the baseline environmental record.
# Topographic scalar variables are intentionally not included here because
# they are optional and should be validated separately.
ALL_ENV_VARS = WC_VARS + ["ghm"]

TOPO_IMAGE_BANDS = [
    "elevation",
    "slope",
    "northness",
    "eastness",
]

# These columns are model-ready normalized values unless
# --disable-topo-normalization is supplied.
TOPO_SCALAR_COLUMNS = [
    "elev_mean",
    "elev_sd",
    "elev_min",
    "elev_max",
    "slope_mean",
    "slope_sd",
    "slope_min",
    "slope_max",
    "northness_mean",
    "eastness_mean",
    "topo_valid_frac",
]

# Embedded default topographic normalization statistics caluclated from the 3DEP product at 30m resolution.
# These are 3DEP 30 m statistics and are used unless --topo-normalization-stats
# points to a replacement JSON file with the same schema.
DEFAULT_TOPO_NORM_STATS_3DEP_30M = {
    "elevation": {
        "mean": 726.2603002867634,
        "std": 726.1498602385226,
        "min": -137.69664001464844,
        "max": 4412.6640625,
        "count": 9540833243,
    },
    "slope": {
        "mean": 4.919028345565722,
        "std": 7.374072011456101,
        "min": 0.0,
        "max": 85.33364868164062,
        "count": 9540832992,
    },
    "northness": {
        "mean": -0.02528306975286833,
        "std": 0.6941432105633398,
        "min": -1.0,
        "max": 1.0,
        "count": 8784324997,
    },
    "eastness": {
        "mean": 0.02390086174964258,
        "std": 0.7189956314423461,
        "min": -1.0,
        "max": 1.0,
        "count": 8784324997,
    },
}

# ---------------------------------------------------------------------
# Argument parsing and config setup
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser(
        description="Build uniform and spatial block CV SDM datasets for one or more species.")

    # INPUTS
    parser.add_argument("--occurrence-file", type=str, default=None, help="Path to a single species occurrence CSV.")

    parser.add_argument("--occurrence-dir", type=str, default=None, help="Directory containing multiple species occurrence CSVs.")

    parser.add_argument("--species-name-col", type=str, default="species")
    
    parser.add_argument("--lat-col", type=str, default="decimalLatitude")
    
    parser.add_argument("--lon-col", type=str, default="decimalLongitude")
    
    parser.add_argument("--source-col", type=str, default="Source", help="Optional source column, for example iNaturalist or GBIF.")

    # SHARED OUTPUTS AND RESOURCES
    parser.add_argument("--output-root", type=str, required=True, help="Root output directory.")

    parser.add_argument("--tileindex", type=str, required=True, help="NAIP tile index shapefile.")

    parser.add_argument("--naip-folder", type=str, required=True, help="Folder containing downloaded NAIP .tif files.")
    
    parser.add_argument("--worldclim-folder", type=str, required=True, help="Folder containing wc2.1_30s_bio_1.tif ... wc2.1_30s_bio_19.tif")

    parser.add_argument("--ghm-raster", type=str, required=True, help="Path to Global Human Modification raster.")

    # BACKGROUND SAMPLING
    parser.add_argument("--background-inner-buffer-km", type=float, default=0.0, help="Optional inner exclusion buffer radius in kilometers around presences. --background-buffer-km. Use 5 for a 5-50 km doughnut.")

    parser.add_argument("--background-buffer-km", type=float, default=50.0, help="Outer buffer radius in kilometers for sampling background points around presences.")

    parser.add_argument("--background-multiplier", type=float, default=3.0, help="Oversampling multiplier when drawing candidate background points.")

    parser.add_argument("--background-max-sampling-rounds", type=int, default=25, help="Maximum candidate-sampling rounds before failing. Increase if the doughnut sampling area is fragmented or very small.")
    
    # PROCESSING SETTINGS
    parser.add_argument("--chip-size", type=int, default=256, help="NAIP chip size in pixels. At 2 m resolution, 256 pixels is approximately 512 m.")

    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    parser.add_argument("--train-frac", type=float, default=0.70, help="Fraction of data for train split.")

    parser.add_argument("--val-frac", type=float, default=0.15, help="Fraction of data for validation split.")

    parser.add_argument("--test-frac", type=float, default=0.15, help="Fraction of data for test split.")

    parser.add_argument("--n-folds", type=int, default=5, help="Number of spatial CV folds.")
    
    parser.add_argument("--block-size-m", type=float, default=200000, help="Spatial block size in meters for block CV. Default = 200 km.")

    parser.add_argument("--spatial-thin-distance-m", type=float, default=800.0, help="Minimum allowed distance in meters between any presence or background samples after the combined PA dataset is built.")

    # TOPOGRAPHIC COVARIATE SETTINGS
    parser.add_argument("--topo-mode", choices=["none", "scalar", "chip", "both"], default="none", help="Whether to extract no topography, scalar summaries, topo chips, or both.")

    parser.add_argument("--topo-normalization-stats", type=str, default=None, help= "Optional path to JSON file containing pre-computed topographic normalization statistics. If omitted, embedded 3DEP 30 m statistics are used.")

    parser.add_argument("--disable-topo-normalization",action="store_true", help="If set, write raw topographic chip values and raw topo scalar columns")

    parser.add_argument("--dem-raster", type=str, default=None, help="Elevation raster.")

    parser.add_argument("--slope-raster", type=str, default=None, help="Slope raster in degrees.")

    parser.add_argument("--aspect-raster", type=str, default=None, help="Optional aspect raster in degrees, used only if northness/eastness rasters are omitted.")

    parser.add_argument("--northness-raster", type=str, default=None, help="Optional precomputed northness raster.")

    parser.add_argument("--eastness-raster", type=str, default=None, help="Optional precomputed eastness raster.")

    parser.add_argument("--topo-chip-size", type=int, default=64, help="Topographic chip size in pixels.")

    parser.add_argument("--topo-min-valid-frac", type=float, default=0.90, help="Minimum fraction of valid pixels required across all four topo layers.")

    args = parser.parse_args()
    
    validate_args(args, parser)
    
    return args

def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Validate arguments early so long batch jobs fail fast."""
    if not args.occurrence_file and not args.occurrence_dir:
        parser.error("Provide either --occurrence-file or --occurrence-dir.")

    if args.occurrence_file and args.occurrence_dir:
        parser.error("Provide only one of --occurrence-file or --occurrence-dir.")

    total = args.train_frac + args.val_frac + args.test_frac
    if not np.isclose(total, 1.0):
        parser.error("--train-frac + --val-frac + --test-frac must sum to 1.0.")

    if args.chip_size <= 0:
        parser.error("--chip-size must be > 0.")

    if args.n_folds < 2:
        parser.error("--n-folds must be >= 2.")

    if args.background_inner_buffer_km < 0:
        parser.error("--background-inner-buffer-km must be >= 0.")

    if args.background_buffer_km <= 0:
        parser.error("--background-buffer-km must be > 0.")

    if args.background_inner_buffer_km >= args.background_buffer_km:
        parser.error("--background-inner-buffer-km must be smaller than --background-buffer-km.")

    if args.background_multiplier <= 0:
        parser.error("--background-multiplier must be > 0.")

    if args.spatial_thin_distance_m < 0:
        parser.error("--spatial-thin-distance-m must be >= 0. Use 0 to disable adjacent point thinning.")

    if args.topo_mode != "none":
        if args.topo_chip_size <= 0:
            parser.error("--topo-chip-size must be > 0 when --topo-mode is not none.")

        if not (0.0 <= args.topo_min_valid_frac <= 1.0):
            parser.error("--topo-min-valid-frac must be in [0, 1].")

        missing = []
        if not args.dem_raster:
            missing.append("--dem-raster")
        if not args.slope_raster:
            missing.append("--slope-raster")
        if missing:
            parser.error(f"Missing required topo raster argument(s): {', '.join(missing)}")

        has_north_east = bool(args.northness_raster and args.eastness_raster)
        has_aspect = bool(args.aspect_raster)
        if not has_north_east and not has_aspect:
            parser.error(
                "When --topo-mode is not none, provide either both --northness-raster "
                "and --eastness-raster, or provide --aspect-raster."
            )

        if (args.northness_raster and not args.eastness_raster) or (args.eastness_raster and not args.northness_raster):
            parser.error("Provide both --northness-raster and --eastness-raster, or neither.")

        if args.topo_normalization_stats is not None and not Path(args.topo_normalization_stats).exists():
            parser.error(f"Topographic normalization JSON not found: {args.topo_normalization_stats}")

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

def slugify_species_name(name: str) -> str:
    """Convert species name to a consistent filesystem-safe slug."""
    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")

def display_name_from_slug(slug: str) -> str:
    """Convert a slug to a title-style label while preserving underscores."""
    return "_".join([x.capitalize() for x in slug.split("_") if x])

def infer_species_slug_from_filename(csv_path: Path) -> str:
    """Infer a species slug from common occurrence CSV filename patterns."""
    stem = csv_path.stem
    stem = re.sub(r"_thinned$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"_thin.*$", "", stem, flags=re.IGNORECASE)
    return slugify_species_name(stem)

# ---------------------------------------------------------------------
# I/O and geometry helpers
# ---------------------------------------------------------------------

def build_naip_file_index(naip_folder: str) -> dict[str, str]:
    """
    Recursively index all NAIP .tif/.TIF files under naip_folder.

    Returns
    -------
    dict
        Mapping from lowercase basename filename to full file path.

    Notes
    -----
    If duplicate basenames exist, later matches overwrite earlier ones. Keep
    your archive organized so basenames are unique within the active project.
    """
    naip_index: dict[str, str] = {}

    for root, _, files in os.walk(naip_folder):
        for filename in files:
            if filename.lower().endswith(".tif"):
                naip_index[filename.lower()] = os.path.join(root, filename)

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
        Full tile index in EPSG:4326.
    downloaded_tiles : GeoDataFrame
        Tile index subset with matched local NAIP files.
    naip_file_index : dict
        Mapping from lowercase basename to full path on disk.
    """
    tileindex = gpd.read_file(tileindex_fp).to_crs("EPSG:4326")

    if "filename" not in tileindex.columns:
        raise ValueError("Tile index must contain a 'filename' column.")

    tileindex["filename"] = tileindex["filename"].apply(os.path.basename)
    tileindex["filename_lower"] = tileindex["filename"].str.lower()

    naip_file_index = build_naip_file_index(naip_folder)

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

def load_occurrence_csv(
    csv_path: Path,
    species_name_col: str,
    lat_col: str,
    lon_col: str,
    source_col: str,
) -> gpd.GeoDataFrame:
    """Load a species occurrence CSV and standardize the required columns."""
    df = pd.read_csv(csv_path)

    missing = [c for c in [lat_col, lon_col] if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name} is missing required columns: {missing}")

    df = df.copy().rename(columns={lat_col: "lat", lon_col: "lon"})

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

    # Keep plausible WGS84 coordinates only.
    df = df[(df["lat"].between(-90, 90)) & (df["lon"].between(-180, 180))].copy()

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    )
    return gdf

# ---------------------------------------------------------------------
# Presence thinning and background sampling
# ---------------------------------------------------------------------

def deduplicate_presences_to_worldclim_cells(
    occurrences_gdf: gpd.GeoDataFrame,
    wc_reference_raster_fp: str,
    seed: int,
) -> gpd.GeoDataFrame:
    """
    Deduplicate occurrence points to WorldClim raster cells.

    One random presence is retained per raster cell. This prevents multiple
    records in the same coarse environmental cell from acting as pseudo-replicates.
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
        occ.geometry.y.values,
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

    A maximum batch limit prevents infinite loops when sampling areas are tiny
    or fragmented relative to their bounding boxes.
    """
    if polygon is None or polygon.is_empty:
        raise RuntimeError("Cannot sample random points from an empty polygon.")

    polygon = polygon.buffer(0)
    if polygon.is_empty:
        raise RuntimeError("Sampling polygon became empty after geometry repair.")

    rng = np.random.default_rng(seed)
    points: list[Point] = []
    minx, miny, maxx, maxy = polygon.bounds

    n_batches = 0
    while len(points) < n and n_batches < max_batches:
        needed = n - len(points)
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

def sample_background_points(
    presence_gdf: gpd.GeoDataFrame,
    downloaded_tiles_gdf: gpd.GeoDataFrame,
    wc_reference_raster_fp: str,
    background_multiplier: float,
    seed: int,
    buffer_km: float,
    inner_buffer_km: float = 0.0,
    max_sampling_rounds: int = 25,
) -> gpd.GeoDataFrame:
    """
    Sample one background point per presence using an optional doughnut method.

    Background points are constrained to:
      - downloaded NAIP footprint
      - farther than inner_buffer_km from any presence
      - within buffer_km of any presence
      - unique WorldClim cells
      - non-overlapping with presence WorldClim cells

    Example:
      inner_buffer_km = 5
      buffer_km = 50

    gives a 5-50 km annulus around known presences.
    """
    n_background = len(presence_gdf)
    if n_background == 0:
        raise ValueError("No presence points available after thinning.")

    if inner_buffer_km < 0:
        raise ValueError("--background-inner-buffer-km must be >= 0.")

    if inner_buffer_km >= buffer_km:
        raise ValueError(
            f"Inner buffer ({inner_buffer_km} km) must be smaller than outer buffer ({buffer_km} km)."
        )

    with rasterio.open(wc_reference_raster_fp) as src:
        affine = src.transform
        raster_crs = src.crs

    # 1. Build doughnut sampling area in a projected CRS.
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
        gpd.GeoSeries([sampling_zone_proj], crs="EPSG:5070").to_crs("EPSG:4326").iloc[0]
    )

    # 2. Restrict to downloaded NAIP footprint.
    footprint = downloaded_tiles_gdf.to_crs("EPSG:4326").geometry.union_all()
    sampling_area = footprint.intersection(sampling_zone_wgs84).buffer(0)

    if sampling_area.is_empty:
        raise RuntimeError(
            f"Sampling area is empty after intersecting NAIP footprint with "
            f"{inner_buffer_km}-{buffer_km} km presence doughnut."
        )

    sampling_area_km2 = (
        gpd.GeoSeries([sampling_area], crs="EPSG:4326").to_crs("EPSG:5070").area.iloc[0]
        / 1_000_000.0
    )

    print(f"Background sampling rule: {inner_buffer_km:g}-{buffer_km:g} km from presences")
    print(f"Background sampling area: {sampling_area_km2:,.1f} km²")

    # 3. Identify WorldClim cells already occupied by presences.
    pres_for_cells = presence_gdf.to_crs(raster_crs).copy()
    rows_pr, cols_pr = rasterio.transform.rowcol(
        affine,
        pres_for_cells.geometry.x.values,
        pres_for_cells.geometry.y.values,
    )
    pres_for_cells["cell_id"] = [f"{r}_{c}" for r, c in zip(rows_pr, cols_pr)]
    presence_cells = set(pres_for_cells["cell_id"])

    # 4. Iteratively sample candidates until enough unique valid cells exist.
    valid_candidate_chunks = []
    n_candidates_per_round = int(math.ceil(n_background * background_multiplier))
    pooled = None

    for round_i in range(max_sampling_rounds):
        round_seed = seed + round_i
        candidate_points = random_points_in_polygon(
            sampling_area,
            n=n_candidates_per_round,
            seed=round_seed,
        )

        candidates = gpd.GeoDataFrame(geometry=candidate_points, crs="EPSG:4326").to_crs(raster_crs)

        rows_bg, cols_bg = rasterio.transform.rowcol(
            affine,
            candidates.geometry.x.values,
            candidates.geometry.y.values,
        )
        candidates["cell_id"] = [f"{r}_{c}" for r, c in zip(rows_bg, cols_bg)]
        candidates = candidates[~candidates["cell_id"].isin(presence_cells)].copy()

        if not candidates.empty:
            valid_candidate_chunks.append(candidates)

        if valid_candidate_chunks:
            pooled = pd.concat(valid_candidate_chunks, ignore_index=True)
            pooled = gpd.GeoDataFrame(pooled, geometry="geometry", crs=raster_crs)
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
        available = len(pooled) if pooled is not None else 0
        raise RuntimeError(
            f"Only {available} valid unique background cells available after "
            f"{max_sampling_rounds} rounds, but {n_background} are needed. "
            "Try increasing --background-multiplier, increasing --background-buffer-km, "
            "or decreasing --background-inner-buffer-km."
        )

    if pooled is None or len(pooled) < n_background:
        raise RuntimeError("Insufficient background points after candidate sampling.")

    # 5. Final sample: one background point per presence.
    background_sampled = pooled.sample(n=n_background, random_state=seed).copy()

    bg_for_dist = background_sampled.to_crs("EPSG:5070").copy()
    presence_union_proj = presence_proj.geometry.union_all()
    background_sampled["nearest_presence_km"] = bg_for_dist.geometry.distance(presence_union_proj) / 1000.0

    background_sampled = background_sampled.to_crs("EPSG:4326")
    background_sampled["lat"] = background_sampled.geometry.y
    background_sampled["lon"] = background_sampled.geometry.x
    background_sampled["source"] = "background"
    background_sampled["background_sampling_rule"] = f"doughnut_{inner_buffer_km:g}_{buffer_km:g}_km"

    print(
        "Final background nearest-presence distance summary, km: "
        f"min={background_sampled['nearest_presence_km'].min():.2f}, "
        f"median={background_sampled['nearest_presence_km'].median():.2f}, "
        f"max={background_sampled['nearest_presence_km'].max():.2f}"
    )

    return background_sampled

def spatially_thin_points(
    points_gdf: gpd.GeoDataFrame,
    min_distance_m: float,
    seed: int,
    projected_crs: str = "EPSG:5070",
) -> gpd.GeoDataFrame:
    """
    Greedily thin all points so no two retained samples are within min_distance_m.

    This is intentionally simple and conservative. It treats presences and
    backgrounds together as one pool, so class balance may change slightly after
    thinning. That is acceptable here because the goal is to avoid inflated
    performance metrics caused by overlapping NAIP/topographic chips.

    Parameters
    ----------
    points_gdf : GeoDataFrame
        Combined presence/background points with a valid CRS.
    min_distance_m : float
        Minimum allowed distance between retained sample centers, in meters.
        Use 0 to disable thinning.
    seed : int
        Random seed controlling the order in which candidate samples are visited.
    projected_crs : str
        Projected CRS used for distance calculations. EPSG:5070 is appropriate
        for CONUS-scale analyses.

    Returns
    -------
    GeoDataFrame
        Spatially thinned points in the original CRS.
    """
    if points_gdf.empty:
        return points_gdf.copy()

    if min_distance_m <= 0:
        out = points_gdf.copy()
        out["spatial_thin_distance_m"] = 0.0
        return out

    if points_gdf.crs is None:
        raise ValueError("points_gdf must have a CRS before spatial thinning.")

    gdf = points_gdf.copy().reset_index(drop=True)
    gdf_proj = gdf.to_crs(projected_crs)

    rng = np.random.default_rng(seed)
    order = np.arange(len(gdf_proj))
    rng.shuffle(order)

    sindex = gdf_proj.sindex

    available = np.ones(len(gdf_proj), dtype=bool)
    keep_indices: list[int] = []

    for idx in order:
        if not available[idx]:
            continue

        keep_indices.append(int(idx))

        # Drop all currently available points within the exclusion radius.
        exclusion_geom = gdf_proj.geometry.iloc[idx].buffer(min_distance_m)
        nearby = sindex.query(exclusion_geom, predicate="intersects")
        available[nearby] = False

    keep_indices = sorted(keep_indices)

    out = gdf.iloc[keep_indices].copy().reset_index(drop=True)
    out["spatial_thin_distance_m"] = float(min_distance_m)

    print(
        "Spatial thinning complete: "
        f"retained {len(out):,} of {len(gdf):,} samples "
        f"({len(gdf) - len(out):,} removed) using "
        f"{min_distance_m:,.1f} m minimum spacing."
    )

    print("Class counts before spatial thinning:")
    print(gdf["presence"].value_counts().sort_index())

    print("Class counts after spatial thinning:")
    print(out["presence"].value_counts().sort_index())

    return gpd.GeoDataFrame(out, geometry="geometry", crs=points_gdf.crs)

def attach_naip_filename_or_url(
    points_gdf: gpd.GeoDataFrame,
    tileindex_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Spatially join points to the tileindex to preserve filename/url metadata."""
    cols = ["geometry"]
    for c in ["filename", "url"]:
        if c in tileindex_gdf.columns:
            cols.append(c)

    joined = gpd.sjoin(
        points_gdf,
        tileindex_gdf[cols],
        how="left",
        predicate="within",
    ).drop(columns=["index_right"], errors="ignore")

    return joined

def build_presence_background_points(
    presence_gdf: gpd.GeoDataFrame,
    background_gdf: gpd.GeoDataFrame,
    tileindex_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Combine presences and background into a single GeoDataFrame."""
    pres = presence_gdf.copy().to_crs("EPSG:4326")
    pres["lat"] = pres.geometry.y
    pres["lon"] = pres.geometry.x
    pres["presence"] = 1

    bg = background_gdf.copy().to_crs("EPSG:4326")
    bg["presence"] = 0

    pres = attach_naip_filename_or_url(pres, tileindex_gdf)
    bg = attach_naip_filename_or_url(bg, tileindex_gdf)

    keep_cols = ["species", "source", "lat", "lon", "presence", "geometry"]
    for extra in [
        "filename",
        "url",
        "cell_id",
        "nearest_presence_km",
        "background_sampling_rule",
    ]:
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

def stratified_train_val_test_split(
    df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> pd.DataFrame:
    """Stratified split by presence class so class balance is maintained."""
    rng = np.random.default_rng(seed)
    chunks = []

    for _, sub in df.groupby("presence"):
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

        if len(test_idx) > 0:
            tmp = df.loc[test_idx].copy()
            tmp["split"] = "test"
            chunks.append(tmp)

    out = pd.concat(chunks, ignore_index=False).sort_index().reset_index(drop=True)
    return out

def assign_spatial_blocks(
    points_gdf: gpd.GeoDataFrame,
    block_size_m: float,
    n_folds: int,
    seed: int,
) -> gpd.GeoDataFrame:
    """
    Assign spatial blocks and folds from projected coordinates.

    Uses EPSG:5070, CONUS Albers Equal Area, for U.S.-scale blocking.
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

    block_to_fold = {block_id: (i % n_folds) + 1 for i, block_id in enumerate(shuffled)}
    gdf["fold"] = gdf["block_id"].map(block_to_fold).astype(int)

    return gdf.to_crs("EPSG:4326")

def make_spatial_cv_rounds(
    points_df: pd.DataFrame,
    n_folds: int,
    seed: int,
) -> pd.DataFrame:
    """
    For each held-out fold:
      - test = held-out fold
      - train/val = remaining folds, split stratified by presence
    """
    rounds = []

    for heldout in range(1, n_folds + 1):
        train_val = points_df[points_df["fold"] != heldout].copy()
        test = points_df[points_df["fold"] == heldout].copy()

        split_train_val = stratified_train_val_test_split(
            train_val.drop(columns=["fold"], errors="ignore"),
            train_frac=0.70,
            val_frac=0.30,
            test_frac=0.0,
            seed=seed,
        )

        test = test.copy()
        test["split"] = "test"

        split_train_val = split_train_val.merge(
            train_val[["lat", "lon", "fold", "block_id"]].drop_duplicates(),
            on=["lat", "lon"],
            how="left",
        )

        round_df = pd.concat([split_train_val, test], ignore_index=True)
        round_df["cv_round"] = heldout
        rounds.append(round_df)

    return pd.concat(rounds, ignore_index=True)

# ---------------------------------------------------------------------
# Climate, GHM, and topographic helpers
# ---------------------------------------------------------------------
def add_normalized_lat_lon(df: pd.DataFrame) -> pd.DataFrame:
    """Add dataset-level normalized latitude and longitude columns."""
    df = df.copy()
    lat_std = df["lat"].std()
    lon_std = df["lon"].std()

    if lat_std == 0 or not np.isfinite(lat_std):
        df["lat_norm"] = 0.0
    else:
        df["lat_norm"] = (df["lat"] - df["lat"].mean()) / lat_std

    if lon_std == 0 or not np.isfinite(lon_std):
        df["lon_norm"] = 0.0
    else:
        df["lon_norm"] = (df["lon"] - df["lon"].mean()) / lon_std

    return df

def compute_worldclim_stats(worldclim_folder: str, study_geom, buffer_deg: float = 0.01) -> dict:
    """Compute mean/std for each WorldClim variable inside the study footprint."""
    stats: dict[str, dict[str, float]] = {}

    buffered_geom = gpd.GeoSeries([study_geom], crs="EPSG:4326").buffer(buffer_deg).iloc[0]
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

def open_worldclim_datasets(worldclim_folder: str) -> dict[str, rasterio.io.DatasetReader]:
    """Open all WorldClim rasters once for repeated point sampling."""
    return {
        var: rasterio.open(os.path.join(worldclim_folder, f"{var}.tif"))
        for var in WC_VARS
    }

def extract_worldclim_vars_for_point(
    lon: float,
    lat: float,
    wc_datasets: dict,
    normalization_stats: dict,
) -> dict[str, float]:
    """Extract normalized WorldClim values for one point."""
    vals: dict[str, float] = {}
    for varname, ds in wc_datasets.items():
        raw_value = list(ds.sample([(lon, lat)]))[0][0]
        mean = normalization_stats[varname]["mean"]
        std = normalization_stats[varname]["std"]
        vals[varname] = float((raw_value - mean) / std)
    return vals

def extract_ghm_for_point(lon: float, lat: float, ghm_ds) -> float:
    """Extract a clamped GHM value for one point."""
    raw_value = list(ghm_ds.sample([(lon, lat)]))[0][0]
    if not np.isfinite(raw_value):
        return np.nan
    return float(np.clip(raw_value, 0.0, 1.0))

def is_valid_env_record(env_vars: dict, min_val: float = -100, max_val: float = 100) -> bool:
    """Check that baseline environmental variables are finite and plausible."""
    for v in ALL_ENV_VARS:
        val = env_vars.get(v, np.nan)
        if not np.isfinite(val):
            return False
        if val < min_val or val > max_val:
            return False
    return True

def load_topo_normalization_stats(stats_fp: str | None) -> dict:
    """
    Load and validate topographic normalization statistics.

    If stats_fp is None, returns the embedded 3DEP 30 m statistics. The JSON
    schema should be:

        {
          "elevation": {"mean": ..., "std": ...},
          "slope":     {"mean": ..., "std": ...},
          "northness": {"mean": ..., "std": ...},
          "eastness":  {"mean": ..., "std": ...}
        }
    """
    if stats_fp is None:
        stats = DEFAULT_TOPO_NORM_STATS_3DEP_30M.copy()
    else:
        with open(stats_fp, "r", encoding="utf-8") as f:
            stats = json.load(f)

    validate_topo_normalization_stats(stats)
    return stats

def validate_topo_normalization_stats(stats: dict) -> None:
    """Ensure all required topo normalization statistics exist and are usable."""
    missing = [name for name in TOPO_IMAGE_BANDS if name not in stats]
    if missing:
        raise ValueError(f"Topographic normalization stats missing variables: {missing}")

    for name in TOPO_IMAGE_BANDS:
        for key in ["mean", "std"]:
            if key not in stats[name]:
                raise ValueError(f"Topographic normalization stats for {name} missing key: {key}")
        mean = float(stats[name]["mean"])
        std = float(stats[name]["std"])
        if not np.isfinite(mean):
            raise ValueError(f"Invalid topographic mean for {name}: {mean}")
        if not np.isfinite(std) or std <= 0:
            raise ValueError(f"Invalid topographic std for {name}: {std}")

def normalize_topo_stack(stack: np.ndarray, topo_norm_stats: dict) -> np.ndarray:
    """
    Z-score normalize a 4-band topo stack using global/precomputed stats.

    Parameters
    ----------
    stack : np.ndarray
        Raw stack with shape (4, H, W) and band order matching TOPO_IMAGE_BANDS.
    topo_norm_stats : dict
        Precomputed stats keyed by elevation/slope/northness/eastness.

    Returns
    -------
    np.ndarray
        Normalized float32 stack with NaN preserved where input was invalid.
    """
    out = stack.astype(np.float32, copy=True)
    for band_idx, name in enumerate(TOPO_IMAGE_BANDS): # elevation, slope, northness, eastness
        mean = float(topo_norm_stats[name]["mean"])
        std = float(topo_norm_stats[name]["std"])
        valid = np.isfinite(out[band_idx])
        out[band_idx, valid] = (out[band_idx, valid] - mean) / std
    return out

def summarize_topo_stack(stack: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    """
    Summarize a topo stack using the same units as the supplied stack.

    If stack has already been normalized, the returned scalar summaries are
    also normalized. This keeps topo scalar columns ready for deep learning.
    """
    stats: dict[str, float] = {"topo_valid_frac": float(valid.mean())}

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
        stats["topo_valid_frac"] = 0.0

    return stats

def open_topo_sources(args: argparse.Namespace) -> dict[str, rasterio.io.DatasetReader] | None:
    """Open topographic rasters once and return a dictionary of rasterio datasets."""
    if args.topo_mode == "none":
        return None

    topo_sources: dict[str, rasterio.io.DatasetReader] = {
        "elevation": rasterio.open(args.dem_raster),
        "slope": rasterio.open(args.slope_raster),
    }

    if args.northness_raster and args.eastness_raster:
        topo_sources["northness"] = rasterio.open(args.northness_raster)
        topo_sources["eastness"] = rasterio.open(args.eastness_raster)
    elif args.aspect_raster:
        topo_sources["aspect"] = rasterio.open(args.aspect_raster)

    return topo_sources

def close_dataset_dict(datasets: dict | None) -> None:
    """Close all rasterio datasets in a dictionary."""
    if not datasets:
        return
    for ds in datasets.values():
        ds.close()

# ---------------------------------------------------------------------
# NAIP chip extraction
# ---------------------------------------------------------------------

def extract_naip_chip_for_point(
    lon: float,
    lat: float,
    out_fp: str,
    chip_size: int,
    tileindex_gdf: gpd.GeoDataFrame,
    naip_file_index: dict,
) -> bool:
    """Extract a NAIP chip around a point from one or more overlapping tiles."""
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
            max(chip_top, chip_bottom),
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
            # Rare cross-CRS chips are skipped because a single merge in native CRS
            # would be invalid without explicit reprojection.
            return False

        xres, yres = datasets[0].res

        mosaic, _ = merge(
            datasets,
            bounds=(
                min(chip_left, chip_right),
                min(chip_top, chip_bottom),
                max(chip_left, chip_right),
                max(chip_top, chip_bottom),
            ),
            res=(xres, yres),
            nodata=0,
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
            yres,
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
            transform=out_transform,
            compress="deflate",
            tiled=True,
            BIGTIFF="IF_SAFER",
        ) as dst:
            dst.write(chip)

        return True

    finally:
        for ds in datasets:
            ds.close()

# ---------------------------------------------------------------------
# Topo chip extraction
# ---------------------------------------------------------------------

def extract_topo_for_naip_chip(
    naip_chip_fp: str,
    topo_sources: dict,
    out_fp: str | None,
    topo_chip_size: int,
    topo_norm_stats: dict | None,
    normalize_topo: bool,
) -> dict[str, float]:
    """
    Extract a topographic chip over the same spatial footprint as a NAIP chip.

    Output band order:
      1 elevation
      2 slope
      3 northness
      4 eastness

    By default, the written chip and returned scalar summaries are z-score
    normalized with topo_norm_stats. Set normalize_topo=False to preserve raw
    physical units.

    Note: We use an affine transformation to reproject/ resample the topo raster to the NAIP chip's spatial footprint.
    So a 256 x 256 NAIP tile at 2-meter resolution (512 x 512 meters) will yield a topo chip of 64 x 64 pixels at ~8-meter resolution, regardless of the original topo raster resolutions. 
    This ensures consistent spatial coverage and alignment between the NAIP and topo chips, while allowing for flexible input topo sources.
    """
    out_nodata = -9999.0

    with rasterio.open(naip_chip_fp) as naip:
        dst_crs = naip.crs
        dst_transform = naip.transform * Affine.scale(
            naip.width / topo_chip_size,
            naip.height / topo_chip_size,
        )

    layers: dict[str, np.ndarray] = {}

    for name in ["elevation", "slope", "northness", "eastness"]:
        src = topo_sources.get(name)
        if src is None:
            continue

        dst = np.full((topo_chip_size, topo_chip_size), np.nan, dtype=np.float32)

        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

        layers[name] = dst

    # Fallback only if northness/eastness rasters were not supplied.
    # Prefer precomputed northness/eastness rasters when possible.
    if ("northness" not in layers or "eastness" not in layers) and topo_sources.get("aspect") is not None:
        aspect = topo_sources["aspect"]
        a = np.full((topo_chip_size, topo_chip_size), np.nan, dtype=np.float32)

        reproject(
            source=rasterio.band(aspect, 1),
            destination=a,
            src_transform=aspect.transform,
            src_crs=aspect.crs,
            src_nodata=aspect.nodata,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=Resampling.nearest,
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

    raw_stack = np.stack([layers[k].astype(np.float32) for k in TOPO_IMAGE_BANDS], axis=0)
    valid = np.isfinite(raw_stack).all(axis=0)
    valid_frac = float(valid.mean())

    if normalize_topo:
        if topo_norm_stats is None:
            raise RuntimeError("normalize_topo=True but topo_norm_stats is None.")
        model_stack = normalize_topo_stack(raw_stack, topo_norm_stats)
    else:
        model_stack = raw_stack

    stats = summarize_topo_stack(model_stack, valid)
    stats["topo_valid_frac"] = valid_frac
    stats["topo_normalized"] = bool(normalize_topo)

    if out_fp is not None:
        out_stack = np.where(np.isfinite(model_stack), model_stack, out_nodata).astype(np.float32)

        block_size = min(128, topo_chip_size)
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
            "BIGTIFF": "IF_SAFER",
        }

        # Tiled GeoTIFF block sizes must be multiples of 16.
        if block_size >= 16 and block_size % 16 == 0:
            profile.update({
                "tiled": True,
                "blockxsize": block_size,
                "blockysize": block_size,
            })

        Path(out_fp).parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(out_fp, "w", **profile) as dst:
            dst.write(out_stack)
            for band_idx, name in enumerate(TOPO_IMAGE_BANDS, start=1):
                dst.set_band_description(band_idx, name)

    return stats

# ---------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------

def build_dataset_csv(
    points_df: pd.DataFrame,
    topo_sources: dict | None,
    topo_mode: str,
    topo_chip_size: int,
    topo_min_valid_frac: float,
    topo_norm_stats: dict | None,
    normalize_topo: bool,
    dataset_root: Path,
    tileindex_gdf: gpd.GeoDataFrame,
    naip_file_index: dict,
    wc_datasets: dict,
    wc_stats: dict,
    ghm_ds,
    chip_size: int,
    species_label: str,
    suffix: str = "",
    cv_round: int | None = None,
) -> pd.DataFrame:
    """
    Extract NAIP chips, optional topo chips/scalars, environmental vars, and
    save the final modeling dataset CSV.
    """
    chip_dir = dataset_root / "images"
    chip_dir.mkdir(parents=True, exist_ok=True)

    topo_chip_dir = dataset_root / "topo_chips"
    if topo_mode in {"chip", "both"}:
        topo_chip_dir.mkdir(parents=True, exist_ok=True)

    records = []

    for idx, row in tqdm(points_df.iterrows(), total=len(points_df), desc=f"{species_label}{suffix}"):
        lat = float(row["lat"])
        lon = float(row["lon"])
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
            naip_file_index=naip_file_index,
        )
        if not ok:
            continue

        env = extract_worldclim_vars_for_point(lon, lat, wc_datasets, wc_stats)
        env["ghm"] = extract_ghm_for_point(lon, lat, ghm_ds)

        if not is_valid_env_record(env):
            continue

        topo_chip_rel = None
        topo_stats: dict[str, float] = {}
        if topo_mode != "none":
            if topo_sources is None:
                raise RuntimeError("topo_mode is not none but topo_sources is None.")

            topo_chip_fp = None
            if topo_mode in {"chip", "both"}:
                topo_chip_fp = topo_chip_dir / chip_fn.replace("chip_", "topo_chip_")
                topo_chip_rel = os.path.relpath(topo_chip_fp, dataset_root)

            topo_stats = extract_topo_for_naip_chip(
                naip_chip_fp=str(chip_fp),
                topo_sources=topo_sources,
                out_fp=str(topo_chip_fp) if topo_chip_fp else None,
                topo_chip_size=topo_chip_size,
                topo_norm_stats=topo_norm_stats,
                normalize_topo=normalize_topo,
            )

            if topo_stats.get("topo_valid_frac", 0.0) < topo_min_valid_frac:
                # Skip rows whose topographic coverage is too incomplete.
                continue

        rec = {
            "sample_id": sample_id,
            "chip_path": os.path.relpath(chip_fp, dataset_root),
            "topo_chip_path": topo_chip_rel,
            "split": split,
            "presence": presence,
            "lat": lat,
            "lon": lon,
            "source": source,
        }

        for optional_col in [
            "nearest_presence_km",
            "background_sampling_rule",
            "filename",
            "url",
            "block_id",
            "fold",
            "spatial_thin_distance_m",
        ]:
            if optional_col in row.index:
                rec[optional_col] = row.get(optional_col)

        if cv_round is not None:
            rec["cv_round"] = cv_round

        rec.update(env)
        if topo_mode in {"scalar", "both", "chip"}:
            rec.update(topo_stats)

        records.append(rec)

    df = pd.DataFrame(records)

    if df.empty:
        return df

    # Keep only rows whose NAIP chip exists. Also validate topo chip paths when
    # they are expected to be present.
    df["chip_path_abs"] = df["chip_path"].apply(lambda p: os.path.join(dataset_root, p))
    df = df[df["chip_path_abs"].apply(os.path.exists)].drop(columns=["chip_path_abs"])

    if topo_mode in {"chip", "both"} and "topo_chip_path" in df.columns:
        df["topo_chip_path_abs"] = df["topo_chip_path"].apply(lambda p: os.path.join(dataset_root, p))
        df = df[df["topo_chip_path_abs"].apply(os.path.exists)].drop(columns=["topo_chip_path_abs"])

    df = add_normalized_lat_lon(df)
    return df


def write_json(obj: dict, out_fp: Path) -> None:
    """Write a small JSON metadata file."""
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    with open(out_fp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# ---------------------------------------------------------------------
# Species-level pipeline
# ---------------------------------------------------------------------

def process_species(
    csv_path: Path,
    args: argparse.Namespace,
    tileindex_gdf: gpd.GeoDataFrame,
    downloaded_tiles_gdf: gpd.GeoDataFrame,
    wc_stats: dict,
    topo_norm_stats: dict | None,
    naip_file_index: dict,
) -> None:
    species_slug = infer_species_slug_from_filename(csv_path)
    species_label = display_name_from_slug(species_slug)

    print("\n" + "=" * 80)
    print(f"Processing species: {species_slug}")
    print("=" * 80)

    # Output roots
    project_root = Path(args.output_root)
    datasets_root = project_root / f"{species_label}_Datasets"
    occurrences_root = project_root / f"{species_label}_US_Occurrences"

    datasets_root.mkdir(parents=True, exist_ok=True)
    occurrences_root.mkdir(parents=True, exist_ok=True)

    uniform_name = f"{species_label}_US_Uniform_PA_NAIP_{args.chip_size}_topo_norm_may2026"
    blockcv_name = f"{species_label}_US_BlockCV_PA_NAIP_{args.chip_size}_topo_norm_may2026"

    uniform_root = datasets_root / uniform_name
    blockcv_root = datasets_root / blockcv_name
    uniform_root.mkdir(parents=True, exist_ok=True)
    blockcv_root.mkdir(parents=True, exist_ok=True)

    if args.topo_mode != "none":
        topo_metadata = {
            "topo_mode": args.topo_mode,
            "topo_chip_size": args.topo_chip_size,
            "topo_min_valid_frac": args.topo_min_valid_frac,
            "topo_normalized": not args.disable_topo_normalization,
            "topo_band_order": TOPO_IMAGE_BANDS,
            "topo_scalar_columns": TOPO_SCALAR_COLUMNS,
            "topo_normalization_stats": topo_norm_stats,
        }
        write_json(topo_metadata, uniform_root / "topography_metadata.json")
        write_json(topo_metadata, blockcv_root / "topography_metadata.json")

    # WorldClim reference raster
    wc_reference = os.path.join(args.worldclim_folder, "wc2.1_30s_bio_1.tif")

    # Load and clean occurrences
    occurrences = load_occurrence_csv(
        csv_path,
        species_name_col=args.species_name_col,
        lat_col=args.lat_col,
        lon_col=args.lon_col,
        source_col=args.source_col,
    )

    print(f"Loaded {len(occurrences):,} raw occurrences from {csv_path.name}")

    occurrences = occurrences.drop_duplicates(subset=["lat", "lon"]).copy()
    print(f"After exact lat/lon deduplication: {len(occurrences):,}")

    presences = deduplicate_presences_to_worldclim_cells(
        occurrences_gdf=occurrences,
        wc_reference_raster_fp=wc_reference,
        seed=args.seed,
    )
    print(f"After WorldClim-cell thinning: {len(presences):,}")

    backgrounds = sample_background_points(
        presence_gdf=presences,
        downloaded_tiles_gdf=downloaded_tiles_gdf,
        wc_reference_raster_fp=wc_reference,
        background_multiplier=args.background_multiplier,
        buffer_km=args.background_buffer_km,
        inner_buffer_km=args.background_inner_buffer_km,
        max_sampling_rounds=args.background_max_sampling_rounds,
        seed=args.seed,
    )
    print(f"Sampled {len(backgrounds):,} background points")

    pa_points = build_presence_background_points(
    presence_gdf=presences,
    background_gdf=backgrounds,
    tileindex_gdf=tileindex_gdf,)

    print(f"Combined presence/background dataset before spatial thinning: {len(pa_points):,}")

    pa_points = spatially_thin_points(
        points_gdf=pa_points,
        min_distance_m=args.spatial_thin_distance_m,
        seed=args.seed,
        projected_crs="EPSG:5070",
    )

    print(f"Combined presence/background dataset after spatial thinning: {len(pa_points):,}")

    pb_points_csv = occurrences_root / f"{species_label}_US_Presence_Background_Points.csv"
    pa_points.drop(columns=["geometry"]).to_csv(pb_points_csv, index=False)
    print(f"Saved: {pb_points_csv}")

    # Uniform train/val/test points.
    uniform_points = stratified_train_val_test_split(
        pa_points.drop(columns=["geometry"]),
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )

    uniform_points_gdf = gpd.GeoDataFrame(
        uniform_points,
        geometry=gpd.points_from_xy(uniform_points["lon"], uniform_points["lat"]),
        crs="EPSG:4326",
    )

    uniform_points_csv = uniform_root / f"{species_label}_Pres_Bg_US_Uniform_Train_Val_Test_Points.csv"
    uniform_points_gdf.drop(columns=["geometry"]).to_csv(uniform_points_csv, index=False)
    print(f"Saved: {uniform_points_csv}")

    # Spatial block CV points.
    pa_points_blocks = assign_spatial_blocks(
        gpd.GeoDataFrame(pa_points.copy(), geometry="geometry", crs="EPSG:4326"),
        block_size_m=args.block_size_m,
        n_folds=args.n_folds,
        seed=args.seed,
    )

    block_points_csv = blockcv_root / f"{species_label}_Pres_Bg_US_SpatialCV_Train_Val_Test_Points.csv"
    cv_rounds = make_spatial_cv_rounds(
        points_df=pa_points_blocks.drop(columns=["geometry"]),
        n_folds=args.n_folds,
        seed=args.seed,
    )
    cv_rounds.to_csv(block_points_csv, index=False)
    print(f"Saved: {block_points_csv}")

    for round_num in range(1, args.n_folds + 1):
        fold_df = cv_rounds[cv_rounds["cv_round"] == round_num].copy()
        fold_csv = blockcv_root / f"{species_label}_Pres_Bg_US_SpatialCV_Blocks_Fold_{round_num}.csv"
        fold_df.to_csv(fold_csv, index=False)
        print(f"Saved: {fold_csv}")

    # Open shared rasters once for final dataset extraction.
    wc_datasets = open_worldclim_datasets(args.worldclim_folder)
    ghm_ds = rasterio.open(args.ghm_raster)
    topo_sources = open_topo_sources(args)

    try:
        normalize_topo = args.topo_mode != "none" and not args.disable_topo_normalization

        # Uniform final dataset.
        uniform_final = build_dataset_csv(
            points_df=uniform_points,
            topo_sources=topo_sources,
            topo_mode=args.topo_mode,
            topo_chip_size=args.topo_chip_size,
            topo_min_valid_frac=args.topo_min_valid_frac,
            topo_norm_stats=topo_norm_stats,
            normalize_topo=normalize_topo,
            dataset_root=uniform_root,
            tileindex_gdf=tileindex_gdf,
            naip_file_index=naip_file_index,
            wc_datasets=wc_datasets,
            wc_stats=wc_stats,
            ghm_ds=ghm_ds,
            chip_size=args.chip_size,
            species_label=species_label,
            suffix="_uniform",
            cv_round=None,
        )

        uniform_dataset_csv = uniform_root / f"{species_label}_Pres_Bg_US_Uniform_Train_Val_Test_Dataset.csv"
        uniform_final.to_csv(uniform_dataset_csv, index=False)
        print(f"Saved: {uniform_dataset_csv}")
        print(f"Uniform final rows retained after chip/env/topo filtering: {len(uniform_final):,}")

        # Block CV final datasets.
        for round_num in range(1, args.n_folds + 1):
            round_df = cv_rounds[cv_rounds["cv_round"] == round_num].copy()
            cv_dir = blockcv_root / f"CV_{round_num}"
            cv_dir.mkdir(parents=True, exist_ok=True)

            if args.topo_mode != "none":
                write_json(
                    {
                        "topo_mode": args.topo_mode,
                        "topo_chip_size": args.topo_chip_size,
                        "topo_min_valid_frac": args.topo_min_valid_frac,
                        "topo_normalized": not args.disable_topo_normalization,
                        "topo_band_order": TOPO_IMAGE_BANDS,
                        "topo_scalar_columns": TOPO_SCALAR_COLUMNS,
                        "topo_normalization_stats": topo_norm_stats,
                    },
                    cv_dir / "topography_metadata.json",
                )

            cv_final = build_dataset_csv(
                points_df=round_df,
                topo_sources=topo_sources,
                topo_mode=args.topo_mode,
                topo_chip_size=args.topo_chip_size,
                topo_min_valid_frac=args.topo_min_valid_frac,
                topo_norm_stats=topo_norm_stats,
                normalize_topo=normalize_topo,
                dataset_root=cv_dir,
                tileindex_gdf=tileindex_gdf,
                naip_file_index=naip_file_index,
                wc_datasets=wc_datasets,
                wc_stats=wc_stats,
                ghm_ds=ghm_ds,
                chip_size=args.chip_size,
                species_label=species_label,
                suffix=f"_CV{round_num}",
                cv_round=round_num,
            )

            cv_dataset_csv = cv_dir / f"{species_label}_Train_Val_Test_US_BlockCV_{round_num}.csv"
            cv_final.to_csv(cv_dataset_csv, index=False)
            print(f"Saved: {cv_dataset_csv}")
            print(f"CV {round_num} final rows retained after chip/env/topo filtering: {len(cv_final):,}")

    finally:
        close_dataset_dict(wc_datasets)
        ghm_ds.close()
        close_dataset_dict(topo_sources)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    tileindex_gdf, downloaded_tiles_gdf, naip_file_index = load_tileindex(args.tileindex, args.naip_folder)
    downloaded_union = downloaded_tiles_gdf.geometry.union_all()

    print("Computing WorldClim normalization statistics once from downloaded NAIP footprint...")
    wc_stats = compute_worldclim_stats(args.worldclim_folder, downloaded_union)

    topo_norm_stats = None
    if args.topo_mode != "none":
        topo_norm_stats = load_topo_normalization_stats(args.topo_normalization_stats)
        if args.disable_topo_normalization:
            print("Topographic normalization disabled: topo chips/scalars will be raw physical values.")
        else:
            source = args.topo_normalization_stats or "embedded DEFAULT_TOPO_NORM_STATS_3DEP_30M"
            print(f"Topographic normalization enabled using: {source}")

    if args.occurrence_file:
        csv_files = [Path(args.occurrence_file)]
    else:
        csv_files = sorted(Path(args.occurrence_dir).glob("*.csv"))

    if not csv_files:
        raise RuntimeError("No occurrence CSV files found.")

    print(f"Found {len(csv_files)} occurrence CSV file(s).")

    failures = []
    for csv_path in csv_files:
        try:
            process_species(
                csv_path=csv_path,
                args=args,
                tileindex_gdf=tileindex_gdf,
                downloaded_tiles_gdf=downloaded_tiles_gdf,
                wc_stats=wc_stats,
                topo_norm_stats=topo_norm_stats,
                naip_file_index=naip_file_index,
            )
        except Exception as e:
            failures.append((csv_path.name, str(e)))
            print(f"\n[ERROR] Failed processing {csv_path.name}: {e}\n")

    if failures:
        print("\nCompleted with failures:")
        for filename, error in failures:
            print(f"  - {filename}: {error}")
    else:
        print("\nDone.")


if __name__ == "__main__":
    main()


# EOF
