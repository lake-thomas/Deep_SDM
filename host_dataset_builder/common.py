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

python /mnt/rsstu_dsfas_gsv_naip/Host_NAIP_SDM/Deep_SDM/create_host_datasets.py \
  --occurrence-file "/mnt/rsstu_dsfas_gsv_naip/Promit_Host_Occurrences/Fully_thinned_data/coordinate_uncertainty_under_256/prunus_cerasus_thinned.csv" \
  --output-root "/mnt/rsstu_dsfas_gsv_naip/Host_NAIP_SDM/Host_Datasets" \
  --tileindex "/mnt/rsstu_dsfas_gsv_naip/Host_NAIP_SDM/NAIP_Imagery_Tile_Indices/NAIP_US_Local_Archive_Tile_Index_ByTileKey.gpkg" \
  --naip-folder "/mnt/rsstu_dsfas_gsv_naip/Host_NAIP_SDM/NAIP_Archive" \
  --worldclim-folder "/mnt/rsstu_dsfas_gsv_naip/Host_NAIP_SDM/Env_Data/Worldclim" \
  --ghm-raster "/mnt/rsstu_dsfas_gsv_naip/Host_NAIP_SDM/Env_Data/Global_Human_Modification/gHM_WGS84.tif" \
  --topo-normalization-stats "/mnt/rsstu_dsfas_gsv_naip/Host_NAIP_SDM/Env_Data/Topography_3DEP/mosaic/topo_norm_stats_3dep_30m.json" \
  --dem-raster "/mnt/rsstu_dsfas_gsv_naip/Host_NAIP_SDM/Env_Data/Topography_3DEP/mosaic/dem_3dep_13_epsg5070_30m.tif" \
  --slope-raster "/mnt/rsstu_dsfas_gsv_naip/Host_NAIP_SDM/Env_Data/Topography_3DEP/mosaic/slope_degrees_3dep_13_epsg5070_30m.tif" \
  --northness-raster "/mnt/rsstu_dsfas_gsv_naip/Host_NAIP_SDM/Env_Data/Topography_3DEP/mosaic/northness_3dep_13_epsg5070_30m.tif" \
  --eastness-raster "/mnt/rsstu_dsfas_gsv_naip/Host_NAIP_SDM/Env_Data/Topography_3DEP/mosaic/eastness_3dep_13_epsg5070_30m.tif" \
  --background-inner-buffer-km 5 \
  --background-buffer-km 50 \
  --background-multiplier 10 \
  --topo-mode both \
  --spatial-thin-distance-m 800 \
  --background-sampling-mode polygon


"""

from __future__ import annotations
import os

if os.name == "nt":
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
from scipy.spatial import cKDTree
from shapely.geometry import Point, box
from tqdm import tqdm


# ---------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------

# 19 Bioclimatic variables at 30 arc-second resolution (approximately 1 km at the equator) from WorldClim version 2.1.
WC_VARS = [f"wc2.1_30s_bio_{i}" for i in range(1, 20)]

# Add Global Human Modification (GHM) to the list of environmental variables. GHM is a global raster dataset that quantifies human modification of terrestrial lands, with values from 0 (no modification) to 1 (complete modification).
ALL_ENV_VARS = WC_VARS + ["ghm"]

# Topographic image chip bands and scalar summary columns. The same topo normalization stats are applied to both the chips and the scalar summaries, so they share the same variable names (e.g. "elev_mean" and "elevation" for the chip band).
TOPO_IMAGE_BANDS = [
    "elevation",
    "slope",
    "northness",
    "eastness",
]

# These columns are model-ready normalized values using the below statistics calculated from the 3DEP product at 30m resolution, 
# unless --disable-topo-normalization is supplied.
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

