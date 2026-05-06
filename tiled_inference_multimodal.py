#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tiled inference for Host NAIP SDM multimodal models.

This script runs raster-tiled inference over NAIP GeoTIFF tiles for the seven
model groups used in the Host NAIP SDM workflow:

    image_only
    tabular_only
    topo_only
    image_tabular
    topo_tabular
    image_topo
    image_topo_tabular

The script is designed to match the revised training code where models expose
`forward_from_batch(batch)` and are constructed with `model.build_model(...)`.
It also remains tolerant of older checkpoints that store only a class name in
`checkpoint["model_type"]`.

Outputs per input NAIP tile:
    <tile>_prediction.tif     mean predicted probability
    <tile>_uncertainty.tif    mean Monte Carlo dropout std. dev. per pixel
    <tile>_count.tif          number of chip predictions contributing per pixel

Notes
-----
1. Even tabular-only and topo-only models are tiled over the NAIP raster grid so
   predictions share the same spatial support, transform, CRS, and dimensions as
   the NAIP imagery.
2. Topographic chips are extracted over the exact NAIP chip footprint and
   resampled to `--topo-chip-size` pixels. For example, a 256 x 256 NAIP chip at
   2 m resolution covers ~512 m; a 64 x 64 topo chip over the same footprint has
   ~8 m effective output pixels.
3. Topographic scalar summaries are computed from the same stack used by the
   model. If topography normalization is enabled, scalar summaries are also on
   the normalized scale, matching the dataset builder convention.
"""

from __future__ import annotations

import os

conda_env = r"C:\Users\talake2\AppData\Local\anaconda3\envs\naip_ailanthus_env"
os.environ["GDAL_DATA"] = os.path.join(conda_env, "Library", "share", "gdal")
os.environ["PROJ_LIB"] = os.path.join(conda_env, "Library", "share", "proj")
os.environ["PATH"] += os.pathsep + os.path.join(conda_env, "Library", "bin")

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

import rasterio
from rasterio.enums import Resampling
from rasterio.transform import rowcol
from rasterio.warp import reproject, transform as transform_coords
from rasterio.windows import Window

# Project-local imports. Keep model.py in the same directory or on PYTHONPATH.
from model import build_model, normalize_model_type  # noqa: E402


# -----------------------------------------------------------------------------
# Default normalization statistics
# -----------------------------------------------------------------------------

WC_VARS = [f"wc2.1_30s_bio_{i}" for i in range(1, 20)]

BASE_ENV_VARS = [
    *WC_VARS,
    "ghm",
    "lat_norm",
    "lon_norm",
]

TOPO_IMAGE_BANDS = ["elevation", "slope", "northness", "eastness"]

# These are model-ready normalized columns when the dataset builder was run with
# topographic normalization enabled. `topo_valid_frac` is kept for compatibility
# but is usually excluded from model features.
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

DEFAULT_MODEL_TOPO_SCALAR_COLUMNS = [
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
]

# These WorldClim and lat/lon values should be replaced with the exact stats used
# during dataset construction/training when available. They are kept here as a
# practical default for existing US-scale experiments.
DEFAULT_WORLDCLIM_STATS = {
    "wc2.1_30s_bio_1": {"mean": 10.9748, "std": 5.0872},
    "wc2.1_30s_bio_2": {"mean": 13.6538, "std": 2.1714},
    "wc2.1_30s_bio_3": {"mean": 36.8663, "std": 6.5792},
    "wc2.1_30s_bio_4": {"mean": 881.2029, "std": 180.3222},
    "wc2.1_30s_bio_5": {"mean": 30.3126, "std": 3.8458},
    "wc2.1_30s_bio_6": {"mean": -7.2765, "std": 6.7536},
    "wc2.1_30s_bio_7": {"mean": 37.5892, "std": 5.4061},
    "wc2.1_30s_bio_8": {"mean": 15.7360, "std": 7.9419},
    "wc2.1_30s_bio_9": {"mean": 5.9556, "std": 11.2411},
    "wc2.1_30s_bio_10": {"mean": 21.6966, "std": 4.2591},
    "wc2.1_30s_bio_11": {"mean": -0.0010, "std": 6.6096},
    "wc2.1_30s_bio_12": {"mean": 764.9769, "std": 421.6841},
    "wc2.1_30s_bio_13": {"mean": 99.6282, "std": 50.9529},
    "wc2.1_30s_bio_14": {"mean": 33.7521, "std": 27.9409},
    "wc2.1_30s_bio_15": {"mean": 39.4090, "std": 21.1094},
    "wc2.1_30s_bio_16": {"mean": 271.0835, "std": 144.2669},
    "wc2.1_30s_bio_17": {"mean": 117.5644, "std": 92.1923},
    "wc2.1_30s_bio_18": {"mean": 213.2364, "std": 114.1300},
    "wc2.1_30s_bio_19": {"mean": 165.2916, "std": 151.7543},
}

DEFAULT_LAT_LON_STATS = {
    "lat": {"mean": 39.34043295, "std": 4.24367101},
    "lon": {"mean": -90.41125129, "std": 15.36062434},
}

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

MODEL_REQUIREMENTS = {
    "image_only": {"image": True, "topo": False, "env": False},
    "tabular_only": {"image": False, "topo": False, "env": True},
    "topo_only": {"image": False, "topo": True, "env": False},
    "image_tabular": {"image": True, "topo": False, "env": True},
    "topo_tabular": {"image": False, "topo": True, "env": True},
    "image_topo": {"image": True, "topo": True, "env": False},
    "image_topo_tabular": {"image": True, "topo": True, "env": True},
}


def model_requires(model_type: str) -> dict[str, bool]:
    """Return required input branches for a canonical or aliased model type."""
    key = normalize_model_type(model_type)
    return MODEL_REQUIREMENTS[key]


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def load_json(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def coerce_stats_dict(raw: dict[str, Any], default: dict[str, Any]) -> dict[str, Any]:
    """
    Accept either a direct stats dictionary or common wrapped schemas.

    Examples accepted:
        {"wc2.1_30s_bio_1": {"mean": ..., "std": ...}, ...}
        {"worldclim_stats": {...}}
        {"lat_lon_stats": {...}}
        {"topo_normalization_stats": {...}}
    """
    if not raw:
        return default
    for key in ["worldclim_stats", "lat_lon_stats", "topo_normalization_stats", "stats"]:
        if isinstance(raw, dict) and key in raw and isinstance(raw[key], dict):
            return raw[key]
    return raw


def validate_mean_std_stats(stats: dict[str, Any], names: list[str], label: str) -> None:
    for name in names:
        if name not in stats:
            raise ValueError(f"Missing {label} stats for '{name}'.")
        for key in ["mean", "std"]:
            if key not in stats[name]:
                raise ValueError(f"Missing {label} stats key '{key}' for '{name}'.")
        mean = float(stats[name]["mean"])
        std = float(stats[name]["std"])
        if not np.isfinite(mean):
            raise ValueError(f"Invalid {label} mean for {name}: {mean}")
        if not np.isfinite(std) or std <= 0:
            raise ValueError(f"Invalid {label} std for {name}: {std}")


def parse_feature_list(value: Any, include_topo_scalars: bool = True) -> list[str]:
    """Parse env feature specification from config or CLI."""
    if value is None or value == "auto":
        features = list(BASE_ENV_VARS)
        if include_topo_scalars:
            features.extend(DEFAULT_MODEL_TOPO_SCALAR_COLUMNS)
        return features
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() == "auto":
            return parse_feature_list(None, include_topo_scalars=include_topo_scalars)
        # Allow JSON list passed as a string.
        if stripped.startswith("["):
            parsed = json.loads(stripped)
            if not isinstance(parsed, list):
                raise ValueError("env_features JSON string must decode to a list.")
            return [str(x) for x in parsed]
        # Allow comma-separated features.
        return [x.strip() for x in stripped.split(",") if x.strip()]
    raise TypeError(f"Unsupported env_features value: {type(value)}")


def normalize_coord(value: float, varname: str, lat_lon_stats: dict[str, Any]) -> float:
    mean = float(lat_lon_stats[varname]["mean"])
    std = float(lat_lon_stats[varname]["std"])
    return float((value - mean) / std)


def list_naip_tiles(naip_folder: str | None, tile_fp: str | None, recursive: bool = False) -> list[Path]:
    """Return one or more NAIP tile paths."""
    if tile_fp:
        path = Path(tile_fp)
        if not path.exists():
            raise FileNotFoundError(path)
        return [path]
    if not naip_folder:
        raise ValueError("Provide either --tile-fp or --naip-folder.")
    root = Path(naip_folder)
    if not root.exists():
        raise FileNotFoundError(root)
    pattern = "**/*.tif" if recursive else "*.tif"
    tiles = sorted(root.glob(pattern)) + sorted(root.glob(pattern.upper()))
    # Remove duplicates from case-insensitive filesystems.
    seen: set[str] = set()
    out: list[Path] = []
    for p in tiles:
        key = str(p.resolve()).lower()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


# -----------------------------------------------------------------------------
# Raster loading and extraction
# -----------------------------------------------------------------------------


def preload_and_normalize_env_rasters(
    worldclim_dir: str,
    ghm_path: str,
    wc_stats: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Preload WorldClim rasters as normalized arrays and GHM as a clipped array."""
    validate_mean_std_stats(wc_stats, WC_VARS, "WorldClim")

    wc_rasters: dict[str, dict[str, Any]] = {}
    for varname in WC_VARS:
        raster_fp = Path(worldclim_dir) / f"{varname}.tif"
        if not raster_fp.exists():
            raise FileNotFoundError(f"WorldClim raster not found: {raster_fp}")
        with rasterio.open(raster_fp) as ds:
            arr = ds.read(1).astype(np.float32)
            if ds.nodata is not None:
                arr = np.where(arr == ds.nodata, np.nan, arr)
            mean = float(wc_stats[varname]["mean"])
            std = float(wc_stats[varname]["std"])
            arr_norm = (arr - mean) / std
            wc_rasters[varname] = {
                "array": arr_norm.astype(np.float32),
                "transform": ds.transform,
                "crs": ds.crs,
                "nodata": ds.nodata,
            }

    with rasterio.open(ghm_path) as ds:
        arr = ds.read(1).astype(np.float32)
        if ds.nodata is not None:
            arr = np.where(arr == ds.nodata, np.nan, arr)
        arr = np.clip(arr, 0.0, 1.0)
        ghm_raster = {
            "array": arr.astype(np.float32),
            "transform": ds.transform,
            "crs": ds.crs,
            "nodata": ds.nodata,
        }

    return wc_rasters, ghm_raster


def extract_raster_value(lon: float, lat: float, raster_info: dict[str, Any]) -> float:
    """Sample a preloaded raster dictionary at lon/lat."""
    xs, ys = transform_coords("EPSG:4326", raster_info["crs"], [lon], [lat])
    row, col = rowcol(raster_info["transform"], xs[0], ys[0])
    arr = raster_info["array"]
    if row < 0 or col < 0 or row >= arr.shape[0] or col >= arr.shape[1]:
        return float("nan")
    return float(arr[row, col])


def extract_worldclim_vars_for_point(
    lon: float,
    lat: float,
    wc_rasters: dict[str, dict[str, Any]],
) -> dict[str, float]:
    return {var: extract_raster_value(lon, lat, wc_rasters[var]) for var in WC_VARS}


def extract_ghm_for_point(lon: float, lat: float, ghm_raster: dict[str, Any]) -> float:
    return extract_raster_value(lon, lat, ghm_raster)


def open_topo_sources(args: argparse.Namespace) -> dict[str, rasterio.io.DatasetReader] | None:
    """Open topographic rasters only when a selected model/features require them."""
    if not args.dem_raster or not args.slope_raster:
        return None

    sources: dict[str, rasterio.io.DatasetReader] = {
        "elevation": rasterio.open(args.dem_raster),
        "slope": rasterio.open(args.slope_raster),
    }

    if args.northness_raster and args.eastness_raster:
        sources["northness"] = rasterio.open(args.northness_raster)
        sources["eastness"] = rasterio.open(args.eastness_raster)
    elif args.aspect_raster:
        sources["aspect"] = rasterio.open(args.aspect_raster)
    else:
        # Close already-open sources before raising.
        for ds in sources.values():
            ds.close()
        raise ValueError(
            "Topography is required but northness/eastness rasters or an aspect raster were not supplied."
        )

    return sources


def close_raster_sources(sources: dict[str, Any] | None) -> None:
    if not sources:
        return
    for src in sources.values():
        try:
            src.close()
        except Exception:
            pass


def normalize_topo_stack(stack: np.ndarray, topo_norm_stats: dict[str, Any]) -> np.ndarray:
    """Z-score normalize a 4-band topo stack using precomputed stats."""
    validate_mean_std_stats(topo_norm_stats, TOPO_IMAGE_BANDS, "topographic")
    out = stack.astype(np.float32, copy=True)
    for band_idx, name in enumerate(TOPO_IMAGE_BANDS):
        mean = float(topo_norm_stats[name]["mean"])
        std = float(topo_norm_stats[name]["std"])
        valid = np.isfinite(out[band_idx])
        out[band_idx, valid] = (out[band_idx, valid] - mean) / std
    return out


def summarize_topo_stack(stack: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    """Summarize the stack actually used by the model."""
    stats: dict[str, float] = {"topo_valid_frac": float(valid.mean())}
    if not valid.any():
        for col in TOPO_SCALAR_COLUMNS:
            stats[col] = float("nan")
        stats["topo_valid_frac"] = 0.0
        return stats

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
    return stats


def extract_topo_for_naip_window(
    naip_src: rasterio.io.DatasetReader,
    window: Window,
    topo_sources: dict[str, rasterio.io.DatasetReader],
    topo_chip_size: int = 64,
    topo_min_valid_frac: float = 0.90,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Extract raw 4-band topo stack over the exact NAIP window footprint.

    Returns
    -------
    raw_stack : np.ndarray
        Shape (4, topo_chip_size, topo_chip_size), raw physical values.
    valid : np.ndarray
        Boolean valid mask where all four bands are finite.
    ok : bool
        True if valid fraction passes topo_min_valid_frac.
    """
    naip_bounds = rasterio.windows.bounds(window, naip_src.transform)
    dst_transform = rasterio.transform.from_bounds(
        *naip_bounds,
        width=topo_chip_size,
        height=topo_chip_size,
    )
    dst_crs = naip_src.crs

    layers: dict[str, np.ndarray] = {}
    for name in TOPO_IMAGE_BANDS:
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

    # Fallback if aspect was supplied instead of northness/eastness.
    if ("northness" not in layers or "eastness" not in layers) and topo_sources.get("aspect") is not None:
        aspect = topo_sources["aspect"]
        aspect_arr = np.full((topo_chip_size, topo_chip_size), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(aspect, 1),
            destination=aspect_arr,
            src_transform=aspect.transform,
            src_crs=aspect.crs,
            src_nodata=aspect.nodata,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=Resampling.nearest,
        )
        valid_aspect = np.isfinite(aspect_arr) & (aspect_arr >= 0.0) & (aspect_arr <= 360.0)
        rad = np.deg2rad(aspect_arr)
        north = np.full_like(aspect_arr, np.nan, dtype=np.float32)
        east = np.full_like(aspect_arr, np.nan, dtype=np.float32)
        north[valid_aspect] = np.cos(rad[valid_aspect])
        east[valid_aspect] = np.sin(rad[valid_aspect])
        layers.setdefault("northness", north)
        layers.setdefault("eastness", east)

    missing = [name for name in TOPO_IMAGE_BANDS if name not in layers]
    if missing:
        raise RuntimeError(f"Missing required topographic layers: {missing}")

    raw_stack = np.stack([layers[name].astype(np.float32) for name in TOPO_IMAGE_BANDS], axis=0)
    valid = np.isfinite(raw_stack).all(axis=0)
    ok = float(valid.mean()) >= topo_min_valid_frac
    return raw_stack, valid, ok


def build_env_vector(
    lon: float,
    lat: float,
    wc_rasters: dict[str, dict[str, Any]],
    ghm_raster: dict[str, Any],
    lat_lon_stats: dict[str, Any],
    env_vars: list[str],
    topo_stats: dict[str, float] | None = None,
) -> list[float] | None:
    """Build one ordered tabular feature vector for inference."""
    wc_values = extract_worldclim_vars_for_point(lon, lat, wc_rasters)
    ghm_value = extract_ghm_for_point(lon, lat, ghm_raster)

    feature_map: dict[str, float] = dict(wc_values)
    feature_map.update({
        "ghm": float(ghm_value),
        "lat_norm": normalize_coord(lat, "lat", lat_lon_stats),
        "lon_norm": normalize_coord(lon, "lon", lat_lon_stats),
    })
    if topo_stats:
        feature_map.update(topo_stats)

    vec: list[float] = []
    for col in env_vars:
        val = feature_map.get(col, float("nan"))
        if not np.isfinite(val):
            return None
        vec.append(float(val))
    return vec


def naip_window_is_valid(chip: np.ndarray, min_valid_frac: float) -> bool:
    """
    Basic NAIP validity check.

    Assumes 0 is nodata for many NAIP products. This is conservative: a pixel is
    valid if at least one band is finite and nonzero.
    """
    if min_valid_frac <= 0:
        return True
    finite = np.isfinite(chip).all(axis=0)
    nonzero = np.any(chip != 0, axis=0)
    valid_frac = float((finite & nonzero).mean())
    return valid_frac >= min_valid_frac


# -----------------------------------------------------------------------------
# Model loading and prediction helpers
# -----------------------------------------------------------------------------


def infer_model_type_from_checkpoint(checkpoint: dict[str, Any], config: dict[str, Any], cli_model_type: str | None) -> str:
    """Resolve model type from CLI, config, or checkpoint metadata."""
    if cli_model_type:
        return normalize_model_type(cli_model_type)
    if config.get("model_type"):
        return normalize_model_type(config["model_type"])
    if checkpoint.get("canonical_model_type"):
        return normalize_model_type(checkpoint["canonical_model_type"])
    if checkpoint.get("model_type"):
        return normalize_model_type(checkpoint["model_type"])
    return normalize_model_type("image_topo_tabular")


def load_checkpoint_state(checkpoint_fp: str, device: torch.device) -> dict[str, Any]:
    """Load a PyTorch checkpoint safely across torch versions."""
    try:
        return torch.load(checkpoint_fp, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_fp, map_location=device)


def resolve_env_features_for_inference(
    config: dict[str, Any],
    req: dict[str, bool],
    cli_env_features: str | None,
    include_topo_scalars: bool,
) -> list[str]:
    """Resolve the exact feature order used by the tabular branch."""
    if not req["env"]:
        return []

    if cli_env_features:
        return parse_feature_list(cli_env_features, include_topo_scalars=include_topo_scalars)

    # Preferred key written by revised training main.py.
    if isinstance(config.get("resolved_env_features"), list):
        return [str(x) for x in config["resolved_env_features"]]

    # Training config key.
    env_features = config.get("env_features", "auto")
    return parse_feature_list(env_features, include_topo_scalars=include_topo_scalars)


def build_and_load_model(
    checkpoint_fp: str,
    config: dict[str, Any],
    model_type: str,
    num_env_features: int,
    device: torch.device,
    strict: bool = True,
) -> torch.nn.Module:
    """Build the selected model architecture and load checkpoint weights."""
    model = build_model(
        model_type=model_type,
        num_env_features=num_env_features,
        hidden_dim=int(config.get("hidden_dim", 256)),
        dropout=float(config.get("dropout", 0.25)),
        naip_channels=int(config.get("naip_channels", 4)),
        topo_channels=int(config.get("topo_channels", 4)),
        topo_feature_dim=int(config.get("topo_feature_dim", 128)),
        env_feature_dim=int(config.get("env_feature_dim", 128)),
        # During inference/checkpoint loading, avoid requesting ImageNet weights.
        pretrained_image=False,
    ).to(device)

    checkpoint = load_checkpoint_state(checkpoint_fp, device)
    state = checkpoint.get("model_state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if not strict:
        if missing:
            print(f"[WARNING] Missing keys while loading checkpoint: {missing}")
        if unexpected:
            print(f"[WARNING] Unexpected keys while loading checkpoint: {unexpected}")

    model.eval()
    return model


def enable_mc_dropout(model: torch.nn.Module) -> None:
    """Enable dropout layers while leaving other modules in eval mode."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def predict_batch(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    model_type: str,
    req: dict[str, bool],
) -> torch.Tensor:
    """Return logits for a batch using the model's preferred batch interface."""
    if hasattr(model, "forward_from_batch"):
        return model.forward_from_batch(batch)

    # Legacy fallback. The revised model.py should use forward_from_batch.
    if model_type == "image_only":
        return model(batch["image"])
    if model_type == "tabular_only":
        return model(batch["env"])
    if model_type == "topo_only":
        return model(batch["topo"])
    if model_type == "image_tabular":
        return model(batch["image"], batch["env"])
    if model_type == "topo_tabular":
        return model(batch["topo"], batch["env"])
    if model_type == "image_topo":
        return model(batch["image"], batch["topo"])
    if model_type == "image_topo_tabular":
        return model(batch["image"], batch["topo"], batch["env"])
    raise ValueError(f"Unsupported model_type: {model_type}; requirements={req}")


# -----------------------------------------------------------------------------
# Tiled inference
# -----------------------------------------------------------------------------


@torch.inference_mode()
def run_inference_on_tile(
    tile_fp: Path,
    output_dir: Path,
    model: torch.nn.Module,
    model_type: str,
    req: dict[str, bool],
    device: torch.device,
    wc_rasters: dict[str, dict[str, Any]] | None,
    ghm_raster: dict[str, Any] | None,
    lat_lon_stats: dict[str, Any],
    env_vars: list[str],
    topo_sources: dict[str, rasterio.io.DatasetReader] | None,
    topo_norm_stats: dict[str, Any],
    chip_size: int = 256,
    stride: int = 256,
    batch_size: int = 64,
    topo_chip_size: int = 64,
    topo_min_valid_frac: float = 0.90,
    normalize_topography: bool = True,
    naip_scale_255: bool = True,
    min_image_valid_frac: float = 0.25,
    mc_samples: int = 2,
    use_amp: bool = True,
    skip_existing: bool = False,
    write_count: bool = True,
    naip_channels: int = 4,
) -> dict[str, Any]:
    """Run tiled inference for one NAIP tile."""
    tile_name = tile_fp.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_fp = output_dir / f"{tile_name}_prediction.tif"
    uncertainty_fp = output_dir / f"{tile_name}_uncertainty.tif"
    count_fp = output_dir / f"{tile_name}_count.tif"

    if skip_existing and prediction_fp.exists() and uncertainty_fp.exists():
        print(f"[SKIP] Existing outputs found for {tile_name}")
        return {"tile": str(tile_fp), "status": "skipped_existing"}

    if req["env"] and (wc_rasters is None or ghm_raster is None):
        raise ValueError("Selected model requires tabular/env inputs but WorldClim/GHM rasters are unavailable.")

    topo_needed_for_env = req["env"] and any(col in TOPO_SCALAR_COLUMNS for col in env_vars)
    topo_needed = req["topo"] or topo_needed_for_env
    if topo_needed and topo_sources is None:
        raise ValueError("Selected model/features require topography but topo_sources is None.")

    if normalize_topography:
        validate_mean_std_stats(topo_norm_stats, TOPO_IMAGE_BANDS, "topographic")

    start_time = time.time()
    model.eval()
    if mc_samples > 1:
        enable_mc_dropout(model)

    processed = 0
    skipped_image = 0
    skipped_topo = 0
    skipped_env = 0

    with rasterio.open(tile_fp) as src:
        width, height = src.width, src.height
        out_profile = src.profile.copy()

        sum_array = np.zeros((height, width), dtype=np.float32)
        uncertainty_sum_array = np.zeros((height, width), dtype=np.float32)
        count_array = np.zeros((height, width), dtype=np.uint16)

        rows = range(0, height - chip_size + 1, stride)
        cols = range(0, width - chip_size + 1, stride)
        chip_iter = [(r, c) for r in rows for c in cols]

        image_batch: list[torch.Tensor] = []
        topo_batch: list[torch.Tensor] = []
        env_batch: list[torch.Tensor] = []
        locs: list[tuple[int, int]] = []

        def flush_batch() -> None:
            nonlocal processed
            if not locs:
                return

            batch: dict[str, torch.Tensor] = {}
            if req["image"]:
                batch["image"] = torch.cat(image_batch, dim=0).to(device, non_blocking=True)
            if req["topo"]:
                batch["topo"] = torch.cat(topo_batch, dim=0).to(device, non_blocking=True)
            if req["env"]:
                batch["env"] = torch.cat(env_batch, dim=0).to(device, non_blocking=True)

            probs_by_pass: list[np.ndarray] = []
            amp_enabled = use_amp and device.type == "cuda"
            for _ in range(max(1, mc_samples)):
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                    logits = predict_batch(model, batch, model_type=model_type, req=req)
                    probs = torch.sigmoid(logits).detach().float().cpu().numpy().reshape(-1)
                probs_by_pass.append(probs)

            probs_stack = np.stack(probs_by_pass, axis=0)
            mean_probs = probs_stack.mean(axis=0)
            std_probs = probs_stack.std(axis=0)

            for (row, col), mean_p, std_p in zip(locs, mean_probs, std_probs):
                sum_array[row: row + chip_size, col: col + chip_size] += float(mean_p)
                uncertainty_sum_array[row: row + chip_size, col: col + chip_size] += float(std_p)
                count_array[row: row + chip_size, col: col + chip_size] += 1
                processed += 1

            image_batch.clear()
            topo_batch.clear()
            env_batch.clear()
            locs.clear()

        for row, col in tqdm(chip_iter, desc=f"{tile_name}", leave=False):
            window = Window(col_off=col, row_off=row, width=chip_size, height=chip_size)

            chip_tensor: torch.Tensor | None = None
            if req["image"]:
                # Read only the requested NAIP channels. Most NAIP products are 4-band.
                if src.count < int(naip_channels):
                    raise ValueError(
                        f"Expected at least {naip_channels} NAIP bands, found {src.count} in {tile_fp}"
                    )
                chip = src.read(indexes=list(range(1, int(naip_channels) + 1)), window=window).astype(np.float32)
                if not naip_window_is_valid(chip, min_image_valid_frac=min_image_valid_frac):
                    skipped_image += 1
                    continue
                if naip_scale_255:
                    chip = chip / 255.0
                chip = np.nan_to_num(chip, nan=0.0, posinf=0.0, neginf=0.0)
                chip_tensor = torch.from_numpy(chip.astype(np.float32)).unsqueeze(0)

            # Chip center coordinates in WGS84 for tabular predictors.
            center_x, center_y = src.transform * (col + chip_size / 2.0, row + chip_size / 2.0)
            lons, lats = transform_coords(src.crs, "EPSG:4326", [center_x], [center_y])
            lon, lat = float(lons[0]), float(lats[0])

            topo_tensor: torch.Tensor | None = None
            topo_stats: dict[str, float] | None = None
            if topo_needed:
                try:
                    raw_topo_stack, valid_topo, topo_ok = extract_topo_for_naip_window(
                        naip_src=src,
                        window=window,
                        topo_sources=topo_sources or {},
                        topo_chip_size=topo_chip_size,
                        topo_min_valid_frac=topo_min_valid_frac,
                    )
                except Exception:
                    skipped_topo += 1
                    continue
                if not topo_ok:
                    skipped_topo += 1
                    continue

                if normalize_topography:
                    model_topo_stack = normalize_topo_stack(raw_topo_stack, topo_norm_stats)
                else:
                    model_topo_stack = raw_topo_stack.astype(np.float32, copy=True)

                topo_stats = summarize_topo_stack(model_topo_stack, valid_topo)
                topo_stats["topo_normalized"] = float(bool(normalize_topography))

                if req["topo"]:
                    topo_for_model = np.nan_to_num(model_topo_stack, nan=0.0, posinf=0.0, neginf=0.0)
                    topo_tensor = torch.from_numpy(topo_for_model.astype(np.float32)).unsqueeze(0)

            env_tensor: torch.Tensor | None = None
            if req["env"]:
                env_vector = build_env_vector(
                    lon=lon,
                    lat=lat,
                    wc_rasters=wc_rasters or {},
                    ghm_raster=ghm_raster or {},
                    lat_lon_stats=lat_lon_stats,
                    env_vars=env_vars,
                    topo_stats=topo_stats,
                )
                if env_vector is None:
                    skipped_env += 1
                    continue
                env_tensor = torch.tensor(env_vector, dtype=torch.float32).unsqueeze(0)

            if req["image"] and chip_tensor is not None:
                image_batch.append(chip_tensor)
            if req["topo"] and topo_tensor is not None:
                topo_batch.append(topo_tensor)
            if req["env"] and env_tensor is not None:
                env_batch.append(env_tensor)
            locs.append((row, col))

            if len(locs) >= batch_size:
                flush_batch()

        flush_batch()

        nodata = -9999.0
        prediction = np.full((height, width), nodata, dtype=np.float32)
        uncertainty = np.full((height, width), nodata, dtype=np.float32)
        valid_count = count_array > 0
        prediction[valid_count] = sum_array[valid_count] / count_array[valid_count]
        uncertainty[valid_count] = uncertainty_sum_array[valid_count] / count_array[valid_count]

        profile = out_profile.copy()
        profile.update({
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": "float32",
            "nodata": nodata,
            "compress": "deflate",
            "predictor": 2,
            "BIGTIFF": "IF_SAFER",
        })

        with rasterio.open(prediction_fp, "w", **profile) as dst:
            dst.write(prediction, 1)
            dst.set_band_description(1, "predicted_presence_probability")

        with rasterio.open(uncertainty_fp, "w", **profile) as dst:
            dst.write(uncertainty, 1)
            dst.set_band_description(1, "mc_dropout_probability_std")

        if write_count:
            count_profile = out_profile.copy()
            count_profile.update({
                "driver": "GTiff",
                "height": height,
                "width": width,
                "count": 1,
                "dtype": "uint16",
                "nodata": 0,
                "compress": "deflate",
                "BIGTIFF": "IF_SAFER",
            })
            with rasterio.open(count_fp, "w", **count_profile) as dst:
                dst.write(count_array, 1)
                dst.set_band_description(1, "prediction_count")

    elapsed = time.time() - start_time
    summary = {
        "tile": str(tile_fp),
        "status": "ok",
        "prediction_fp": str(prediction_fp),
        "uncertainty_fp": str(uncertainty_fp),
        "count_fp": str(count_fp) if write_count else None,
        "processed_chips": processed,
        "skipped_image_chips": skipped_image,
        "skipped_topo_chips": skipped_topo,
        "skipped_env_chips": skipped_env,
        "elapsed_seconds": elapsed,
    }
    print(
        f"[OK] {tile_name}: processed={processed:,}, "
        f"skipped image/topo/env={skipped_image:,}/{skipped_topo:,}/{skipped_env:,}, "
        f"elapsed={elapsed:.1f}s"
    )
    return summary


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run tiled multimodal inference for Host NAIP SDM models."
    )

    parser.add_argument("--config", type=str, default=None, help="Training/inference config JSON. Optional but recommended.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint .tar/.pt file.")
    parser.add_argument("--model-type", type=str, default=None, help="Override model_type. Otherwise read from config/checkpoint.")

    parser.add_argument("--tile-fp", type=str, default=None, help="Run inference on one NAIP GeoTIFF tile.")
    parser.add_argument("--naip-folder", type=str, default=None, help="Folder containing NAIP GeoTIFF tiles.")
    parser.add_argument("--recursive", action="store_true", help="Search --naip-folder recursively for .tif files.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for prediction rasters.")

    parser.add_argument("--worldclim-dir", type=str, default=None, help="Folder containing wc2.1_30s_bio_*.tif rasters.")
    parser.add_argument("--ghm-raster", type=str, default=None, help="Path to GHM raster.")
    parser.add_argument("--worldclim-stats", type=str, default=None, help="Optional JSON with WorldClim mean/std stats.")
    parser.add_argument("--lat-lon-stats", type=str, default=None, help="Optional JSON with lat/lon mean/std stats.")
    parser.add_argument("--env-features", type=str, default=None, help="Comma-separated, JSON-list, or 'auto' env features override.")
    parser.add_argument("--include-topo-scalars", action="store_true", default=None, help="Include topo scalar summaries when env_features='auto'.")
    parser.add_argument("--exclude-topo-scalars", action="store_true", help="Exclude topo scalar summaries when env_features='auto'.")

    parser.add_argument("--dem-raster", type=str, default=None, help="Elevation raster for topo models/features.")
    parser.add_argument("--slope-raster", type=str, default=None, help="Slope raster in degrees.")
    parser.add_argument("--northness-raster", type=str, default=None, help="Precomputed northness raster.")
    parser.add_argument("--eastness-raster", type=str, default=None, help="Precomputed eastness raster.")
    parser.add_argument("--aspect-raster", type=str, default=None, help="Aspect raster in degrees; fallback for northness/eastness.")
    parser.add_argument("--topo-normalization-stats", type=str, default=None, help="Optional JSON with topo mean/std stats.")
    parser.add_argument("--disable-topo-normalization", action="store_true", help="Use raw topo values instead of z-scored topo values.")
    parser.add_argument("--topo-chip-size", type=int, default=None, help="Topo chip size. Default from config or 64.")
    parser.add_argument("--topo-min-valid-frac", type=float, default=None, help="Minimum valid fraction for topo chips. Default from config or 0.90.")

    parser.add_argument("--chip-size", type=int, default=None, help="NAIP chip size in pixels. Default from config or 256.")
    parser.add_argument("--stride", type=int, default=None, help="Sliding-window stride in pixels. Default = chip-size.")
    parser.add_argument("--batch-size", type=int, default=None, help="Inference batch size. Default from config or 64.")
    parser.add_argument("--mc-samples", type=int, default=2, help="MC dropout passes for uncertainty. Use 1 for deterministic inference.")
    parser.add_argument("--min-image-valid-frac", type=float, default=0.25, help="Minimum valid nonzero NAIP pixel fraction for image models.")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA automatic mixed precision.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip tiles whose prediction and uncertainty rasters already exist.")
    parser.add_argument("--no-count", action="store_true", help="Do not write prediction-count rasters.")
    parser.add_argument("--strict-checkpoint", action="store_true", help="Require exact checkpoint/model key matching.")
    parser.add_argument("--conda-env", type=str, default=None, help="Optional Windows conda env path for GDAL/PROJ variables.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_json(args.config) or {}
    checkpoint_meta = load_checkpoint_state(args.checkpoint, torch.device("cpu"))
    model_type = infer_model_type_from_checkpoint(checkpoint_meta, config, args.model_type) # model_type like "image_topo_tabular"
    req = model_requires(model_type)

    if args.exclude_topo_scalars:
        include_topo_scalars = False
    elif args.include_topo_scalars is not None:
        include_topo_scalars = bool(args.include_topo_scalars)
    else:
        include_topo_scalars = bool(config.get("include_topo_scalars", True))

    env_vars = resolve_env_features_for_inference(
        config=config,
        req=req,
        cli_env_features=args.env_features,
        include_topo_scalars=include_topo_scalars,
    )

    print(f"Canonical model_type: {model_type}")
    print(f"Model input requirements: {req}")
    print(f"Number of tabular features: {len(env_vars)}")
    if env_vars:
        print("Tabular feature order:", env_vars)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_and_load_model(
        checkpoint_fp=args.checkpoint,
        config=config,
        model_type=model_type,
        num_env_features=len(env_vars),
        device=device,
        strict=bool(args.strict_checkpoint),
    )

    if args.mc_samples > 1:
        print(f"Monte Carlo dropout uncertainty enabled with {args.mc_samples} passes.")
    else:
        print("Deterministic inference: Monte Carlo dropout disabled.")

    wc_rasters = None
    ghm_raster = None
    if req["env"]:
        worldclim_dir = args.worldclim_dir or config.get("worldclim_dir") or config.get("worldclim_folder")
        ghm_path = args.ghm_raster or config.get("ghm_raster") or config.get("ghm_path")
        if not worldclim_dir or not ghm_path:
            raise ValueError("Selected model requires env inputs. Provide --worldclim-dir and --ghm-raster.")

        wc_stats_raw = load_json(args.worldclim_stats) or config.get("worldclim_stats") or DEFAULT_WORLDCLIM_STATS
        wc_stats = coerce_stats_dict(wc_stats_raw, DEFAULT_WORLDCLIM_STATS)
        print("Preloading and normalizing WorldClim/GHM rasters...")
        wc_rasters, ghm_raster = preload_and_normalize_env_rasters(worldclim_dir, ghm_path, wc_stats)

    lat_lon_stats_raw = load_json(args.lat_lon_stats) or config.get("lat_lon_stats") or DEFAULT_LAT_LON_STATS
    lat_lon_stats = coerce_stats_dict(lat_lon_stats_raw, DEFAULT_LAT_LON_STATS)
    validate_mean_std_stats(lat_lon_stats, ["lat", "lon"], "lat/lon")

    topo_needed_for_env = req["env"] and any(col in TOPO_SCALAR_COLUMNS for col in env_vars)
    topo_needed = req["topo"] or topo_needed_for_env

    topo_sources = None
    if topo_needed:
        # Allow config fallback for topo paths.
        for attr, keys in {
            "dem_raster": ["dem_raster", "dem_raster_fp"],
            "slope_raster": ["slope_raster", "slope_raster_fp"],
            "northness_raster": ["northness_raster", "northness_raster_fp"],
            "eastness_raster": ["eastness_raster", "eastness_raster_fp"],
            "aspect_raster": ["aspect_raster", "aspect_raster_fp"],
        }.items():
            if getattr(args, attr) is None:
                for key in keys:
                    if config.get(key):
                        setattr(args, attr, config[key])
                        break
        topo_sources = open_topo_sources(args)

    topo_stats_raw = load_json(args.topo_normalization_stats) or config.get("topo_normalization_stats") or DEFAULT_TOPO_NORM_STATS_3DEP_30M
    topo_norm_stats = coerce_stats_dict(topo_stats_raw, DEFAULT_TOPO_NORM_STATS_3DEP_30M)
    normalize_topography = not bool(args.disable_topo_normalization or config.get("disable_topo_normalization", False))

    chip_size = int(args.chip_size or config.get("chip_size", 256))
    stride = int(args.stride or config.get("stride", chip_size))
    batch_size = int(args.batch_size or config.get("batch_size", 64))
    topo_chip_size = int(args.topo_chip_size or config.get("topo_chip_size", 64))
    topo_min_valid_frac = float(args.topo_min_valid_frac or config.get("topo_min_valid_frac", 0.90))
    naip_scale_255 = bool(config.get("naip_scale_255", True))

    output_dir = Path(args.output_dir)
    tiles = list_naip_tiles(args.naip_folder, args.tile_fp, recursive=args.recursive)
    if not tiles:
        raise RuntimeError("No NAIP .tif files found for inference.")
    print(f"Found {len(tiles):,} NAIP tile(s) for inference.")

    summaries: list[dict[str, Any]] = []
    try:
        for tile_fp in tqdm(tiles, desc="Processing tiles"):
            try:
                summary = run_inference_on_tile(
                    tile_fp=tile_fp,
                    output_dir=output_dir,
                    model=model,
                    model_type=model_type,
                    req=req,
                    device=device,
                    wc_rasters=wc_rasters,
                    ghm_raster=ghm_raster,
                    lat_lon_stats=lat_lon_stats,
                    env_vars=env_vars,
                    topo_sources=topo_sources,
                    topo_norm_stats=topo_norm_stats,
                    chip_size=chip_size,
                    stride=stride,
                    batch_size=batch_size,
                    topo_chip_size=topo_chip_size,
                    topo_min_valid_frac=topo_min_valid_frac,
                    normalize_topography=normalize_topography,
                    naip_scale_255=naip_scale_255,
                    min_image_valid_frac=float(args.min_image_valid_frac),
                    mc_samples=int(args.mc_samples),
                    use_amp=not bool(args.no_amp),
                    skip_existing=bool(args.skip_existing),
                    write_count=not bool(args.no_count),
                    naip_channels=int(config.get("naip_channels", 4)),
                )
                summaries.append(summary)
            except Exception as exc:
                print(f"[ERROR] Failed on {tile_fp}: {exc}")
                summaries.append({"tile": str(tile_fp), "status": "error", "error": str(exc)})
    finally:
        close_raster_sources(topo_sources)

    summary_fp = output_dir / "tiled_inference_summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_fp, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(f"Saved inference summary: {summary_fp}")


if __name__ == "__main__":
    main()
