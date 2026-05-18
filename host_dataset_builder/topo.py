from .common import *

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
    print("Topo Normalization Stats:", stats)

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

