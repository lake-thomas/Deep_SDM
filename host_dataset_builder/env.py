from .common import *
from .utils import geometry_union

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

def load_or_compute_worldclim_stats(args: argparse.Namespace, downloaded_tiles_gdf: gpd.GeoDataFrame) -> dict:
    """Load cached WorldClim stats when available, otherwise compute and cache."""
    cache_fp = Path(args.worldclim_stats_json) if args.worldclim_stats_json else None
    if cache_fp is not None and cache_fp.exists():
        print(f"Loading cached WorldClim normalization statistics from {cache_fp}")
        with open(cache_fp, "r", encoding="utf-8") as f:
            return json.load(f)

    print("Computing WorldClim normalization statistics once from downloaded NAIP footprint...")
    downloaded_union = geometry_union(downloaded_tiles_gdf.geometry)
    stats = compute_worldclim_stats(args.worldclim_folder, downloaded_union)

    if cache_fp is not None:
        cache_fp.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_fp, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved WorldClim normalization statistics cache to {cache_fp}")

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

