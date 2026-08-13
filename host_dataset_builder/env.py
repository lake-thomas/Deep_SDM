from .common import *
from .utils import geometry_union

from concurrent.futures import ThreadPoolExecutor, as_completed
from rasterio.warp import transform as warp_transform


def _sample_raw_environment_raster(
    path: str | Path,
    points: pd.DataFrame,
    column: str,
) -> tuple[str, np.ndarray, dict]:
    """Sample one raster without converting its no-data sentinel to a value."""
    path = Path(path)
    with rasterio.open(path) as src:
        x = points["lon"].to_numpy(dtype=float)
        y = points["lat"].to_numpy(dtype=float)
        if src.crs is not None and src.crs.to_string() != "EPSG:4326":
            x, y = warp_transform("EPSG:4326", src.crs, x.tolist(), y.tolist())
        values = np.full(len(points), np.nan, dtype=float)
        for index, value in enumerate(src.sample(zip(x, y), indexes=1, masked=True)):
            scalar = value[0]
            if np.ma.is_masked(scalar):
                continue
            numeric = float(scalar)
            if src.nodata is not None and np.isclose(
                numeric,
                float(src.nodata),
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            ):
                continue
            if np.isfinite(numeric):
                values[index] = numeric
        metadata = {
            "column": column,
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "mtime": pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC").isoformat(),
            "crs": src.crs.to_string() if src.crs else None,
            "nodata": src.nodata,
            "dtype": src.dtypes[0],
            "width": src.width,
            "height": src.height,
        }
    return column, values, metadata


def sample_raw_environment_for_points(
    points: pd.DataFrame,
    worldclim_folder: str | Path,
    ghm_raster: str | Path,
    workers: int = 4,
) -> tuple[pd.DataFrame, list[dict]]:
    """Return raw WorldClim/GHM values and raster provenance for WGS84 points."""
    required = {"lat", "lon"}
    missing = required.difference(points.columns)
    if missing:
        raise ValueError(f"Environmental sampling points lack columns: {sorted(missing)}")
    if workers <= 0:
        raise ValueError("workers must be positive")

    raster_specs: list[tuple[Path, str]] = []
    worldclim = Path(worldclim_folder)
    for column in WC_VARS:
        path = worldclim / f"{column}.tif"
        if not path.is_file():
            raise FileNotFoundError(path)
        raster_specs.append((path, column))
    ghm_path = Path(ghm_raster)
    if not ghm_path.is_file():
        raise FileNotFoundError(ghm_path)
    raster_specs.append((ghm_path, "ghm"))

    sampled: dict[str, np.ndarray] = {}
    metadata: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_sample_raw_environment_raster, path, points, column): column
            for path, column in raster_specs
        }
        for future in as_completed(futures):
            column, values, raster_metadata = future.result()
            sampled[column] = values
            metadata.append(raster_metadata)

    result = points.copy()
    for column in ALL_ENV_VARS:
        result[column] = sampled[column]
    return result, sorted(metadata, key=lambda item: item["column"])


def raw_environment_valid_mask(values: pd.DataFrame) -> pd.Series:
    """Identify rows with complete, plausible raw environmental support."""
    finite = np.isfinite(values[ALL_ENV_VARS].to_numpy(dtype=float)).all(axis=1)
    ghm = pd.to_numeric(values["ghm"], errors="coerce").to_numpy(dtype=float)
    return pd.Series(finite & (ghm >= 0.0) & (ghm <= 1.0), index=values.index)

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
        sampled = list(ds.sample([(lon, lat)], masked=True))[0][0]
        if np.ma.is_masked(sampled):
            vals[varname] = np.nan
            continue
        raw_value = float(sampled)
        if ds.nodata is not None and np.isclose(
            raw_value,
            float(ds.nodata),
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        ):
            vals[varname] = np.nan
            continue
        mean = normalization_stats[varname]["mean"]
        std = normalization_stats[varname]["std"]
        vals[varname] = float((raw_value - mean) / std)
    return vals

def extract_ghm_for_point(lon: float, lat: float, ghm_ds) -> float:
    """Extract a clamped GHM value for one point."""
    sampled = list(ghm_ds.sample([(lon, lat)], masked=True))[0][0]
    if np.ma.is_masked(sampled):
        return np.nan
    raw_value = float(sampled)
    if ghm_ds.nodata is not None and np.isclose(
        raw_value,
        float(ghm_ds.nodata),
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    ):
        return np.nan
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
