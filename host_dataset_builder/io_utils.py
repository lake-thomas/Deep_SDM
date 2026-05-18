from .common import *
from .utils import slugify_species_name, display_name_from_slug, infer_species_slug_from_filename

def planned_output_paths(csv_path: Path, output_root: str, chip_size: int) -> dict[str, Path]:
    """Return the versioned output paths for a species without creating them."""
    species_slug = infer_species_slug_from_filename(csv_path)
    species_label = display_name_from_slug(species_slug)
    project_root = Path(output_root)
    datasets_root = project_root / f"{species_label}_Datasets"
    occurrences_root = project_root / f"{species_label}_US_Occurrences"
    uniform_name = f"{species_label}_US_Uniform_PA_NAIP_{chip_size}_topo_norm_may2026"
    blockcv_name = f"{species_label}_US_BlockCV_PA_NAIP_{chip_size}_topo_norm_may2026"

    return {
        "datasets_root": datasets_root,
        "occurrences_root": occurrences_root,
        "presence_background_points": occurrences_root / f"{species_label}_US_Presence_Background_Points.csv",
        "background_audit": occurrences_root / f"{species_label}_US_Presence_Background_Audit.csv",
        "uniform_root": datasets_root / uniform_name,
        "blockcv_root": datasets_root / blockcv_name,
    }

def resolve_column(
    columns,
    requested: str | None,
    candidates: list[str],
    label: str,
    required: bool,
) -> str | None:
    """Resolve an input CSV column with common aliases."""
    column_set = set(columns)

    if requested and requested != "auto":
        if requested in column_set:
            return requested
        if required:
            raise ValueError(f"Requested {label} column '{requested}' is missing.")
        return None

    for candidate in candidates:
        if candidate in column_set:
            return candidate

    if required:
        raise ValueError(
            f"Could not auto-detect {label} column. Tried: {candidates}. "
            f"Available columns: {list(columns)}"
        )
    return None

def build_naip_file_index(naip_folder: str) -> dict[str, str]:
    """
    Recursively index all NAIP .tif/.TIF files under naip_folder.

    Returns
    -------
    dict
        Mapping from lowercase basename and date-independent tile key to full
        file path. For example, both m_3908512_ne_16_060_20200612.tif and
        m_3908512_ne_16_060 point to the same local file.

    Notes
    -----
    If duplicate basenames exist, later matches overwrite earlier ones. Keep
    your archive organized so basenames are unique within the active project.
    """
    naip_index: dict[str, str] = {}
    file_count = 0
    duplicate_tile_keys = 0

    for root, _, files in os.walk(naip_folder):
        for filename in sorted(files):
            if filename.lower().endswith(".tif"):
                path = os.path.join(root, filename)
                basename = filename.lower()
                tile_key = naip_tile_key(filename)
                file_count += 1

                naip_index[basename] = path
                if tile_key:
                    duplicate_tile_keys += int(tile_key in naip_index)
                    existing = naip_index.get(tile_key)
                    if existing is None or basename > os.path.basename(existing).lower():
                        naip_index[tile_key] = path

    if not naip_index:
        raise RuntimeError(f"No .tif files found recursively under: {naip_folder}")

    print(f"Indexed {file_count:,} NAIP TIFF files under {naip_folder}")
    if duplicate_tile_keys:
        print(
            f"Found {duplicate_tile_keys:,} duplicate date-independent NAIP tile keys; "
            "using the latest dated local filename for each key."
        )
    return naip_index

def naip_tile_key(filename: str | None) -> str | None:
    """Return stable NAIP tile key without acquisition date, e.g. m_3908512_ne_16_060."""
    if not isinstance(filename, str) or not filename:
        return None
    stem = os.path.splitext(os.path.basename(filename).lower())[0]
    return re.sub(r"(?:_\d{8})+$", "", stem)

def resolve_naip_local_path(row: pd.Series, naip_file_index: dict[str, str]) -> tuple[str | None, str | None]:
    """Prefer existing local_path, then exact filename, then date-independent tile key."""
    local_path = row.get("local_path")
    if isinstance(local_path, str) and os.path.exists(local_path):
        return local_path, "local_path"

    filename = str(row.get("filename", "")).lower()
    local_path = naip_file_index.get(filename)
    if local_path and os.path.exists(local_path):
        return local_path, "exact_filename"

    tile_key = row.get("tile_key") or naip_tile_key(filename)
    local_path = naip_file_index.get(tile_key)
    if local_path and os.path.exists(local_path):
        return local_path, "tile_key"

    return None, None

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
    tileindex["tile_key"] = tileindex["filename"].apply(naip_tile_key)

    naip_file_index = build_naip_file_index(naip_folder)

    resolved = tileindex.apply(
        lambda row: resolve_naip_local_path(row, naip_file_index),
        axis=1,
        result_type="expand",
    )
    resolved.columns = ["local_path", "local_match_method"]
    tileindex[["local_path", "local_match_method"]] = resolved

    tileindex["downloaded"] = tileindex["local_path"].notna()
    downloaded_tiles = tileindex[tileindex["downloaded"]].copy()

    print(f"Tile index rows: {len(tileindex):,}")
    print(f"Downloaded tiles matched: {len(downloaded_tiles):,}")
    print("Local NAIP match methods:")
    print(tileindex["local_match_method"].fillna("unmatched").value_counts().sort_index())
    fallback_examples = tileindex[tileindex["local_match_method"] == "tile_key"][
        ["filename", "local_path"]
    ].head(5)
    if not fallback_examples.empty:
        print("Example date-independent tile-key matches:")
        print(fallback_examples.to_string(index=False))

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
    coordinate_uncertainty_col: str | None,
    max_coordinate_uncertainty_m: float | None,
) -> gpd.GeoDataFrame:
    """Load a species occurrence CSV and standardize the required columns."""
    df = pd.read_csv(csv_path)

    lat_col = resolve_column(
        df.columns,
        lat_col,
        ["decimalLatitude", "latitude", "lat", "Lat", "LAT"],
        "latitude",
        required=True,
    )
    lon_col = resolve_column(
        df.columns,
        lon_col,
        ["decimalLongitude", "longitude", "lon", "Long", "LON"],
        "longitude",
        required=True,
    )
    uncertainty_col = resolve_column(
        df.columns,
        coordinate_uncertainty_col,
        [
            "coordinateUncertaintyInMeters",
            "coordinate_uncertainty_m",
            "coordinate_uncertainty",
            "coordinateUncertainty",
            "uncertainty",
        ],
        "coordinate uncertainty",
        required=False,
    )

    print(
        "Occurrence schema: "
        f"lat='{lat_col}', lon='{lon_col}', "
        f"source='{source_col if source_col in df.columns else 'unknown'}', "
        f"coordinate_uncertainty='{uncertainty_col or 'not detected'}'"
    )

    df = df.copy().rename(columns={lat_col: "lat", lon_col: "lon"})

    if species_name_col in df.columns:
        df["species"] = df[species_name_col]
    else:
        df["species"] = infer_species_slug_from_filename(csv_path).replace("_", " ").title()

    if source_col in df.columns:
        df["source"] = df[source_col].fillna("unknown")
    else:
        df["source"] = "unknown"

    if uncertainty_col is not None:
        df["coordinate_uncertainty_m"] = pd.to_numeric(df[uncertainty_col], errors="coerce")
        before_uncertainty = len(df)
        if max_coordinate_uncertainty_m is not None:
            df = df[
                df["coordinate_uncertainty_m"].notna()
                & (df["coordinate_uncertainty_m"] <= max_coordinate_uncertainty_m)
            ].copy()
            print(
                "After coordinate uncertainty filter "
                f"<= {max_coordinate_uncertainty_m:g} m: {len(df):,} "
                f"({before_uncertainty - len(df):,} removed)"
            )
    else:
        df["coordinate_uncertainty_m"] = np.nan

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
