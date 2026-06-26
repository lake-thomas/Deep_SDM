from .common import *
from .splits import add_normalized_lat_lon
from .env import extract_worldclim_vars_for_point, extract_ghm_for_point, is_valid_env_record
from .naip import extract_naip_chip_for_point
from .topo import extract_topo_for_naip_chip

def relative_posix_path(path: Path, root: Path) -> str:
    """Return a portable relative path for dataset CSV path columns.

    Dataset CSVs may be read on Linux workstations or Windows-mounted storage.
    Normalizing generated relative paths to POSIX separators avoids creating
    CSV rows that work only on the operating system used for dataset creation.
    Existing input metadata path columns are preserved as source provenance.
    """
    return os.path.relpath(path, root).replace(os.sep, "/")

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
                topo_chip_rel = relative_posix_path(topo_chip_fp, dataset_root)

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

        # Columns with paths are relative to the dataset root for better portability
        rec = {
            "sample_id": sample_id,
            "chip_path": relative_posix_path(chip_fp, dataset_root),
            "topo_chip_path": topo_chip_rel,
            "split": split,
            "presence": presence,
            "lat": lat,
            "lon": lon,
            "source": source,
        }

        for optional_col in [
            "coordinate_uncertainty_m",
            "nearest_presence_km",
            "background_sampling_rule",
            "filename",
            "url",
            "tile_key",
            "local_path",
            "local_match_method",
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
