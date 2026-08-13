from .common import *
from .io_utils import (
    SOURCE_METADATA_COLUMNS,
    add_canonical_naip_source_metadata,
    naip_tile_key,
)
from rasterio.features import geometry_mask
from rasterio.warp import transform_geom


def _add_legacy_source_aliases(points: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Retain June column names while making exact source fields authoritative."""
    out = points.copy()
    alias_map = {
        "filename": "source_filename",
        "tile_key": "source_tile_key",
        "local_path": "source_local_path",
    }
    for legacy, source in alias_map.items():
        if source in out.columns:
            if legacy not in out.columns:
                out[legacy] = out[source]
            else:
                out[legacy] = out[legacy].where(out[legacy].notna(), out[source])
    if "source_local_path" in out.columns:
        out["local_match_method"] = out.get(
            "local_match_method",
            pd.Series(index=out.index, dtype=object),
        ).fillna("source_locked_exact_local_path")
    return out


def _canonical_tileindex_view(tileindex_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Reuse pre-normalized production indexes instead of copying them per chip."""
    if set(SOURCE_METADATA_COLUMNS).issubset(tileindex_gdf.columns):
        return tileindex_gdf
    return add_canonical_naip_source_metadata(tileindex_gdf)


def attach_naip_filename_or_url(
    points_gdf: gpd.GeoDataFrame,
    tileindex_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Assign one deterministic exact local acquisition to each point.

    Backgrounds sampled by the August USBg sampler already carry their selected
    footprint and are returned unchanged. Other records (normally presences)
    are spatially joined to the canonical local tile index. If overlapping
    footprints contain a point, the newest exact acquisition date is selected
    deterministically and the policy is recorded explicitly.
    """
    points = points_gdf.copy()
    if "source_local_path" in points.columns and points["source_local_path"].notna().all():
        return _add_legacy_source_aliases(points)

    tiles = _canonical_tileindex_view(tileindex_gdf)
    tile_columns = ["geometry", *SOURCE_METADATA_COLUMNS]
    if "url" in tiles.columns:
        tile_columns.append("url")

    points = points.drop(columns=[c for c in SOURCE_METADATA_COLUMNS if c in points.columns])
    points = points.reset_index(drop=False).rename(columns={"index": "_original_point_index"})
    points["_point_row_id"] = np.arange(len(points))
    joined = gpd.sjoin(
        points,
        tiles[tile_columns],
        how="left",
        predicate="within",
    )
    joined["_source_date_sort"] = joined["source_acq_date"].fillna("")
    joined["_source_key_sort"] = joined["source_tile_key"].fillna("")
    joined = joined.sort_values(
        ["_point_row_id", "_source_date_sort", "_source_key_sort"],
        ascending=[True, False, True],
        kind="mergesort",
    ).drop_duplicates("_point_row_id", keep="first")
    joined["source_selection_method"] = np.where(
        joined["source_local_path"].notna(),
        "spatial_join_latest_exact_acquisition",
        "no_local_source_match",
    )
    joined = joined.sort_values("_point_row_id").drop(
        columns=[
            "index_right",
            "_point_row_id",
            "_source_date_sort",
            "_source_key_sort",
        ],
        errors="ignore",
    )
    joined = joined.set_index("_original_point_index")
    joined.index.name = points_gdf.index.name
    return _add_legacy_source_aliases(joined)


def _source_record_value(source_record, key: str):
    if source_record is None:
        return None
    if hasattr(source_record, "get"):
        return source_record.get(key)
    return None


def _path_identity(path: str) -> str:
    """Normalize a local source path for exact provenance comparisons."""
    return os.path.normcase(os.path.realpath(path))


def _chip_output_profile(source_profile: dict, chip, chip_size: int, transform, crs) -> dict:
    """Return a valid GeoTIFF profile independent of source-scene block sizes."""
    profile = source_profile.copy()
    profile.update(
        driver="GTiff",
        height=chip_size,
        width=chip_size,
        count=chip.shape[0],
        dtype=chip.dtype,
        crs=crs,
        transform=transform,
        compress="deflate",
        BIGTIFF="IF_SAFER",
    )
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    if chip_size >= 16:
        block_size = min(256, (chip_size // 16) * 16)
        profile.update(tiled=True, blockxsize=block_size, blockysize=block_size)
    else:
        profile.pop("tiled", None)
    return profile


def _strict_sources_are_compatible(datasets) -> bool:
    """Reject silent band, dtype, or ground-resolution changes within a chip.

    Same-acquisition NAIP source tiles are not guaranteed to share one exact
    pixel-grid origin. They may still be composited deterministically onto the
    intended source grid with nearest-neighbor resampling.
    """
    reference = datasets[0]
    if reference.count != 4:
        return False
    ref_res = np.asarray(reference.res, dtype=float)
    for dataset in datasets[1:]:
        if dataset.count != reference.count or dataset.dtypes != reference.dtypes:
            return False
        if not np.allclose(np.asarray(dataset.res, dtype=float), ref_res, rtol=0.0, atol=1e-9):
            return False
    return True


def _strict_source_chip(
    lon: float,
    lat: float,
    out_fp: str,
    chip_size: int,
    tileindex_gdf: gpd.GeoDataFrame,
    source_record,
    min_valid_frac: float,
) -> dict | None:
    """Write one chip using only the selected state/acquisition domain."""
    source_path = _source_record_value(source_record, "source_local_path")
    source_state = _source_record_value(source_record, "source_archive_state")
    source_date = _source_record_value(source_record, "source_acq_date")
    source_tile_key = _source_record_value(source_record, "source_tile_key")
    if not isinstance(source_path, str) or not os.path.exists(source_path):
        return None
    if pd.isna(source_state) or pd.isna(source_date):
        return None
    source_state = str(source_state)
    source_date = str(source_date)

    point_wgs84 = Point(lon, lat)
    with rasterio.open(source_path) as source_ds:
        point_proj = gpd.GeoSeries([point_wgs84], crs="EPSG:4326").to_crs(source_ds.crs).iloc[0]
        row, col = source_ds.index(point_proj.x, point_proj.y)
        if row < 0 or col < 0 or row >= source_ds.height or col >= source_ds.width:
            return None
        half = chip_size // 2
        chip_window = Window(
            col_off=col - half,
            row_off=row - half,
            width=chip_size,
            height=chip_size,
        )
        chip_bounds = rasterio.windows.bounds(chip_window, source_ds.transform)
        chip_transform = rasterio.windows.transform(chip_window, source_ds.transform)
        chip_crs = source_ds.crs
        source_res = source_ds.res

    bbox = gpd.GeoSeries([box(*chip_bounds)], crs=chip_crs).to_crs("EPSG:4326").iloc[0]
    tiles = _canonical_tileindex_view(tileindex_gdf)
    candidate_positions = tiles.sindex.query(bbox, predicate="intersects")
    candidates = tiles.iloc[candidate_positions]
    same_domain = candidates[
        (candidates["source_archive_state"].astype(str) == source_state)
        & (candidates["source_acq_date"].astype(str) == source_date)
    ].copy()
    if same_domain.empty:
        return None


    source_identity = _path_identity(source_path)
    same_domain["_source_path_identity"] = same_domain["source_local_path"].apply(
        lambda path: _path_identity(str(path)) if pd.notna(path) else None
    )
    if source_identity not in set(same_domain["_source_path_identity"].dropna()):
        # Strict mode never accepts a source record that is absent from the
        # canonical tile-index rows intersecting the requested chip.
        return None

    ordered_paths = []
    if source_tile_key is not None:
        source_rows = same_domain[same_domain["source_tile_key"].astype(str) == str(source_tile_key)]
        ordered_paths.extend(source_rows["source_local_path"].dropna().astype(str).tolist())
    ordered_paths.append(source_path)
    ordered_paths.extend(same_domain["source_local_path"].dropna().astype(str).sort_values().tolist())
    source_paths = []
    seen_source_paths = set()
    for path in ordered_paths:
        if not os.path.exists(path):
            continue
        identity = _path_identity(path)
        if identity in seen_source_paths:
            continue
        seen_source_paths.add(identity)
        source_paths.append(path)
    if not source_paths:
        return None


    datasets = []
    try:
        for path in source_paths:
            datasets.append(rasterio.open(path))
        if not datasets or not _strict_sources_are_compatible([datasets[0]]):
            return None

        # Composite only the portion each same-domain raster actually covers.
        # The intended file is first and therefore authoritative wherever it
        # has support. Neighbors fill only uncovered pixels. This avoids the
        # prior all-or-nothing rejection when an irrelevant overlapping index
        # polygon referred to a fractional-grid-offset neighbor.
        reference = datasets[0]
        window_inside_intended = (
            chip_window.col_off >= 0
            and chip_window.row_off >= 0
            and chip_window.col_off + chip_window.width <= reference.width
            and chip_window.row_off + chip_window.height <= reference.height
        )
        if window_inside_intended:
            # The common path needs no mosaic or reprojection. Preserve the
            # intended NAIP bytes exactly and ignore imprecise neighboring
            # footprint polygons that happen to intersect the chip bbox.
            chip = reference.read(window=chip_window)
            if chip.shape != (reference.count, chip_size, chip_size):
                return None
            covered = np.ones((chip_size, chip_size), dtype=bool)
            used_paths: list[str] = [source_paths[0]]
        else:
            chip = np.zeros(
                (reference.count, chip_size, chip_size),
                dtype=np.dtype(reference.dtypes[0]),
            )
            covered = np.zeros((chip_size, chip_size), dtype=bool)
            used_paths = []
        resampled_neighbor = False
        for path, dataset in zip(source_paths, datasets):
            if covered.all():
                break
            if not _strict_sources_are_compatible([reference, dataset]):
                continue
            raster_polygon = box(*dataset.bounds).__geo_interface__
            if dataset.crs != chip_crs:
                raster_polygon = transform_geom(
                    dataset.crs,
                    chip_crs,
                    raster_polygon,
                    precision=-1,
                )
            source_coverage = geometry_mask(
                [raster_polygon],
                out_shape=(chip_size, chip_size),
                transform=chip_transform,
                invert=True,
                all_touched=False,
            )
            new_coverage = source_coverage & ~covered
            if not new_coverage.any():
                continue

            warped = np.zeros_like(chip)
            for band_index in range(1, reference.count + 1):
                reproject(
                    source=rasterio.band(dataset, band_index),
                    destination=warped[band_index - 1],
                    src_transform=dataset.transform,
                    src_crs=dataset.crs,
                    src_nodata=None,
                    dst_transform=chip_transform,
                    dst_crs=chip_crs,
                    dst_nodata=0,
                    resampling=Resampling.nearest,
                )
            chip[:, new_coverage] = warped[:, new_coverage]
            covered |= source_coverage
            used_paths.append(path)

            same_grid = dataset.crs == chip_crs
            if same_grid:
                x_offset = (dataset.transform.c - chip_transform.c) / source_res[0]
                y_offset = (dataset.transform.f - chip_transform.f) / source_res[1]
                same_grid = np.isclose(x_offset, round(x_offset), atol=1e-6) and np.isclose(
                    y_offset,
                    round(y_offset),
                    atol=1e-6,
                )
            resampled_neighbor |= not bool(same_grid)

        valid_frac = float(covered.mean())
        if valid_frac + 1e-12 < min_valid_frac:
            return None
        if not used_paths:
            return None

        used_path_identities = {_path_identity(path) for path in used_paths}
        used_rows = same_domain[
            same_domain["_source_path_identity"].isin(used_path_identities)
        ].copy()
        actual_tile_keys = sorted(
            used_rows["source_tile_key"].dropna().astype(str).unique()
        )
        actual_filenames = sorted(
            used_rows["source_filename"].dropna().astype(str).unique()
        )
        intended_source_used = source_identity in used_path_identities
        Path(out_fp).parent.mkdir(parents=True, exist_ok=True)
        partial_fp = f"{out_fp}.partial"
        profile = _chip_output_profile(
            source_profile=datasets[0].profile,
            chip=chip,
            chip_size=chip_size,
            transform=chip_transform,
            crs=chip_crs,
        )
        with rasterio.open(partial_fp, "w", **profile) as dst:
            dst.write(chip)
            dst.update_tags(
                naip_source_lock_mode="strict",
                naip_source_archive_state=source_state,
                naip_source_acq_date=source_date,
                naip_source_paths=";".join(used_paths),
                naip_valid_frac=f"{valid_frac:.12f}",
                naip_neighbor_resampled=str(bool(resampled_neighbor)).lower(),
            )
        os.replace(partial_fp, out_fp)

        return {
            "actual_source_lock_mode": "strict",
            "actual_source_archive_state": source_state,
            "actual_source_acq_date": source_date,
            "actual_source_paths": ";".join(used_paths),
            "actual_source_paths_json": json.dumps(used_paths),
            "actual_source_tile_keys": ";".join(actual_tile_keys),
            "actual_source_filenames": ";".join(actual_filenames),
            "actual_source_count": len(used_paths),
            "actual_multi_state": False,
            "actual_multi_acquisition": False,
            "intended_source_used": intended_source_used,
            "source_provenance_match": intended_source_used,
            "naip_valid_frac": valid_frac,
            "actual_source_resampled": bool(resampled_neighbor),
            "actual_source_resampling_method": (
                "nearest_to_intended_source_grid" if resampled_neighbor else "none"
            ),
        }
    finally:
        for dataset in datasets:
            dataset.close()


def _legacy_chip(
    lon: float,
    lat: float,
    out_fp: str,
    chip_size: int,
    tileindex_gdf: gpd.GeoDataFrame,
    naip_file_index: dict,
) -> dict | None:
    """Preserve the June multi-tile extraction path for compatibility."""
    point_wgs84 = Point(lon, lat)
    matches = tileindex_gdf[tileindex_gdf.geometry.intersects(point_wgs84)]
    if matches.empty:
        return None

    naip_fp = None
    for _, row in matches.iterrows():
        candidate = row.get("local_path")
        if not isinstance(candidate, str):
            candidate = naip_file_index.get(str(row["filename"]).lower())
        if not candidate:
            candidate = naip_file_index.get(row.get("tile_key") or naip_tile_key(row.get("filename")))
        if candidate and os.path.exists(candidate):
            naip_fp = candidate
            break
    if naip_fp is None:
        return None

    with rasterio.open(naip_fp) as src:
        point_proj = gpd.GeoSeries([point_wgs84], crs="EPSG:4326").to_crs(src.crs).iloc[0]
        row, col = src.index(point_proj.x, point_proj.y)
        half = chip_size // 2
        chip_window = Window(col_off=col - half, row_off=row - half, width=chip_size, height=chip_size)
        chip_bounds = rasterio.windows.bounds(chip_window, src.transform)
        bbox_gdf = gpd.GeoDataFrame(geometry=[box(*chip_bounds)], crs=src.crs).to_crs("EPSG:4326")

    overlapping = tileindex_gdf[tileindex_gdf.geometry.intersects(bbox_gdf.geometry.iloc[0])]
    naip_paths = []
    for _, row in overlapping.iterrows():
        path = row.get("local_path")
        if not isinstance(path, str):
            path = naip_file_index.get(str(row["filename"]).lower())
        if not path:
            path = naip_file_index.get(row.get("tile_key") or naip_tile_key(row.get("filename")))
        if path and os.path.exists(path):
            naip_paths.append(path)
    naip_paths = list(dict.fromkeys(naip_paths))
    if not naip_paths:
        return None

    datasets = []
    try:
        for path in naip_paths:
            datasets.append(rasterio.open(path))
        if len({dataset.crs.to_string() for dataset in datasets}) != 1:
            return None
        mosaic, out_transform = merge(
            datasets,
            bounds=chip_bounds,
            res=datasets[0].res,
            nodata=0,
        )
        chip = mosaic[:, :chip_size, :chip_size]
        if chip.shape[1] < chip_size or chip.shape[2] < chip_size:
            padded = np.zeros((chip.shape[0], chip_size, chip_size), dtype=chip.dtype)
            padded[:, :chip.shape[1], :chip.shape[2]] = chip
            chip = padded

        Path(out_fp).parent.mkdir(parents=True, exist_ok=True)
        profile = _chip_output_profile(
            source_profile=datasets[0].profile,
            chip=chip,
            chip_size=chip.shape[1],
            transform=out_transform,
            crs=datasets[0].crs,
        )
        with rasterio.open(out_fp, "w", **profile) as dst:
            dst.write(chip)
            dst.update_tags(naip_source_lock_mode="legacy")
        return {
            "actual_source_lock_mode": "legacy",
            "actual_source_paths": ";".join(naip_paths),
            "actual_source_paths_json": json.dumps(naip_paths),
            "actual_source_count": len(naip_paths),
            "actual_multi_state": float("nan"),
            "actual_multi_acquisition": float("nan"),
            "intended_source_used": float("nan"),
            "source_provenance_match": float("nan"),
            "naip_valid_frac": float("nan"),
        }
    finally:
        for dataset in datasets:
            dataset.close()


def extract_naip_chip_for_point(
    lon: float,
    lat: float,
    out_fp: str,
    chip_size: int,
    tileindex_gdf: gpd.GeoDataFrame,
    naip_file_index: dict,
    source_record=None,
    source_lock_mode: str = "legacy",
    min_valid_frac: float = 1.0,
) -> dict | None:
    """Extract a NAIP chip and return its actual source-provenance record."""
    if source_lock_mode == "strict":
        return _strict_source_chip(
            lon=lon,
            lat=lat,
            out_fp=out_fp,
            chip_size=chip_size,
            tileindex_gdf=tileindex_gdf,
            source_record=source_record,
            min_valid_frac=min_valid_frac,
        )
    if source_lock_mode == "legacy":
        return _legacy_chip(
            lon=lon,
            lat=lat,
            out_fp=out_fp,
            chip_size=chip_size,
            tileindex_gdf=tileindex_gdf,
            naip_file_index=naip_file_index,
        )
    raise ValueError(f"Unsupported NAIP source lock mode: {source_lock_mode}")
