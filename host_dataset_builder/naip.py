from .common import *
from .io_utils import resolve_naip_local_path

def attach_naip_filename_or_url(
    points_gdf: gpd.GeoDataFrame,
    tileindex_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Spatially join points to the tileindex to preserve filename/url metadata."""
    cols = ["geometry"]
    for c in ["filename", "url", "tile_key", "local_path", "local_match_method"]:
        if c in tileindex_gdf.columns:
            cols.append(c)

    joined = gpd.sjoin(
        points_gdf,
        tileindex_gdf[cols],
        how="left",
        predicate="within",
    ).drop(columns=["index_right"], errors="ignore")

    return joined

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
        candidate = row.get("local_path")
        if not isinstance(candidate, str):
            candidate = naip_file_index.get(str(row["filename"]).lower())
        if not candidate:
            candidate = naip_file_index.get(row.get("tile_key") or naip_tile_key(row.get("filename")))
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
        fp = r.get("local_path")
        if not isinstance(fp, str):
            fp = naip_file_index.get(str(r["filename"]).lower())
        if not fp:
            fp = naip_file_index.get(r.get("tile_key") or naip_tile_key(r.get("filename")))
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

