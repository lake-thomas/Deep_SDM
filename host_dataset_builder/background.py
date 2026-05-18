from .common import *
from .utils import geometry_union
from .naip import attach_naip_filename_or_url

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

def assign_presence_ids(presence_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Attach stable per-run presence IDs used to audit local backgrounds."""
    pres = presence_gdf.copy().reset_index(drop=True)
    if "presence_id" not in pres.columns:
        pres["presence_id"] = [f"presence_{i:06d}" for i in range(len(pres))]
    return pres

def _sample_one_background_per_presence(
    candidates: gpd.GeoDataFrame,
    presence_gdf: gpd.GeoDataFrame,
    seed: int,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """
    Select at most one background per presence while keeping WorldClim cells unique.

    Candidate generation can find several local points for a presence, and the
    same raster cell can be reachable from nearby presences. This greedy pass
    keeps the output auditable without changing the one-background-per-presence
    target when enough valid local candidates exist.
    """
    required_cols = {"paired_presence_id", "cell_id"}
    missing = required_cols.difference(candidates.columns)
    if missing:
        raise ValueError(f"Background candidates missing required columns: {sorted(missing)}")

    if candidates.empty:
        selected = candidates.copy()
    else:
        shuffled = candidates.sample(frac=1.0, random_state=seed).copy()
        selected_rows = []
        used_cells: set[str] = set()
        for presence_id in presence_gdf["presence_id"].tolist():
            local = shuffled[
                (shuffled["paired_presence_id"] == presence_id)
                & (~shuffled["cell_id"].isin(used_cells))
            ]
            if local.empty:
                continue
            row = local.iloc[0].copy()
            used_cells.add(str(row["cell_id"]))
            selected_rows.append(row)

        selected = gpd.GeoDataFrame(selected_rows, geometry="geometry", crs=candidates.crs)

    found = set(selected["paired_presence_id"]) if not selected.empty else set()
    audit_rows = []
    for _, row in presence_gdf.to_crs("EPSG:4326").iterrows():
        if row["presence_id"] in found:
            continue
        audit_rows.append({
            "presence_id": row["presence_id"],
            "species": row.get("species"),
            "source": row.get("source"),
            "lat": float(row.geometry.y),
            "lon": float(row.geometry.x),
            "coordinate_uncertainty_m": row.get("coordinate_uncertainty_m"),
            "audit_reason": "no_valid_unique_local_background",
        })

    return selected, pd.DataFrame(audit_rows)

def sample_background_points_radial(
    presence_gdf: gpd.GeoDataFrame,
    wc_reference_raster_fp: str,
    background_multiplier: float,
    seed: int,
    buffer_km: float,
    inner_buffer_km: float = 0.0,
    max_sampling_rounds: int = 5,
) -> gpd.GeoDataFrame:
    """
    Fast doughnut background sampling for large national host datasets.

    Candidates are generated around every presence in EPSG:5070, then a KD-tree
    enforces the nearest-presence distance constraint. This avoids constructing
    one huge union/difference polygon from all presence buffers, which is the
    slowest step for high-record hosts, while still making the local
    background-to-presence pairing auditable.
    """
    presence_gdf = assign_presence_ids(presence_gdf)
    n_background = len(presence_gdf)
    if n_background == 0:
        raise ValueError("No presence points available after thinning.")

    # Buffer around presences in meters, used for both candidate generation and distance-based filtering.
    inner_m = inner_buffer_km * 1000.0
    outer_m = buffer_km * 1000.0
    if inner_m >= outer_m:
        raise ValueError("inner_buffer_km must be smaller than buffer_km.")

    with rasterio.open(wc_reference_raster_fp) as src:
        affine = src.transform
        raster_crs = src.crs

    pres_for_cells = presence_gdf.to_crs(raster_crs).copy()
    rows_pr, cols_pr = rasterio.transform.rowcol(
        affine,
        pres_for_cells.geometry.x.values,
        pres_for_cells.geometry.y.values,
    )
    presence_cells = {f"{r}_{c}" for r, c in zip(rows_pr, cols_pr)}

    presence_proj = presence_gdf.to_crs("EPSG:5070").copy()
    presence_xy = np.column_stack([presence_proj.geometry.x.values, presence_proj.geometry.y.values])
    presence_ids = presence_proj["presence_id"].to_numpy()
    presence_tree = cKDTree(presence_xy)

    candidates_per_presence = max(1, int(math.ceil(background_multiplier)))
    valid_candidate_chunks = []
    pooled = None

    for round_i in range(max_sampling_rounds):
        rng = np.random.default_rng(seed + round_i)
        origin_idx = np.repeat(np.arange(len(presence_xy)), candidates_per_presence)
        origins = presence_xy[origin_idx]
        n_candidates_this_round = len(origin_idx)
        angles = rng.uniform(0.0, 2.0 * np.pi, size=n_candidates_this_round)
        # Area-uniform radial distance within the annulus.
        distances = np.sqrt(rng.uniform(inner_m ** 2, outer_m ** 2, size=n_candidates_this_round))
        candidate_xy = np.column_stack([
            origins[:, 0] + distances * np.cos(angles),
            origins[:, 1] + distances * np.sin(angles),
        ])

        nearest_m, _ = presence_tree.query(candidate_xy, k=1)
        keep = (nearest_m >= inner_m) & (nearest_m <= outer_m)
        if not np.any(keep):
            continue

        candidates_proj = gpd.GeoDataFrame(
            {
                "nearest_presence_km": nearest_m[keep] / 1000.0,
                "paired_presence_id": presence_ids[origin_idx[keep]],
            },
            geometry=gpd.points_from_xy(candidate_xy[keep, 0], candidate_xy[keep, 1]),
            crs="EPSG:5070",
        )

        candidates = candidates_proj.to_crs(raster_crs)
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
                pooled.groupby(["paired_presence_id", "cell_id"], group_keys=False)
                .apply(lambda x: x.sample(n=1, random_state=seed))
                .reset_index(drop=True)
            )

            paired_count = pooled["paired_presence_id"].nunique()
            print(
                f"Radial sampling round {round_i + 1}: "
                f"{len(pooled):,} valid local candidate cells for "
                f"{paired_count:,} of {n_background:,} presences"
            )

            selected, audit = _sample_one_background_per_presence(pooled, presence_gdf, seed)
            if len(audit) == 0:
                break
    else:
        selected, audit = _sample_one_background_per_presence(
            pooled if pooled is not None else gpd.GeoDataFrame(
                columns=["paired_presence_id", "cell_id", "geometry"],
                geometry="geometry",
                crs=raster_crs,
            ),
            presence_gdf,
            seed,
        )
        print(
            f"Only {len(selected):,} valid unique radial local backgrounds were available "
            f"for {n_background:,} presences after {max_sampling_rounds} rounds. "
            "Continuing with retained pairs and writing a background audit."
        )

    background_sampled = selected.copy()
    background_sampled = background_sampled.to_crs("EPSG:4326")
    background_sampled["lat"] = background_sampled.geometry.y
    background_sampled["lon"] = background_sampled.geometry.x
    background_sampled["source"] = "background"
    background_sampled["background_sampling_rule"] = f"radial_doughnut_{inner_buffer_km:g}_{buffer_km:g}_km"
    background_sampled.attrs["background_audit"] = audit

    if not background_sampled.empty:
        print(
            "Final radial background nearest-presence distance summary, km: "
            f"min={background_sampled['nearest_presence_km'].min():.2f}, "
            f"median={background_sampled['nearest_presence_km'].median():.2f}, "
            f"max={background_sampled['nearest_presence_km'].max():.2f}"
        )

    return background_sampled

def sample_background_points(
    presence_gdf: gpd.GeoDataFrame,
    downloaded_tiles_gdf: gpd.GeoDataFrame,
    wc_reference_raster_fp: str,
    background_multiplier: float,
    seed: int,
    buffer_km: float,
    inner_buffer_km: float = 0.0,
    max_sampling_rounds: int = 25,
    max_naip_footprint_union_tiles: int = 50000,
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
    presence_gdf = assign_presence_ids(presence_gdf)
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

    outer_buffer_proj = geometry_union(presence_proj.buffer(outer_buffer_m))
    if inner_buffer_km > 0:
        inner_buffer_proj = geometry_union(presence_proj.buffer(inner_buffer_m))
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

    # 2. Restrict to downloaded NAIP footprint when feasible. For national
    # tile indexes, unioning hundreds of thousands of tile polygons can dominate
    # runtime. In that case, sample from the presence doughnut and let the later
    # NAIP chip extraction step enforce downloaded-tile availability.
    if len(downloaded_tiles_gdf) <= max_naip_footprint_union_tiles:
        footprint = geometry_union(downloaded_tiles_gdf.to_crs("EPSG:4326").geometry)
        sampling_area = footprint.intersection(sampling_zone_wgs84).buffer(0)
    else:
        print(
            f"Skipping expensive NAIP footprint union for {len(downloaded_tiles_gdf):,} "
            "downloaded tiles; NAIP chip extraction will enforce tile availability."
        )
        sampling_area = sampling_zone_wgs84.buffer(0)

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
    presence_union_proj = geometry_union(presence_proj.geometry)
    background_sampled["nearest_presence_km"] = bg_for_dist.geometry.distance(presence_union_proj) / 1000.0

    background_sampled = background_sampled.to_crs("EPSG:4326")
    background_sampled["lat"] = background_sampled.geometry.y
    background_sampled["lon"] = background_sampled.geometry.x
    background_sampled["source"] = "background"
    background_sampled["background_sampling_rule"] = f"doughnut_{inner_buffer_km:g}_{buffer_km:g}_km"
    background_sampled.attrs["background_audit"] = pd.DataFrame()

    print(
        "Final background nearest-presence distance summary, km: "
        f"min={background_sampled['nearest_presence_km'].min():.2f}, "
        f"median={background_sampled['nearest_presence_km'].median():.2f}, "
        f"max={background_sampled['nearest_presence_km'].max():.2f}"
    )

    return background_sampled

def build_presence_background_points(
    presence_gdf: gpd.GeoDataFrame,
    background_gdf: gpd.GeoDataFrame,
    tileindex_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Combine presences and background into a single GeoDataFrame."""
    pres = assign_presence_ids(presence_gdf).to_crs("EPSG:4326")
    pres["lat"] = pres.geometry.y
    pres["lon"] = pres.geometry.x
    pres["presence"] = 1

    bg = background_gdf.copy().to_crs("EPSG:4326")
    bg["presence"] = 0

    pres = attach_naip_filename_or_url(pres, tileindex_gdf)
    bg = attach_naip_filename_or_url(bg, tileindex_gdf)

    keep_cols = ["presence_id", "paired_presence_id", "species", "source", "lat", "lon", "presence", "geometry"]
    for extra in [
        "coordinate_uncertainty_m",
        "filename",
        "url",
        "tile_key",
        "local_path",
        "local_match_method",
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

def audit_presences_without_background(
    points_gdf: gpd.GeoDataFrame,
    reason: str,
) -> pd.DataFrame:
    """Return presences that do not have a retained paired local background."""
    if "presence_id" not in points_gdf.columns:
        return pd.DataFrame()
    pres = points_gdf[points_gdf["presence"] == 1].copy()
    if pres.empty:
        return pd.DataFrame()
    if "paired_presence_id" not in points_gdf.columns:
        found = set()
    else:
        found = set(points_gdf.loc[points_gdf["presence"] == 0, "paired_presence_id"].dropna())

    missing = pres[~pres["presence_id"].isin(found)].copy()
    if missing.empty:
        return pd.DataFrame()
    missing = missing.to_crs("EPSG:4326")
    rows = []
    for _, row in missing.iterrows():
        rows.append({
            "presence_id": row["presence_id"],
            "species": row.get("species"),
            "source": row.get("source"),
            "lat": float(row.geometry.y),
            "lon": float(row.geometry.x),
            "coordinate_uncertainty_m": row.get("coordinate_uncertainty_m"),
            "audit_reason": reason,
        })
    return pd.DataFrame(rows)

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
