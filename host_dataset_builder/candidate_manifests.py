"""Build traceable background superpools and nested manifests only.

This module intentionally stops before NAIP/environment/topography extraction.
It creates the evidence needed to review sampling coverage, nesting, source
availability, and split leakage before any expensive dataset build begins.
"""

from .common import *

import hashlib
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

from .background import (
    _allocate_domain_counts,
    assign_presence_ids,
    deduplicate_presences_to_worldclim_cells,
    sample_background_points_us_naip,
    spatially_thin_points,
)
from .env import raw_environment_valid_mask, sample_raw_environment_for_points
from .io_utils import add_canonical_naip_source_metadata, load_occurrence_csv
from .naip import attach_naip_filename_or_url
from .splits import assign_spatial_blocks, stratified_train_val_test_split


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state(workdir: Path) -> dict:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=workdir,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        return {
            "commit": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current"),
            "dirty": bool(run("status", "--porcelain")),
        }
    except Exception as error:
        return {"error": str(error)}


def _builder_source_hashes(package_dir: Path) -> dict[str, str]:
    """Hash the exact local builder sources, including uncommitted versions."""
    files = sorted(package_dir.glob("*.py"))
    readme = package_dir / "README.md"
    if readme.is_file():
        files.append(readme)
    return {path.name: _sha256(path) for path in files}


def _sampling_domain(frame: pd.DataFrame, domain_columns: list[str]) -> pd.Series:
    missing = [column for column in domain_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Candidate-domain columns are missing: {missing}")
    if frame[domain_columns].isna().any(axis=None):
        raise ValueError("Candidate domains contain missing values.")
    return frame[domain_columns].astype(str).agg("|".join, axis=1)


def _allocate_with_capacity(
    weights: pd.Series,
    total: int,
    minimum_per_domain: int,
    capacities: pd.Series,
) -> pd.Series:
    """Allocate an exact total while respecting finite superpool capacity."""
    weights = weights.reindex(capacities.index).fillna(0.0).astype(float)
    capacities = capacities.astype(int)
    if int(capacities.sum()) < total:
        raise RuntimeError(
            f"Superpool capacity is {int(capacities.sum()):,}, below requested {total:,}."
        )
    desired = _allocate_domain_counts(weights, total, minimum_per_domain)
    allocation = pd.Series(
        np.minimum(desired.to_numpy(dtype=int), capacities.to_numpy(dtype=int)),
        index=capacities.index,
        dtype=int,
    )
    deficit = total - int(allocation.sum())
    while deficit > 0:
        remaining_capacity = capacities - allocation
        eligible = remaining_capacity[remaining_capacity > 0]
        if eligible.empty:
            raise RuntimeError("Unable to redistribute a capacity-limited domain allocation.")
        additions = _allocate_domain_counts(weights.loc[eligible.index], deficit, 0)
        additions = pd.Series(
            np.minimum(additions.to_numpy(dtype=int), eligible.to_numpy(dtype=int)),
            index=eligible.index,
            dtype=int,
        )
        if int(additions.sum()) == 0:
            chosen = weights.loc[eligible.index].sort_values(
                ascending=False, kind="mergesort"
            ).index[0]
            additions.loc[chosen] = 1
        allocation.loc[additions.index] += additions
        deficit = total - int(allocation.sum())
    return allocation


def assign_weighted_random_order(
    pool: pd.DataFrame,
    domain_areas: pd.Series,
    seed: int,
) -> pd.DataFrame:
    """Rank a stratified proposal pool to recover an area-weighted random arm."""
    out = pool.copy()
    counts = out["background_sampling_domain"].value_counts()
    per_candidate_weight = domain_areas.reindex(counts.index).fillna(0.0) / counts
    if (per_candidate_weight <= 0).any():
        bad = per_candidate_weight[per_candidate_weight <= 0].index.tolist()
        raise RuntimeError(f"Non-positive area weights for candidate domains: {bad[:10]}")
    weights = out["background_sampling_domain"].map(per_candidate_weight).to_numpy(float)
    rng = np.random.default_rng(seed)
    uniform = np.clip(rng.random(len(out)), np.finfo(float).tiny, 1.0)
    out["random_priority"] = -np.log(uniform) / weights
    out = out.sort_values(
        ["random_priority", "candidate_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    out["random_proposal_rank"] = np.arange(1, len(out) + 1)
    return out


def assign_balanced_order(
    pool: pd.DataFrame,
    domain_areas: pd.Series,
    target: int,
    minimum_per_domain: int,
    seed: int,
) -> pd.DataFrame:
    """Create one nested minimum-depth then area-residual balanced ordering."""
    out = pool.copy()
    capacities = out["background_sampling_domain"].value_counts().sort_index()
    allocation = _allocate_with_capacity(
        weights=domain_areas,
        total=target,
        minimum_per_domain=minimum_per_domain,
        capacities=capacities,
    )

    rng = np.random.default_rng(seed)
    out["_within_domain_random"] = rng.random(len(out))
    out = out.sort_values(
        ["background_sampling_domain", "_within_domain_random", "candidate_id"],
        kind="mergesort",
    )
    out["_within_domain_rank"] = (
        out.groupby("background_sampling_domain").cumcount() + 1
    )

    domain_order = pd.Series(
        rng.permutation(len(capacities)),
        index=capacities.index,
        dtype=int,
    )
    out["_domain_order"] = out["background_sampling_domain"].map(domain_order)
    desired = out["background_sampling_domain"].map(allocation).astype(int)
    warm_count = np.minimum(desired, minimum_per_domain)
    warm = out[out["_within_domain_rank"] <= warm_count].copy()
    warm["_balanced_sort"] = (
        (warm["_within_domain_rank"] - 1) * len(capacities) + warm["_domain_order"]
    )
    warm["_balanced_phase"] = 0

    residual = out[
        (out["_within_domain_rank"] > warm_count)
        & (out["_within_domain_rank"] <= desired)
    ].copy()
    residual_denominator = (desired - warm_count).clip(lower=1)
    residual["_balanced_sort"] = (
        (residual["_within_domain_rank"] - warm_count) / residual_denominator
        + residual["_domain_order"] * 1e-9
    )
    residual["_balanced_phase"] = 1

    remainder = out[out["_within_domain_rank"] > desired].copy()
    remainder["_balanced_sort"] = remainder["_within_domain_random"]
    remainder["_balanced_phase"] = 2

    # GeoPandas propagates the sampler's DataFrame-valued audit attribute and
    # pandas cannot compare that object while concatenating slices. Candidate
    # provenance lives in columns, so clear non-tabular attrs before concat.
    for frame in (warm, residual, remainder):
        frame.attrs = {}
    ordered = pd.concat([warm, residual, remainder], ignore_index=True)
    ordered = ordered.sort_values(
        ["_balanced_phase", "_balanced_sort", "candidate_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    ordered["balanced_proposal_rank"] = np.arange(1, len(ordered) + 1)
    ordered["balanced_target_domain_count"] = ordered["background_sampling_domain"].map(
        allocation
    )
    return ordered.drop(
        columns=[
            "_within_domain_random",
            "_within_domain_rank",
            "_domain_order",
            "_balanced_sort",
            "_balanced_phase",
        ]
    )


def spatially_filter_order(
    ordered_pool: gpd.GeoDataFrame,
    presences: gpd.GeoDataFrame,
    target: int,
    min_distance_m: float,
) -> pd.DataFrame:
    """Greedily retain ranked backgrounds while preserving every presence."""
    if min_distance_m <= 0:
        selected = ordered_pool.head(target).copy()
        selected["nested_rank"] = np.arange(1, len(selected) + 1)
        return selected

    backgrounds_proj = ordered_pool.to_crs("EPSG:5070")
    presences_proj = presences.to_crs("EPSG:5070")
    cell_size = float(min_distance_m)
    occupied: dict[tuple[int, int], list[tuple[float, float]]] = {}

    def cell_for(x: float, y: float) -> tuple[int, int]:
        return (math.floor(x / cell_size), math.floor(y / cell_size))

    def add_point(x: float, y: float) -> None:
        occupied.setdefault(cell_for(x, y), []).append((x, y))

    for point in presences_proj.geometry:
        add_point(float(point.x), float(point.y))

    selected_positions = []
    threshold_sq = cell_size * cell_size
    for position, point in enumerate(backgrounds_proj.geometry):
        x, y = float(point.x), float(point.y)
        cell_x, cell_y = cell_for(x, y)
        valid = True
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for other_x, other_y in occupied.get((cell_x + dx, cell_y + dy), []):
                    if (x - other_x) ** 2 + (y - other_y) ** 2 < threshold_sq:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if not valid:
            continue
        add_point(x, y)
        selected_positions.append(position)
        if len(selected_positions) >= target:
            break

    if len(selected_positions) < target:
        raise RuntimeError(
            f"Only {len(selected_positions):,} spatially eligible ranked backgrounds "
            f"were available for target {target:,}; increase --pool-factor."
        )
    selected = ordered_pool.iloc[selected_positions].copy().reset_index(drop=True)
    selected["nested_rank"] = np.arange(1, len(selected) + 1)
    return selected


def _verify_paths(paths: pd.Series, workers: int) -> pd.DataFrame:
    unique_paths = sorted(set(paths.dropna().astype(str)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        exists = list(executor.map(os.path.isfile, unique_paths))
    return pd.DataFrame({"source_local_path": unique_paths, "exists": exists})


def _candidate_qa(
    manifests: dict[tuple[str, int], pd.DataFrame],
    split_assignments: pd.DataFrame,
    path_checks: pd.DataFrame,
    targets: list[int],
    allowed_states: list[str] | None,
    min_distance_m: float,
) -> dict[str, pd.DataFrame | dict]:
    findings = []

    def finding(check: str, count: int, details: str) -> None:
        findings.append(
            {
                "check": check,
                "severity": "hard",
                "status": "fail" if count else "pass",
                "count": int(count),
                "details": details,
            }
        )

    all_rows = []
    class_split_rows = []
    domain_rows = []
    target_failures = 0
    split_class_failures = 0
    state_failures = 0
    duplicate_failures = 0
    cell_failures = 0
    distance_failures = 0
    for (arm, target), manifest in sorted(manifests.items()):
        backgrounds = manifest[manifest["presence"] == 0]
        presences = manifest[manifest["presence"] == 1]
        target_failures += int(len(backgrounds) != target)
        duplicate_failures += int(manifest["sample_id"].duplicated().sum())
        duplicate_failures += int(manifest.duplicated(["lat", "lon", "presence"]).sum())
        cell_failures += int(backgrounds["cell_id"].duplicated().sum())
        cell_failures += len(set(backgrounds["cell_id"]) & set(presences["cell_id"]))
        if min_distance_m > 0:
            projected = gpd.GeoDataFrame(
                manifest[["lat", "lon"]].copy(),
                geometry=gpd.points_from_xy(manifest["lon"], manifest["lat"]),
                crs="EPSG:4326",
            ).to_crs("EPSG:5070")
            xy = np.column_stack([projected.geometry.x, projected.geometry.y])
            nearest, _ = cKDTree(xy).query(xy, k=2)
            distance_failures += int((nearest[:, 1] < min_distance_m - 1e-6).sum())
        if allowed_states:
            state_failures += int(
                (~backgrounds["source_archive_state"].isin(allowed_states)).sum()
            )
        split_counts = (
            manifest.groupby(["split", "presence"]).size().rename("row_count").reset_index()
        )
        expected = {(split, label) for split in ["train", "val", "test"] for label in [0, 1]}
        observed = set(zip(split_counts["split"], split_counts["presence"]))
        split_class_failures += len(expected - observed)
        split_counts.insert(0, "target", target)
        split_counts.insert(0, "arm", arm)
        class_split_rows.append(split_counts)

        domains = (
            backgrounds.groupby(
                ["source_archive_state", "source_acq_year", "source_acq_month", "source_acq_date"],
                dropna=False,
            )
            .size()
            .rename("background_count")
            .reset_index()
        )
        domains.insert(0, "target", target)
        domains.insert(0, "arm", arm)
        domain_rows.append(domains)
        all_rows.append(
            {
                "arm": arm,
                "target": target,
                "rows": len(manifest),
                "presences": len(presences),
                "backgrounds": len(backgrounds),
                "unique_source_tiles": manifest["source_tile_key"].nunique(),
                "unique_background_domains": len(domains),
            }
        )

    nested_failures = []
    for arm in ["random", "domain_balanced"]:
        for smaller, larger in zip(targets[:-1], targets[1:]):
            small_ids = set(
                manifests[(arm, smaller)].loc[
                    manifests[(arm, smaller)]["presence"] == 0, "sample_id"
                ]
            )
            large_ids = set(
                manifests[(arm, larger)].loc[
                    manifests[(arm, larger)]["presence"] == 0, "sample_id"
                ]
            )
            if not small_ids.issubset(large_ids):
                nested_failures.append(
                    {"arm": arm, "smaller": smaller, "larger": larger, "missing": len(small_ids - large_ids)}
                )

    tile_uniform = (
        split_assignments.groupby("source_tile_key")["split"].nunique().reset_index(name="split_count")
    )
    tile_fold = (
        split_assignments.groupby("source_tile_key")["fold"].nunique().reset_index(name="fold_count")
    )
    tile_leakage = tile_uniform[tile_uniform["split_count"] > 1].merge(
        tile_fold[tile_fold["fold_count"] > 1],
        on="source_tile_key",
        how="outer",
    )

    finding("background_target_counts", target_failures, "manifests not matching exact target")
    finding("nested_background_sets", len(nested_failures), "smaller target not subset of larger")
    finding("duplicate_rows_or_ids", duplicate_failures, "duplicate IDs or coordinates")
    finding("worldclim_cell_conflicts", cell_failures, "duplicate or presence-overlap cells")
    finding(
        "minimum_sample_spacing",
        distance_failures,
        f"rows with a neighbor closer than {min_distance_m:g} m",
    )
    finding("missing_split_classes", split_class_failures, "absent split × class combinations")
    finding("background_state_restriction", state_failures, "backgrounds outside allowed states")
    finding("source_tile_split_or_fold_leakage", len(tile_leakage), "source tiles spanning split/fold")
    finding("missing_source_paths", int((~path_checks["exists"]).sum()), "source paths not found")

    domain_depth = pd.concat(domain_rows, ignore_index=True)
    domain_summary = (
        domain_depth.groupby(["arm", "target"])["background_count"]
        .agg(
            background_count="sum",
            domain_count="size",
            median_backgrounds_per_domain="median",
            min_backgrounds_per_domain="min",
            max_backgrounds_per_domain="max",
        )
        .reset_index()
    )
    thresholds = (
        domain_depth.assign(
            domains_ge_5=domain_depth["background_count"] >= 5,
            domains_ge_10=domain_depth["background_count"] >= 10,
        )
        .groupby(["arm", "target"])[["domains_ge_5", "domains_ge_10"]]
        .sum()
        .reset_index()
    )
    domain_summary = domain_summary.merge(thresholds, on=["arm", "target"])
    findings_df = pd.DataFrame(findings)
    return {
        "findings": findings_df,
        "manifest_summary": pd.DataFrame(all_rows),
        "class_split_counts": pd.concat(class_split_rows, ignore_index=True),
        "domain_depth": domain_depth,
        "domain_depth_summary": domain_summary,
        "nested_failures": pd.DataFrame(nested_failures),
        "source_tile_leakage": tile_leakage,
        "source_path_checks": path_checks,
        "status": {
            "production_ready_for_extraction_review": not bool(
                ((findings_df["severity"] == "hard") & (findings_df["status"] == "fail")).any()
            )
        },
    }


def _write_qa(qa: dict[str, pd.DataFrame | dict], output_dir: Path) -> None:
    qa_dir = output_dir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    for name, value in qa.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(qa_dir / f"{name}.csv", index=False)
        else:
            with open(qa_dir / f"{name}.json", "w", encoding="utf-8") as stream:
                json.dump(value, stream, indent=2)


def build_candidate_manifests(args: argparse.Namespace) -> Path:
    targets = sorted(set(args.targets))
    max_target = max(targets)
    pool_target = int(math.ceil(max_target * args.pool_factor))
    reserve_target = int(math.ceil(max_target * args.reserve_factor))
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing candidate output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    builder_root = Path(__file__).resolve().parents[1]
    builder_git_at_start = _git_state(builder_root)
    builder_sources_at_start = _builder_source_hashes(Path(__file__).resolve().parent)

    try:
        tiles = gpd.read_file(args.tileindex).to_crs("EPSG:4326")
        tiles = add_canonical_naip_source_metadata(tiles)
        downloaded = tiles[tiles["source_local_path"].notna()].copy()
        if args.allowed_states:
            allowed_states = [state.upper() for state in args.allowed_states]
        else:
            allowed_states = None

        occurrences = load_occurrence_csv(
            Path(args.occurrence_file),
            species_name_col=args.species_name_col,
            lat_col=args.lat_col,
            lon_col=args.lon_col,
            source_col=args.source_col,
            coordinate_uncertainty_col=args.coordinate_uncertainty_col,
            max_coordinate_uncertainty_m=args.max_coordinate_uncertainty_m,
        ).drop_duplicates(["lat", "lon"])
        worldclim_reference = Path(args.worldclim_folder) / "wc2.1_30s_bio_1.tif"
        presences = deduplicate_presences_to_worldclim_cells(
            occurrences,
            str(worldclim_reference),
            seed=args.seed,
        )
        presences = assign_presence_ids(presences).reset_index(drop=True)
        presences = attach_naip_filename_or_url(presences, tiles)
        presences["lat"] = presences.geometry.y
        presences["lon"] = presences.geometry.x
        presences["presence"] = 1
        presences["sample_id"] = presences["presence_id"]
        missing_presence_source = presences["source_local_path"].isna()
        presence_source_audit = presences.loc[missing_presence_source].copy()
        presence_source_audit["audit_reason"] = "no_local_naip_source_footprint"
        presences = presences.loc[~missing_presence_source].copy().reset_index(drop=True)
        if presences.empty:
            raise RuntimeError("No source-backed presences remain after NAIP footprint filtering.")
        presence_environment_screen, environment_raster_metadata = (
            sample_raw_environment_for_points(
                presences,
                worldclim_folder=args.worldclim_folder,
                ghm_raster=args.ghm_raster,
                workers=args.environment_workers,
            )
        )
        presence_environment_valid = raw_environment_valid_mask(
            presence_environment_screen
        )
        presence_environment_screen["environment_valid"] = presence_environment_valid
        presence_environment_screen["nonfinite_variable_count"] = (
            ~np.isfinite(
                presence_environment_screen[ALL_ENV_VARS].to_numpy(dtype=float)
            )
        ).sum(axis=1)
        presence_environment_audit = presence_environment_screen.loc[
            ~presence_environment_valid
        ].copy()
        presence_environment_audit["audit_reason"] = (
            "incomplete_or_invalid_worldclim_ghm_support"
        )
        presences = presences.loc[presence_environment_valid].copy().reset_index(drop=True)
        if presences.empty:
            raise RuntimeError("No environmentally valid presences remain.")
        source_backed_presences = presences.copy()
        presence_ids_before_thinning = set(presences["presence_id"])
        presences = spatially_thin_points(
            presences,
            min_distance_m=args.spatial_thin_distance_m,
            seed=args.seed,
            projected_crs="EPSG:5070",
        )
        presence_spatial_attrition_ids = presence_ids_before_thinning - set(
            presences["presence_id"]
        )
        presence_spatial_attrition = source_backed_presences[
            source_backed_presences["presence_id"].isin(presence_spatial_attrition_ids)
        ].copy()
        presence_spatial_attrition["audit_reason"] = "presence_removed_by_800m_thinning"

        pool_before_environment = sample_background_points_us_naip(
            presence_gdf=presences,
            downloaded_tiles_gdf=downloaded,
            wc_reference_raster_fp=str(worldclim_reference),
            background_multiplier=args.background_multiplier,
            seed=args.seed,
            inner_buffer_km=args.background_inner_buffer_km,
            max_sampling_rounds=args.background_max_sampling_rounds,
            target_count=pool_target,
            allowed_states=allowed_states,
            sampling_design="domain_balanced",
            domain_columns=args.domain_columns,
            domain_min_count=args.domain_min_count,
        ).reset_index(drop=True)
        pool_before_environment["candidate_id"] = [
            f"background_{index:07d}" for index in range(len(pool_before_environment))
        ]
        pool_before_environment["sample_id"] = pool_before_environment["candidate_id"]
        pool_before_environment["presence"] = 0
        background_environment_screen, background_environment_raster_metadata = (
            sample_raw_environment_for_points(
                pool_before_environment,
                worldclim_folder=args.worldclim_folder,
                ghm_raster=args.ghm_raster,
                workers=args.environment_workers,
            )
        )
        if background_environment_raster_metadata != environment_raster_metadata:
            raise RuntimeError("Environmental raster provenance changed within one build.")
        background_environment_valid = raw_environment_valid_mask(
            background_environment_screen
        )
        background_environment_screen["environment_valid"] = background_environment_valid
        background_environment_screen["nonfinite_variable_count"] = (
            ~np.isfinite(
                background_environment_screen[ALL_ENV_VARS].to_numpy(dtype=float)
            )
        ).sum(axis=1)
        background_environment_audit = background_environment_screen.loc[
            ~background_environment_valid
        ].copy()
        background_environment_audit["audit_reason"] = (
            "incomplete_or_invalid_worldclim_ghm_support"
        )
        pool = pool_before_environment.loc[background_environment_valid].copy().reset_index(
            drop=True
        )
        if len(pool) < reserve_target:
            raise RuntimeError(
                f"Only {len(pool):,} environmentally valid backgrounds remain, below "
                f"the reserve target {reserve_target:,}; increase --pool-factor."
            )

        tiles_for_area = downloaded
        if allowed_states:
            tiles_for_area = tiles_for_area[
                tiles_for_area["source_archive_state"].isin(allowed_states)
            ].copy()
        tiles_for_area = tiles_for_area.to_crs("EPSG:5070")
        tiles_for_area["_candidate_domain"] = _sampling_domain(
            tiles_for_area,
            args.domain_columns,
        )
        tiles_for_area["_area_m2"] = tiles_for_area.geometry.area
        domain_areas = tiles_for_area.groupby("_candidate_domain")["_area_m2"].sum()

        random_order = assign_weighted_random_order(pool, domain_areas, seed=args.seed + 101)
        balanced_order = assign_balanced_order(
            pool,
            domain_areas,
            target=max_target,
            minimum_per_domain=args.domain_min_count,
            seed=args.seed + 202,
        )
        random_selected = spatially_filter_order(
            gpd.GeoDataFrame(random_order, geometry="geometry", crs=pool.crs),
            presences,
            target=reserve_target,
            min_distance_m=args.spatial_thin_distance_m,
        )
        balanced_selected = spatially_filter_order(
            gpd.GeoDataFrame(balanced_order, geometry="geometry", crs=pool.crs),
            presences,
            target=reserve_target,
            min_distance_m=args.spatial_thin_distance_m,
        )

        selected_union_ids = set(random_selected["candidate_id"]) | set(
            balanced_selected["candidate_id"]
        )
        selected_union = pool[pool["candidate_id"].isin(selected_union_ids)].copy()
        split_frame = pd.concat(
            [presences, selected_union],
            ignore_index=True,
            sort=False,
        )
        split_frame = gpd.GeoDataFrame(split_frame, geometry="geometry", crs="EPSG:4326")
        uniform = stratified_train_val_test_split(
            split_frame.drop(columns="geometry"),
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            seed=args.seed,
            group_column="source_tile_key",
        )
        blocks = assign_spatial_blocks(
            split_frame,
            block_size_m=args.block_size_m,
            n_folds=args.n_folds,
            seed=args.seed,
            group_column="source_tile_key",
        ).drop(columns="geometry")
        split_assignments = uniform[["sample_id", "split"]].merge(
            blocks[["sample_id", "block_id", "fold"]],
            on="sample_id",
            how="inner",
            validate="one_to_one",
        )

        presence_table = presences.drop(columns="geometry").merge(
            split_assignments,
            on="sample_id",
            how="left",
            validate="one_to_one",
        )
        manifests: dict[tuple[str, int], pd.DataFrame] = {}
        ordered_by_arm = {
            "random": random_selected,
            "domain_balanced": balanced_selected,
        }
        manifest_dir = temporary / "manifests"
        manifest_dir.mkdir(parents=True)
        for arm, ordered in ordered_by_arm.items():
            ordered_table = ordered.drop(columns="geometry").merge(
                split_assignments,
                on="sample_id",
                how="left",
                validate="one_to_one",
            )
            for target in targets:
                backgrounds = ordered_table[ordered_table["nested_rank"] <= target].copy()
                backgrounds["arm"] = arm
                backgrounds["background_target"] = target
                pres = presence_table.copy()
                pres["arm"] = arm
                pres["background_target"] = target
                pres["nested_rank"] = 0
                manifest = pd.concat([pres, backgrounds], ignore_index=True, sort=False)
                manifest["manifest_id"] = f"{args.host_slug}_{arm}_{target:05d}"
                manifests[(arm, target)] = manifest
                manifest.to_csv(
                    manifest_dir / f"{args.host_slug}_{arm}_{target:05d}.csv",
                    index=False,
                )

        reserve_dir = temporary / "ranked_reserves"
        reserve_dir.mkdir(parents=True)
        for arm, ordered in ordered_by_arm.items():
            reserve_table = ordered.drop(columns="geometry").merge(
                split_assignments,
                on="sample_id",
                how="left",
                validate="one_to_one",
            )
            reserve_table["arm"] = arm
            reserve_table["primary_target_max"] = max_target
            reserve_table["is_primary_candidate"] = (
                reserve_table["nested_rank"] <= max_target
            )
            reserve_table.to_csv(
                reserve_dir / f"{args.host_slug}_{arm}_ranked_to_{reserve_target:05d}.csv",
                index=False,
            )

        extraction_backgrounds = selected_union.drop(columns="geometry").merge(
            split_assignments,
            on="sample_id",
            how="left",
            validate="one_to_one",
        )
        extraction_backgrounds["random_nested_rank"] = extraction_backgrounds[
            "sample_id"
        ].map(random_selected.set_index("sample_id")["nested_rank"])
        extraction_backgrounds["domain_balanced_nested_rank"] = extraction_backgrounds[
            "sample_id"
        ].map(balanced_selected.set_index("sample_id")["nested_rank"])
        extraction_backgrounds["candidate_role"] = "background_primary_or_reserve"
        extraction_presences = presence_table.copy()
        extraction_presences["random_nested_rank"] = 0
        extraction_presences["domain_balanced_nested_rank"] = 0
        extraction_presences["candidate_role"] = "presence"
        extraction_candidates = pd.concat(
            [extraction_presences, extraction_backgrounds],
            ignore_index=True,
            sort=False,
        )
        extraction_candidates.to_csv(temporary / "extraction_candidates.csv", index=False)
        environment_values = pd.concat(
            [
                presence_environment_screen.loc[
                    presence_environment_valid,
                    ["sample_id", "lat", "lon", *ALL_ENV_VARS],
                ],
                background_environment_screen.loc[
                    background_environment_valid,
                    ["sample_id", "lat", "lon", *ALL_ENV_VARS],
                ],
            ],
            ignore_index=True,
        )
        extraction_environment = extraction_candidates[
            ["sample_id", "lat", "lon"]
        ].merge(
            environment_values,
            on=["sample_id", "lat", "lon"],
            how="left",
            validate="one_to_one",
        )
        if not raw_environment_valid_mask(extraction_environment).all():
            raise RuntimeError("Extraction candidates lost environmental support after ranking.")
        extraction_environment.to_csv(
            temporary / "extraction_environment_raw.csv",
            index=False,
        )

        pool.drop(columns="geometry").to_csv(temporary / "background_superpool.csv", index=False)
        presence_table.to_csv(temporary / "presence_manifest.csv", index=False)
        presence_source_audit.drop(columns="geometry").to_csv(
            temporary / "presence_source_attrition_audit.csv",
            index=False,
        )
        presence_spatial_attrition.drop(columns="geometry").to_csv(
            temporary / "presence_spatial_attrition_audit.csv",
            index=False,
        )
        presence_environment_audit.drop(columns="geometry").to_csv(
            temporary / "presence_environment_attrition_audit.csv",
            index=False,
        )
        background_environment_audit.drop(columns="geometry").to_csv(
            temporary / "background_environment_attrition_audit.csv",
            index=False,
        )
        environment_screen_summary = pd.DataFrame(
            [
                {
                    "class": "presence",
                    "screened": len(presence_environment_screen),
                    "valid": int(presence_environment_valid.sum()),
                    "invalid": int((~presence_environment_valid).sum()),
                },
                {
                    "class": "background_superpool",
                    "screened": len(background_environment_screen),
                    "valid": int(background_environment_valid.sum()),
                    "invalid": int((~background_environment_valid).sum()),
                },
            ]
        )
        qa_dir = temporary / "qa"
        qa_dir.mkdir(parents=True, exist_ok=True)
        environment_screen_summary.to_csv(
            qa_dir / "candidate_environment_screen_summary.csv", index=False
        )
        with open(
            qa_dir / "candidate_environment_raster_provenance.json",
            "w",
            encoding="utf-8",
        ) as stream:
            json.dump(environment_raster_metadata, stream, indent=2)
        split_assignments.to_csv(temporary / "frozen_split_assignments.csv", index=False)

        paths_to_check = pd.concat(
            [presences["source_local_path"], selected_union["source_local_path"]],
            ignore_index=True,
        )
        path_checks = _verify_paths(paths_to_check, workers=args.path_check_workers)
        qa = _candidate_qa(
            manifests,
            split_assignments=split_frame.drop(columns="geometry")[["sample_id", "source_tile_key"]]
            .merge(split_assignments, on="sample_id", validate="one_to_one"),
            path_checks=path_checks,
            targets=targets,
            allowed_states=allowed_states,
            min_distance_m=args.spatial_thin_distance_m,
        )
        _write_qa(qa, temporary)
        if not qa["status"]["production_ready_for_extraction_review"]:
            raise RuntimeError("Candidate-manifest QA failed; rejected output will be preserved.")

        metadata = {
            "created_at": pd.Timestamp.now(tz="America/New_York").isoformat(),
            "host_slug": args.host_slug,
            "occurrence_file": str(Path(args.occurrence_file).resolve()),
            "occurrence_sha256": _sha256(args.occurrence_file),
            "tileindex": str(Path(args.tileindex).resolve()),
            "tileindex_sha256": _sha256(args.tileindex),
            "worldclim_reference": str(worldclim_reference.resolve()),
            "targets": targets,
            "max_target": max_target,
            "pool_factor": args.pool_factor,
            "pool_target": pool_target,
            "reserve_factor": args.reserve_factor,
            "reserve_target_per_arm": reserve_target,
            "pool_rows_before_environment_filter": len(pool_before_environment),
            "pool_rows": len(pool),
            "presence_rows": len(presences),
            "presence_rows_before_source_filter": (
                len(presences)
                + len(presence_source_audit)
                + len(presence_environment_audit)
                + len(presence_spatial_attrition)
            ),
            "presence_source_attrition_rows": len(presence_source_audit),
            "presence_environment_attrition_rows": len(presence_environment_audit),
            "presence_spatial_attrition_rows": len(presence_spatial_attrition),
            "background_environment_attrition_rows": len(background_environment_audit),
            "selected_union_background_rows": len(selected_union),
            "allowed_states": allowed_states,
            "domain_columns": args.domain_columns,
            "domain_min_count": args.domain_min_count,
            "spatial_thin_distance_m": args.spatial_thin_distance_m,
            "seed": args.seed,
            "builder_git": builder_git_at_start,
            "builder_source_sha256": builder_sources_at_start,
            "environment_raster_provenance": environment_raster_metadata,
            "soft_cap_note": "20,000 is a planning soft cap, not a hard builder limit.",
        }
        with open(temporary / "manifest_metadata.json", "w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2)

        temporary.rename(output)
        return output
    except Exception as error:
        if temporary.exists():
            rejected_root = output.parent / "rejected_builds"
            rejected_root.mkdir(parents=True, exist_ok=True)
            timestamp = pd.Timestamp.now(tz="America/New_York").strftime("%Y%m%dT%H%M%S")
            rejected = rejected_root / f"{output.name}_failed_{timestamp}"
            with open(temporary / "build_failure.json", "w", encoding="utf-8") as stream:
                json.dump(
                    {
                        "failed_at": pd.Timestamp.now(
                            tz="America/New_York"
                        ).isoformat(),
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "builder_git": builder_git_at_start,
                        "builder_source_sha256": builder_sources_at_start,
                    },
                    stream,
                    indent=2,
                )
            temporary.rename(rejected)
            print(f"Preserved rejected candidate build for review: {rejected}")
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build nested candidate manifests without extracting chips."
    )
    parser.add_argument("--host-slug", required=True)
    parser.add_argument("--occurrence-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tileindex", required=True)
    parser.add_argument("--worldclim-folder", required=True)
    parser.add_argument("--ghm-raster", required=True)
    parser.add_argument("--targets", type=int, nargs="+", required=True)
    parser.add_argument("--pool-factor", type=float, default=2.5)
    parser.add_argument(
        "--reserve-factor",
        type=float,
        default=1.20,
        help="Rank this multiple of the largest target per arm for extraction attrition.",
    )
    parser.add_argument("--allowed-states", nargs="+", default=None)
    parser.add_argument(
        "--domain-columns",
        nargs="+",
        default=["source_archive_state", "source_acq_year", "source_acq_month"],
    )
    parser.add_argument("--domain-min-count", type=int, default=5)
    parser.add_argument("--background-multiplier", type=float, default=1.5)
    parser.add_argument("--background-inner-buffer-km", type=float, default=0.0)
    parser.add_argument("--background-max-sampling-rounds", type=int, default=10)
    parser.add_argument("--spatial-thin-distance-m", type=float, default=800.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--block-size-m", type=float, default=200000.0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--path-check-workers", type=int, default=32)
    parser.add_argument("--environment-workers", type=int, default=4)
    parser.add_argument("--species-name-col", default="species")
    parser.add_argument("--lat-col", default="auto")
    parser.add_argument("--lon-col", default="auto")
    parser.add_argument("--source-col", default="Source")
    parser.add_argument("--coordinate-uncertainty-col", default="auto")
    parser.add_argument("--max-coordinate-uncertainty-m", type=float, default=128.0)
    args = parser.parse_args()

    if any(target <= 0 for target in args.targets):
        parser.error("--targets must all be positive.")
    if args.pool_factor <= 1:
        parser.error("--pool-factor must exceed 1.")
    if args.reserve_factor < 1:
        parser.error("--reserve-factor must be at least 1.")
    if args.domain_min_count < 0:
        parser.error("--domain-min-count must be non-negative.")
    if args.path_check_workers <= 0:
        parser.error("--path-check-workers must be positive.")
    if args.environment_workers <= 0:
        parser.error("--environment-workers must be positive.")
    if args.n_folds < 3:
        parser.error("--n-folds must be at least 3.")
    if not np.isclose(args.train_frac + args.val_frac + args.test_frac, 1.0):
        parser.error("train/val/test fractions must sum to one.")
    if args.allowed_states:
        args.allowed_states = [state.strip().upper() for state in args.allowed_states]
    return args


def main() -> None:
    args = parse_args()
    output = build_candidate_manifests(args)
    print(f"Candidate manifest package complete: {output}")


if __name__ == "__main__":
    main()
