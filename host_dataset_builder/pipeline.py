

from .common import *
from .cli import parse_args
from .utils import set_seed, infer_species_slug_from_filename, display_name_from_slug
from .io_utils import planned_output_paths, load_tileindex, load_occurrence_csv
from .background import (
    audit_presences_without_background,
    build_presence_background_points,
    deduplicate_presences_to_worldclim_cells,
    sample_background_points,
    sample_background_points_radial,
    sample_background_points_us_naip,
    spatially_thin_points,
)
from .splits import stratified_train_val_test_split, assign_spatial_blocks, make_spatial_cv_rounds
from .env import load_or_compute_worldclim_stats, open_worldclim_datasets
from .topo import load_topo_normalization_stats, open_topo_sources, close_dataset_dict
from .dataset_builders import build_dataset_csv, write_json

def process_species(
    csv_path: Path,
    args: argparse.Namespace,
    tileindex_gdf: gpd.GeoDataFrame,
    downloaded_tiles_gdf: gpd.GeoDataFrame,
    wc_stats: dict,
    topo_norm_stats: dict | None,
    naip_file_index: dict,
) -> None:
    species_slug = infer_species_slug_from_filename(csv_path)
    species_label = display_name_from_slug(species_slug)

    print("\n" + "=" * 80)
    print(f"Processing species: {species_slug}")
    print("=" * 80)

    # Output roots
    paths = planned_output_paths(
        csv_path,
        args.output_root,
        args.chip_size,
        dataset_tag=args.dataset_tag,
        background_label=args.background_label,
    )
    datasets_root = paths["datasets_root"]
    occurrences_root = paths["occurrences_root"]
    uniform_root = paths["uniform_root"]
    blockcv_root = paths["blockcv_root"]
    build_blockcv = args.build_variant == "all"
    pb_points_csv = paths["presence_background_points"]
    background_audit_csv = paths["background_audit"]

    conflicts = []
    if pb_points_csv.exists():
        conflicts.append(pb_points_csv)
    output_dirs = [uniform_root]
    if build_blockcv:
        output_dirs.append(blockcv_root)
    for output_dir in output_dirs:
        if output_dir.exists() and any(output_dir.iterdir()):
            conflicts.append(output_dir)
    if conflicts and not args.allow_existing_output:
        conflict_list = "\n  - ".join(str(p) for p in conflicts)
        raise FileExistsError(
            "Refusing to write into existing output artifact(s). "
            "Use a new --output-root or pass --allow-existing-output after review.\n"
            f"  - {conflict_list}"
        )

    datasets_root.mkdir(parents=True, exist_ok=True)
    occurrences_root.mkdir(parents=True, exist_ok=True)
    uniform_root.mkdir(parents=True, exist_ok=True)
    if build_blockcv:
        blockcv_root.mkdir(parents=True, exist_ok=True)

    if args.topo_mode != "none":
        topo_metadata = {
            "topo_mode": args.topo_mode,
            "topo_chip_size": args.topo_chip_size,
            "topo_min_valid_frac": args.topo_min_valid_frac,
            "topo_normalized": not args.disable_topo_normalization,
            "topo_band_order": TOPO_IMAGE_BANDS,
            "topo_scalar_columns": TOPO_SCALAR_COLUMNS,
            "topo_normalization_stats": topo_norm_stats,
        }
        write_json(topo_metadata, uniform_root / "topography_metadata.json")
        if build_blockcv:
            write_json(topo_metadata, blockcv_root / "topography_metadata.json")

    # WorldClim reference raster
    wc_reference = os.path.join(args.worldclim_folder, "wc2.1_30s_bio_1.tif")

    # Load and clean occurrences
    occurrences = load_occurrence_csv(
        csv_path,
        species_name_col=args.species_name_col,
        lat_col=args.lat_col,
        lon_col=args.lon_col,
        source_col=args.source_col,
        coordinate_uncertainty_col=args.coordinate_uncertainty_col,
        max_coordinate_uncertainty_m=args.max_coordinate_uncertainty_m,
    )

    print(f"Loaded {len(occurrences):,} raw occurrences from {csv_path.name}")

    occurrences = occurrences.drop_duplicates(subset=["lat", "lon"]).copy()
    print(f"After exact lat/lon deduplication: {len(occurrences):,}")

    presences = deduplicate_presences_to_worldclim_cells(
        occurrences_gdf=occurrences,
        wc_reference_raster_fp=wc_reference,
        seed=args.seed,
    )
    print(f"After WorldClim-cell thinning: {len(presences):,}")

    if args.max_presences_after_thinning is not None and len(presences) > args.max_presences_after_thinning:
        before_cap = len(presences)
        presences = presences.sample(n=args.max_presences_after_thinning, random_state=args.seed).copy()
        print(
            "Applied bounded validation cap after WorldClim-cell thinning: "
            f"{len(presences):,} of {before_cap:,} presences retained"
        )

    if args.background_sampling_mode == "polygon":
        backgrounds = sample_background_points(
            presence_gdf=presences,
            downloaded_tiles_gdf=downloaded_tiles_gdf,
            wc_reference_raster_fp=wc_reference,
            background_multiplier=args.background_multiplier,
            buffer_km=args.background_buffer_km,
            inner_buffer_km=args.background_inner_buffer_km,
            max_sampling_rounds=args.background_max_sampling_rounds,
            max_naip_footprint_union_tiles=args.max_naip_footprint_union_tiles,
            seed=args.seed,
        )
    elif args.background_sampling_mode == "us_naip":
        backgrounds = sample_background_points_us_naip(
            presence_gdf=presences,
            downloaded_tiles_gdf=downloaded_tiles_gdf,
            wc_reference_raster_fp=wc_reference,
            background_multiplier=args.background_multiplier,
            inner_buffer_km=args.background_inner_buffer_km,
            max_sampling_rounds=args.background_max_sampling_rounds,
            seed=args.seed,
            target_count=args.background_target_count,
        )
    else:
        backgrounds = sample_background_points_radial(
            presence_gdf=presences,
            wc_reference_raster_fp=wc_reference,
            background_multiplier=args.background_multiplier,
            buffer_km=args.background_buffer_km,
            inner_buffer_km=args.background_inner_buffer_km,
            max_sampling_rounds=args.background_max_sampling_rounds,
            seed=args.seed,
        )
    print(f"Sampled {len(backgrounds):,} background points")
    sampling_audit = backgrounds.attrs.get("background_audit", pd.DataFrame())
    if args.background_sampling_mode == "us_naip":
        target_label = args.background_target_count if args.background_target_count is not None else len(presences)
        print(
            "Background sampling retained broad unpaired USBg backgrounds: "
            f"{len(backgrounds):,} of target {target_label:,}"
        )
    elif sampling_audit is not None and not sampling_audit.empty:
        print(
            "Background sampling retained local backgrounds for "
            f"{len(presences) - len(sampling_audit):,} of {len(presences):,} presences; "
            f"{len(sampling_audit):,} presences will be written to the audit CSV."
        )
    else:
        print(f"Background sampling retained local backgrounds for {len(backgrounds):,} of {len(presences):,} presences")

    pa_points = build_presence_background_points(
        presence_gdf=presences,
        background_gdf=backgrounds,
        tileindex_gdf=tileindex_gdf,
    )

    print(f"Combined presence/background dataset before spatial thinning: {len(pa_points):,}")

    pa_points = spatially_thin_points(
        points_gdf=pa_points,
        min_distance_m=args.spatial_thin_distance_m,
        seed=args.seed,
        projected_crs="EPSG:5070",
    )

    print(f"Combined presence/background dataset after spatial thinning: {len(pa_points):,}")

    if args.background_sampling_mode == "us_naip":
        post_thin_audit = pd.DataFrame()
    else:
        post_thin_audit = audit_presences_without_background(
            pa_points,
            reason="no_retained_paired_background_after_spatial_thinning",
        )
    audit_parts = []
    if sampling_audit is not None and not sampling_audit.empty:
        audit_parts.append(sampling_audit)
    if not post_thin_audit.empty:
        audit_parts.append(post_thin_audit)
    if audit_parts:
        background_audit = pd.concat(audit_parts, ignore_index=True).drop_duplicates(
            subset=["presence_id", "audit_reason"]
        )
    else:
        background_audit = pd.DataFrame(
            columns=[
                "presence_id",
                "species",
                "source",
                "lat",
                "lon",
                "coordinate_uncertainty_m",
                "audit_reason",
            ]
        )
    background_audit.to_csv(background_audit_csv, index=False)
    print(f"Saved background audit: {background_audit_csv} ({len(background_audit):,} rows)")

    pa_points.drop(columns=["geometry"]).to_csv(pb_points_csv, index=False)
    print(f"Saved: {pb_points_csv}")

    # Uniform train/val/test points.
    uniform_points = stratified_train_val_test_split(
        pa_points.drop(columns=["geometry"]),
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )

    uniform_points_gdf = gpd.GeoDataFrame(
        uniform_points,
        geometry=gpd.points_from_xy(uniform_points["lon"], uniform_points["lat"]),
        crs="EPSG:4326",
    )

    uniform_points_csv = uniform_root / f"{species_label}_Pres_Bg_US_Uniform_Train_Val_Test_Points.csv"
    uniform_points_gdf.drop(columns=["geometry"]).to_csv(uniform_points_csv, index=False)
    print(f"Saved: {uniform_points_csv}")

    cv_rounds = None
    if build_blockcv:
        # Spatial block CV points.
        pa_points_blocks = assign_spatial_blocks(
            gpd.GeoDataFrame(pa_points.copy(), geometry="geometry", crs="EPSG:4326"),
            block_size_m=args.block_size_m,
            n_folds=args.n_folds,
            seed=args.seed,
        )

        block_points_csv = blockcv_root / f"{species_label}_Pres_Bg_US_SpatialCV_Train_Val_Test_Points.csv"
        cv_rounds = make_spatial_cv_rounds(
            points_df=pa_points_blocks.drop(columns=["geometry"]),
            n_folds=args.n_folds,
            seed=args.seed,
        )
        cv_rounds.to_csv(block_points_csv, index=False)
        print(f"Saved: {block_points_csv}")

        for round_num in range(1, args.n_folds + 1):
            fold_df = cv_rounds[cv_rounds["cv_round"] == round_num].copy()
            fold_csv = blockcv_root / f"{species_label}_Pres_Bg_US_SpatialCV_Blocks_Fold_{round_num}.csv"
            fold_df.to_csv(fold_csv, index=False)
            print(f"Saved: {fold_csv}")

    # Open shared rasters once for final dataset extraction.
    wc_datasets = open_worldclim_datasets(args.worldclim_folder)
    ghm_ds = rasterio.open(args.ghm_raster)
    topo_sources = open_topo_sources(args)

    try:
        normalize_topo = args.topo_mode != "none" and not args.disable_topo_normalization

        # Uniform final dataset.
        uniform_final = build_dataset_csv(
            points_df=uniform_points,
            topo_sources=topo_sources,
            topo_mode=args.topo_mode,
            topo_chip_size=args.topo_chip_size,
            topo_min_valid_frac=args.topo_min_valid_frac,
            topo_norm_stats=topo_norm_stats,
            normalize_topo=normalize_topo,
            dataset_root=uniform_root,
            tileindex_gdf=tileindex_gdf,
            naip_file_index=naip_file_index,
            wc_datasets=wc_datasets,
            wc_stats=wc_stats,
            ghm_ds=ghm_ds,
            chip_size=args.chip_size,
            species_label=species_label,
            suffix="_uniform",
            cv_round=None,
        )

        uniform_dataset_csv = uniform_root / f"{species_label}_Pres_Bg_US_Uniform_Train_Val_Test_Dataset.csv"
        uniform_final.to_csv(uniform_dataset_csv, index=False)
        print(f"Saved: {uniform_dataset_csv}")
        print(f"Uniform final rows retained after chip/env/topo filtering: {len(uniform_final):,}")

        if build_blockcv:
            # Block CV final datasets.
            for round_num in range(1, args.n_folds + 1):
                round_df = cv_rounds[cv_rounds["cv_round"] == round_num].copy()
                cv_dir = blockcv_root / f"CV_{round_num}"
                cv_dir.mkdir(parents=True, exist_ok=True)

                if args.topo_mode != "none":
                    write_json(
                        {
                            "topo_mode": args.topo_mode,
                            "topo_chip_size": args.topo_chip_size,
                            "topo_min_valid_frac": args.topo_min_valid_frac,
                            "topo_normalized": not args.disable_topo_normalization,
                            "topo_band_order": TOPO_IMAGE_BANDS,
                            "topo_scalar_columns": TOPO_SCALAR_COLUMNS,
                            "topo_normalization_stats": topo_norm_stats,
                        },
                        cv_dir / "topography_metadata.json",
                    )

                cv_final = build_dataset_csv(
                    points_df=round_df,
                    topo_sources=topo_sources,
                    topo_mode=args.topo_mode,
                    topo_chip_size=args.topo_chip_size,
                    topo_min_valid_frac=args.topo_min_valid_frac,
                    topo_norm_stats=topo_norm_stats,
                    normalize_topo=normalize_topo,
                    dataset_root=cv_dir,
                    tileindex_gdf=tileindex_gdf,
                    naip_file_index=naip_file_index,
                    wc_datasets=wc_datasets,
                    wc_stats=wc_stats,
                    ghm_ds=ghm_ds,
                    chip_size=args.chip_size,
                    species_label=species_label,
                    suffix=f"_CV{round_num}",
                    cv_round=round_num,
                )

                cv_dataset_csv = cv_dir / f"{species_label}_Train_Val_Test_US_BlockCV_{round_num}.csv"
                cv_final.to_csv(cv_dataset_csv, index=False)
                print(f"Saved: {cv_dataset_csv}")
                print(f"CV {round_num} final rows retained after chip/env/topo filtering: {len(cv_final):,}")

    finally:
        close_dataset_dict(wc_datasets)
        ghm_ds.close()
        close_dataset_dict(topo_sources)

def print_plan_only(csv_files: list[Path], args: argparse.Namespace) -> None:
    """Print occurrence QA and output targets without creating files."""
    print("Plan-only mode: no directories or files will be created.")
    for csv_path in csv_files:
        print("\n" + "=" * 80)
        print(f"Planning species file: {csv_path}")
        print("=" * 80)

        occurrences = load_occurrence_csv(
            csv_path,
            species_name_col=args.species_name_col,
            lat_col=args.lat_col,
            lon_col=args.lon_col,
            source_col=args.source_col,
            coordinate_uncertainty_col=args.coordinate_uncertainty_col,
            max_coordinate_uncertainty_m=args.max_coordinate_uncertainty_m,
        )
        exact_deduped = occurrences.drop_duplicates(subset=["lat", "lon"])
        source_counts = exact_deduped["source"].value_counts(dropna=False).to_dict()
        uncertainty = exact_deduped["coordinate_uncertainty_m"]

        print(f"Rows after schema/coordinate cleaning: {len(occurrences):,}")
        print(f"Rows after exact lat/lon deduplication: {len(exact_deduped):,}")
        print(f"Source counts after exact deduplication: {source_counts}")
        if uncertainty.notna().any():
            print(
                "Coordinate uncertainty summary, m: "
                f"min={uncertainty.min():.2f}, "
                f"median={uncertainty.median():.2f}, "
                f"max={uncertainty.max():.2f}"
            )
        else:
            print("Coordinate uncertainty summary, m: not available")

        paths = planned_output_paths(
            csv_path,
            args.output_root,
            args.chip_size,
            dataset_tag=args.dataset_tag,
            background_label=args.background_label,
        )
        print("Planned output paths:")
        for label, path in paths.items():
            print(f"  {label}: {path}")

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.occurrence_file:
        csv_files = [Path(args.occurrence_file)]
    else:
        csv_files = sorted(Path(args.occurrence_dir).glob("*.csv"))

    if not csv_files:
        raise RuntimeError("No occurrence CSV files found.")

    print(f"Found {len(csv_files)} occurrence CSV file(s).")

    if args.plan_only:
        print_plan_only(csv_files, args)
        print("\nDone. Plan-only mode did not write files.")
        return

    tileindex_gdf, downloaded_tiles_gdf, naip_file_index = load_tileindex(args.tileindex, args.naip_folder)
    wc_stats = load_or_compute_worldclim_stats(args, downloaded_tiles_gdf)

    topo_norm_stats = None
    if args.topo_mode != "none":
        topo_norm_stats = load_topo_normalization_stats(args.topo_normalization_stats)
        if args.disable_topo_normalization:
            print("Topographic normalization disabled: topo chips/scalars will be raw physical values.")
        else:
            source = args.topo_normalization_stats or "embedded DEFAULT_TOPO_NORM_STATS_3DEP_30M"
            print(f"Topographic normalization enabled using: {source}")

    failures = []
    for csv_path in csv_files:
        if args.build_variant == "uniform" and args.skip_completed_uniform:
            species_slug = infer_species_slug_from_filename(csv_path)
            species_label = display_name_from_slug(species_slug)
            paths = planned_output_paths(
                csv_path,
                args.output_root,
                args.chip_size,
                dataset_tag=args.dataset_tag,
                background_label=args.background_label,
            )
            uniform_dataset_csv = paths["uniform_root"] / f"{species_label}_Pres_Bg_US_Uniform_Train_Val_Test_Dataset.csv"
            if uniform_dataset_csv.exists():
                print(f"\nSkipping {csv_path.name}: completed uniform dataset exists at {uniform_dataset_csv}")
                continue

        try:
            process_species(
                csv_path=csv_path,
                args=args,
                tileindex_gdf=tileindex_gdf,
                downloaded_tiles_gdf=downloaded_tiles_gdf,
                wc_stats=wc_stats,
                topo_norm_stats=topo_norm_stats,
                naip_file_index=naip_file_index,
            )
        except Exception as e:
            failures.append((csv_path.name, str(e)))
            print(f"\n[ERROR] Failed processing {csv_path.name}: {e}\n")

    if failures:
        print("\nCompleted with failures:")
        for filename, error in failures:
            print(f"  - {filename}: {error}")
    else:
        print("\nDone.")
