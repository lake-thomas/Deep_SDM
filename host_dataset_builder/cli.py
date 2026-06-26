from .common import *

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Build uniform and spatial block CV SDM datasets for one or more species.")

    # INPUTS
    parser.add_argument("--occurrence-file", type=str, default=None, help="Path to a single species occurrence CSV.")

    parser.add_argument("--occurrence-dir", type=str, default=None, help="Directory containing multiple species occurrence CSVs.")

    parser.add_argument("--species-name-col", type=str, default="species")

    parser.add_argument("--lat-col", type=str, default="auto", help="Latitude column. Use auto to detect decimalLatitude/latitude/lat.")

    parser.add_argument("--lon-col", type=str, default="auto", help="Longitude column. Use auto to detect decimalLongitude/longitude/lon.")

    parser.add_argument("--source-col", type=str, default="Source", help="Optional source column, for example iNaturalist or GBIF.")

    parser.add_argument("--coordinate-uncertainty-col", type=str, default="auto", help="Optional coordinate uncertainty column. Use auto to detect common GBIF/iNat names.")

    parser.add_argument("--max-coordinate-uncertainty-m", type=float, default=None, help="Optional maximum coordinate uncertainty in meters. Rows above this value are removed before thinning.")

    parser.add_argument("--max-presences-after-thinning", type=int, default=None, help="Optional cap on presences after WorldClim-cell thinning. Use for small bounded validation runs before full dataset generation.")

    parser.add_argument("--plan-only", action="store_true", help="Inspect occurrence inputs and print planned output paths without loading rasters, creating directories, or writing files.")

    # SHARED OUTPUTS AND RASTER SAMPLING RESOURCES
    parser.add_argument("--output-root", type=str, required=True, help="Root output directory.")

    parser.add_argument("--tileindex", type=str, required=True, help="NAIP tile index shapefile/ geopackage.")

    parser.add_argument("--naip-folder", type=str, required=True, help="Folder containing downloaded NAIP .tif files.")

    parser.add_argument("--worldclim-folder", type=str, required=True, help="Folder containing wc2.1_30s_bio_1.tif ... wc2.1_30s_bio_19.tif")

    parser.add_argument("--topo-mode", choices=["none", "scalar", "chip", "both"], default="none", help="Whether to extract no topography, scalar summaries, topo chips, or both.")

    parser.add_argument("--ghm-raster", type=str, required=True, help="Path to Global Human Modification raster.")

    parser.add_argument("--dem-raster", type=str, default=None, help="Elevation raster.")

    parser.add_argument("--slope-raster", type=str, default=None, help="Slope raster in degrees.")

    parser.add_argument("--aspect-raster", type=str, default=None, help="Optional aspect raster in degrees, used only if northness/eastness rasters are omitted.")

    parser.add_argument("--northness-raster", type=str, default=None, help="Optional precomputed northness raster.")

    parser.add_argument("--eastness-raster", type=str, default=None, help="Optional precomputed eastness raster.")

    # BACKGROUND SAMPLING
    parser.add_argument("--background-inner-buffer-km", type=float, default=0.0, help="Optional inner exclusion buffer radius in kilometers around presences. --background-buffer-km. Use 5 for a 5-50 km doughnut.")

    parser.add_argument("--background-buffer-km", type=float, default=50.0, help="Outer buffer radius in kilometers for sampling background points around presences.")

    parser.add_argument("--background-multiplier", type=float, default=3.0, help="Oversampling multiplier when drawing candidate background points.")

    parser.add_argument("--background-target-count", type=int, default=None, help="Optional explicit number of background points to sample. Used by us_naip for MaxEnt-style broad background sampling.")

    parser.add_argument("--background-max-sampling-rounds", type=int, default=25, help="Maximum candidate-sampling rounds before failing. Increase if the doughnut sampling area is fragmented or very small.")

    parser.add_argument("--background-sampling-mode", choices=["radial", "polygon", "us_naip"], default="radial", help="Radial samples local candidates around presences; polygon uses a union/difference doughnut polygon; us_naip samples broad US backgrounds from downloaded NAIP tile footprints.")

    parser.add_argument("--dataset-tag", type=str, default="may2026", help="Version tag used in output dataset directory names.")

    parser.add_argument("--background-label", type=str, default="PA", help="Short background label used in output dataset directory names, e.g. PA or USBg.")

    parser.add_argument("--max-naip-footprint-union-tiles", type=int, default=50000, help="Maximum downloaded tile rows for building one NAIP footprint union during background sampling. Larger tile indexes skip the union and rely on chip extraction to enforce NAIP availability.")

    parser.add_argument("--allow-existing-output", action="store_true", help="Allow writing into non-empty output directories or existing point CSVs. By default this script fails before overwriting versioned outputs.")

    parser.add_argument("--build-variant", choices=["all", "uniform"], default="all", help="Dataset variants to build. Use uniform to skip spatial BlockCV point and chip extraction.")

    parser.add_argument("--skip-completed-uniform", action="store_true", help="When --build-variant uniform is active, skip species whose final uniform dataset CSV already exists.")

    # PROCESSING SETTINGS
    parser.add_argument("--chip-size", type=int, default=256, help="NAIP chip size in pixels. At 2 m resolution, 256 pixels is approximately 512 m.")

    parser.add_argument("--topo-chip-size", type=int, default=64, help="Topographic chip size in pixels.")

    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    parser.add_argument("--train-frac", type=float, default=0.70, help="Fraction of data for train split.")

    parser.add_argument("--val-frac", type=float, default=0.15, help="Fraction of data for validation split.")

    parser.add_argument("--test-frac", type=float, default=0.15, help="Fraction of data for test split.")

    parser.add_argument("--n-folds", type=int, default=5, help="Number of spatial CV folds.")

    parser.add_argument("--block-size-m", type=float, default=200000, help="Spatial block size in meters for block CV. Default = 200 km.")

    parser.add_argument("--spatial-thin-distance-m", type=float, default=800.0, help="Minimum allowed distance in meters between any presence or background samples after the combined PA dataset is built.")

    parser.add_argument("--topo-min-valid-frac", type=float, default=0.90, help="Minimum fraction of valid pixels required across all four topo layers.")

    # NORMALIZATION SETTINGS
    parser.add_argument("--worldclim-stats-json", type=str, default=None, help="Optional JSON cache for WorldClim normalization stats. If it exists, it is loaded; otherwise stats are computed and written there.")

    parser.add_argument("--topo-normalization-stats", type=str, default=None, help= "Optional path to JSON file containing pre-computed topographic normalization statistics. If omitted, embedded 3DEP 30 m statistics are used.")

    parser.add_argument("--disable-topo-normalization",action="store_true", help="If set, write raw topographic chip values and raw topo scalar columns")

    args = parser.parse_args()

    validate_args(args, parser)

    return args

def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Validate arguments early so long batch jobs fail fast."""
    if not args.occurrence_file and not args.occurrence_dir:
        parser.error("Provide either --occurrence-file or --occurrence-dir.")

    if args.occurrence_file and args.occurrence_dir:
        parser.error("Provide only one of --occurrence-file or --occurrence-dir.")

    total = args.train_frac + args.val_frac + args.test_frac
    if not np.isclose(total, 1.0):
        parser.error("--train-frac + --val-frac + --test-frac must sum to 1.0.")

    if args.chip_size <= 0:
        parser.error("--chip-size must be > 0.")

    if args.n_folds < 2:
        parser.error("--n-folds must be >= 2.")

    if args.background_inner_buffer_km < 0:
        parser.error("--background-inner-buffer-km must be >= 0.")

    if args.background_buffer_km <= 0:
        parser.error("--background-buffer-km must be > 0.")

    if args.background_sampling_mode != "us_naip" and args.background_inner_buffer_km >= args.background_buffer_km:
        parser.error("--background-inner-buffer-km must be smaller than --background-buffer-km.")

    if args.background_multiplier <= 0:
        parser.error("--background-multiplier must be > 0.")

    if args.background_target_count is not None and args.background_target_count <= 0:
        parser.error("--background-target-count must be > 0 when supplied.")

    if args.spatial_thin_distance_m < 0:
        parser.error("--spatial-thin-distance-m must be >= 0. Use 0 to disable adjacent point thinning.")

    if args.max_presences_after_thinning is not None and args.max_presences_after_thinning <= 0:
        parser.error("--max-presences-after-thinning must be > 0 when supplied.")

    if args.topo_mode != "none":
        if args.topo_chip_size <= 0:
            parser.error("--topo-chip-size must be > 0 when --topo-mode is not none.")

        if not (0.0 <= args.topo_min_valid_frac <= 1.0):
            parser.error("--topo-min-valid-frac must be in [0, 1].")

        missing = []
        if not args.dem_raster:
            missing.append("--dem-raster")
        if not args.slope_raster:
            missing.append("--slope-raster")
        if missing:
            parser.error(f"Missing required topo raster argument(s): {', '.join(missing)}")

        has_north_east = bool(args.northness_raster and args.eastness_raster)
        has_aspect = bool(args.aspect_raster)
        if not has_north_east and not has_aspect:
            parser.error(
                "When --topo-mode is not none, provide either both --northness-raster "
                "and --eastness-raster, or provide --aspect-raster."
            )

        if (args.northness_raster and not args.eastness_raster) or (args.eastness_raster and not args.northness_raster):
            parser.error("Provide both --northness-raster and --eastness-raster, or neither.")

        if args.topo_normalization_stats is not None and not Path(args.topo_normalization_stats).exists():
            parser.error(f"Topographic normalization JSON not found: {args.topo_normalization_stats}")
