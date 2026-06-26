# Host Dataset Builder

This package is the current modular dataset builder for Host NAIP SDMs. It
replaces the older monolithic workflow with explicit modules for occurrence
loading, background sampling, NAIP extraction, environmental variables,
topography, split creation, and final dataset CSV writing.

## Entry Point

Run the package from the Deep_SDM root:

```bash
python -m host_dataset_builder \
  --occurrence-file /path/to/species_thinned.csv \
  --output-root /mnt/rsstu_dsfas_gsv_naip/Host_NAIP_SDM/Host_Datasets \
  --tileindex /path/to/naip_tileindex.gpkg \
  --naip-folder /mnt/rsstu_dsfas_gsv_naip/Host_NAIP_SDM/NAIP_Archive \
  --worldclim-folder /path/to/worldclim \
  --ghm-raster /path/to/ghm.tif \
  --topo-mode both \
  --dem-raster /path/to/dem.tif \
  --slope-raster /path/to/slope.tif \
  --aspect-raster /path/to/aspect.tif
```

Use `--occurrence-dir` instead of `--occurrence-file` for a batch of species
CSV files.

## Safety And Review Options

- `--plan-only` performs occurrence schema checks and prints planned output
  paths without creating directories or loading rasters.
- `--max-presences-after-thinning` caps presences after WorldClim-cell thinning
  and is intended for bounded validation runs before a full build.
- `--allow-existing-output` is required to write into existing output
  artifacts. Without it, the builder refuses to overwrite populated output
  directories or existing presence/background point CSVs.

## Expected Occurrence Inputs

The loader can auto-detect common coordinate aliases:

- latitude: `decimalLatitude`, `latitude`, or `lat`
- longitude: `decimalLongitude`, `longitude`, or `lon`
- coordinate uncertainty: common GBIF/iNaturalist uncertainty field names

Recommended priority inputs for the May 2026 host builds are the strict
`coordinate_uncertainty_under_128` thinned iNaturalist + GBIF files unless a
mentor explicitly chooses a looser threshold.

## Output Structure

For each species, the builder writes:

- occurrence-level presence/background points and background audit tables
- one uniform train/validation/test dataset
- five spatial BlockCV fold datasets
- NAIP chip paths and optional topographic chip paths relative to each dataset
  root
- WorldClim, GHM, latitude/longitude, source, coordinate uncertainty, and
  topographic summary columns when available

Relative paths in newly generated CSVs are written with forward slashes so the
CSV remains portable across Linux and Windows path conventions.

## Acceptance QA Before Modeling

Before using a newly built dataset for training, verify:

- row counts and columns
- train/validation/test counts and class balance
- unique `sample_id` values and no split overlap
- presence/source counts and retained coordinate uncertainty fields
- bounded existence checks for `chip_path` and `topo_chip_path`
- latitude/longitude domain checks, especially for hosts with known input
  coordinate-risk flags
- BlockCV fold existence, fold row counts, and fold class balance

Do not compare model outputs until prediction sample IDs, split definitions,
dataset CSV paths, and evaluation denominators are traced.
