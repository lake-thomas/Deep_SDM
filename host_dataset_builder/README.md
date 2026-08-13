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
- `--background-domain-plan-only` reads only the tile index, applies any
  `--background-allowed-states` restriction, prints the available acquisition
  domains, and exits without indexing imagery or writing files.
- `--max-presences-after-thinning` caps presences after WorldClim-cell thinning
  and is intended for bounded validation runs before a full build.
- `--allow-existing-output` is required to write into existing output
  artifacts. Without it, the builder refuses to overwrite populated output
  directories or existing presence/background point CSVs.

## August 2026 Acquisition-Aware Path

The August interfaces are opt-in so the June behavior remains reproducible.
An August candidate build should combine:

- `--background-sampling-mode us_naip`;
- `--background-sampling-design domain_balanced` for the balanced arm, while
  retaining an equal-sized `area_weighted` arm;
- `--naip-source-lock-mode strict`;
- `--uniform-split-group-column source_tile_key`;
- `--block-group-column source_tile_key`;
- `--blockcv-validation-mode heldout_fold`;
- `--manifest-qa-mode strict`.

The source-locked sampler carries the selected local tile, archive state,
archive year, exact acquisition date, year, month, and domain ID with each
background. Strict extraction uses only the selected state and exact
acquisition date. A boundary chip may use multiple constituent tiles only when
they share that state/date domain. The final dataset records both intended and
actual source paths, filenames, tile keys, state/date, constituent count, valid
pixel fraction, and an explicit provenance-match flag.

Domain-balanced sampling first supplies the configured minimum per available
coarse domain and then allocates the residual by accessible footprint area.
The default coarse domain is state × acquisition year × month. Exact-date depth
is retained and reported in QA; it can also be selected explicitly through
`--background-domain-columns` after reviewing feasibility.

The nominal 20,000-background target is a soft planning cap, not a builder
limit or absolute maximum. A candidate pool or final set may exceed 20,000 when
coverage and validation evidence show that additional samples remain useful.
Current planned nested screens are approximately 6k/10k/15k/20k for Vitis and
5k/10k/15k for Tanoak. Tanoak must add:

```bash
--background-allowed-states CA OR
```

and must not use national backgrounds in the current design.

The manifest-only planner is available as
`python -m host_dataset_builder.candidate_manifests`. It requires both
`--worldclim-folder` and `--ghm-raster`, removes candidates or presences with
any missing environmental predictor before ranking, writes the attrition
audits, and produces frozen nested manifests plus ranked extraction reserves.
The default reserve depth is 1.2 times the largest requested target per arm;
adjust it with `--reserve-factor` when measured extraction attrition warrants
more or less headroom. Candidate packages remain non-model-ready until strict
source-locked chip/topography materialization and final manifest QA pass.

Reviewed packages are materialized with
`python -m host_dataset_builder.materialize_candidates`. It extracts the union
of both arms only once, writes checkpointed and atomic NAIP/topography chips,
uses the frozen raw environmental values, fills any failed primary background
from its arm's ordered reserve, and recomputes leakage-safe uniform/BlockCV
assignments from the actual constituent source-tile components. Use `--resume`
only with an exactly matching candidate/configuration identity.

Before any extraction, inspect the accessible Tanoak domains with the standard
required input arguments plus:

```bash
--background-sampling-mode us_naip \
--background-allowed-states CA OR \
--background-sampling-design domain_balanced \
--background-domain-columns source_archive_state source_acq_year source_acq_month \
--background-domain-plan-only
```

The strict manifest gate fails early at argument validation unless strict
source locking and leakage-safe grouping are selected. After extraction it
writes a `qa/` package and rejects hard findings involving required fields,
duplicates, conflicting labels, source-tile or spatial-block leakage, missing
paths, non-finite predictors, cross-domain chips, or intended/actual source
mismatch. Acquisition-domain depth and extreme prevalence are reported as
diagnostics rather than universal exclusion rules.

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
- strict intended-versus-actual NAIP source agreement
- no source-tile, exact-coordinate, or spatial-block leakage between evaluation
  partitions
- state × archive year × exact acquisition date/month depth, including median
  backgrounds per domain and counts of domains with at least 5 or 10
  backgrounds
- no multi-state or multi-acquisition chip constituents in strict mode

Do not compare model outputs until prediction sample IDs, split definitions,
dataset CSV paths, and evaluation denominators are traced.
