# Data

This guide describes the inputs Deep SDM needs, how to build a training dataset from them and what the finished dataset looks like.

## Inputs

| Input | Description |
|---|---|
| Occurrence records | A CSV of presence locations for one species. Latitude and longitude columns are detected automatically (`decimalLatitude`, `latitude` or `lat`, and the matching longitude names). Optional columns: `species`, `Source` and a coordinate uncertainty field. |
| NAIP imagery | 4-band GeoTIFF tiles (red, green, blue, near-infrared), all at one resolution, and a GeoPackage tile index with each tile's footprint and acquisition details. The framework was developed with NAIP resampled to 2 m. |
| Climate | WorldClim 2.1 bioclimatic variables at 30 arc seconds (`wc2.1_30s_bio_1` to `wc2.1_30s_bio_19`). |
| Human land use | The Global Human Modification raster (gHM). |
| Terrain | Elevation, slope (degrees), northness and eastness rasters, plus a JSON file with the mean and standard deviation of each, used to standardize them. |

Clean the occurrence records before you start. Records with large coordinate uncertainty place the image chip in the wrong spot, so remove them (`--max-coordinate-uncertainty-m 128` works well for 512 m chips) and thin clustered records.

## Building a dataset

The builder loads the occurrences, samples background points, cuts an image chip and a terrain chip around every point, attaches the climate, land use and terrain values, and writes the training, validation and test splits:

```bash
python -m host_dataset_builder \
  --occurrence-file occurrences/species.csv \
  --output-root datasets/ \
  --tileindex naip/tile_index.gpkg \
  --naip-folder naip/ \
  --worldclim-folder env/worldclim/ \
  --worldclim-stats-json datasets/worldclim_stats.json \
  --ghm-raster env/gHM_WGS84.tif \
  --topo-mode both \
  --dem-raster env/dem.tif --slope-raster env/slope.tif \
  --northness-raster env/northness.tif --eastness-raster env/eastness.tif \
  --topo-normalization-stats env/topo_norm_stats.json \
  --max-coordinate-uncertainty-m 128 \
  --background-sampling-mode us_naip \
  --background-sampling-design domain_balanced \
  --naip-source-lock-mode strict \
  --uniform-split-group-column source_tile_key \
  --block-group-column source_tile_key \
  --blockcv-validation-mode heldout_fold \
  --manifest-qa-mode strict
```

The last seven options are the recommended settings for datasets that span many NAIP acquisitions:

- **Background sampling across the imagery.** `us_naip` draws background points from the whole area covered by the imagery. `domain_balanced` first places a minimum number of points in every combination of state, acquisition year and month, then spreads the rest by area. The model then sees every season and sensor that the presences might come from. To limit backgrounds to some states, add `--background-allowed-states CA OR`.
- **One image per chip.** `--naip-source-lock-mode strict` reads each chip from a single acquisition and records the exact file it came from. A chip that straddles two images is never stitched from different flight dates.
- **Splits that respect image tiles.** Grouping by `source_tile_key` keeps all points from one NAIP tile in the same split and the same spatial fold. Without this, a model can score well by recognizing the tile instead of the habitat.
- **Strict quality checks.** `--manifest-qa-mode strict` writes a `qa/` folder and stops if it finds duplicates, conflicting labels, leakage between splits, missing files or chips that mix acquisitions.

Two options help you check a build before it runs. `--plan-only` validates the occurrence file and prints the planned outputs without reading any rasters. `--background-domain-plan-only` lists the acquisition groups available for background sampling.

For large builds you can split the work in two. `python -m host_dataset_builder.candidate_manifests` samples the points, checks their environmental values and writes nested manifests for several background sizes (for example `--targets 10000 15000 20000`) without extracting any imagery. After reviewing them, `python -m host_dataset_builder.materialize_candidates` extracts the chips once for the chosen package. Run either module with `--help` to see its options.

## Dataset format

Each species gets one dataset folder containing the image chips, the terrain chips and a CSV with one row per point.

| Column | Meaning |
|---|---|
| `split` | `train`, `val` or `test` |
| `presence` | 1 for a presence, 0 for a background point |
| `lat`, `lon` | Point location in decimal degrees |
| `chip_path` | NAIP chip, relative to the dataset folder |
| `topo_chip_path` | Terrain chip, relative to the dataset folder |
| `wc2.1_30s_bio_1` to `wc2.1_30s_bio_19` | Standardized bioclimatic variables |
| `ghm` | Human modification, from 0 to 1 |
| `lat_norm`, `lon_norm` | Standardized latitude and longitude |
| `elev_mean`, `elev_sd`, `elev_min`, `elev_max`, `slope_mean`, `slope_sd`, `slope_min`, `slope_max`, `northness_mean`, `eastness_mean` | Standardized terrain summaries over the chip |

The builder also records where each chip came from (source tile, state and acquisition date) and the spatial block of each point. Training ignores these columns.

**Image chips** are 4-band 8-bit GeoTIFFs of 256 by 256 pixels. Chips are cut at the imagery's own resolution, so with 2 m imagery each chip covers about 512 m on a side. **Terrain chips** cover the same footprint at 64 by 64 pixels with four standardized bands: elevation, slope, northness and eastness.

**Spatial cross-validation.** Besides the main split (70% training, 15% validation and 15% test), the builder writes five fold files based on 200 km spatial blocks. Use them to check how well a model transfers to regions it has not seen.

## Standardization statistics

The climate variables are standardized with means and standard deviations computed over the imagery footprint, and latitude and longitude with the dataset's own mean and standard deviation. To map new areas, inference must apply exactly the same scaling. Copy these values into the training configuration (see [Training](training.md)) so they travel with the model:

- `worldclim_stats`: from the file given to `--worldclim-stats-json`
- `lat_lon_stats`: the mean and standard deviation of the `lat` and `lon` columns
- `topo_normalization_stats`: from the file given to `--topo-normalization-stats`
