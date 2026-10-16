# Inference

Inference turns a trained model into maps. It reads NAIP tiles, scores overlapping windows across each tile and writes three rasters per tile on the tile's own grid and coordinate system.

## Running inference

```bash
python inference_multimodal_production.py \
  --config runs/my_species/config.input.json \
  --checkpoint runs/my_species/checkpoints/best.pt --strict-checkpoint \
  --tile-list tiles.txt \
  --output-dir maps/my_species/ \
  --worldclim-dir env/worldclim/ --ghm-raster env/gHM_WGS84.tif --env-clip-conus \
  --dem-raster env/dem.tif --slope-raster env/slope.tif \
  --northness-raster env/northness.tif --eastness-raster env/eastness.tif \
  --topo-normalization-stats env/topo_norm_stats.json \
  --chip-size 256 --stride 64 --batch-size 128 \
  --mc-samples 10 --seed 42 \
  --blend-window hann --blend-edge-floor 0.05
```

These are the recommended settings. The sections below explain them.

## Inputs

- **Model.** `--checkpoint` is the trained model. `--config` is the `config.input.json` written by training. It supplies the feature list, the standardization statistics and any image harmonization settings, so inference scales new data exactly as in training. `--strict-checkpoint` stops with an error if the checkpoint and the model definition do not match exactly.
- **Tiles.** Give one tile (`--tile-fp`), a text file with one tile path per line (`--tile-list`) or a folder (`--naip-folder`, with `--recursive` to search subfolders).
- **Climate and land use.** `--worldclim-dir` and `--ghm-raster`. `--env-clip-conus` loads only the part of these global rasters that covers the lower 48 states, which saves memory. Use `--env-clip-bounds` for another region.
- **Terrain.** Models with terrain chips or terrain summaries need `--dem-raster`, `--slope-raster`, `--northness-raster` and `--eastness-raster` (or `--aspect-raster` in place of the last two), with `--topo-normalization-stats`.

Statistics given on the command line (`--worldclim-stats`, `--lat-lon-stats`, `--topo-normalization-stats`) take precedence over those in the configuration.

## Overlapping windows

The model scores square windows of `--chip-size` pixels that step across the tile every `--stride` pixels. With 256-pixel windows and a 64-pixel step, each pixel is covered by up to 16 windows, and the final value at each pixel combines all of them. Overlap smooths the map and reduces the influence of any single window.

A narrow strip along the right and bottom edges of a tile, narrower than one window, stays empty. Neighboring NAIP tiles overlap, so this strip is filled when tiles are mosaicked.

## Hann blending

The model gives one score for each window, and that score describes the whole window. A pixel near the center of a window is described well by it, while a pixel near its edge is better described by the windows centered closer to it. By default every window that covers a pixel counts equally (`--blend-window flat`). `--blend-window hann` instead weights each window's contribution by a smooth dome that is highest at the window's center and lowest at its edges. Each pixel then takes its value mostly from the windows centered nearest to it, and the map changes smoothly from one window to the next.

The weight never falls all the way to zero. `--blend-edge-floor 0.05` keeps a minimum weight of 5% at the window edges, so pixels covered only by window edges still receive a value. Suitability and uncertainty use the same weights. The count raster stays an unweighted count of windows.

## Uncertainty

`--mc-samples` sets how many times each window is scored with dropout left on (Monte Carlo dropout). Each pass gives a slightly different answer. The suitability map is the mean of the passes, and the uncertainty map is their standard deviation, combined across windows with the same blending weights. Ten passes give stable estimates. Use `--mc-samples 1` for a single deterministic pass without uncertainty.

Results are reproducible. Each tile gets its own random seed, derived from `--seed` and the tile's path, so a tile gives the same result no matter which other tiles run with it or in what order.

## Outputs

For each input tile `<tile>.tif`:

| File | Contents | Type, no-data value |
|---|---|---|
| `<tile>_prediction.tif` | Suitability from 0 to 1 | float32, −9999 |
| `<tile>_uncertainty.tif` | Standard deviation across Monte Carlo passes | float32, −9999 |
| `<tile>_count.tif` | Number of windows that scored each pixel | uint16, 0 |

The blending method, edge floor, seed and number of passes are written into each raster's metadata and into `tiled_inference_summary.json`. With `--skip-existing`, tiles that already have outputs are skipped. If the existing outputs were made with different settings, inference stops with an error instead of mixing results.

To apply a Platt calibrator fitted during training (`postfit_calibration: "platt"`), pass `--platt-calibrator`. It is applied to every pass before the mean and standard deviation are taken.

## Mosaicking

Outputs stay on each NAIP tile's own grid, in the UTM zone of that tile. To build a regional map, mosaic the tiles of each UTM zone separately and then reproject the zone mosaics to one grid with an equal area projection (for example EPSG:5070 for the conterminous United States). Mosaicking zone by zone avoids seams where zones meet. Build the suitability, uncertainty and count mosaics the same way so they line up cell for cell.

## Notes

- The suitability score is relative. Use it to compare places, and read it together with the uncertainty map.
- `--fast-topo-scalars` and `--conda-env` are accepted for compatibility but currently have no effect.
