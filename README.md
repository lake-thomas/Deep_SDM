# Deep SDM

Deep SDM builds species distribution models that learn from aerial imagery. Each model looks at a small aerial image of a place, together with the climate, terrain and human land use there, and predicts how suitable that place is for a species. Once trained, a model can map habitat suitability at fine resolution across large regions.

The framework was developed to map host plants of invasive pests and pathogens across the United States using 4-band NAIP aerial imagery (red, green, blue and near-infrared). It works for any species with reliable occurrence records inside the imagery's coverage.

## What it does

- **Builds training datasets** from occurrence records. It samples background points, cuts an image chip and a terrain chip around every point, attaches climate and land use variables, and splits the data into training, validation and test sets with spatial blocks to limit leakage.
- **Trains multimodal models.** An image network (ResNet-18), a small terrain network and a network for tabular variables are combined into one classifier. Seven model types let you use any combination of the three inputs.
- **Keeps predictions stable across image boundaries.** Aerial imagery is collected in separate flights, so the same ground can look different in neighboring images. Same-ground consistency training teaches the model to give the same answer for the same place in both images, which removes blocky seams from the final maps.
- **Maps large regions.** Inference slides overlapping windows across each NAIP tile, blends the overlapping predictions smoothly and estimates uncertainty with Monte Carlo dropout.
- **Records how every result was made.** Each run saves its configuration, a checksum of its dataset, the code version and its random seeds.

## Repository layout

| Path | Contents |
|---|---|
| `main.py` | Training entry point |
| `model.py` | Model types, losses and the same-ground consistency term |
| `datasets.py`, `transforms.py` | Data loading and augmentation |
| `same_ground_pairs.py` | Loader for same-ground image pairs |
| `image_harmonization.py` | Optional band standardization for each chip |
| `train_utils.py`, `eval_utils.py`, `calibration_utils.py` | Training loop, evaluation metrics and calibration |
| `inference_multimodal_production.py` | Tiled inference over NAIP tiles |
| `host_dataset_builder/` | Dataset builder |
| `examples/train_config.json` | Example training configuration |
| `docs/` | Guides |
| `tests/` | Unit tests and short training runs on synthetic data |

## Installation

Deep SDM is tested with Python 3.10 and PyTorch 2.10. Install PyTorch for your hardware first (see [pytorch.org](https://pytorch.org)), then the remaining packages:

```bash
pip install -r requirements.txt
```

Packages that depend on GDAL (rasterio, geopandas) are often easier to install with conda:

```bash
conda install -c conda-forge rasterio geopandas shapely scipy
```

Run every command from the repository root, because the modules import each other directly.

## Quick start

**1. Build a dataset** from an occurrence file, the NAIP imagery and the environmental rasters:

```bash
python -m host_dataset_builder \
  --occurrence-file occurrences/species.csv \
  --output-root datasets/ \
  --tileindex naip/tile_index.gpkg \
  --naip-folder naip/ \
  --worldclim-folder env/worldclim/ \
  --ghm-raster env/gHM_WGS84.tif \
  --topo-mode both \
  --dem-raster env/dem.tif --slope-raster env/slope.tif \
  --northness-raster env/northness.tif --eastness-raster env/eastness.tif \
  --topo-normalization-stats env/topo_norm_stats.json
```

**2. Train a model.** Copy `examples/train_config.json`, point it at your dataset and run:

```bash
python main.py --config my_config.json
```

**3. Map suitability** over NAIP tiles with the trained model:

```bash
python inference_multimodal_production.py \
  --config runs/my_species/config.input.json \
  --checkpoint runs/my_species/checkpoints/best.pt \
  --naip-folder naip/ --recursive \
  --worldclim-dir env/worldclim/ --ghm-raster env/gHM_WGS84.tif --env-clip-conus \
  --dem-raster env/dem.tif --slope-raster env/slope.tif \
  --northness-raster env/northness.tif --eastness-raster env/eastness.tif \
  --topo-normalization-stats env/topo_norm_stats.json \
  --output-dir maps/my_species/ \
  --chip-size 256 --stride 64 --mc-samples 10 --seed 42 \
  --blend-window hann --blend-edge-floor 0.05
```

Each tile produces a suitability map, an uncertainty map and a coverage count map.

## Recommended approach

For new species, start from `examples/train_config.json`. It trains an image and tabular model (`image_tabular`) with a pretrained image network, random rotations and flips, mild random color changes and same-ground consistency training. For inference, use 256-pixel windows with a 64-pixel step, Hann blending and 10 Monte Carlo dropout passes, as in the quick start. The guides explain each choice and which settings to tune.

## Guides

- [Data](docs/data.md): inputs, building datasets and the dataset format
- [Training](docs/training.md): configuration, model types and outputs
- [Same-ground consistency](docs/same_ground_consistency.md): stabilizing predictions across image boundaries
- [Inference](docs/inference.md): mapping, blending, uncertainty and outputs

## Tests

```bash
python -m unittest discover -s tests
```

The tests use small synthetic rasters and run on a CPU in about a minute.

## Citation

If you use Deep SDM in your work, please cite this repository:

```bibtex
@software{lake_deep_sdm,
  author = {Lake, Thomas},
  title  = {Deep SDM: species distribution models from aerial imagery},
  url    = {https://github.com/lake-thomas/Deep_SDM},
  year   = {2026}
}
```

## License

MIT. See [LICENSE](LICENSE).
