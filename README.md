# Deep_SDM

Deep_SDM is a research-oriented deep learning framework for species distribution modeling that combines spatial context with structured predictors in a unified workflow. The project is designed to support end-to-end experimentation—from dataset preparation and split generation, to multimodal model training, evaluation, and inference—while keeping components modular so new data sources, model branches, and study systems can be added with minimal refactoring.

## Topography (USGS 3DEP 1/3 arc-second)

This repo supports optional topography integration using USGS 3DEP 1/3 arc-second DEM tiles, derived terrain bands, and/or scalar summaries.

### 1) Download 3DEP tiles and create a VRT mosaic

Use the downloader with a recommended CONUS extent:

```bash
python download_3dep_13arcsec_dem.py \
  --bounds -125 24 -66 50 \
  --out-dir Env_Data/Topography/USGS_3DEP_13_CONUS \
  --mosaic-name USGS_3DEP_13_CONUS_DEM
```

A VRT is preferred initially because it references tile files directly and avoids creating a very large physical GeoTIFF mosaic too early.

### 2) Prepare terrain derivatives in a projected CRS

Derivatives should be created after reprojection (default EPSG:5070), not directly from geographic-degree DEM grids.

```bash
python prepare_topography_3dep_derivatives.py \
  --dem-input Env_Data/Topography/USGS_3DEP_13_CONUS/mosaic/USGS_3DEP_13_CONUS_DEM.vrt \
  --out-dir Env_Data/Topography/USGS_3DEP_13_CONUS/derived \
  --target-crs EPSG:5070
```

Outputs include elevation, slope, aspect, northness (`cos(aspect)`), and eastness (`sin(aspect)`).

### 3) Build datasets with topography

Topography can be disabled (default) or enabled as scalar summaries, image chips, or both:

```bash
python create_host_datasets.py ... \
  --topo-mode both \
  --dem-raster .../dem_3dep_13_epsg5070.tif \
  --slope-raster .../slope_degrees_3dep_13_epsg5070.tif \
  --aspect-raster .../aspect_degrees_3dep_13_epsg5070.tif \
  --northness-raster .../northness_3dep_13_epsg5070.tif \
  --eastness-raster .../eastness_3dep_13_epsg5070.tif
```

### 4) Train baseline vs topography-enabled models

Model types include:
- baseline imagery + climate: `image_climate`
- imagery only: `image_only`
- climate only: `climate_only`
- topography chip only: `topo_only`
- full imagery + topo chip + climate: `image_topo_climate` (also `naip_topo_climate`)

Input modes include `baseline`, `topo_scalar`, `topo_chip`, and `full`.
