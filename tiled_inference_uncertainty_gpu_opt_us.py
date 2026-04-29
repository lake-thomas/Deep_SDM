# Inference Script for US-Trained Host Model with Tiled NAIP Imagery and Uncertainty Estimation
# Thomas Lake, January 2026

# Imports

import os

conda_env = r"C:\Users\talake2\AppData\Local\anaconda3\envs\naip_ailanthus_env"
os.environ["GDAL_DATA"] = os.path.join(conda_env, "Library", "share", "gdal")
os.environ["PROJ_LIB"] = os.path.join(conda_env, "Library", "share", "proj")
os.environ["PATH"] += os.pathsep + os.path.join(conda_env, "Library", "bin")

import numpy as np #noqa: E402
import pandas as pd #noqa: E402
from tqdm import tqdm #noqa: E402
import rasterio #noqa: E402
from rasterio.windows import Window #noqa: E402
from rasterio.transform import rowcol #noqa: E402
from rasterio.warp import transform, reproject, Resampling, calculate_default_transform #noqa: E402
from shapely.geometry import Point #noqa: E402
import geopandas as gpd #noqa: E402
import time #noqa: E402

import torch #noqa: E402
from torchvision import transforms #noqa: E402

from model import HostImageryClimateModel, HostClimateOnlyModel, HostImageryOnlyModel #noqa: E402
from train_utils import get_default_device, load_model_from_checkpoint #noqa: E402


# --- Updated WorldClim normalization stats for US models (Jan 2026) ---
WORLDCLIM_STATS = {
    "wc2.1_30s_bio_1":  {"mean": 10.9748, "std": 5.0872},
    "wc2.1_30s_bio_2":  {"mean": 13.6538, "std": 2.1714},
    "wc2.1_30s_bio_3":  {"mean": 36.8663, "std": 6.5792},
    "wc2.1_30s_bio_4":  {"mean": 881.2029, "std": 180.3222},
    "wc2.1_30s_bio_5":  {"mean": 30.3126, "std": 3.8458},
    "wc2.1_30s_bio_6":  {"mean": -7.2765, "std": 6.7536},
    "wc2.1_30s_bio_7":  {"mean": 37.5892, "std": 5.4061},
    "wc2.1_30s_bio_8":  {"mean": 15.7360, "std": 7.9419},
    "wc2.1_30s_bio_9":  {"mean": 5.9556, "std": 11.2411},
    "wc2.1_30s_bio_10": {"mean": 21.6966, "std": 4.2591},
    "wc2.1_30s_bio_11": {"mean": -0.0010, "std": 6.6096},
    "wc2.1_30s_bio_12": {"mean": 764.9769, "std": 421.6841},
    "wc2.1_30s_bio_13": {"mean": 99.6282, "std": 50.9529},
    "wc2.1_30s_bio_14": {"mean": 33.7521, "std": 27.9409},
    "wc2.1_30s_bio_15": {"mean": 39.4090, "std": 21.1094},
    "wc2.1_30s_bio_16": {"mean": 271.0835, "std": 144.2669},
    "wc2.1_30s_bio_17": {"mean": 117.5644, "std": 92.1923},
    "wc2.1_30s_bio_18": {"mean": 213.2364, "std": 114.1300},
    "wc2.1_30s_bio_19": {"mean": 165.2916, "std": 151.7543},
}

# --- Updated Latitude/Longitude normalization stats for US models (Jan 2026) ---
LAT_LON_STATS = {
    "lat": {"mean": 39.34043295, "std": 4.24367101},
    "lon": {"mean": -90.41125129, "std": 15.36062434},
}


def normalize_coord(value, varname):
    """Normalize latitude or longitude value using hard-coded stats from the US."""
    mean = LAT_LON_STATS[varname]['mean']
    std = LAT_LON_STATS[varname]['std']
    return (value - mean) / std

# Extract WorldClim variables for a given lon/lat point from preloaded rasters
def extract_worldclim_vars_for_point(lon, lat, wc_rasters):
    """
    Extract WorldClim variables for a given lon/lat point from preloaded rasters.
    """
    values = []
    for bio in range(1, 20):
        varname = f"wc2.1_30s_bio_{bio}"
        r = wc_rasters[varname]

        # Reproject point to raster CRS
        x, y = transform("EPSG:4326", r["crs"], [lon], [lat])
        x, y = x[0], y[0]

        # Pixel indices
        row, col = rowcol(r["transform"], x, y)
        try:
            val = r["array"][row, col]
        except IndexError:
            val = np.nan
        values.append(val)
    return values

# Extract GHM value for a given lon/lat point from preloaded raster
def extract_ghm_for_point(lon, lat, ghm_raster):
    """
    Extract GHM value for a given lon/lat point from preloaded raster.
    """
    r = ghm_raster
    x, y = transform("EPSG:4326", r["crs"], [lon], [lat])
    x, y = x[0], y[0]

    row, col = rowcol(r["transform"], x, y)
    try:
        val = r["array"][row, col]
    except IndexError:
        val = np.nan
    return val

# Preload and normalize all WorldClim rasters into memory
def preload_and_normalize_rasters(worldclim_dir, ghm_path, wc_stats):
    """
    Preload Worldclim and GHM rasters into memory.
    Normalize Worldclim rasters using provided statistics.
    """
    wc_rasters = {}
    for bio in range(1, 20):
        varname = f"wc2.1_30s_bio_{bio}"
        path = f"{worldclim_dir}/{varname}.tif"
        with rasterio.open(path) as ds:
            arr = ds.read(1).astype(np.float32)
            # Mask nodata
            arr = np.where(arr == ds.nodata, np.nan, arr)
            # Normalize using hard-coded stats
            mean = wc_stats[varname]['mean']
            std = wc_stats[varname]['std']
            arr_norm = (arr - mean) / std

            wc_rasters[varname] = {
                "array": arr_norm,
                "transform": ds.transform,
                "crs": ds.crs,
                "nodata": ds.nodata
            }

    # GHM raster (already normalized 0-1)
    with rasterio.open(ghm_path) as ds:
        arr = ds.read(1).astype(np.float32)
        arr = np.where(arr == ds.nodata, np.nan, arr)
        ghm_raster = {
            "array": arr,
            "transform": ds.transform,
            "crs": ds.crs,
            "nodata": ds.nodata,
        }

    return wc_rasters, ghm_raster

def enable_mc_dropout(model):
    """Enable dropout layers during inference"""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()

@torch.inference_mode() # Disable gradient calculations for inference-only
def run_inference_on_tile(
    tile_fp, output_dir, model, device, image_transform, wc_rasters, ghm_raster, chip_size=256, stride=256
):
    
    # Track inference time
    start_time = time.time()

    # Output file path
    tile_name = os.path.splitext(os.path.basename(tile_fp))[0]
    output_fp = os.path.join(output_dir, f"{tile_name}_predictions.tif")

    with rasterio.open(tile_fp) as src:
        profile = src.profile
        width, height = src.width, src.height
        transform = src.transform
        crs = src.crs

        # Init prediction accumulation arrays
        sum_array = np.zeros((height, width), dtype=np.float32)
        std_sum_array = np.zeros((height, width), dtype=np.float32)
        count_array = np.zeros((height, width), dtype=np.uint16)

        # Chip grid
        rows = range(0, height - chip_size + 1, stride)
        cols = range(0, width - chip_size + 1, stride)
        chip_iter = [(r, c) for r in rows for c in cols]

        # Batch buffers
        chip_batch, env_batch, locs = [], [], []

        def flush_batch():
            '''
            Run model with batch inference.
            '''

            chips = torch.cat(chip_batch, dim=0).to(device) if chip_batch else None
            envs = torch.cat(env_batch, dim=0).to(device) if env_batch else None

            # Monte Carlo Dropout
            T = 2 # Number of stochastic forward passes for uncertainty estimation
            probs = []
            for _ in range(T):
                if isinstance(model, HostImageryClimateModel):
                    with torch.inference_mode(), torch.autocast("cuda"): # Mixed precision for faster inference
                        p = model(chips, envs)
                elif isinstance(model, HostImageryOnlyModel):
                     with torch.inference_mode(), torch.autocast("cuda"):
                        p = model(chips)
                elif isinstance(model, HostClimateOnlyModel):
                     with torch.inference_mode(), torch.autocast("cuda"):
                        p = model(envs)
                else:
                    raise NotImplementedError("Unknown model type used in inference.")
                
                p_prob = torch.sigmoid(p) # Convert logits to probabilities
                probs.append(p_prob.detach().cpu().numpy()) # Store the probability for one forward pass

            probs = np.stack(probs, axis=0) # Shape: (T, batch_size, 1)
            mean_probs = probs.mean(axis=0).squeeze() # Shape: (batch_size,)
            std_probs = probs.std(axis=0).squeeze() # Shape: (batch_size,)

            # Write batch back to accumulators
            for (row, col), mean_p, std_p in zip(locs, mean_probs, std_probs):
                sum_array[row:row + chip_size, col:col + chip_size] += float(mean_p)
                std_sum_array[row:row + chip_size, col:col + chip_size] += float(std_p)
                count_array[row:row + chip_size, col:col + chip_size] += 1

            # Clear batch buffers
            chip_batch.clear()
            env_batch.clear()
            locs.clear()

        # Iterate over chips for tiled inference
        for row, col in chip_iter:
            window = Window(col_off=col, row_off=row, width=chip_size, height=chip_size)
            chip = src.read(window=window).astype(np.float32) / 255.0 # Normalize to [0, 1]
            chip_tensor = image_transform(np.moveaxis(chip, 0, -1)).unsqueeze(0) # Tensor Shape: (1, C, H, W)

            # Center coords -> pixel indices for env rasters
            cx, cy = src.transform * (col + chip_size // 2, row + chip_size // 2)
            lon, lat = rasterio.warp.transform(crs, "EPSG:4326", [cx], [cy])
            lon, lat = lon[0], lat[0]

            # Collect env features
            # Use preloaded rasters instead of opening files to get Worldclim and GHM values from NAIP chip center coords
            wc_values = extract_worldclim_vars_for_point(lon, lat, wc_rasters)
            ghm_value = extract_ghm_for_point(lon, lat, ghm_raster)

            # Normalize coordinates (us-scale stats)
            lat_norm = normalize_coord(lat, "lat")
            lon_norm = normalize_coord(lon, "lon")

            env_tensor = torch.tensor([[ *wc_values, ghm_value, lat_norm, lon_norm ]], dtype=torch.float32).to(device)

            # Accumulate in batch
            chip_batch.append(chip_tensor)
            env_batch.append(env_tensor)
            locs.append((row, col))

            if len(chip_batch) >= 512: # Batch size - adjust based on GPU memory
                flush_batch()
        
        # Flush any remaining chips in batch
        flush_batch()

        # Compute final averages
        avg_array = np.divide(sum_array, count_array, out=np.full_like(sum_array, np.nan), where=(count_array > 0))
        std_avg_array = np.divide(std_sum_array, count_array, out=np.full_like(std_sum_array, np.nan), where=(count_array > 0))

        # --- Reproject setup ---
        dst_crs = "EPSG:5070"
        dst_transform, dst_width, dst_height = calculate_default_transform(
            crs, dst_crs, src.width, src.height, *src.bounds
        )

        # --- Reproject average prediction to float32 ---
        reprojected_avg_array = np.full((dst_height, dst_width), -9999, dtype=np.float32)

        reproject(
            source=avg_array,
            destination=reprojected_avg_array,
            src_transform=src.transform,
            src_crs=crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=-9999
        )

        # --- Reproject standard deviation to float32 ---
        reprojected_std_array = np.full((dst_height, dst_width), -9999, dtype=np.float32)

        reproject(
            source=std_avg_array,
            destination=reprojected_std_array,
            src_transform=src.transform,
            src_crs=crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=-9999
        )

        # --- Common raster profile ---
        float_profile = profile.copy()
        float_profile.update({
            "driver": "GTiff",
            "dtype": "float32",
            "count": 1,
            "width": dst_width,
            "height": dst_height,
            "crs": dst_crs,
            "transform": dst_transform,
            "nodata": -9999,
            "compress": "lzw"
        })

        # --- Save mean prediction raster ---
        with rasterio.open(output_fp, "w", **float_profile) as dst:
            dst.write(reprojected_avg_array, 1)

        print(f"[{tile_name}] Saved float32 prediction to {output_fp}")

        # --- Save standard deviation (uncertainty) raster ---
        std_output_fp = os.path.join(output_dir, f"{tile_name}_uncertainty.tif")

        with rasterio.open(std_output_fp, "w", **float_profile) as dst:
            dst.write(reprojected_std_array, 1)

        print(f"[{tile_name}] Saved float32 uncertainty to {std_output_fp}")

        # Track inference time
        end_time = time.time()
        print(f"[{tile_name}] Inference completed in {end_time - start_time:.2f} seconds.")


def main():
    # Paths
    csv_path = r"C:\Users\talake2\Desktop\Host_NAIP_Datasets\Notholithofagus_Densiflorus_US_Uniform_PA_NAIP_256_april2026\Notholithofagus_Densiflorus_Pres_Bg_US_Uniform_Train_Val_Test_Dataset.csv"
    naip_folder = r"Y:\Host_NAIP_SDM\NAIP_Archive\states\CA\2022\raw"
    worldclim_folder = r"Y:\Host_NAIP_SDM\Env_Data\Worldclim"
    ghm_raster_fp = r"Y:\Host_NAIP_SDM\Env_Data\Global_Human_Modification\gHM_WGS84.tif"
    output_dir = r"Y:\Host_NAIP_SDM\Outputs_Host_NAIP_SDMs\tanoak_inat_image_climate_uniform_bioall_ghm_latlon_20ep_experiment_april2026\inference_tanoak_ca"

    # Output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load NAIP data from paths
    naip_files = set(os.listdir(naip_folder))

    # Environment variables used by the model
    temp_df = pd.read_csv(csv_path)
    env_vars = [col for col in temp_df.columns if col.startswith('wc2.1_30s') or col in ['ghm', 'lat_norm', 'lon_norm']]
    print("Length of env_vars:", len(env_vars))

    # Load model from checkpoint
    checkpoint_path = r"Y:\Host_NAIP_SDM\Outputs_Host_NAIP_SDMs\tanoak_inat_image_climate_uniform_bioall_ghm_latlon_20ep_experiment_april2026\checkpoints\checkpoint_epoch_19.tar"
    model, _ = load_model_from_checkpoint(checkpoint_path, env_vars, hidden_dim=256, dropout=0.25)

    model.eval()  # Keeps BatchNorm layers in eval mode
    enable_mc_dropout(model)  # Activates dropout for uncertainty estimation

    device = get_default_device() # Use GPU
    print(f"Loaded model type: {model.__class__.__name__}")
    print(f"Loaded model from {checkpoint_path} on device {device}")

    image_transform = transforms.Compose([transforms.ToTensor()])

    # Preload and normalize WorldClim and GHM rasters into memory
    wc_rasters, ghm_raster = preload_and_normalize_rasters(worldclim_folder, ghm_raster_fp, WORLDCLIM_STATS)
    print("Preloaded and normalized Worldclim and GHM rasters into memory.")

    # Get all NAIP .tif files
    naip_files = [os.path.join(naip_folder, f) for f in os.listdir(naip_folder) if f.endswith(".tif")]
    print(f"Found {len(naip_files)} NAIP tiles to process.")

    # Loop over each NAIP tile and run inference
    for tile_fp in tqdm(naip_files, desc="Processing Tiles"): # or use missing_tile_paths for only missing tiles if restart needed
        try:
            run_inference_on_tile(
                tile_fp, output_dir, model, device, image_transform,
                wc_rasters = wc_rasters,
                ghm_raster = ghm_raster,
                chip_size=256, stride=256
            )
        except Exception as e:
            print(f"[ERROR] Failed on {tile_fp}: {e}")


if __name__ == "__main__":
    main()


# EOF