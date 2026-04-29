# Python Driver Script for Launching Model Runs (CV Splits)
# Thomas Lake, January 2026

import subprocess
import json
import shutil

# --- Base config template ---
BASE_CONFIG = "model_config.json"   # your template JSON
TMP_CONFIG  = "tmp_config.json"     # temporary file we overwrite for each run

# Experiment details
experiment_name = [
    "ailanthus_us_image_climate_nolatlon_noghm_bio171215_blockcv_10ep_cv1_jan2926",
    "ailanthus_us_image_climate_nolatlon_noghm_bio171215_blockcv_10ep_cv2_jan2926",
    "ailanthus_us_image_climate_nolatlon_noghm_bio171215_blockcv_10ep_cv3_jan2926",
    "ailanthus_us_image_climate_nolatlon_noghm_bio171215_blockcv_10ep_cv4_jan2926",
    "ailanthus_us_image_climate_nolatlon_noghm_bio171215_blockcv_10ep_cv5_jan2926",
    "ailanthus_us_image_only_nolatlon_noghm_bio171215_blockcv_10ep_cv1_jan2926",
    "ailanthus_us_image_only_nolatlon_noghm_bio171215_blockcv_10ep_cv2_jan2926",
    "ailanthus_us_image_only_nolatlon_noghm_bio171215_blockcv_10ep_cv3_jan2926",
    "ailanthus_us_image_only_nolatlon_noghm_bio171215_blockcv_10ep_cv4_jan2926",
    "ailanthus_us_image_only_nolatlon_noghm_bio171215_blockcv_10ep_cv5_jan2926",
    "ailanthus_us_climate_only_nolatlon_noghm_bio171215_blockcv_10ep_cv1_jan2926",
    "ailanthus_us_climate_only_nolatlon_noghm_bio171215_blockcv_10ep_cv2_jan2926",
    "ailanthus_us_climate_only_nolatlon_noghm_bio171215_blockcv_10ep_cv3_jan2926",
    "ailanthus_us_climate_only_nolatlon_noghm_bio171215_blockcv_10ep_cv4_jan2926",
    "ailanthus_us_climate_only_nolatlon_noghm_bio171215_blockcv_10ep_cv5_jan2926",
]

model_type = [
    "image_climate", "image_climate", "image_climate", "image_climate", "image_climate", # image-climate model: CV 1-5
    "image_only", "image_only", "image_only", "image_only", "image_only", # image-only model: CV 1-5
    "climate_only", "climate_only", "climate_only", "climate_only", "climate_only", # climate-only model: CV 1-5
]

csv_paths = [
    f"C:/Users/talake2/Desktop/Datasets/Ailanthus_US_BlockCV_PA_NAIP_256_jan29/CV_{i}/Ailanthus_Train_Val_Test_US_BlockCV_{i}_Jan26.csv"
    for i in [1,2,3,4,5,1,2,3,4,5,1,2,3,4,5]
]

image_dirs = [
    f"C:/Users/talake2/Desktop/Datasets/Ailanthus_US_BlockCV_PA_NAIP_256_jan29/CV_{i}"
    for i in [1,2,3,4,5,1,2,3,4,5,1,2,3,4,5]
]

# --- Run loop ---
for i in range(len(experiment_name)):
    print(f"\n>>> Running experiment {i+1}/{len(experiment_name)}: {experiment_name[i]}\n")

    # Load base config
    with open(BASE_CONFIG, "r") as f:
        config = json.load(f)

    # Update values for this run
    config["experiment"] = experiment_name[i]
    config["csv_path"] = csv_paths[i]
    config["image_dir"] = image_dirs[i]
    config["model_type"] = model_type[i]

    # Write temporary config
    with open(TMP_CONFIG, "w") as f:
        json.dump(config, f, indent=4)

    # Call training script
    subprocess.run(["python", "main.py", "--config", TMP_CONFIG])

print("\n✅ All experiments completed.\n")
