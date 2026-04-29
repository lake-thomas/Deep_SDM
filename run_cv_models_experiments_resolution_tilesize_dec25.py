import os
import json
import subprocess
from pathlib import Path

# -----------------------------
# USER SETTINGS
# -----------------------------
BASE_DIR = Path(r"D:/Ailanthus_NAIP_Classification/Datasets")
BASE_CONFIG = r"D:\Ailanthus_NAIP_Classification\NAIP_Host_Model\model_config.json"
TMP_CONFIG  = r"D:\Ailanthus_NAIP_Classification\NAIP_Host_Modeltmp_config.json"

PIXEL_SIZES = ["1m", "2m", "5m"]
TILE_SIZES  = ["64", "128", "256", "512"]
FOLDS       = [1, 2, 3, 4, 5]

MODEL_TYPES = ["image_climate", "image_only", "climate_only"]

# -----------------------------
# Helper: Build dataset folder name
# -----------------------------
def dataset_folder(pixel, tile):
    return BASE_DIR / f"Ailanthus_CrossVal_BlockCV_PA_1300m_thin_naip_{pixel}_{tile}_Dec325"

# -----------------------------
# Loop over all combinations
# -----------------------------
experiments = []

for pixel in PIXEL_SIZES:
    for tile in TILE_SIZES:

        folder = dataset_folder(pixel, tile)
        if not folder.exists():
            print(f"⚠️ Missing dataset: {folder}")
            continue

        for fold in FOLDS:
            cv_dir = folder / f"CV_{fold}"

            csv_name = f"Ailanthus_Train_Val_Test_BlockCV_naip_{pixel}_{fold}_Dec325.csv"
            csv_path = cv_dir / csv_name
            # print(csv_path)

            if not csv_path.exists():
                print(f"⚠️ Missing CSV: {csv_path}")
                continue

            image_dir = cv_dir
            # print(image_dir)

            for model_type in MODEL_TYPES:
                exp_name = f"ailanthus_{model_type}_naip{pixel}_{tile}px_cv{fold}_Dec325"

                experiments.append({
                    "experiment": exp_name,
                    "csv_path": str(csv_path),
                    "image_dir": str(image_dir),
                    "model_type": model_type
                })

print(f"\n🔍 Total experiments prepared: {len(experiments)}\n")

# print(experiments[0])


# -----------------------------
# Run all experiments
# -----------------------------
for i, exp in enumerate(experiments, 1):

    print(f"\n>>> Running {i}/{len(experiments)}: {exp['experiment']}")

    # Load base config
    with open(BASE_CONFIG, "r") as f:
        config = json.load(f)

    # Update config fields
    config["experiment"] = exp["experiment"]
    config["csv_path"]   = exp["csv_path"]
    config["image_dir"]  = exp["image_dir"]
    config["model_type"] = exp["model_type"]

    # Write temporary config
    with open(TMP_CONFIG, "w") as f:
        json.dump(config, f, indent=4)

    # Launch main training script
    subprocess.run(["python", "D:\\Ailanthus_NAIP_Classification\\NAIP_Host_Model\\main.py", "--config", TMP_CONFIG])

print("\n✅ All experimental runs completed.\n")
