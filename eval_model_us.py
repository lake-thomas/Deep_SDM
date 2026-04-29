# Model Evaluation and Visualization for NAIP/ Ailanthus US Model
# January 2026

# Imports
import os

conda_env = r"C:\Users\talake2\AppData\Local\anaconda3\envs\naip_ailanthus_env"
os.environ["GDAL_DATA"] = os.path.join(conda_env, "Library", "share", "gdal")
os.environ["PROJ_LIB"] = os.path.join(conda_env, "Library", "share", "proj")
os.environ["PATH"] += os.pathsep + os.path.join(conda_env, "Library", "bin")

import torch # noqa: E402
from torch.utils.data import DataLoader # noqa: E402
import matplotlib.pyplot as plt # noqa: E402
import pandas as pd # noqa: E402
import numpy as np # noqa: E402 
import geopandas as gpd # noqa: E402
import contextily as ctx # noqa: E402
import rasterio as rio # noqa: E402
from model import HostImageryClimateModel, HostImageryOnlyModel, HostClimateOnlyModel # noqa: E402
from datasets import HostNAIPDataset # noqa: E402
from train_utils import get_default_device,  load_model_from_checkpoint # noqa: E402
from eval_utils import test_model, map_model_errors, plot_roc_curve, plot_accuracies, plot_losses # noqa: E402
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report # noqa: E402
from sklearn.metrics import roc_curve, roc_auc_score, matthews_corrcoef # noqa: E402



def inspect_predictions(model, loader, device, image_base_dir, out_dir, top_n=100):
    """Save chips for high-confidence absences, presences, and uncertain predictions"""
    model.eval()
    results = []

    print("Gathering model predictions for inspection...")
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            envs   = batch["env"].to(device)
            labels = batch["label"]
            lats   = batch["lat"]
            lons   = batch["lon"]
            paths  = batch["path"]

            # Handle model type dynamically
            if isinstance(model, HostImageryClimateModel):
                logits = model(images, envs)
            elif isinstance(model, HostImageryOnlyModel):
                logits = model(images)
            elif isinstance(model, HostClimateOnlyModel):
                logits = model(envs)
            else:
                raise NotImplementedError("Unknown model type in inspect_predictions")

            probs = torch.sigmoid(logits).cpu().numpy()

            for i in range(len(probs)):
                results.append({
                    "path": paths[i],
                    "prob": float(probs[i]),
                    "label": int(labels[i]),
                    "lat": float(lats[i]),
                    "lon": float(lons[i])
                })

    res_df = pd.DataFrame(results)

    # High-confidence absences
    top_absent = res_df.nsmallest(top_n, "prob")
    plot_chips(
        top_absent,
        image_base_dir,
        "High Confidence Absences (Prob ≈ 0)",
        os.path.join(out_dir, "inspection_high_confidence_absent.png")
    )

    # High-confidence presences
    top_present = res_df.nlargest(top_n, "prob")
    plot_chips(
        top_present,
        image_base_dir,
        "High Confidence Presences (Prob ≈ 1)",
        os.path.join(out_dir, "inspection_high_confidence_present.png")
    )

    # Most uncertain predictions
    uncertain = res_df.iloc[(res_df["prob"] - 0.5).abs().argsort()[:top_n]]
    plot_chips(
        uncertain,
        image_base_dir,
        "Most Uncertain Predictions (Prob ≈ 0.5)",
        os.path.join(out_dir, "inspection_uncertain_predictions.png")
    )

    res_df.to_csv(
        os.path.join(out_dir, "naip_ailanthus_test_chip_prediction_probabilities.csv"),
        index=False
    )



def plot_chips(df, image_base_dir, title, out_path, grid_size=(8, 8)):
    """
    Plots a grid of RGB chips from NAIP imagery
    """
    n_images = grid_size[0] * grid_size[1]
    df_subset = df.head(n_images)
    
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(16, 16))
    axes = axes.flatten()

    for i, (_, row) in enumerate(df_subset.iterrows()):
        img_path = os.path.join(image_base_dir, row['path'])
        with rio.open(img_path) as src:
            # Read RGB (Bands 1, 2, 3) and transpose to (H, W, C)
            img = src.read([1, 2, 3]).transpose(1, 2, 0)
            # Simple 2% - 98% clip for visualization
            img = np.clip(img / 255.0, 0, 1)

        axes[i].imshow(img)
        axes[i].set_title(f"Prob: {row['prob']:.3f}\nL: {int(row['label'])}", fontsize=4)
        axes[i].axis('off')

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_path, dpi=300)
    plt.close()



def run_evaluation():
    # Paths (Note: Use raw strings for Windows paths)
    checkpoint_p = r"Y:\Ailanthus_NAIP_SDM\Outputs_NAIP_Ailanthus_Model_US_2026\outputs\ailanthus_us_climate_only_uniform_20ep_jan2926\checkpoints\checkpoint_epoch_18.tar"
    csv_path = r"Y:\Ailanthus_NAIP_SDM\Datasets\Ailanthus_US_Uniform_PA_NAIP_256_jan28\Ailanthus_Pres_Bg_US_Uniform_Train_Val_Test_Dataset.csv"
    image_dir = r"Y:\Ailanthus_NAIP_SDM\Datasets\Ailanthus_US_Uniform_PA_NAIP_256_jan28"
    output_dir = r"Y:\Ailanthus_NAIP_SDM\Outputs_NAIP_Ailanthus_Model_US_2026\outputs\ailanthus_us_climate_only_uniform_20ep_jan2926"

    device = get_default_device()

    # Load DF to get env_vars
    temp_df = pd.read_csv(csv_path)
    env_vars = [col for col in temp_df.columns if col.startswith('wc2.1_30s') or col in ['ghm', 'lat_norm', 'lon_norm']]

    # Load model
    model, _ = load_model_from_checkpoint(checkpoint_path=checkpoint_p, env_vars=env_vars)
    model.to(device)

    # Dataset & Loader - ensure num_workers=0 if debugging on Windows to avoid pickling errors
    test_ds = HostNAIPDataset(csv_path, image_dir, split='test', environment_features=env_vars)
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    # 1. Standard Metrics
    # test_model(model, test_dl, device, output_dir)

    # 2. Spatial Error Mapping
    # map_model_errors(model, test_dl, device, output_dir)

    # 3. Plot images with high and low confidence sigmoid for inspection
    inspect_predictions(model, test_dl, device, image_base_dir=image_dir, out_dir=output_dir, top_n=100)



if __name__ == "__main__":
    print("Starting evaluation...")

    run_evaluation()

    print("Evaluation complete.")

# EOF