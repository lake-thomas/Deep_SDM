# Automated evaluation utilities for NAIP imagery and environmental variables model
# Thomas Lake, July 2025

import os
import torch
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import contextily as ctx # For basemaps in plotting errors
from model import HostImageryClimateModel, HostImageryOnlyModel, HostClimateOnlyModel
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import roc_curve, roc_auc_score, matthews_corrcoef


@torch.no_grad()
def test_model(model, test_loader, device, out_dir="model_results"):
    """
    Evaluate the model on the testing dataset and save results
    """
    model.to(device)
    model.eval()

    y_probs = []
    y_pred = []
    y_true = []

    for batch in test_loader:
        images = batch["image"].to(device)
        envs   = batch["env"].to(device)
        labels = batch["label"].to(device)

        # Dynamically handle model input
        if isinstance(model, HostImageryClimateModel):
            outputs = model(images, envs)
        elif isinstance(model, HostImageryOnlyModel):
            outputs = model(images)
        elif isinstance(model, HostClimateOnlyModel):
            outputs = model(envs)
        else:
            raise NotImplementedError("Unknown model type for test_model")

        logits = outputs
        probs = torch.sigmoid(logits)
        
        preds = (probs > 0.5).float() # Outputs are Sigmoid. Use 0.5 threshold for binary classification.
        
        y_probs.extend(probs.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

    disp = ConfusionMatrixDisplay(cm, display_labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format="d")
    os.makedirs(out_dir, exist_ok=True)
    plt.title("Binary Confusion Matrix")
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=300)
    plt.close()

    # Classification Report
    report = classification_report(y_true, y_pred, output_dict=True, target_names=["Negative", "Positive"])
    report_df = pd.DataFrame(report).transpose()
    print("Classification Report:\n", report_df)

    report_df.to_csv(os.path.join(out_dir, "classification_report.csv"))

    # ROC Curve (Binary Classification) and AUC
    auc_score = plot_roc_curve(y_true=y_true, y_probs=y_probs, out_dir=out_dir)
    print(f"Overall ROC AUC (all thresholds): {auc_score:.4f}")

    # AUC at threshold 0.5 for comparison (same as accuracy-based metrics)
    auc_at_05 = roc_auc_score(y_true, y_probs)
    print(f"AUC at threshold=0.5: {auc_at_05:.4f}")

    # Matthews Correlation Coefficient
    mcc = matthews_corrcoef(y_true, y_pred)
    print(f"Matthews Correlation Coefficient: {mcc:.4f}")

    # Save MCC to file alongside AUC results
    metrics_summary = {
        "roc_auc_all_thresholds": auc_score,
        "roc_auc_at_0.5": auc_at_05,
        "mcc_at_0.5": mcc
    }
    pd.DataFrame([metrics_summary]).to_csv(
        os.path.join(out_dir, "model_summary_metrics.csv"),
        index=False
    )


@torch.no_grad()
def map_model_errors(model, test_loader, device, out_dir="model_results"):
    """
    Map model errors on the test dataset
    Loads lat/lon from the dataset and plots error points by type
    """

    model.to(device)
    model.eval()

    y_pred = []
    y_true = []
    lat_list = []
    lon_list = []

    for batch in test_loader:
        images = batch["image"].to(device)
        envs   = batch["env"].to(device)
        labels = batch["label"].to(device)

        lats = batch["lat"].cpu().numpy()
        lons = batch["lon"].cpu().numpy()
        # paths = batch["path"]

        # Dynamically handle model input
        if isinstance(model, HostImageryClimateModel):
            outputs = model(images, envs)
        elif isinstance(model, HostImageryOnlyModel):
            outputs = model(images)
        elif isinstance(model, HostClimateOnlyModel):
            outputs = model(envs)
        else:
            raise NotImplementedError("Unknown model type")
        
        logits = outputs
        probs = torch.sigmoid(logits)
        
        preds = (probs > 0.5).float() # Outputs are Sigmoid. Use 0.5 threshold for binary classification.
        
        y_pred.extend(preds.cpu().numpy())
        y_true.extend(labels.cpu().numpy())
        
        # Collect lat/lon from the batch
        lat_list.extend(lats.tolist())
        lon_list.extend(lons.tolist())

    # Build Dataframe
    df = pd.DataFrame({
        "lat": lat_list,
        "lon": lon_list,
        "true_label": y_true,
        "predicted_label": y_pred})
    
    # Label Prediction Types by Error
    df['error_type'] = "UNDEF"
    df.loc[(df.true_label == 1) & (df.predicted_label == 1), "error_type"] = "TP"
    df.loc[(df.true_label == 0) & (df.predicted_label == 0), "error_type"] = "TN"
    df.loc[(df.true_label == 0) & (df.predicted_label == 1), "error_type"] = "FP"
    df.loc[(df.true_label == 1) & (df.predicted_label == 0), "error_type"] = "FN"

    # Save DataFrame to CSV
    df.to_csv(os.path.join(out_dir, "ailanthus_spatial_prediction_errors.csv"), index=False)

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.lon, df.lat),
        crs="EPSG:4326"  # WGS84
    )
    gdf = gdf.to_crs(epsg=3857)  # Web Mercator for basemaps

    # Color and marker
    color_map = {"TP": "green", "TN": "orange", "FP": "blue", "FN": "purple"}
    marker_map = {0: "o", 1: "^"}

    # Dynamically calculate bounds from the full dataset, with padding:
    buffer = 100000  # meters
    xmin, ymin, xmax, ymax = gdf.total_bounds
    extent = [xmin - buffer, ymin - buffer, xmax + buffer, ymax + buffer]

    # --- Plot 1: Presences only ---
    gdf_pres = gdf[gdf["true_label"] == 1]
    fig1, ax1 = plt.subplots(figsize=(10, 10))
    ax1.set_xlim(extent[0], extent[2])
    ax1.set_ylim(extent[1], extent[3])

    for err_type in ["TP", "FN"]:
        subset = gdf_pres[gdf_pres["error_type"] == err_type]
        if not subset.empty:
            subset.plot(
                ax=ax1,
                color=color_map[err_type],
                marker=marker_map[1],  # triangle
                markersize=20,
                label=err_type,
                alpha=0.7
            )

    ctx.add_basemap(ax1, crs=gdf_pres.crs)
    ax1.set_title("Ailanthus Prediction Errors: Presences", fontsize=14)
    ax1.axis("off")
    ax1.legend(loc="lower left", fontsize=9, title="Error Type")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ailanthus_spatial_errors_testing_presences.png"), dpi=300)
    plt.close()

    # --- Plot 2: Absences only ---
    gdf_abs = gdf[gdf["true_label"] == 0]
    fig2, ax2 = plt.subplots(figsize=(10, 10))
    ax2.set_xlim(extent[0], extent[2])
    ax2.set_ylim(extent[1], extent[3])

    for err_type in ["TN", "FP"]:
        subset = gdf_abs[gdf_abs["error_type"] == err_type]
        if not subset.empty:
            subset.plot(
                ax=ax2,
                color=color_map[err_type],
                marker=marker_map[0],  # circle
                markersize=20,
                label=err_type,
                alpha=0.7
            )

    ctx.add_basemap(ax2, crs=gdf_abs.crs)
    ax2.set_title("Ailanthus Prediction Errors: Pseudoabsences", fontsize=14)
    ax2.axis("off")
    ax2.legend(loc="lower left", fontsize=9, title="Error Type")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ailanthus_spatial_errors_testing_absences.png"), dpi=300)
    plt.close()


def plot_roc_curve(y_true, y_probs, out_dir="model_results"):
    """
    Plot ROC curve and save the figure
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    auc_score = roc_auc_score(y_true, y_probs)
    
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.4f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)

    os.makedirs(out_dir, exist_ok=True)
    roc_path = os.path.join(out_dir, "roc_curve.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()

    # --- Export data to CSV ---
    roc_df = pd.DataFrame({
        "threshold": thresholds,
        "fpr": fpr,
        "tpr": tpr
    })

    roc_df.to_csv(os.path.join(out_dir, "model_roc_data.csv"), index=False)

    return auc_score


def plot_accuracies(history, outpath):
    """
    Plot history of model accuracy
    """
    outpath = os.path.join(outpath, 'model_accuracy.png')
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('validation accuracy')
    plt.title('Accuracy vs. No. of epochs')
    plt.autoscale()
    plt.margins(0.2)
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_losses(history, outpath):
    """
    Plot history of model losses
    """
    outpath = os.path.join(outpath, 'model_losses.png')
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')
    plt.autoscale()
    plt.margins(0.2)
    plt.savefig(outpath, dpi=300)
    plt.close()