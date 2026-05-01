"""
Evaluation utilities for Host NAIP SDM models.

This module supports all seven multimodal model groups used in the Host NAIP
SDM workflow:

    image_only
    tabular_only
    topo_only
    image_tabular
    topo_tabular
    image_topo
    image_topo_tabular

The key design change is that evaluation no longer checks model classes with
``isinstance``. Instead, it calls ``model.forward_from_batch(batch)`` when that
method is available. The revised model classes implement this method, so the
same evaluation functions work regardless of whether a model uses NAIP chips,
topographic chips, tabular predictors, or any combination of those inputs.

Outputs written by ``test_model``
---------------------------------
    prediction_results.csv
        One row per test sample with label, probability, predicted class, lat,
        lon, sample_id, and optional source/path metadata.

    confusion_matrix.png
    classification_report.csv
    model_summary_metrics.csv
    model_roc_data.csv, roc_curve.png
        Only when both classes are present in y_true.

    model_pr_data.csv, precision_recall_curve.png
        Only when both classes are present in y_true.

Outputs written by ``map_model_errors``
---------------------------------------
    spatial_prediction_errors.csv
    spatial_errors_testing_presences.png
    spatial_errors_testing_background.png
    spatial_errors_testing_all.png


"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

try:
    import geopandas as gpd
except Exception:  # pragma: no cover - optional plotting dependency
    gpd = None

try:
    import contextily as ctx
except Exception:  # pragma: no cover - optional plotting dependency
    ctx = None


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------


def _ensure_out_dir(out_dir: str | os.PathLike) -> Path:
    """Create and return an output directory as a Path."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path


def _to_numpy_1d(x: Any) -> np.ndarray:
    """Convert a tensor/list/scalar to a flattened NumPy array."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x).reshape(-1)


def _batch_get_list(batch: dict, key: str, n: int, default: Any = "") -> list:
    """
    Extract metadata from a DataLoader batch as a Python list of length n.

    PyTorch's default collate function keeps strings as lists/tuples and stacks
    numeric values into tensors. This helper handles both cases.
    """
    if key not in batch:
        return [default] * n

    value = batch[key]
    if isinstance(value, torch.Tensor):
        arr = _to_numpy_1d(value)
        return arr.tolist()

    if isinstance(value, np.ndarray):
        arr = value.reshape(-1)
        return arr.tolist()

    if isinstance(value, (list, tuple)):
        vals = list(value)
        if len(vals) == n:
            return vals
        if len(vals) == 1:
            return vals * n
        return vals[:n] + [default] * max(0, n - len(vals))

    return [value] * n


def _forward_logits(model: torch.nn.Module, batch: dict, device: torch.device) -> torch.Tensor:
    """
    Return logits for a batch from either the revised or legacy model API.

    Preferred path:
        model.forward_from_batch(batch)

    Fallback path:
        route by class name and available batch keys. This keeps older
        checkpoints/scripts usable while the codebase transitions.
    """
    if hasattr(model, "forward_from_batch"):
        return model.forward_from_batch(batch)

    # Legacy fallback for older model.py versions.
    class_name = model.__class__.__name__

    image = batch.get("image")
    topo = batch.get("topo")
    env = batch.get("env", batch.get("tabular"))

    if image is not None:
        image = image.to(device)
    if topo is not None:
        topo = topo.to(device)
    if env is not None:
        env = env.to(device)

    if class_name in {"HostImageryOnlyModel"}:
        return model(image)
    if class_name in {"HostClimateOnlyModel", "HostTabularOnlyModel"}:
        return model(env)
    if class_name in {"HostTopoOnlyModel"}:
        return model(topo)
    if class_name in {"HostImageryClimateModel", "HostImageTabularModel"}:
        return model(image, env)
    if class_name in {"HostTopoClimateModel", "HostTopoTabularModel"}:
        return model(topo, env)
    if class_name in {"HostImageTopoModel", "HostImageryTopoModel"}:
        return model(image, topo)
    if class_name in {
        "HostImageClimateTopoModel",
        "HostImageryClimateTopoModel",
        "HostImageTopoClimateModel",
        "HostNAIPTopoClimateModel",
    }:
        return model(image, topo, env)

    raise NotImplementedError(
        f"Cannot evaluate model class '{class_name}'. Add forward_from_batch() "
        "to the model or extend _forward_logits()."
    )


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Run inference over a dataloader and return sample-level predictions.

    Parameters
    ----------
    model
        Trained PyTorch model. Revised models should implement
        ``forward_from_batch(batch)``.
    data_loader
        DataLoader returning dictionaries from HostNAIPDataset.
    device
        CPU or CUDA device.
    threshold
        Probability threshold used to convert probabilities to binary labels.

    Returns
    -------
    pandas.DataFrame
        Sample-level predictions and metadata.
    """
    model.to(device)
    model.eval()

    rows = []

    for batch in data_loader:
        labels = batch["label"].to(device).float()
        logits = _forward_logits(model, batch, device)
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()

        y_true = _to_numpy_1d(labels)
        y_prob = _to_numpy_1d(probs)
        y_pred = _to_numpy_1d(preds)
        n = len(y_true)

        sample_ids = _batch_get_list(batch, "sample_id", n, default="")
        lats = _batch_get_list(batch, "lat", n, default=np.nan)
        lons = _batch_get_list(batch, "lon", n, default=np.nan)
        paths = _batch_get_list(batch, "path", n, default="")
        topo_paths = _batch_get_list(batch, "topo_path", n, default="")
        sources = _batch_get_list(batch, "source", n, default="")

        for i in range(n):
            rows.append(
                {
                    "sample_id": sample_ids[i],
                    "lat": lats[i],
                    "lon": lons[i],
                    "path": paths[i],
                    "topo_path": topo_paths[i],
                    "source": sources[i],
                    "true_label": int(y_true[i]),
                    "probability": float(y_prob[i]),
                    "predicted_label": int(y_pred[i]),
                    "threshold": float(threshold),
                }
            )

    if not rows:
        raise RuntimeError("No predictions were collected. Check that the dataloader is not empty.")

    df = pd.DataFrame(rows)
    df["error_type"] = classify_error_types(df["true_label"], df["predicted_label"])
    return df


def classify_error_types(y_true: Iterable, y_pred: Iterable) -> list[str]:
    """Return TP/TN/FP/FN labels for binary predictions."""
    true = np.asarray(list(y_true)).astype(int)
    pred = np.asarray(list(y_pred)).astype(int)

    out = np.full(true.shape, "UNDEF", dtype=object)
    out[(true == 1) & (pred == 1)] = "TP"
    out[(true == 0) & (pred == 0)] = "TN"
    out[(true == 0) & (pred == 1)] = "FP"
    out[(true == 1) & (pred == 0)] = "FN"
    return out.tolist()


def _safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Return ROC AUC, or NaN if only one class is present."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def _safe_average_precision(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Return average precision, or NaN if only one class is present."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_prob))


def _safe_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return specificity/TNR for binary predictions."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, _, _ = cm.ravel()
    denom = tn + fp
    return float(tn / denom) if denom > 0 else float("nan")


def summarize_binary_metrics(pred_df: pd.DataFrame) -> dict:
    """Compute thresholded and threshold-independent binary metrics."""
    y_true = pred_df["true_label"].astype(int).to_numpy()
    y_pred = pred_df["predicted_label"].astype(int).to_numpy()
    y_prob = pred_df["probability"].astype(float).to_numpy()

    metrics = {
        "n_samples": int(len(y_true)),
        "n_positive": int(np.sum(y_true == 1)),
        "n_negative": int(np.sum(y_true == 0)),
        "threshold": float(pred_df["threshold"].iloc[0]) if len(pred_df) else 0.5,
        "accuracy_at_threshold": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy_at_threshold": float(balanced_accuracy_score(y_true, y_pred))
        if len(np.unique(y_true)) > 1
        else float("nan"),
        "precision_at_threshold": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_sensitivity_at_threshold": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity_at_threshold": _safe_specificity(y_true, y_pred),
        "f1_at_threshold": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc_at_threshold": float(matthews_corrcoef(y_true, y_pred))
        if len(np.unique(y_true)) > 1
        else float("nan"),
        "roc_auc_all_thresholds": _safe_roc_auc(y_true, y_prob),
        "average_precision_all_thresholds": _safe_average_precision(y_true, y_prob),
        "brier_score": float(brier_score_loss(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "mean_probability": float(np.mean(y_prob)),
        "mean_probability_positive": float(np.mean(y_prob[y_true == 1])) if np.any(y_true == 1) else float("nan"),
        "mean_probability_negative": float(np.mean(y_prob[y_true == 0])) if np.any(y_true == 0) else float("nan"),
    }

    return metrics


# -----------------------------------------------------------------------------
# Main evaluation functions
# -----------------------------------------------------------------------------


@torch.no_grad()
def test_model(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
    out_dir: str | os.PathLike = "model_results",
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Evaluate a trained model on the testing dataset and save results.

    This function is model-type agnostic as long as the model implements
    ``forward_from_batch(batch)`` or matches one of the legacy class names in
    ``_forward_logits``.

    Returns
    -------
    pandas.DataFrame
        Sample-level prediction results. Returning this DataFrame makes it easy
        to inspect predictions interactively in notebooks.
    """
    out_path = _ensure_out_dir(out_dir)

    pred_df = collect_predictions(model, test_loader, device, threshold=threshold)
    pred_csv = out_path / "prediction_results.csv"
    pred_df.to_csv(pred_csv, index=False)

    y_true = pred_df["true_label"].astype(int).to_numpy()
    y_pred = pred_df["predicted_label"].astype(int).to_numpy()
    y_prob = pred_df["probability"].astype(float).to_numpy()

    # Confusion matrix with fixed binary labels so outputs stay comparable even
    # for tiny test sets that may contain only one class.
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print("Confusion Matrix:\n", cm)

    disp = ConfusionMatrixDisplay(cm, display_labels=["Background", "Presence"])
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format="d", colorbar=False)
    ax.set_title("Binary Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_path / "confusion_matrix.png", dpi=300)
    plt.close(fig)

    # Classification report.
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Background", "Presence"],
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()
    print("Classification Report:\n", report_df)
    report_df.to_csv(out_path / "classification_report.csv")

    # Threshold-independent curves.
    auc_score = plot_roc_curve(y_true=y_true, y_probs=y_prob, out_dir=out_path)
    ap_score = plot_precision_recall_curve(y_true=y_true, y_probs=y_prob, out_dir=out_path)

    metrics_summary = summarize_binary_metrics(pred_df)
    metrics_summary["roc_auc_all_thresholds"] = auc_score
    metrics_summary["average_precision_all_thresholds"] = ap_score

    metrics_df = pd.DataFrame([metrics_summary])
    metrics_df.to_csv(out_path / "model_summary_metrics.csv", index=False)

    print("Summary metrics:\n", metrics_df.transpose())
    return pred_df


@torch.no_grad()
def map_model_errors(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
    out_dir: str | os.PathLike = "model_results",
    threshold: float = 0.5,
    add_basemap: bool = True,
    species_label: str | None = None,
) -> pd.DataFrame:
    """
    Map model errors on the test dataset.

    The function saves a generic prediction-error CSV and three maps:
      1. test presences only: TP/FN
      2. test background/pseudoabsence only: TN/FP
      3. all test samples: TP/TN/FP/FN

    If geopandas is unavailable, the function still writes the CSV and skips
    map plotting. If contextily is unavailable or basemap fetching fails, plots
    are saved without a basemap.
    """
    out_path = _ensure_out_dir(out_dir)
    pred_df = collect_predictions(model, test_loader, device, threshold=threshold)

    csv_path = out_path / "spatial_prediction_errors.csv"
    pred_df.to_csv(csv_path, index=False)

    if gpd is None:
        print("[WARNING] geopandas is not available. Saved CSV but skipped error maps.")
        return pred_df

    if "lat" not in pred_df.columns or "lon" not in pred_df.columns:
        print("[WARNING] lat/lon are not available. Saved CSV but skipped error maps.")
        return pred_df

    if pred_df[["lat", "lon"]].isna().any(axis=None):
        print("[WARNING] Some lat/lon values are missing. Dropping missing coordinates for maps.")
        pred_df = pred_df.dropna(subset=["lat", "lon"]).copy()

    if pred_df.empty:
        print("[WARNING] No mapped rows remain after dropping missing coordinates.")
        return pred_df

    gdf = gpd.GeoDataFrame(
        pred_df.copy(),
        geometry=gpd.points_from_xy(pred_df["lon"], pred_df["lat"]),
        crs="EPSG:4326",
    ).to_crs(epsg=3857)

    title_prefix = species_label or "Host SDM"

    _plot_error_map(
        gdf[gdf["true_label"] == 1],
        out_path / "spatial_errors_testing_presences.png",
        title=f"{title_prefix}: Test Presences",
        error_order=["TP", "FN"],
        add_basemap=add_basemap,
    )

    _plot_error_map(
        gdf[gdf["true_label"] == 0],
        out_path / "spatial_errors_testing_background.png",
        title=f"{title_prefix}: Test Background / Pseudoabsences",
        error_order=["TN", "FP"],
        add_basemap=add_basemap,
    )

    _plot_error_map(
        gdf,
        out_path / "spatial_errors_testing_all.png",
        title=f"{title_prefix}: All Test Predictions",
        error_order=["TP", "TN", "FP", "FN"],
        add_basemap=add_basemap,
    )

    return pred_df


# -----------------------------------------------------------------------------
# Plotting utilities
# -----------------------------------------------------------------------------


def plot_roc_curve(y_true, y_probs, out_dir: str | os.PathLike = "model_results") -> float:
    """
    Plot ROC curve and save curve data.

    Returns NaN when the true labels contain only one class.
    """
    out_path = _ensure_out_dir(out_dir)
    y_true = np.asarray(y_true).astype(int)
    y_probs = np.asarray(y_probs).astype(float)

    if len(np.unique(y_true)) < 2:
        print("[WARNING] ROC AUC is undefined because y_true contains only one class.")
        pd.DataFrame(columns=["threshold", "fpr", "tpr"]).to_csv(
            out_path / "model_roc_data.csv",
            index=False,
        )
        return float("nan")

    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    auc_score = float(roc_auc_score(y_true, y_probs))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_path / "roc_curve.png", dpi=300)
    plt.close(fig)

    roc_df = pd.DataFrame({"threshold": thresholds, "fpr": fpr, "tpr": tpr})
    roc_df.to_csv(out_path / "model_roc_data.csv", index=False)

    return auc_score


def plot_precision_recall_curve(
    y_true,
    y_probs,
    out_dir: str | os.PathLike = "model_results",
) -> float:
    """
    Plot precision-recall curve and save curve data.

    Returns NaN when the true labels contain only one class.
    """
    out_path = _ensure_out_dir(out_dir)
    y_true = np.asarray(y_true).astype(int)
    y_probs = np.asarray(y_probs).astype(float)

    if len(np.unique(y_true)) < 2:
        print("[WARNING] Average precision is undefined because y_true contains only one class.")
        pd.DataFrame(columns=["threshold", "precision", "recall"]).to_csv(
            out_path / "model_pr_data.csv",
            index=False,
        )
        return float("nan")

    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    ap_score = float(average_precision_score(y_true, y_probs))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(recall, precision, label=f"AP = {ap_score:.4f}")
    ax.set_xlabel("Recall / Sensitivity")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_path / "precision_recall_curve.png", dpi=300)
    plt.close(fig)

    # precision and recall have one more element than thresholds.
    pr_df = pd.DataFrame(
        {
            "threshold": np.r_[thresholds, np.nan],
            "precision": precision,
            "recall": recall,
        }
    )
    pr_df.to_csv(out_path / "model_pr_data.csv", index=False)

    return ap_score


def _plot_error_map(
    gdf,
    out_fp: Path,
    title: str,
    error_order: list[str],
    add_basemap: bool = True,
) -> None:
    """Plot a GeoDataFrame of TP/TN/FP/FN points."""
    if gdf.empty:
        print(f"[WARNING] No rows available for map: {out_fp.name}")
        return

    color_map = {
        "TP": "green",
        "TN": "orange",
        "FP": "blue",
        "FN": "purple",
        "UNDEF": "gray",
    }
    marker_map = {
        "TP": "^",
        "FN": "^",
        "TN": "o",
        "FP": "o",
        "UNDEF": "x",
    }

    xmin, ymin, xmax, ymax = gdf.total_bounds
    width = xmax - xmin
    height = ymax - ymin
    pad = max(width, height, 1000) * 0.05
    extent = [xmin - pad, xmax + pad, ymin - pad, ymax + pad]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    for err_type in error_order:
        subset = gdf[gdf["error_type"] == err_type]
        if subset.empty:
            continue
        subset.plot(
            ax=ax,
            color=color_map.get(err_type, "gray"),
            marker=marker_map.get(err_type, "o"),
            markersize=28,
            label=err_type,
            alpha=0.75,
        )

    if add_basemap and ctx is not None:
        try:
            ctx.add_basemap(ax, crs=gdf.crs)
        except Exception as exc:  # pragma: no cover - depends on internet/tile provider
            print(f"[WARNING] Could not add basemap to {out_fp.name}: {exc}")
    elif add_basemap and ctx is None:
        print("[WARNING] contextily is not available. Saving map without basemap.")

    ax.set_title(title, fontsize=14)
    ax.axis("off")
    ax.legend(loc="lower left", fontsize=9, title="Prediction Type")
    fig.tight_layout()
    fig.savefig(out_fp, dpi=300)
    plt.close(fig)


def plot_accuracies(history, outpath: str | os.PathLike) -> None:
    """Plot validation accuracy across epochs."""
    out_path = _ensure_out_dir(outpath)
    if not history:
        print("[WARNING] Empty training history. Skipping accuracy plot.")
        return

    accuracies = [x.get("val_acc", np.nan) for x in history]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(range(1, len(accuracies) + 1), accuracies, marker="x")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Accuracy vs. Epoch")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_path / "model_accuracy.png", dpi=300)
    plt.close(fig)


def plot_losses(history, outpath: str | os.PathLike) -> None:
    """Plot training and validation losses across epochs."""
    out_path = _ensure_out_dir(outpath)
    if not history:
        print("[WARNING] Empty training history. Skipping loss plot.")
        return

    train_losses = [x.get("train_loss", np.nan) for x in history]
    val_losses = [x.get("val_loss", np.nan) for x in history]
    epochs = range(1, len(history) + 1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, train_losses, marker="x", label="Training")
    ax.plot(epochs, val_losses, marker="x", label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss vs. Epoch")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_path / "model_losses.png", dpi=300)
    plt.close(fig)


# EOF
