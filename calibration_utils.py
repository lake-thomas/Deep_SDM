"""Calibration and validation-threshold utilities."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, matthews_corrcoef


def _binary_arrays(labels, probabilities) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=int).reshape(-1)
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    if len(y) != len(p) or len(y) == 0:
        raise ValueError("labels and probabilities must be nonempty with equal length")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("labels must be binary")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must be finite and in [0, 1]")
    return y, p


def probability_logit(probabilities, epsilon: float = 1e-6) -> np.ndarray:
    p = np.asarray(probabilities, dtype=float)
    p = np.clip(p, epsilon, 1.0 - epsilon)
    return np.log(p / (1.0 - p))


def derive_mcc_threshold(labels, probabilities) -> dict:
    """Choose a probability threshold by maximum validation MCC in O(n log n)."""
    y, p = _binary_arrays(labels, probabilities)
    if len(np.unique(y)) < 2:
        raise ValueError("MCC threshold derivation requires both classes")
    order = np.argsort(-p, kind="mergesort")
    sorted_y = y[order]
    sorted_p = p[order]
    tp = np.cumsum(sorted_y == 1)
    fp = np.cumsum(sorted_y == 0)
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    fn = positives - tp
    tn = negatives - fp
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(
        tp * tn - fp * fn,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator > 0,
    )
    # Only the final occurrence of a tied probability represents predictions
    # made with probability >= that threshold.
    unique_boundary = np.r_[sorted_p[:-1] != sorted_p[1:], True]
    positions = np.flatnonzero(unique_boundary)
    best_value = float(np.max(mcc[positions]))
    best_positions = positions[np.isclose(mcc[positions], best_value, atol=1e-12)]
    thresholds = sorted_p[best_positions]
    best_position = best_positions[np.argmin(np.abs(thresholds - 0.5))]
    threshold = float(sorted_p[best_position])
    predictions = (p >= threshold).astype(int)
    return {
        "method": "maximum_validation_mcc",
        "threshold": threshold,
        "validation_mcc": float(matthews_corrcoef(y, predictions)),
        "validation_count": len(y),
        "validation_positive_count": positives,
        "validation_negative_count": negatives,
        "tie_break": "closest_to_0.5",
    }


def reliability_table(labels, probabilities, n_bins: int = 10) -> pd.DataFrame:
    y, p = _binary_arrays(labels, probabilities)
    if n_bins < 2:
        raise ValueError("n_bins must be at least two")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_index = np.minimum(np.searchsorted(edges, p, side="right") - 1, n_bins - 1)
    rows = []
    for index in range(n_bins):
        mask = bin_index == index
        rows.append(
            {
                "bin": index + 1,
                "lower": edges[index],
                "upper": edges[index + 1],
                "count": int(mask.sum()),
                "mean_probability": float(p[mask].mean()) if mask.any() else np.nan,
                "observed_fraction": float(y[mask].mean()) if mask.any() else np.nan,
                "absolute_gap": (
                    float(abs(p[mask].mean() - y[mask].mean())) if mask.any() else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def calibration_metrics(labels, probabilities, n_bins: int = 10) -> dict:
    y, p = _binary_arrays(labels, probabilities)
    table = reliability_table(y, p, n_bins=n_bins)
    nonempty = table[table["count"] > 0]
    ece = float(
        (nonempty["count"] / len(y) * nonempty["absolute_gap"]).sum()
    )
    metrics = {
        "n_samples": len(y),
        "positive_count": int((y == 1).sum()),
        "negative_count": int((y == 0).sum()),
        "brier_score": float(brier_score_loss(y, p)),
        "binary_log_loss": float(log_loss(y, p, labels=[0, 1])),
        "expected_calibration_error": ece,
        "calibration_intercept": np.nan,
        "calibration_slope": np.nan,
    }
    if len(np.unique(y)) == 2:
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
        model.fit(probability_logit(p).reshape(-1, 1), y)
        metrics["calibration_intercept"] = float(model.intercept_[0])
        metrics["calibration_slope"] = float(model.coef_[0, 0])
    return metrics


def fit_platt_calibrator(labels, probabilities) -> dict:
    y, p = _binary_arrays(labels, probabilities)
    if len(np.unique(y)) < 2:
        raise ValueError("Platt calibration requires both classes")
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    model.fit(probability_logit(p).reshape(-1, 1), y)
    return {
        "method": "platt_logistic_on_validation_logits",
        "intercept": float(model.intercept_[0]),
        "slope": float(model.coef_[0, 0]),
        "validation_count": len(y),
    }


def apply_platt_calibrator(probabilities, calibrator: dict) -> np.ndarray:
    logits = probability_logit(probabilities)
    calibrated_logits = float(calibrator["intercept"]) + float(calibrator["slope"]) * logits
    calibrated = np.empty_like(calibrated_logits, dtype=float)
    nonnegative = calibrated_logits >= 0
    calibrated[nonnegative] = 1.0 / (1.0 + np.exp(-calibrated_logits[nonnegative]))
    exp_logits = np.exp(calibrated_logits[~nonnegative])
    calibrated[~nonnegative] = exp_logits / (1.0 + exp_logits)
    return calibrated


def write_calibration_package(
    labels,
    probabilities,
    output_dir: str | Path,
    prefix: str,
    n_bins: int = 10,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics = calibration_metrics(labels, probabilities, n_bins=n_bins)
    table = reliability_table(labels, probabilities, n_bins=n_bins)
    table.to_csv(output / f"{prefix}_reliability.csv", index=False)
    with open(output / f"{prefix}_calibration_metrics.json", "w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)

    plotted = table[table["count"] > 0]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideal")
    ax.plot(
        plotted["mean_probability"],
        plotted["observed_fraction"],
        marker="o",
        label=prefix,
    )
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted score", ylabel="Observed fraction")
    ax.set_title("Reliability diagram")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / f"{prefix}_reliability.png", dpi=300)
    plt.close(fig)
    return metrics
