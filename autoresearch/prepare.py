#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Locked Deep_SDM autoresearch benchmark harness.

This file is intentionally the fixed data/evaluation layer for an
``autoresearch``-style workflow. During autonomous experimentation, agents
should edit ``train.py`` only. They should not edit this file, dataset CSVs,
image chips, topographic chips, normalization statistics, split/fold
assignments, or evaluation metric definitions.

What this file does
-------------------
1. Loads a benchmark config.
2. Accepts either a compact autoresearch config with ``datasets`` or a regular
   Deep_SDM training config with top-level ``csv_path`` and ``image_dir``.
3. Resolves model inputs and tabular feature order using Deep_SDM utilities.
4. Builds train/validation dataloaders for one or more fixed dataset specs.
5. Trains one candidate model from ``train.py`` under a fixed epoch/time budget.
6. Evaluates fixed validation metrics.
7. Appends one row per candidate to ``results.tsv``.

Run from the root of the Deep_SDM repository, for example:

    python sdm_autoresearch/prepare.py --write-example-config sdm_autoresearch/configs/scout_template.json
    python sdm_autoresearch/train.py --config sdm_autoresearch/configs/test_species_mini_scout.json
"""
from __future__ import annotations

import os

conda_env = r"C:\Users\talake2\AppData\Local\anaconda3\envs\naip_ailanthus_env"
os.environ["GDAL_DATA"] = os.path.join(conda_env, "Library", "share", "gdal")
os.environ["PROJ_LIB"] = os.path.join(conda_env, "Library", "share", "proj")
os.environ["PATH"] += os.pathsep + os.path.join(conda_env, "Library", "bin")

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

# Make imports robust when this file is executed as sdm_autoresearch/prepare.py.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from datasets import HostNAIPDataset, model_requires, normalize_model_type, resolve_tabular_features
from transforms import RandomAugment4Band, RandomPairedAugment4Band

try:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Deep_SDM autoresearch requires scikit-learn for fixed evaluation metrics."
    ) from exc


# -----------------------------------------------------------------------------
# Locked benchmark constants
# -----------------------------------------------------------------------------

MODEL_TYPES = [
    "image_only",
    "tabular_only",
    "topo_only",
    "image_tabular",
    "topo_tabular",
    "image_topo",
    "image_topo_tabular",
]

# These columns should never be used as tabular predictors. This mirrors the
# leakage guardrails in datasets.py and keeps the benchmark resistant to
# accidental pseudoabsence-design leakage.
LEAKAGE_COLUMNS = {
    "sample_id",
    "presence",
    "split",
    "chip_path",
    "topo_chip_path",
    "filename",
    "url",
    "nearest_presence_km",
    "background_sampling_rule",
    "block_id",
    "block_id_x",
    "block_id_y",
    "block_x",
    "block_y",
    "fold",
    "cv_round",
    "topo_valid_frac",
    "topo_normalized",
}

EXAMPLE_CONFIG = {
    "benchmark_name": "test_species_mini_scout",
    "description": "Fast scout benchmark for Deep_SDM autoresearch. Edit paths before running.",
    "output_dir": "autoresearch_runs",
    "results_tsv": "results.tsv",
    "primary_metric": "mean_val_mcc",
    "maximize_primary_metric": True,
    "epochs": 5,
    "max_minutes_per_dataset": None,
    "seed": 42,
    "batch_size": 16,
    "num_workers": 0,
    "pin_memory": True,
    "use_fp16": True,
    "include_topo_scalars": True,
    "allow_topo_spatial_augmentation": False,
    "naip_scale_255": True,
    "strict_dataset": True,
    "env_features": "auto",
    "rotation_degrees": 0,
    "hflip_prob": 0.0,
    "vflip_prob": 0.0,
    "datasets": [
        {
            "name": "uniform_test_species_mini",
            "csv_path": "C:/path/to/Test_Species_Mini_Pres_Bg_US_Uniform_Train_Val_Test_Dataset.csv",
            "image_dir": "C:/path/to/Test_Species_Mini_US_Uniform_PA_NAIP_256_topo_norm_may2026",
        }
    ],
}


# -----------------------------------------------------------------------------
# Result containers
# -----------------------------------------------------------------------------

@dataclass
class FoldMetrics:
    dataset_name: str
    n_train: int
    n_val: int
    val_loss: float
    val_acc: float
    val_balanced_acc: float
    val_mcc: float
    val_auc: float
    val_avg_precision: float
    val_precision: float
    val_sensitivity: float
    val_specificity: float


@dataclass
class TrialResult:
    run_id: str
    benchmark_name: str
    model_type: str
    primary_metric: str
    primary_value: float
    mean_val_loss: float
    mean_val_acc: float
    mean_val_balanced_acc: float
    mean_val_mcc: float
    mean_val_auc: float
    mean_val_avg_precision: float
    mean_val_precision: float
    mean_val_sensitivity: float
    mean_val_specificity: float
    elapsed_seconds: float
    epochs_requested: int
    epochs_completed: int
    n_datasets: int
    optimizer: str
    scheduler: str
    learning_rate: float
    weight_decay: float
    batch_size: int
    seed: int
    search_space_json: str
    env_features_json: str
    run_dir: str


# -----------------------------------------------------------------------------
# CLI and config handling
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Locked Deep_SDM autoresearch harness")
    parser.add_argument("--config", type=str, default=None, help="Benchmark config JSON")
    parser.add_argument(
        "--write-example-config",
        type=str,
        default=None,
        help="Write an example benchmark config JSON and exit.",
    )
    parser.add_argument(
        "--coerce-training-config",
        type=str,
        default=None,
        help=(
            "Load a regular Deep_SDM training config with csv_path/image_dir and "
            "write an autoresearch-compatible config to --write-example-config."
        ),
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "deep_sdm_benchmark"


def write_json(path: str | os.PathLike[str], obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    print(f"Wrote: {path}")


def write_example_config(path: str | os.PathLike[str], source_config: Optional[dict[str, Any]] = None) -> None:
    if source_config is None:
        cfg = EXAMPLE_CONFIG
    else:
        cfg = coerce_config(source_config)
    write_json(path, cfg)


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        raw_config = json.load(f)
    config = coerce_config(raw_config)
    validate_config(config)
    return config


def coerce_config(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Convert either accepted config shape into the locked benchmark shape.

    Accepted shape A: autoresearch benchmark config with ``datasets``.
    Accepted shape B: regular Deep_SDM training config with top-level
    ``csv_path`` and ``image_dir``.
    """
    config = dict(raw)

    if "datasets" in config:
        config.setdefault("benchmark_name", slugify(config.get("experiment", "deep_sdm_benchmark")))
        config.setdefault("output_dir", "autoresearch_runs")
        config.setdefault("results_tsv", "results.tsv")
        config.setdefault("primary_metric", "mean_val_mcc")
        config.setdefault("maximize_primary_metric", True)
        config.setdefault("max_minutes_per_dataset", config.get("max_minutes", None))
        return config

    if "csv_path" in config and "image_dir" in config:
        experiment = config.get("experiment", "deep_sdm_autoresearch")
        # Keep regular model outputs separate from autoresearch outputs.
        base_output = Path(config.get("output_dir", "autoresearch_runs"))
        output_dir = str(base_output / "autoresearch_runs")
        coerced = dict(config)
        coerced["benchmark_name"] = slugify(experiment)
        coerced["output_dir"] = output_dir
        coerced.setdefault("results_tsv", "results.tsv")
        coerced.setdefault("primary_metric", "mean_val_mcc")
        coerced.setdefault("maximize_primary_metric", True)
        coerced.setdefault("max_minutes_per_dataset", coerced.get("max_minutes", None))
        coerced["datasets"] = [
            {
                "name": "uniform_or_single_dataset",
                "csv_path": config["csv_path"],
                "image_dir": config["image_dir"],
            }
        ]
        # Preserve the original top-level values for provenance, but the locked
        # harness reads from datasets[...].
        return coerced

    raise ValueError(
        "Config must either contain a 'datasets' list or top-level 'csv_path' and 'image_dir'."
    )


def validate_config(config: dict[str, Any]) -> None:
    required = ["benchmark_name", "output_dir", "datasets"]
    missing = [k for k in required if k not in config]
    if missing:
        raise ValueError(f"Benchmark config missing required keys: {missing}")

    if not isinstance(config["datasets"], list) or len(config["datasets"]) == 0:
        raise ValueError("Benchmark config must include at least one dataset spec.")

    for spec in config["datasets"]:
        for key in ["name", "csv_path", "image_dir"]:
            if key not in spec:
                raise ValueError(f"Dataset spec missing key '{key}': {spec}")
        if not Path(spec["csv_path"]).exists():
            raise FileNotFoundError(f"Dataset CSV not found: {spec['csv_path']}")
        if not Path(spec["image_dir"]).exists():
            raise FileNotFoundError(f"Dataset image_dir not found: {spec['image_dir']}")


# -----------------------------------------------------------------------------
# Reproducibility, feature resolution, and dataloaders
# -----------------------------------------------------------------------------

def get_param(config: dict[str, Any], search_space: dict[str, Any], key: str, default: Any = None) -> Any:
    """Return search-space override first, then config value, then default."""
    value = search_space.get(key, None)
    if value is not None:
        return value
    return config.get(key, default)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_hash(obj: Any, n: int = 10) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:n]


def make_run_id(config: dict[str, Any], search_space: dict[str, Any]) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{stable_hash({'config': config, 'search_space': search_space})}"


def resolve_env_vars(config: dict[str, Any], model_type: str) -> list[str]:
    req = model_requires(model_type)
    if not req["env"]:
        return []

    first_csv = config["datasets"][0]["csv_path"]
    env_vars = resolve_tabular_features(
        first_csv,
        env_features=config.get("env_features", "auto"),
        include_topo_scalars=bool(config.get("include_topo_scalars", True)),
    )
    leakage = [c for c in env_vars if c in LEAKAGE_COLUMNS]
    if leakage:
        raise ValueError(f"Resolved tabular features include leakage columns: {leakage}")
    return list(env_vars)


def make_dataloaders(
    dataset_spec: dict[str, Any],
    config: dict[str, Any],
    search_space: dict[str, Any],
    model_type: str,
    env_vars: list[str],
) -> tuple[DataLoader, DataLoader, int, int]:
    req = model_requires(model_type)

    image_transform = None
    paired_transform = None
    topo_transform = None

    rotation_degrees = float(get_param(config, search_space, "rotation_degrees", 0.0))
    hflip_prob = float(get_param(config, search_space, "hflip_prob", 0.0))
    vflip_prob = float(get_param(config, search_space, "vflip_prob", 0.0))
    allow_topo_aug = bool(get_param(config, search_space, "allow_topo_spatial_augmentation", False))

    # For models using topo chips, spatial augmentation is disabled by default
    # because northness/eastness encode absolute direction.
    if req["image"] and not req["topo"]:
        image_transform = RandomAugment4Band(
            rotation_degrees=rotation_degrees,
            hflip_prob=hflip_prob,
            vflip_prob=vflip_prob,
        )
    elif req["image"] and req["topo"] and allow_topo_aug:
        paired_transform = RandomPairedAugment4Band(
            rotation_degrees=rotation_degrees,
            hflip_prob=hflip_prob,
            vflip_prob=vflip_prob,
        )
        print(
            "[WARNING] Paired NAIP/topo augmentation is enabled. Confirm this "
            "is appropriate for northness/eastness channels."
        )

    common = dict(
        csv_path=dataset_spec["csv_path"],
        image_base_dir=dataset_spec["image_dir"],
        environment_features=env_vars if req["env"] else None,
        model_type=model_type,
        include_topo_scalars=bool(config.get("include_topo_scalars", True)),
        naip_scale_255=bool(get_param(config, search_space, "naip_scale_255", True)),
        strict=bool(get_param(config, search_space, "strict_dataset", True)),
    )

    train_ds = HostNAIPDataset(
        split="train",
        transform=image_transform,
        topo_transform=topo_transform,
        paired_transform=paired_transform,
        **common,
    )
    val_ds = HostNAIPDataset(split="val", **common)

    batch_size = int(get_param(config, search_space, "batch_size", 64))
    num_workers = int(get_param(config, search_space, "num_workers", 0))
    pin_memory = bool(get_param(config, search_space, "pin_memory", torch.cuda.is_available()))

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_dl, val_dl, len(train_ds), len(val_ds)


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


# -----------------------------------------------------------------------------
# Fixed validation metrics
# -----------------------------------------------------------------------------

def evaluate_fixed(model: torch.nn.Module, val_loader: DataLoader, device: torch.device) -> dict[str, float]:
    """Evaluate a model on the fixed validation split without changing threshold rules."""
    model.eval()
    total_loss = 0.0
    total_n = 0
    y_true: list[float] = []
    y_prob: list[float] = []

    with torch.inference_mode():
        for batch in val_loader:
            batch = move_batch_to_device(batch, device)
            labels = batch["label"].float()
            logits = model.forward_from_batch(batch)
            loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="sum")
            probs = torch.sigmoid(logits)
            total_loss += float(loss.item())
            total_n += int(labels.numel())
            y_true.extend(labels.detach().cpu().numpy().astype(float).tolist())
            y_prob.extend(probs.detach().cpu().numpy().astype(float).tolist())

    if total_n == 0:
        raise RuntimeError("Validation loader produced zero samples.")

    y_true_arr = np.asarray(y_true, dtype=np.float32)
    y_prob_arr = np.asarray(y_prob, dtype=np.float32)
    y_pred_arr = (y_prob_arr >= 0.5).astype(np.float32)

    positives = y_true_arr == 1
    negatives = y_true_arr == 0
    has_two_classes = len(np.unique(y_true_arr)) == 2

    specificity = float(np.mean(y_pred_arr[negatives] == 0)) if np.any(negatives) else math.nan
    sensitivity = float(np.mean(y_pred_arr[positives] == 1)) if np.any(positives) else math.nan

    auc = math.nan
    avg_precision = math.nan
    if has_two_classes:
        try:
            auc = float(roc_auc_score(y_true_arr, y_prob_arr))
        except ValueError:
            auc = math.nan
        try:
            avg_precision = float(average_precision_score(y_true_arr, y_prob_arr))
        except ValueError:
            avg_precision = math.nan

    return {
        "val_loss": total_loss / total_n,
        "val_acc": float(accuracy_score(y_true_arr, y_pred_arr)),
        "val_balanced_acc": float(balanced_accuracy_score(y_true_arr, y_pred_arr)) if has_two_classes else math.nan,
        "val_mcc": float(matthews_corrcoef(y_true_arr, y_pred_arr)) if has_two_classes else math.nan,
        "val_auc": auc,
        "val_avg_precision": avg_precision,
        "val_precision": float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
        "val_sensitivity": sensitivity,
        "val_specificity": specificity,
    }


def _scheduler_step(scheduler: Any, metrics: dict[str, float]) -> None:
    if scheduler is None:
        return
    if isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(metrics["val_loss"])
    else:
        scheduler.step()


def train_fixed_budget(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    epochs: int,
    max_seconds: float | None,
    use_fp16: bool,
) -> tuple[dict[str, float], int]:
    """Train using the fixed autoresearch budget and return final validation metrics."""
    scaler = torch.cuda.amp.GradScaler(enabled=bool(use_fp16 and device.type == "cuda"))
    start = time.time()
    last_metrics: dict[str, float] | None = None
    completed = 0

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=bool(use_fp16 and device.type == "cuda"),
            ):
                labels = batch["label"].float()
                logits = model.forward_from_batch(batch)
                loss = F.binary_cross_entropy_with_logits(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss: {loss.item()}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        completed = epoch + 1
        last_metrics = evaluate_fixed(model, val_loader, device)
        _scheduler_step(scheduler, last_metrics)
        lr = optimizer.param_groups[0].get("lr", math.nan)
        print(
            f"epoch={completed:03d} lr={lr:.8g} "
            f"val_loss={last_metrics['val_loss']:.5f} "
            f"val_mcc={last_metrics['val_mcc']:.4f} "
            f"val_auc={last_metrics['val_auc']:.4f}"
        )
        if max_seconds is not None and (time.time() - start) >= max_seconds:
            print(f"Reached fixed time budget after epoch {completed}.")
            break

    if last_metrics is None:
        last_metrics = evaluate_fixed(model, val_loader, device)
    return last_metrics, completed


# -----------------------------------------------------------------------------
# Results logging
# -----------------------------------------------------------------------------

def aggregate_fold_metrics(folds: list[FoldMetrics]) -> dict[str, float]:
    def nanmean(values: list[float]) -> float:
        arr = np.asarray(values, dtype=float)
        return float(np.nanmean(arr)) if np.isfinite(arr).any() else math.nan

    return {
        "mean_val_loss": nanmean([f.val_loss for f in folds]),
        "mean_val_acc": nanmean([f.val_acc for f in folds]),
        "mean_val_balanced_acc": nanmean([f.val_balanced_acc for f in folds]),
        "mean_val_mcc": nanmean([f.val_mcc for f in folds]),
        "mean_val_auc": nanmean([f.val_auc for f in folds]),
        "mean_val_avg_precision": nanmean([f.val_avg_precision for f in folds]),
        "mean_val_precision": nanmean([f.val_precision for f in folds]),
        "mean_val_sensitivity": nanmean([f.val_sensitivity for f in folds]),
        "mean_val_specificity": nanmean([f.val_specificity for f in folds]),
    }


def append_results_tsv(path: str | os.PathLike[str], result: TrialResult) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = asdict(result)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()), delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_trial_artifacts(
    run_dir: Path,
    config: dict[str, Any],
    search_space: dict[str, Any],
    folds: list[FoldMetrics],
    result: TrialResult,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "benchmark_config.json", config)
    write_json(run_dir / "search_space.json", search_space)
    pd.DataFrame([asdict(f) for f in folds]).to_csv(run_dir / "fold_metrics.csv", index=False)
    write_json(run_dir / "trial_result.json", asdict(result))


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except Exception:
        return default


def run_trial(
    config: dict[str, Any],
    search_space: dict[str, Any],
    build_model_fn: Callable[..., torch.nn.Module],
    make_optimizer_fn: Callable[[torch.nn.Module, dict[str, Any]], torch.optim.Optimizer],
    make_scheduler_fn: Optional[Callable[[torch.optim.Optimizer, dict[str, Any]], Any]] = None,
) -> TrialResult:
    """Run one fixed benchmark trial for the candidate specified in train.py."""
    seed = int(get_param(config, search_space, "seed", 42))
    set_global_seed(seed)

    model_type = normalize_model_type(get_param(config, search_space, "model_type", "image_topo_tabular"))
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unsupported model_type for autoresearch: {model_type}")

    env_vars = resolve_env_vars(config, model_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = int(get_param(config, search_space, "epochs", 3))
    max_minutes = get_param(config, search_space, "max_minutes_per_dataset", None)
    max_seconds = None if max_minutes is None else float(max_minutes) * 60.0
    use_fp16 = bool(get_param(config, search_space, "use_fp16", True))
    batch_size = int(get_param(config, search_space, "batch_size", 64))

    run_id = make_run_id(config, search_space)
    run_dir = Path(config["output_dir"]) / config["benchmark_name"] / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    fold_metrics: list[FoldMetrics] = []
    start = time.time()
    epochs_completed_values: list[int] = []

    for dataset_spec in config["datasets"]:
        print(f"\n=== Dataset: {dataset_spec['name']} | model_type={model_type} ===")
        train_loader, val_loader, n_train, n_val = make_dataloaders(
            dataset_spec=dataset_spec,
            config=config,
            search_space=search_space,
            model_type=model_type,
            env_vars=env_vars,
        )

        model = build_model_fn(
            model_type=model_type,
            num_env_features=len(env_vars),
            search_space=search_space,
            device=device,
        ).to(device)
        optimizer = make_optimizer_fn(model, search_space)
        scheduler = make_scheduler_fn(optimizer, search_space) if make_scheduler_fn is not None else None

        metrics, epochs_completed = train_fixed_budget(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epochs=epochs,
            max_seconds=max_seconds,
            use_fp16=use_fp16,
        )
        epochs_completed_values.append(epochs_completed)
        fold_metrics.append(
            FoldMetrics(
                dataset_name=dataset_spec["name"],
                n_train=n_train,
                n_val=n_val,
                val_loss=metrics["val_loss"],
                val_acc=metrics["val_acc"],
                val_balanced_acc=metrics["val_balanced_acc"],
                val_mcc=metrics["val_mcc"],
                val_auc=metrics["val_auc"],
                val_avg_precision=metrics["val_avg_precision"],
                val_precision=metrics["val_precision"],
                val_sensitivity=metrics["val_sensitivity"],
                val_specificity=metrics["val_specificity"],
            )
        )

    elapsed = time.time() - start
    agg = aggregate_fold_metrics(fold_metrics)
    primary_metric = str(config.get("primary_metric", "mean_val_mcc"))
    if primary_metric not in agg:
        raise ValueError(f"Unknown primary_metric '{primary_metric}'. Available: {sorted(agg)}")
    primary_value = float(agg[primary_metric])

    result = TrialResult(
        run_id=run_id,
        benchmark_name=config["benchmark_name"],
        model_type=model_type,
        primary_metric=primary_metric,
        primary_value=primary_value,
        elapsed_seconds=elapsed,
        epochs_requested=epochs,
        epochs_completed=int(min(epochs_completed_values) if epochs_completed_values else 0),
        n_datasets=len(config["datasets"]),
        optimizer=str(search_space.get("optimizer", "adamw")),
        scheduler=str(search_space.get("scheduler", "none")),
        learning_rate=safe_float(search_space.get("learning_rate", config.get("learning_rate", math.nan))),
        weight_decay=safe_float(search_space.get("weight_decay", config.get("weight_decay", 0.0))),
        batch_size=batch_size,
        seed=seed,
        search_space_json=json.dumps(search_space, sort_keys=True),
        env_features_json=json.dumps(env_vars),
        run_dir=str(run_dir),
        **agg,
    )

    save_trial_artifacts(run_dir, config, search_space, fold_metrics, result)
    results_tsv = Path(config["output_dir"]) / config.get("results_tsv", "results.tsv")
    append_results_tsv(results_tsv, result)

    print("\n=== Trial complete ===")
    print(json.dumps(asdict(result), indent=2))
    return result


if __name__ == "__main__":
    args = parse_args()
    if args.write_example_config:
        source = None
        if args.coerce_training_config:
            with open(args.coerce_training_config, "r", encoding="utf-8") as f:
                source = json.load(f)
        write_example_config(args.write_example_config, source_config=source)
    else:
        if args.config is None:
            raise SystemExit("Provide --config or --write-example-config")
        cfg = load_config(args.config)
        print(json.dumps(cfg, indent=2))
        print("Config validated. Run train.py --config <this_config> to launch a candidate trial.")
