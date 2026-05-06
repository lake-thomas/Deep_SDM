#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mutable Deep_SDM autoresearch candidate file.

Agents MAY edit this file. Agents should NOT edit prepare.py, dataset CSVs,
normalization statistics, split assignments, or evaluation metrics.

Typical command from the Deep_SDM repository root:

    python sdm_autoresearch/train.py --config sdm_autoresearch/configs/test_species_mini_scout.json
"""
from __future__ import annotations

import os

conda_env = r"C:\Users\talake2\AppData\Local\anaconda3\envs\naip_ailanthus_env"
os.environ["GDAL_DATA"] = os.path.join(conda_env, "Library", "share", "gdal")
os.environ["PROJ_LIB"] = os.path.join(conda_env, "Library", "share", "proj")
os.environ["PATH"] += os.pathsep + os.path.join(conda_env, "Library", "bin")

import argparse
import sys
from pathlib import Path
from typing import Any

# Make imports robust when running this file as sdm_autoresearch/train.py.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR

import prepare
from model import build_model


# -----------------------------------------------------------------------------
# Mutable research surface
# -----------------------------------------------------------------------------
# Agents should make small, reviewable changes here first. The locked harness in
# prepare.py will train/evaluate this candidate on the fixed benchmark.
#
# A good workflow is to modify exactly one or two fields, run train.py, inspect
# results.tsv, and keep only changes that improve the configured primary metric.
SEARCH_SPACE: dict[str, Any] = {
    # Seven canonical options:
    # image_only, tabular_only, topo_only, image_tabular,
    # topo_tabular, image_topo, image_topo_tabular
    "model_type": "image_topo_tabular",

    # Autoresearch ratchet metric. prepare.py will use the best validation MCC
    # reached across epochs instead of only the final-epoch MCC.
    "primary_metric": "best_val_mcc",
    "epoch_selection_metric": "val_mcc",

    # Training hyperparameters.
    "optimizer": "adamw",
    "scheduler": "cosine",  # none, plateau, cosine, step
    "learning_rate": 1.5e-4,
    "weight_decay": 0,
    "batch_size": 16,
    "epochs": 6,
    "seed": 42,
    "use_fp16": True,

    # Model capacity / regularization.
    "dropout": 0.45,
    "hidden_dim": 192,
    "env_feature_dim": 128,
    "topo_feature_dim": 128,
    "pretrained_image": True,
    "naip_channels": 4,
    "topo_channels": 4,

    # Scheduler-specific knobs.
    "lr_patience": 1,
    "lr_factor": 0.5,
    "step_size": 5,
    "step_gamma": 0.5,

    # Augmentation knobs. prepare.py will ignore spatial augmentation for topo
    # models unless allow_topo_spatial_augmentation is explicitly true.
    "rotation_degrees": 0,
    "hflip_prob": 0.0,
    "vflip_prob": 0.0,
    "allow_topo_spatial_augmentation": False,
}


def resolve_search_space(variant: int | None = None) -> dict[str, Any]:
    """
    Return the base candidate or a deterministic indexed variant.

    Set AUTORESEARCH_VARIANT to an integer to run a compact hyperparameter
    sweep without changing the locked benchmark harness.
    """
    search_space = dict(SEARCH_SPACE)
    raw_variant = os.environ.get("AUTORESEARCH_VARIANT") if variant is None else str(variant)
    if raw_variant is None or raw_variant == "":
        return search_space

    variant_index = int(raw_variant)
    if variant_index >= 1000:
        focused_variants: list[dict[str, Any]] = [
            {"learning_rate": 1.35e-4, "dropout": 0.40},
            {"learning_rate": 1.35e-4, "dropout": 0.45},
            {"learning_rate": 1.35e-4, "dropout": 0.50},
            {"learning_rate": 1.50e-4, "dropout": 0.40},
            {"learning_rate": 1.50e-4, "dropout": 0.45, "hidden_dim": 160},
            {"learning_rate": 1.50e-4, "dropout": 0.45, "hidden_dim": 224},
            {"learning_rate": 1.50e-4, "dropout": 0.50},
            {"learning_rate": 1.65e-4, "dropout": 0.40},
            {"learning_rate": 1.65e-4, "dropout": 0.45},
            {"learning_rate": 1.65e-4, "dropout": 0.50},
            {"learning_rate": 1.80e-4, "dropout": 0.45},
            {"learning_rate": 1.50e-4, "dropout": 0.45, "weight_decay": 1e-5},
            {"learning_rate": 1.50e-4, "dropout": 0.45, "weight_decay": 5e-5},
            {"learning_rate": 1.50e-4, "dropout": 0.45, "eta_min": 1e-6},
            {"learning_rate": 1.50e-4, "dropout": 0.45, "eta_min": 5e-6},
            {"learning_rate": 1.50e-4, "dropout": 0.45, "env_feature_dim": 160},
            {"learning_rate": 1.50e-4, "dropout": 0.45, "topo_feature_dim": 160},
            {"learning_rate": 1.50e-4, "dropout": 0.45, "env_feature_dim": 160, "topo_feature_dim": 160},
            {"learning_rate": 1.50e-4, "dropout": 0.45, "batch_size": 24},
            {"learning_rate": 1.50e-4, "dropout": 0.45, "epochs": 8},
            {"learning_rate": 1.45e-4, "dropout": 0.43},
            {"learning_rate": 1.45e-4, "dropout": 0.45},
            {"learning_rate": 1.45e-4, "dropout": 0.47},
            {"learning_rate": 1.55e-4, "dropout": 0.43},
            {"learning_rate": 1.55e-4, "dropout": 0.45},
            {"learning_rate": 1.55e-4, "dropout": 0.47},
            {"learning_rate": 1.50e-4, "dropout": 0.43, "env_feature_dim": 160},
            {"learning_rate": 1.50e-4, "dropout": 0.47, "env_feature_dim": 160},
            {"learning_rate": 1.50e-4, "dropout": 0.43, "topo_feature_dim": 160},
            {"learning_rate": 1.50e-4, "dropout": 0.47, "topo_feature_dim": 160},
            {"learning_rate": 1.50e-4, "dropout": 0.43, "env_feature_dim": 160, "topo_feature_dim": 160},
            {"learning_rate": 1.50e-4, "dropout": 0.47, "env_feature_dim": 160, "topo_feature_dim": 160},
            {"learning_rate": 1.50e-4, "dropout": 0.45, "env_feature_dim": 144},
            {"learning_rate": 1.50e-4, "dropout": 0.45, "env_feature_dim": 176},
            {"learning_rate": 1.50e-4, "dropout": 0.45, "topo_feature_dim": 144},
            {"learning_rate": 1.50e-4, "dropout": 0.45, "topo_feature_dim": 176},
            {"learning_rate": 1.50e-4, "dropout": 0.45, "hidden_dim": 176},
            {"learning_rate": 1.50e-4, "dropout": 0.45, "hidden_dim": 208},
            {"learning_rate": 1.50e-4, "dropout": 0.45, "hidden_dim": 240},
            {"learning_rate": 1.50e-4, "dropout": 0.45, "epochs": 7},
            {"learning_rate": 1.45e-4, "dropout": 0.45, "env_feature_dim": 160, "epochs": 7},
            {"learning_rate": 1.55e-4, "dropout": 0.45, "env_feature_dim": 160, "epochs": 7},
            {"learning_rate": 1.50e-4, "dropout": 0.43, "env_feature_dim": 160, "epochs": 7},
            {"learning_rate": 1.50e-4, "dropout": 0.47, "env_feature_dim": 160, "epochs": 7},
            {"learning_rate": 1.50e-4, "dropout": 0.45, "seed": 7},
            {"learning_rate": 1.50e-4, "dropout": 0.45, "seed": 13},
            {"learning_rate": 1.50e-4, "dropout": 0.45, "seed": 21},
            {"learning_rate": 1.50e-4, "dropout": 0.45, "seed": 84},
            {"learning_rate": 1.50e-4, "dropout": 0.45, "env_feature_dim": 160, "seed": 7},
            {"learning_rate": 1.50e-4, "dropout": 0.45, "env_feature_dim": 160, "seed": 13},
            {"learning_rate": 1.50e-4, "dropout": 0.45, "env_feature_dim": 160, "seed": 21},
            {"learning_rate": 1.50e-4, "dropout": 0.45, "env_feature_dim": 160, "seed": 84},
            {"learning_rate": 1.50e-4, "dropout": 0.45, "topo_feature_dim": 160, "seed": 7},
            {"learning_rate": 1.50e-4, "dropout": 0.45, "topo_feature_dim": 160, "seed": 13},
            {"learning_rate": 1.50e-4, "dropout": 0.45, "topo_feature_dim": 160, "seed": 21},
            {"learning_rate": 1.50e-4, "dropout": 0.45, "topo_feature_dim": 160, "seed": 84},
            {"learning_rate": 1.50e-4, "dropout": 0.45, "env_feature_dim": 160, "topo_feature_dim": 160, "seed": 7},
            {"learning_rate": 1.50e-4, "dropout": 0.45, "env_feature_dim": 160, "topo_feature_dim": 160, "seed": 13},
            {"learning_rate": 1.50e-4, "dropout": 0.45, "env_feature_dim": 160, "topo_feature_dim": 160, "seed": 21},
            {"learning_rate": 1.50e-4, "dropout": 0.45, "env_feature_dim": 160, "topo_feature_dim": 160, "seed": 84},
        ]
        focused_index = variant_index - 1000
        if focused_index >= len(focused_variants):
            raise ValueError(
                f"Focused variant {variant_index} is out of range. "
                f"Use 1000 through {999 + len(focused_variants)}."
            )
        search_space.update(
            {
                "variant_index": variant_index,
                "focused_variant_index": focused_index,
                "scheduler": "cosine",
                "learning_rate": 1.5e-4,
                "dropout": 0.45,
                "batch_size": 16,
                "hidden_dim": 192,
                "epochs": 6,
                "weight_decay": 0,
                "eta_min": 0.0,
            }
        )
        search_space.update(focused_variants[focused_index])
        return search_space

    learning_rates = [5e-5, 7.5e-5, 1e-4, 1.25e-4, 1.5e-4]
    dropouts = [0.35, 0.45, 0.50, 0.60]
    schedulers = ["plateau", "none", "cosine"]
    batch_sizes = [16, 32, 64]
    hidden_dims = [192, 256, 320]
    lr_patiences = [1, 2]
    lr_factors = [0.5, 0.7]

    n = variant_index
    search_space.update(
        {
            "variant_index": variant_index,
            "learning_rate": learning_rates[n % len(learning_rates)],
            "dropout": dropouts[(n // len(learning_rates)) % len(dropouts)],
            "scheduler": schedulers[(n // (len(learning_rates) * len(dropouts))) % len(schedulers)],
            "batch_size": batch_sizes[
                (n // (len(learning_rates) * len(dropouts) * len(schedulers))) % len(batch_sizes)
            ],
            "hidden_dim": hidden_dims[
                (n // (len(learning_rates) * len(dropouts) * len(schedulers) * len(batch_sizes)))
                % len(hidden_dims)
            ],
            "lr_patience": lr_patiences[
                (
                    n
                    // (
                        len(learning_rates)
                        * len(dropouts)
                        * len(schedulers)
                        * len(batch_sizes)
                        * len(hidden_dims)
                    )
                )
                % len(lr_patiences)
            ],
            "lr_factor": lr_factors[
                (
                    n
                    // (
                        len(learning_rates)
                        * len(dropouts)
                        * len(schedulers)
                        * len(batch_sizes)
                        * len(hidden_dims)
                        * len(lr_patiences)
                    )
                )
                % len(lr_factors)
            ],
        }
    )
    return search_space


def build_research_model(
    model_type: str,
    num_env_features: int,
    search_space: dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    """
    Build one candidate Deep_SDM model.

    Agents may modify this function to test architecture ideas, but should keep
    the function signature unchanged so prepare.py can call it.
    """
    model = build_model(
        model_type=model_type,
        num_env_features=num_env_features,
        hidden_dim=int(search_space.get("hidden_dim", 256)),
        dropout=float(search_space.get("dropout", 0.25)),
        naip_channels=int(search_space.get("naip_channels", 4)),
        topo_channels=int(search_space.get("topo_channels", 4)),
        topo_feature_dim=int(search_space.get("topo_feature_dim", 128)),
        env_feature_dim=int(search_space.get("env_feature_dim", 128)),
        pretrained_image=bool(search_space.get("pretrained_image", True)),
    )
    return model.to(device)


def make_optimizer(model: torch.nn.Module, search_space: dict[str, Any]) -> torch.optim.Optimizer:
    """Build optimizer for one candidate model."""
    opt_name = str(search_space.get("optimizer", "adamw")).lower()
    lr = float(search_space.get("learning_rate", 3e-5))
    weight_decay = float(search_space.get("weight_decay", 0.0))

    if opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if opt_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(search_space.get("momentum", 0.9)),
            weight_decay=weight_decay,
            nesterov=bool(search_space.get("nesterov", True)),
        )
    raise ValueError(f"Unknown optimizer: {opt_name}")


def make_scheduler(optimizer: torch.optim.Optimizer, search_space: dict[str, Any]):
    """
    Build an optional learning-rate scheduler.

    Agents may tune or extend this function, but should not change how validation
    metrics are computed in prepare.py.
    """
    sched_name = str(search_space.get("scheduler", "none")).lower()

    if sched_name in {"none", "off", "false"}:
        return None

    if sched_name == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(search_space.get("lr_factor", 0.5)),
            patience=int(search_space.get("lr_patience", 3)),
        )

    if sched_name == "cosine":
        # T_max should usually match or exceed the requested epoch budget.
        return CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(search_space.get("epochs", 5))),
            eta_min=float(search_space.get("eta_min", 0.0)),
        )

    if sched_name == "step":
        return StepLR(
            optimizer,
            step_size=max(1, int(search_space.get("step_size", 5))),
            gamma=float(search_space.get("step_gamma", 0.5)),
        )

    raise ValueError(f"Unknown scheduler: {sched_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Deep_SDM autoresearch candidate trial")
    parser.add_argument("--config", type=str, required=True, help="Locked benchmark config JSON")
    parser.add_argument("--variant", type=int, default=None, help="Optional indexed search-space variant")
    args = parser.parse_args()

    config = prepare.load_config(args.config)
    search_space = resolve_search_space(args.variant)
    prepare.run_trial(
        config=config,
        search_space=search_space,
        build_model_fn=build_research_model,
        make_optimizer_fn=make_optimizer,
        make_scheduler_fn=make_scheduler,
    )


if __name__ == "__main__":
    main()
