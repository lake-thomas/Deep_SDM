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

    # Training hyperparameters.
    "optimizer": "adamw",
    "scheduler": "plateau",  # none, plateau, cosine, step
    "learning_rate": 1e-4,
    "weight_decay": 0,
    "batch_size": 16,
    "epochs": 20,
    "seed": 42,
    "use_fp16": True,

    # Model capacity / regularization.
    "dropout": 0.50,
    "hidden_dim": 256,
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
    args = parser.parse_args()

    config = prepare.load_config(args.config)
    prepare.run_trial(
        config=config,
        search_space=SEARCH_SPACE,
        build_model_fn=build_research_model,
        make_optimizer_fn=make_optimizer,
        make_scheduler_fn=make_scheduler,
    )


if __name__ == "__main__":
    main()
