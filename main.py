"""
Train Species Distribution Models using CNNs and MLPs with image, tabular, and topographic data.

Supports seven model_type values:
    image_only (naip)
    tabular_only
    topo_only (3dep)
    image_tabular
    topo_tabular
    image_topo
    image_topo_tabular (all)

"""
from __future__ import annotations

import os

conda_env = r"C:\Users\talake2\AppData\Local\anaconda3\envs\naip_ailanthus_env"
os.environ["GDAL_DATA"] = os.path.join(conda_env, "Library", "share", "gdal")
os.environ["PROJ_LIB"] = os.path.join(conda_env, "Library", "share", "proj")
os.environ["PATH"] += os.pathsep + os.path.join(conda_env, "Library", "bin")

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from datasets import HostNAIPDataset, model_requires, normalize_model_type, resolve_tabular_features
from eval_utils import map_model_errors, plot_accuracies, plot_losses, test_model
from logging_utils import setup_logging
from model import build_model
from train_utils import fit
from transforms import RandomAugment4Band, RandomPairedAugment4Band

try:
    import wandb
except Exception:  # pragma: no cover - wandb is optional
    wandb = None

def make_dataloaders(config: dict, model_type: str, env_vars: list[str]):
    """Create train/validation/test dataloaders for the selected model type."""
    req = model_requires(model_type)

    rotation_degrees = config.get("rotation_degrees", 0)
    hflip_prob = config.get("hflip_prob", 0.0)
    vflip_prob = config.get("vflip_prob", 0.0)

    # Spatial augmentation is safe for NAIP-only/image+tabular models. For any
    # model using topo chips, keep it off by default because northness/eastness
    # encode absolute direction. Enable only with explicit opt-in.
    allow_topo_spatial_aug = bool(config.get("allow_topo_spatial_augmentation", False))

    image_transform = None
    topo_transform = None
    paired_transform = None

    if req["image"] and not req["topo"]:
        image_transform = RandomAugment4Band(
            rotation_degrees=rotation_degrees,
            hflip_prob=hflip_prob,
            vflip_prob=vflip_prob,
        )
    elif req["image"] and req["topo"] and allow_topo_spatial_aug:
        paired_transform = RandomPairedAugment4Band(
            rotation_degrees=rotation_degrees,
            hflip_prob=hflip_prob,
            vflip_prob=vflip_prob,
        )
        print(
            "[WARNING] Paired NAIP/topo spatial augmentation is enabled. Confirm "
            "this is appropriate for northness/eastness channels."
        )
    elif req["topo"]:
        print(
            "Topographic chips are used, so spatial rotations/flips are disabled "
            "by default to preserve absolute northness/eastness orientation."
        )

    dataset_kwargs = dict(
        csv_path=config["csv_path"],
        image_base_dir=config["image_dir"],
        environment_features=env_vars if req["env"] else None,
        model_type=model_type,
        include_topo_scalars=bool(config.get("include_topo_scalars", True)),
        naip_scale_255=bool(config.get("naip_scale_255", True)),
        strict=bool(config.get("strict_dataset", True)),
    )

    train_ds = HostNAIPDataset(
        split="train",
        transform=image_transform,
        topo_transform=topo_transform,
        paired_transform=paired_transform,
        **dataset_kwargs,
    )
    
    val_ds = HostNAIPDataset(split="val", **dataset_kwargs)
    test_ds = HostNAIPDataset(split="test", **dataset_kwargs)

    # For Windows, num_workers=0 can be easier for debugging. For Linux/HPC,
    # increase to 4-8 after the smoke test passes.
    num_workers = int(config.get("num_workers", 0))
    pin_memory = bool(config.get("pin_memory", torch.cuda.is_available()))

    train_dl = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_ds, val_ds, test_ds, train_dl, val_dl, test_dl


def main():
    parser = argparse.ArgumentParser(description="Train Host NAIP SDM model")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration JSON file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    model_type = normalize_model_type(config.get("model_type", "full"))
    req = model_requires(model_type) # Example: {"image": True, "topo": False, "env": False}

    experiment_name = config.get("experiment", f"host_sdm_{model_type}")
    experiment_dir = Path(config["output_dir"]) / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = experiment_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_path = experiment_dir / f"{experiment_name}.log"
    setup_logging(str(log_path))

    use_wandb = bool(config.get("use_wandb", False))
    if use_wandb and wandb is not None:
        wandb.init(
            name=experiment_name,
            entity=config.get("wandb_entity", "talake2-ncsu"),
            project=config.get("wandb_project", "host_naip_sdm"),
            config=config,
        )

        # Override selected hyperparameters during W&B sweeps.
        if wandb.run:
            config["epochs"] = wandb.config.get("epochs", config["epochs"])
            config["batch_size"] = wandb.config.get("batch_size", config["batch_size"])
            config["learning_rate"] = wandb.config.get("learning_rate", config["learning_rate"])
            config["dropout"] = wandb.config.get("dropout", config.get("dropout", 0.25))
    elif use_wandb and wandb is None:
        print("[WARNING] use_wandb=True but wandb is not installed. Continuing without W&B.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Canonical model_type: {model_type}")
    print(f"Model inputs: {req}")

    if req["env"]:
        env_vars = resolve_tabular_features(
            config["csv_path"],
            env_features=config.get("env_features", "auto"),
            include_topo_scalars=bool(config.get("include_topo_scalars", True)),
        )
    else:
        env_vars = []

    print(f"Number of tabular features: {len(env_vars)}")
    if env_vars:
        print("Tabular features:", env_vars)

    train_ds, val_ds, test_ds, train_dl, val_dl, test_dl = make_dataloaders(
        config=config,
        model_type=model_type,
        env_vars=env_vars,
    )

    dropout = config.get("dropout", 0.25)
    hidden_dim = config.get("hidden_dim", 256)

    model = build_model(
        model_type=model_type,
        num_env_features=len(env_vars),
        hidden_dim=hidden_dim,
        dropout=dropout,
        naip_channels=config.get("naip_channels", 4),
        topo_channels=config.get("topo_channels", 4),
        topo_feature_dim=config.get("topo_feature_dim", 128),
        env_feature_dim=config.get("env_feature_dim", 128),
        pretrained_image=bool(config.get("pretrained_image", True)),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    config_to_log = dict(config)
    config_to_log["canonical_model_type"] = model_type
    config_to_log["resolved_env_features"] = env_vars
    logging.info("Model arguments: %s", json.dumps(config_to_log, indent=4))

    start_time = time.time()

    history = fit(
        epochs=config["epochs"],
        lr=config["learning_rate"],
        model=model,
        train_loader=train_dl,
        val_loader=val_dl,
        optimizer=optimizer,
        outpath=str(checkpoint_dir),
        lr_patience=config.get("lr_patience", 5),
        es_patience=config.get("es_patience", 10),
        use_fp16=bool(config.get("use_fp16", True)),
        log_wandb=use_wandb,
    )

    elapsed_time = time.time() - start_time
    logging.info("Training completed in %.2f seconds", elapsed_time)

    output_csv = experiment_dir / "training_history.csv"
    pd.DataFrame(history).to_csv(output_csv, index=False)
    logging.info("Training history saved to %s", output_csv)

    plot_accuracies(history, str(experiment_dir))
    plot_losses(history, str(experiment_dir))

    test_model(model, test_dl, device, str(experiment_dir))
    map_model_errors(model, test_dl, device, str(experiment_dir))

    if use_wandb and wandb is not None and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
