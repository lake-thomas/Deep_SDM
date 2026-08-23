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
import sys

# Set geospatial data paths only when the referenced installation is real.
# ``sys.prefix`` is authoritative when an environment's Python executable is
# invoked from a shell whose CONDA_PREFIX still points at the base environment.
for prefix in [sys.prefix, os.environ.get("CONDA_PREFIX")]:
    if not prefix:
        continue
    proj_dir = os.path.join(prefix, "share", "proj")
    gdal_dir = os.path.join(prefix, "share", "gdal")
    if os.path.isfile(os.path.join(proj_dir, "proj.db")):
        os.environ.setdefault("PROJ_LIB", proj_dir)
        os.environ.setdefault("PROJ_DATA", proj_dir)
    if os.path.isdir(gdal_dir):
        os.environ.setdefault("GDAL_DATA", gdal_dir)

legacy_windows_env = r"C:\Users\talake2\AppData\Local\anaconda3\envs\naip_ailanthus_env"
if os.name == "nt" and os.path.isdir(legacy_windows_env):
    os.environ.setdefault(
        "GDAL_DATA", os.path.join(legacy_windows_env, "Library", "share", "gdal")
    )
    os.environ.setdefault(
        "PROJ_LIB", os.path.join(legacy_windows_env, "Library", "share", "proj")
    )
    os.environ["PATH"] += os.pathsep + os.path.join(
        legacy_windows_env, "Library", "bin"
    )

import argparse
import datetime as dt
import hashlib
import json
import logging
import platform
import random
import socket
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from calibration_utils import (
    apply_platt_calibrator,
    derive_mcc_threshold,
    fit_platt_calibrator,
    write_calibration_package,
)
from datasets import (
    BalancedBinaryBatchSampler,
    HostNAIPDataset,
    model_requires,
    normalize_model_type,
    resolve_tabular_features,
)
from eval_utils import (
    collect_predictions,
    map_model_errors,
    plot_accuracies,
    plot_losses,
    summarize_binary_metrics,
    test_model,
)
from logging_utils import setup_logging
from model import build_model
from train_utils import fit
from transforms import (
    Compose4Band,
    PairedIndependentRadiometricViews4Band,
    RandomAugment4Band,
    RandomDihedral4Band,
    RandomPairedAugment4Band,
    RandomRadiometricAugment4Band,
)

try:
    import wandb
except Exception:  # pragma: no cover - wandb is optional
    wandb = None


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    with open(partial, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    os.replace(partial, path)


def _git_runtime_state() -> dict:
    root = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"git_commit": commit, "git_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}


def _runtime_provenance(config_path: Path, config_sha256: str) -> dict:
    cuda_device = None
    if torch.cuda.is_available():
        cuda_device = torch.cuda.get_device_name(torch.cuda.current_device())
    return {
        "started_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cuda_device": cuda_device,
        "config_path": str(config_path.resolve()),
        "config_sha256": config_sha256,
        **_git_runtime_state(),
    }

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
    paired_view_transform = None
    consistency_config = config.get("radiometric_consistency", False)
    consistency_requested = (
        bool(consistency_config.get("enabled", True))
        if isinstance(consistency_config, dict)
        else bool(consistency_config)
    )
    if consistency_requested and not (req["image"] and not req["topo"]):
        raise ValueError(
            "radiometric_consistency currently requires an image-only pathway "
            "without topographic image chips"
        )

    augmentation_regime = str(config.get("spatial_augmentation", "legacy_rotation")).lower()
    if augmentation_regime not in {"legacy_rotation", "dihedral", "none"}:
        raise ValueError(
            "spatial_augmentation must be legacy_rotation, dihedral, or none"
        )
    image_transforms = []
    if req["image"] and not req["topo"]:
        if augmentation_regime == "legacy_rotation":
            image_transforms.append(
                RandomAugment4Band(
                    rotation_degrees=rotation_degrees,
                    hflip_prob=hflip_prob,
                    vflip_prob=vflip_prob,
                )
            )
        elif augmentation_regime == "dihedral":
            image_transforms.append(RandomDihedral4Band())

        radiometric = config.get("radiometric_augmentation", False)
        if isinstance(radiometric, dict):
            radiometric_enabled = bool(radiometric.get("enabled", True))
            radiometric_kwargs = {
                key: value for key, value in radiometric.items() if key != "enabled"
            }
        else:
            radiometric_enabled = bool(radiometric)
            radiometric_kwargs = {}
        consistency_enabled = (
            bool(consistency_config.get("enabled", True))
            if isinstance(consistency_config, dict)
            else bool(consistency_config)
        )
        if consistency_enabled and not radiometric_enabled:
            raise ValueError(
                "radiometric_consistency requires radiometric_augmentation"
            )
        if consistency_enabled:
            paired_view_transform = PairedIndependentRadiometricViews4Band(
                RandomRadiometricAugment4Band(**radiometric_kwargs)
            )
        elif radiometric_enabled:
            image_transforms.append(RandomRadiometricAugment4Band(**radiometric_kwargs))
        image_transform = Compose4Band(image_transforms) if image_transforms else None
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
        paired_view_transform=paired_view_transform,
        **dataset_kwargs,
    )
    
    val_ds = HostNAIPDataset(split="val", **dataset_kwargs)
    test_ds = HostNAIPDataset(split="test", **dataset_kwargs)

    # For Windows, num_workers=0 can be easier for debugging. For Linux/HPC,
    # increase to 4-8 after the smoke test passes.
    num_workers = int(config.get("num_workers", 0))
    pin_memory = bool(config.get("pin_memory", torch.cuda.is_available()))

    seed = int(config.get("seed", 42))
    loss_strategy = str(config.get("loss_strategy", "ordinary_bce")).lower()
    if loss_strategy not in {"ordinary_bce", "pos_weight_bce", "balanced_batches"}:
        raise ValueError(
            "loss_strategy must be ordinary_bce, pos_weight_bce, or balanced_batches"
        )
    generator = torch.Generator()
    generator.manual_seed(seed)
    common_loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": bool(num_workers > 0 and config.get("persistent_workers", True)),
    }
    if loss_strategy == "balanced_batches":
        batch_sampler = BalancedBinaryBatchSampler(
            train_ds.df["presence"].to_numpy(),
            batch_size=int(config["batch_size"]),
            seed=seed,
        )
        train_dl = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            **common_loader_kwargs,
        )
    else:
        train_dl = DataLoader(
            train_ds,
            batch_size=config["batch_size"],
            shuffle=True,
            generator=generator,
            **common_loader_kwargs,
        )
    val_dl = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        **common_loader_kwargs,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        **common_loader_kwargs,
    )

    return train_ds, val_ds, test_ds, train_dl, val_dl, test_dl


def main():
    parser = argparse.ArgumentParser(description="Train Host NAIP SDM model")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration JSON file")
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="Explicitly allow writing into a nonempty experiment directory.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config_bytes = config_path.read_bytes()
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if bool(config.get("deterministic_training", True)):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model_type = normalize_model_type(config.get("model_type", "full"))
    req = model_requires(model_type) # Example: {"image": True, "topo": False, "env": False}

    experiment_name = config.get("experiment", f"host_sdm_{model_type}")
    experiment_dir = Path(config["output_dir"]) / experiment_name
    if experiment_dir.exists() and any(experiment_dir.iterdir()) and not args.allow_existing_output:
        raise FileExistsError(
            f"Refusing to reuse nonempty experiment directory: {experiment_dir}. "
            "Use a new versioned experiment name, or pass --allow-existing-output "
            "only after inspecting the existing run."
        )
    experiment_dir.mkdir(parents=True, exist_ok=True)

    _write_json_atomic(experiment_dir / "config.input.json", config)
    runtime_provenance = _runtime_provenance(config_path, config_sha256)
    _write_json_atomic(experiment_dir / "runtime_provenance.json", runtime_provenance)
    _write_json_atomic(
        experiment_dir / "run_status.json",
        {"status": "running", "started_at": runtime_provenance["started_at"]},
    )

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

    train_counts = train_ds.df["presence"].value_counts().reindex([0, 1], fill_value=0)
    if (train_counts == 0).any():
        raise RuntimeError(f"Training split lacks a class: {train_counts.to_dict()}")
    loss_strategy = str(config.get("loss_strategy", "ordinary_bce")).lower()
    label_smoothing = float(config.get("label_smoothing", 0.0))
    consistency = config.get("radiometric_consistency", False)
    if isinstance(consistency, dict):
        consistency_enabled = bool(consistency.get("enabled", True))
        consistency_weight = float(consistency.get("weight", 0.0))
    else:
        consistency_enabled = bool(consistency)
        consistency_weight = 1.0 if consistency_enabled else 0.0
    if consistency_weight < 0:
        raise ValueError("radiometric consistency weight must be non-negative")
    if consistency_enabled and consistency_weight == 0:
        raise ValueError("enabled radiometric consistency requires positive weight")
    pos_weight = (
        float(train_counts.loc[0] / train_counts.loc[1])
        if loss_strategy == "pos_weight_bce"
        else None
    )
    model.configure_training_loss(
        pos_weight=pos_weight,
        label_smoothing=label_smoothing,
        radiometric_consistency_weight=consistency_weight,
    )
    dataset_sha256 = hashlib.sha256(Path(config["csv_path"]).read_bytes()).hexdigest()
    training_design = {
        "loss_strategy": loss_strategy,
        "training_label_smoothing": label_smoothing,
        "training_pos_weight": pos_weight,
        "validation_loss": "ordinary_bce_unsmoothed_unweighted",
        "train_background_count": int(train_counts.loc[0]),
        "train_presence_count": int(train_counts.loc[1]),
        "train_sampled_background_fraction": (
            0.5
            if loss_strategy == "balanced_batches"
            else float(train_counts.loc[0] / train_counts.sum())
        ),
        "optimization_effective_positive_fraction": (
            0.5
            if loss_strategy in {"pos_weight_bce", "balanced_batches"}
            else float(train_counts.loc[1] / train_counts.sum())
        ),
        "score_interpretation": (
            "Conditional suitability score under the documented presence/background "
            "sampling and optimization design; not a literal occurrence probability."
        ),
        "dataset_csv_sha256": dataset_sha256,
        "spatial_augmentation": config.get("spatial_augmentation", "legacy_rotation"),
        "radiometric_augmentation": config.get("radiometric_augmentation", False),
        "radiometric_consistency": {
            "enabled": consistency_enabled,
            "weight": consistency_weight,
            "loss": "mean_squared_probability_difference",
            "paired_view_rule": (
                "shared exact spatial view with two independent bounded radiometric views"
            ),
        },
        "seed": seed,
    }
    _write_json_atomic(experiment_dir / "training_design.json", training_design)

    config_to_log = dict(config)
    config_to_log["canonical_model_type"] = model_type
    config_to_log["resolved_env_features"] = env_vars
    logging.info("Model arguments: %s", json.dumps(config_to_log, indent=4))

    start_time = time.time()

    output_csv = experiment_dir / "training_history.csv"

    def write_epoch_heartbeat(
        *, epoch, history, result, best_val_loss, early_stop_counter
    ):
        partial = output_csv.with_name(f"{output_csv.name}.partial")
        pd.DataFrame(history).to_csv(partial, index=False)
        os.replace(partial, output_csv)
        _write_json_atomic(
            experiment_dir / "training_heartbeat.json",
            {
                "status": "training",
                "updated_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
                "epoch_zero_based": int(epoch),
                "epoch_one_based": int(epoch) + 1,
                "epochs_requested": int(config["epochs"]),
                "best_val_loss": float(best_val_loss),
                "current_val_loss": float(result["val_loss"]),
                "current_train_loss": float(result["train_loss"]),
                "current_train_supervised_loss": (
                    float(result["train_supervised_loss"])
                    if "train_supervised_loss" in result else None
                ),
                "current_train_radiometric_consistency_loss": (
                    float(result["train_radiometric_consistency_loss"])
                    if "train_radiometric_consistency_loss" in result else None
                ),
                "early_stop_counter": int(early_stop_counter),
                "is_best": bool(result["is_best"]),
            },
        )

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
        restore_best_model=bool(config.get("restore_best_model", True)),
        save_epoch_checkpoints=bool(config.get("save_epoch_checkpoints", False)),
        checkpoint_metadata={
            "config": config,
            "resolved_env_features": env_vars,
            "training_design": training_design,
        },
        epoch_callback=write_epoch_heartbeat,
    )

    elapsed_time = time.time() - start_time
    logging.info("Training completed in %.2f seconds", elapsed_time)

    pd.DataFrame(history).to_csv(output_csv, index=False)
    logging.info("Training history saved to %s", output_csv)
    _write_json_atomic(
        experiment_dir / "training_heartbeat.json",
        {
            "status": "training_complete",
            "updated_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
            "epochs_completed": len(history),
            "epochs_requested": int(config["epochs"]),
            "elapsed_seconds": elapsed_time,
            "best_val_loss": min(float(item["val_loss"]) for item in history),
        },
    )

    plot_accuracies(history, str(experiment_dir))
    plot_losses(history, str(experiment_dir))

    validation_predictions = collect_predictions(model, val_dl, device, threshold=0.5)
    validation_predictions.to_csv(
        experiment_dir / "validation_prediction_results.csv", index=False
    )
    write_calibration_package(
        validation_predictions["true_label"],
        validation_predictions["probability"],
        experiment_dir,
        prefix="validation_raw",
    )

    calibration_method = str(config.get("postfit_calibration", "none")).lower()
    if calibration_method not in {"none", "platt"}:
        raise ValueError("postfit_calibration must be none or platt")
    if calibration_method == "platt":
        calibrator = fit_platt_calibrator(
            validation_predictions["true_label"],
            validation_predictions["probability"],
        )
        validation_score = apply_platt_calibrator(
            validation_predictions["probability"], calibrator
        )
        score_space = "platt_calibrated_validation_score"
        _write_json_atomic(experiment_dir / "platt_calibrator.json", calibrator)
    else:
        calibrator = None
        validation_score = validation_predictions["probability"].to_numpy()
        score_space = "raw_sigmoid_score"
    threshold_record = derive_mcc_threshold(
        validation_predictions["true_label"], validation_score
    )
    threshold_record["score_space"] = score_space
    _write_json_atomic(experiment_dir / "validation_threshold.json", threshold_record)

    if calibrator is None:
        test_predictions = test_model(
            model,
            test_dl,
            device,
            str(experiment_dir),
            threshold=threshold_record["threshold"],
        )
        map_model_errors(
            model,
            test_dl,
            device,
            str(experiment_dir),
            threshold=threshold_record["threshold"],
        )
        write_calibration_package(
            test_predictions["true_label"],
            test_predictions["probability"],
            experiment_dir,
            prefix="test_raw",
        )
    else:
        raw_test = collect_predictions(model, test_dl, device, threshold=0.5)
        raw_test.to_csv(experiment_dir / "prediction_results_raw.csv", index=False)
        write_calibration_package(
            raw_test["true_label"],
            raw_test["probability"],
            experiment_dir,
            prefix="test_raw",
        )
        calibrated = apply_platt_calibrator(raw_test["probability"], calibrator)
        raw_test["raw_probability"] = raw_test["probability"]
        raw_test["probability"] = calibrated
        raw_test["threshold"] = threshold_record["threshold"]
        raw_test["predicted_label"] = (
            raw_test["probability"] >= threshold_record["threshold"]
        ).astype(int)
        raw_test["error_type"] = np.select(
            [
                (raw_test["true_label"] == 1) & (raw_test["predicted_label"] == 1),
                (raw_test["true_label"] == 0) & (raw_test["predicted_label"] == 0),
                (raw_test["true_label"] == 0) & (raw_test["predicted_label"] == 1),
                (raw_test["true_label"] == 1) & (raw_test["predicted_label"] == 0),
            ],
            ["TP", "TN", "FP", "FN"],
            default="UNDEF",
        )
        raw_test.to_csv(experiment_dir / "prediction_results.csv", index=False)
        raw_test.to_csv(experiment_dir / "prediction_results_platt.csv", index=False)
        _write_json_atomic(
            experiment_dir / "test_metrics_platt.json",
            summarize_binary_metrics(raw_test),
        )
        write_calibration_package(
            raw_test["true_label"],
            raw_test["probability"],
            experiment_dir,
            prefix="test_platt",
        )

    if use_wandb and wandb is not None and wandb.run is not None:
        wandb.finish()

    _write_json_atomic(
        experiment_dir / "run_status.json",
        {
            "status": "completed",
            "started_at": runtime_provenance["started_at"],
            "completed_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
            "elapsed_training_seconds": elapsed_time,
            "best_checkpoint": str(checkpoint_dir / "best.pt"),
        },
    )


if __name__ == "__main__":
    main()
