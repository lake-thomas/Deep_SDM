"""Training utilities for Host NAIP SDM models."""

from __future__ import annotations

import os
from typing import Optional

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import build_model, normalize_model_type

try:
    import wandb
except Exception:  # pragma: no cover - wandb is optional at runtime
    wandb = None


def get_default_device():
    """Return CUDA if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(model, epoch, optimizer, path="checkpoints"):
    """Save model and optimizer state."""
    os.makedirs(path, exist_ok=True)
    filename = f"checkpoint_epoch_{epoch}.tar"
    checkpoint_path = os.path.join(path, filename)

    torch.save(
        {
            "model_type": model.__class__.__name__,
            "model_type_key": getattr(model, "model_type_key", model.__class__.__name__),
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )
    print(f"Checkpoint saved to {checkpoint_path}")


def load_model_from_checkpoint(
    checkpoint_path: str,
    env_vars: Optional[list[str]] = None,
    hidden_dim: int = 256,
    dropout: float = 0.25,
    naip_channels: int = 4,
    topo_channels: int = 4,
    topo_feature_dim: int = 128,
    env_feature_dim: int = 128,
    pretrained_image: bool = True,
):
    """Load a model and optimizer state from a checkpoint file."""
    device = get_default_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_type = checkpoint.get("model_type_key", checkpoint.get("model_type", "image_tabular"))
    model_type = normalize_model_type(model_type)

    num_env_features = len(env_vars or [])
    model = build_model(
        model_type=model_type,
        num_env_features=num_env_features,
        hidden_dim=hidden_dim,
        dropout=dropout,
        naip_channels=naip_channels,
        topo_channels=topo_channels,
        topo_feature_dim=topo_feature_dim,
        env_feature_dim=env_feature_dim,
        pretrained_image=pretrained_image,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters())
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model.eval()
    return model, optimizer


@torch.no_grad()
def evaluate(model, val_loader):
    """Evaluate validation loss/accuracy over a dataloader."""
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def _autocast_context(device: torch.device, enabled: bool):
    """Return an autocast context compatible with CPU and CUDA."""
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled)
    return torch.autocast(device_type="cpu", enabled=False)


def fit(
    epochs,
    lr,
    model,
    train_loader,
    val_loader,
    optimizer,
    outpath,
    lr_patience=5,
    es_patience=10,
    use_fp16=True,
    log_wandb=True,
):
    """
    Train a model with ReduceLROnPlateau scheduling and early stopping.

    Mixed precision is enabled only when CUDA is available. This keeps the same
    code usable for quick CPU smoke tests on small mini datasets.
    """
    history = []
    best_val_loss = float("inf")
    early_stop_counter = 0

    device = next(model.parameters()).device
    fp16_enabled = bool(use_fp16 and device.type == "cuda")

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=lr_patience,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=fp16_enabled)

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)

            with _autocast_context(device, fp16_enabled):
                loss = model.training_step(batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.detach())

        result = evaluate(model, val_loader)
        result["train_loss"] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)

        if log_wandb and wandb is not None and wandb.run is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": result["train_loss"],
                    "val_loss": result["val_loss"],
                    "val_acc": result["val_acc"],
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

        scheduler.step(result["val_loss"])
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.8f}")

        if result["val_loss"] < best_val_loss:
            best_val_loss = result["val_loss"]
            early_stop_counter = 0
            save_checkpoint(model, epoch, optimizer, path=outpath)
        else:
            early_stop_counter += 1
            if early_stop_counter >= es_patience:
                print(f"Early stopping at epoch {epoch}.")
                save_checkpoint(model, epoch, optimizer, path=outpath)
                break

    return history
