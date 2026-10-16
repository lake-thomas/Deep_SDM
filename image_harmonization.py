"""Image harmonisation shared by training, same-ground pairs and inference (October 2026).

mode "chip_standardize": each band of a chip (already scaled to 0-1) is standardised to zero mean and
unit SD over that chip, then mapped to a fixed reference mean and SD per band (stored in the model
config, so inference applies exactly the same mapping). This removes chip-level brightness and contrast
differences between NAIP acquisitions while keeping spatial pattern and, approximately, band ratios.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

EPS = 1e-3


def validate(cfg: dict[str, Any] | None) -> dict[str, Any] | None:
    if not cfg:
        return None
    mode = cfg.get("mode")
    if mode in (None, "none"):
        return None
    if mode != "chip_standardize":
        raise ValueError(f"Unknown image_harmonization mode: {mode}")
    ref_mean, ref_std = cfg.get("ref_mean"), cfg.get("ref_std")
    if not (isinstance(ref_mean, list) and isinstance(ref_std, list) and len(ref_mean) == len(ref_std) and len(ref_mean) >= 1):
        raise ValueError("chip_standardize needs ref_mean and ref_std lists (one value per band)")
    return {"mode": mode, "ref_mean": [float(x) for x in ref_mean], "ref_std": [float(x) for x in ref_std]}


def harmonize_torch(x: torch.Tensor, cfg: dict[str, Any] | None) -> torch.Tensor:
    """x: C x H x W or N x C x H x W, scaled 0-1."""
    if not cfg:
        return x
    dims = (-2, -1)
    mean = x.mean(dim=dims, keepdim=True)
    std = x.std(dim=dims, keepdim=True)
    shape = (-1, 1, 1)
    rm = torch.tensor(cfg["ref_mean"], dtype=x.dtype, device=x.device).view(shape)
    rs = torch.tensor(cfg["ref_std"], dtype=x.dtype, device=x.device).view(shape)
    return (x - mean) / (std + EPS) * rs + rm


def harmonize_numpy(x: np.ndarray, cfg: dict[str, Any] | None) -> np.ndarray:
    """x: C x H x W, scaled 0-1."""
    if not cfg:
        return x
    mean = x.mean(axis=(1, 2), keepdims=True)
    std = x.std(axis=(1, 2), ddof=1, keepdims=True)
    rm = np.asarray(cfg["ref_mean"], dtype=x.dtype).reshape(-1, 1, 1)
    rs = np.asarray(cfg["ref_std"], dtype=x.dtype).reshape(-1, 1, 1)
    return ((x - mean) / (std + EPS) * rs + rm).astype(x.dtype)
