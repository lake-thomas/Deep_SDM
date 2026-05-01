"""
PyTorch datasets for Host NAIP SDM models.

This dataset supports three predictor families produced by the dataset builder:

1. NAIP image chips
   - CSV column: chip_path
   - Expected tensor shape: 4 x 256 x 256
   - Values are scaled from 0-255 to 0-1 by default.

2. Topographic image chips
   - CSV column: topo_chip_path
   - Expected tensor shape: 4 x 64 x 64 by default
   - Band order produced by the dataset builder:
       band 1 = elevation
       band 2 = slope
       band 3 = northness
       band 4 = eastness
   - Values are assumed to already be normalized if topo_normalized == TRUE.
   - No per-chip elevation/slope fallback normalization is applied here because
     per-chip normalization removes meaningful absolute topographic gradients.

3. Tabular environmental covariates
   - WorldClim variables
   - GHM
   - lat_norm/lon_norm
   - optional topographic scalar summaries (mean, min, max, sd of elevation/slope/northness/eastness..)

Important leakage guardrails
----------------------------
The following columns are metadata or sampling-design variables and should not be
used as predictors:

    presence, split, sample_id, chip_path, topo_chip_path, filename, url,
    nearest_presence_km, background_sampling_rule, block_id, fold, cv_round,
    topo_valid_frac, topo_normalized

`nearest_presence_km` and `background_sampling_rule` are especially important to
exclude because they encode the pseudoabsence sampling design.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset


WC_VARS = [f"wc2.1_30s_bio_{i}" for i in range(1, 20)]

BASE_TABULAR_COLUMNS = WC_VARS + ["ghm", "lat_norm", "lon_norm"]

# Topographic scalar summaries are already normalized by the revised dataset builder when topo_normalized == TRUE.
TOPO_SCALAR_COLUMNS = [
    "elev_mean",
    "elev_sd",
    "elev_min",
    "elev_max",
    "slope_mean",
    "slope_sd",
    "slope_min",
    "slope_max",
    "northness_mean",
    "eastness_mean",
]

# QA/QC, metadata, target, and sampling-design columns that should never be
# auto-selected as model predictors.
NON_PREDICTOR_COLUMNS = {
    "sample_id",
    "chip_path",
    "topo_chip_path",
    "split",
    "presence",
    "lat",
    "lon",
    "source",
    "filename",
    "url",
    "cell_id",
    "nearest_presence_km",
    "background_sampling_rule",
    "block_x",
    "block_y",
    "block_id",
    "block_id_x",
    "block_id_y",
    "fold",
    "cv_round",
    "topo_valid_frac",
    "topo_normalized",
}

# Seven Core Model Ailases
MODEL_TYPE_ALIASES = {
    # NAIP only
    "image_only": "image_only",
    "naip_only": "image_only",
    "imagery_only": "image_only",

    # tabular only
    "tabular_only": "tabular_only",
    "env_only": "tabular_only",
    "climate_only": "tabular_only",

    # topo chip only
    "topo_only": "topo_only",
    "topography_only": "topo_only",

    # NAIP + tabular
    "image_tabular": "image_tabular",
    "image_climate": "image_tabular",
    "naip_climate": "image_tabular",
    "imagery_climate": "image_tabular",

    # topo chip + tabular
    "topo_tabular": "topo_tabular",
    "topo_climate": "topo_tabular",
    "topography_climate": "topo_tabular",

    # NAIP + topo chip
    "image_topo": "image_topo",
    "naip_topo": "image_topo",
    "imagery_topo": "image_topo",

    # NAIP + topo chip + tabular
    "image_topo_tabular": "image_topo_tabular",
    "image_tabular_topo": "image_topo_tabular",
    "image_climate_topo": "image_topo_tabular",
    "image_topo_climate": "image_topo_tabular",
    "naip_topo_climate": "image_topo_tabular",
    "naip_climate_topo": "image_topo_tabular",
    "full": "image_topo_tabular",
}

MODEL_INPUTS = {
    "image_only": {"image": True, "topo": False, "env": False},
    "tabular_only": {"image": False, "topo": False, "env": True},
    "topo_only": {"image": False, "topo": True, "env": False},
    "image_tabular": {"image": True, "topo": False, "env": True},
    "topo_tabular": {"image": False, "topo": True, "env": True},
    "image_topo": {"image": True, "topo": True, "env": False},
    "image_topo_tabular": {"image": True, "topo": True, "env": True},
}


def normalize_model_type(model_type: str) -> str:
    """Return the canonical model type key used internally."""
    key = str(model_type).strip().lower()
    if key not in MODEL_TYPE_ALIASES:
        valid = ", ".join(sorted(MODEL_TYPE_ALIASES))
        raise ValueError(f"Unknown model_type '{model_type}'. Valid aliases include: {valid}")
    return MODEL_TYPE_ALIASES[key]

def model_requires(model_type: str) -> dict[str, bool]:
    """Return booleans indicating whether a model uses image/topo/env inputs."""
    return MODEL_INPUTS[normalize_model_type(model_type)].copy()

def _as_bool(value) -> bool:
    """Robustly parse bool-like CSV values such as TRUE/FALSE/1/0."""
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}

def resolve_tabular_features(
    csv_path: str | os.PathLike,
    env_features: Optional[Iterable[str] | str] = None,
    include_topo_scalars: bool = True,
) -> list[str]:
    """
    Resolve tabular feature columns from a dataset CSV.

    Parameters
    ----------
    csv_path
        Path to the final dataset CSV produced by the dataset builder.
    env_features
        - None or "auto": use WorldClim + ghm + lat_norm + lon_norm and,
          if requested and available, topographic scalar summaries.
        - list[str]: use the provided feature list exactly, after validation.
    include_topo_scalars
        If True and env_features is None/"auto", include available topo scalar
        summaries in the tabular feature vector.

    Returns
    -------
    list[str]
        Ordered feature names.
    """
    df = pd.read_csv(csv_path, nrows=1)
    columns = set(df.columns)

    if env_features is not None and env_features != "auto":
        features = list(env_features)
    else:
        features = [c for c in BASE_TABULAR_COLUMNS if c in columns]
        if include_topo_scalars:
            features += [c for c in TOPO_SCALAR_COLUMNS if c in columns]

    missing = [c for c in features if c not in columns]
    if missing:
        raise ValueError(
            "The following requested tabular features are missing from the CSV: "
            f"{missing}"
        )

    forbidden = [c for c in features if c in NON_PREDICTOR_COLUMNS]
    if forbidden:
        raise ValueError(
            "The following requested tabular features are metadata, QA/QC, target, "
            f"or sampling-design columns and should not be predictors: {forbidden}"
        )

    return features


class HostNAIPDataset(Dataset):
    """
    Dataset for NAIP, topographic chip, and tabular Host SDM models.

    The dataset returns a dictionary containing only the inputs required by the
    selected model type, plus common metadata used by evaluation functions.

    Returned keys
    -------------
    Always:
        label, sample_id, lat, lon, path

    Conditionally:
        image      if the model uses NAIP imagery
        topo       if the model uses topographic chips
        env        if the model uses tabular predictors
        tabular    alias of env, for clearer downstream naming
    """

    def __init__(
        self,
        csv_path: str | os.PathLike,
        image_base_dir: str | os.PathLike,
        split: str = "train",
        environment_features: Optional[Iterable[str] | str] = None,
        transform=None,
        topo_transform=None,
        paired_transform=None,
        model_type: str = "image_tabular",
        input_mode: Optional[str] = None,
        include_topo_scalars: bool = True,
        naip_scale_255: bool = True,
        strict: bool = True,
    ):
        self.csv_path = Path(csv_path)
        self.image_base_dir = Path(image_base_dir)
        self.split = split
        self.transform = transform
        self.topo_transform = topo_transform
        self.paired_transform = paired_transform
        self.naip_scale_255 = naip_scale_255
        self.strict = strict

        # Backward compatibility with older input_mode values.
        # Prefer model_type in new configs.
        if input_mode is not None and model_type is None:
            model_type = input_mode

        self.model_type = normalize_model_type(model_type)
        req = model_requires(self.model_type)
        self.use_image = req["image"]
        self.use_topo = req["topo"]
        self.use_env = req["env"]

        self.df = pd.read_csv(self.csv_path)
        if "split" not in self.df.columns:
            raise ValueError(f"CSV is missing required 'split' column: {self.csv_path}")
        if "presence" not in self.df.columns:
            raise ValueError(f"CSV is missing required 'presence' column: {self.csv_path}")

        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"No rows found for split='{split}' in {self.csv_path}")

        if self.use_env:
            self.environment_features = resolve_tabular_features(
                self.csv_path,
                env_features=environment_features,
                include_topo_scalars=include_topo_scalars,
            )
        else:
            self.environment_features = []

        self._validate_required_columns()

    def _validate_required_columns(self) -> None:
        required = ["presence", "lat", "lon"]
        if self.use_image:
            required.append("chip_path")
        if self.use_topo:
            required.append("topo_chip_path")
        if self.use_env:
            required.extend(self.environment_features)

        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(
                f"Dataset for model_type='{self.model_type}' is missing columns: {missing}"
            )

        if self.use_topo and "topo_normalized" in self.df.columns:
            values = self.df["topo_normalized"].dropna().unique().tolist()
            if values and not all(_as_bool(v) for v in values):
                print(
                    "[WARNING] Some topo_normalized values are FALSE. The dataset "
                    "class will not apply per-chip topo normalization. Confirm that "
                    "this is intended."
                )

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, rel_or_abs_path) -> Path:
        """Resolve relative CSV paths under image_base_dir, handling Windows slashes."""
        path_str = str(rel_or_abs_path)
        path_str = path_str.replace("\\", os.sep).replace("/", os.sep)
        p = Path(path_str)
        if p.is_absolute():
            return p
        return self.image_base_dir / p

    def _load_raster(self, path: Path, scale255: bool = False) -> np.ndarray:
        """Load a raster as channels-first float32 and replace nodata with zero."""
        if not path.exists():
            msg = f"Raster path does not exist: {path}"
            if self.strict:
                raise FileNotFoundError(msg)
            print(f"[WARNING] {msg}")

        with rasterio.open(path) as src:
            arr = src.read().astype(np.float32)
            nodata = src.nodata

        if nodata is not None and np.isfinite(nodata):
            arr = np.where(arr == nodata, np.nan, arr)

        if scale255:
            arr = arr / 255.0
            arr = np.clip(arr, 0.0, 1.0)

        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr.astype(np.float32)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        label = torch.tensor(float(row["presence"]), dtype=torch.float32)
        sample_id = str(row.get("sample_id", idx))

        batch = {
            "label": label,
            "sample_id": sample_id,
            "lat": torch.tensor(float(row["lat"]), dtype=torch.float32),
            "lon": torch.tensor(float(row["lon"]), dtype=torch.float32),
            "path": str(row.get("chip_path", "")),
        }

        image_tensor = None
        topo_tensor = None

        if self.use_image:
            image_path = self._resolve_path(row["chip_path"])
            image_arr = self._load_raster(image_path, scale255=self.naip_scale_255)
            image_tensor = torch.from_numpy(image_arr)

            if image_tensor.ndim != 3:
                raise ValueError(f"Expected NAIP tensor with 3 dimensions C,H,W; got {image_tensor.shape}")
            if image_tensor.shape[0] != 4:
                raise ValueError(
                    f"Expected 4-band NAIP chip; got shape {tuple(image_tensor.shape)} "
                    f"for {image_path}"
                )

        if self.use_topo:
            topo_path_value = row.get("topo_chip_path", None)
            if pd.isna(topo_path_value) or topo_path_value is None or str(topo_path_value).strip() == "":
                raise ValueError(
                    f"Topographic chip requested for model_type='{self.model_type}', "
                    f"but topo_chip_path is missing for sample_id={sample_id}."
                )

            topo_path = self._resolve_path(topo_path_value)
            topo_arr = self._load_raster(topo_path, scale255=False)
            topo_tensor = torch.from_numpy(topo_arr.astype(np.float32))

            if topo_tensor.ndim != 3:
                raise ValueError(f"Expected topo tensor with 3 dimensions C,H,W; got {topo_tensor.shape}")
            if topo_tensor.shape[0] != 4:
                raise ValueError(
                    f"Expected 4-band topography chip; got shape {tuple(topo_tensor.shape)} "
                    f"for {topo_path}"
                )

        # Apply transforms. For paired image+topo models, paired_transform ensures
        # identical spatial operations when explicitly enabled. By default, main.py
        # disables topo spatial augmentation because northness/eastness encode
        # absolute direction.
        if image_tensor is not None and topo_tensor is not None and self.paired_transform is not None:
            image_tensor, topo_tensor = self.paired_transform(image_tensor, topo_tensor)
        else:
            if image_tensor is not None and self.transform is not None:
                image_tensor = self.transform(image_tensor)
            if topo_tensor is not None and self.topo_transform is not None:
                topo_tensor = self.topo_transform(topo_tensor)

        if image_tensor is not None:
            batch["image"] = image_tensor.float()
        if topo_tensor is not None:
            batch["topo"] = topo_tensor.float()

        if self.use_env:
            env_values = row[self.environment_features].to_numpy(dtype=np.float32)
            env_values = np.nan_to_num(env_values, nan=0.0, posinf=0.0, neginf=0.0)
            env_tensor = torch.tensor(env_values, dtype=torch.float32)
            batch["env"] = env_tensor
            batch["tabular"] = env_tensor  # clearer alias for future code

        return batch
