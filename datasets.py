import os
import pandas as pd
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

TOPO_SCALAR_COLUMNS = ["elev_mean","elev_sd","elev_min","elev_max","slope_mean","slope_sd","slope_min","slope_max","northness_mean","eastness_mean","topo_valid_frac"]

class HostNAIPDataset(Dataset):
    def __init__(self, csv_path, image_base_dir, split='train', environment_features=None, transform=None, input_mode="baseline"):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        self.image_base_dir = image_base_dir
        self.transform = transform
        self.input_mode = input_mode
        base_env = [c for c in self.df.columns if c.startswith('wc2.1_30s') or c in ['ghm', 'lat_norm', 'lon_norm']]
        if input_mode in {"topo_scalar", "full"}:
            base_env += [c for c in TOPO_SCALAR_COLUMNS if c in self.df.columns]
        self.environment_features = environment_features or base_env

    def __len__(self):
        return len(self.df)

    def _load_raster(self, path, scale255=False):
        with rasterio.open(path) as src:
            arr = src.read().astype(np.float32)
        if scale255:
            arr = arr / 255.0
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self._load_raster(os.path.join(self.image_base_dir, row['chip_path']), scale255=True)
        img = torch.from_numpy(img)
        if self.transform:
            img = self.transform(img)

        env_features = torch.tensor(row[self.environment_features].values.astype(np.float32), dtype=torch.float32)
        env_features = torch.nan_to_num(env_features, nan=0.0, posinf=0.0, neginf=0.0)
        label = torch.tensor(row['presence'], dtype=torch.float32)

        batch = {"image": img, "env": env_features, "label": label,
                 "lat": torch.tensor(row['lat'], dtype=torch.float32),
                 "lon": torch.tensor(row['lon'], dtype=torch.float32),
                 "path": str(row['chip_path'])}

        if self.input_mode in {"topo_chip", "full"}:
            topo_path = row.get("topo_chip_path", None)
            if pd.isna(topo_path) or topo_path is None:
                raise ValueError("Topography chip requested but topo_chip_path missing")
            topo = self._load_raster(os.path.join(self.image_base_dir, topo_path), scale255=False)
            if topo.shape[0] != 4:
                raise ValueError(f"Expected 4-band topography chip, got shape {topo.shape} for {topo_path}")
            # elevation z-score fallback
            topo[0] = (topo[0] - float(np.nanmean(topo[0]))) / (float(np.nanstd(topo[0])) + 1e-6)
            # slope scaled to [0, 1]
            topo[1] = topo[1] / 90.0
            topo = np.nan_to_num(topo, nan=0.0, posinf=0.0, neginf=0.0)
            batch["topo"] = torch.from_numpy(topo.astype(np.float32))

        return batch
