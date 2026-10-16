"""Same-ground NAIP chip pairs for the same-ground consistency loss.

A pairs directory holds images_a.npy and images_b.npy (N x 4 x H x W uint8, the same footprint read
from two different NAIP acquisitions), env.npy (N x F, the environmental vector shared by both chips),
keep.npy (N booleans) and meta.csv (a "split" column with "train" or "eval").
See docs/same_ground_consistency.md.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class SameGroundPairDataset(Dataset):
    """Returns sg_image_a, sg_image_b (float, scaled /255 like the NAIP training chips) and sg_env.

    With dihedral=True the same random one of the 8 dihedral transforms is applied to both chips.
    """

    def __init__(self, root: str | Path, split: str = "train", dihedral: bool = True, naip_scale_255: bool = True,
                 image_harmonization: dict | None = None):
        from image_harmonization import validate
        self.harmonization = validate(image_harmonization)
        root = Path(root)
        self.a = np.load(root / "images_a.npy", mmap_mode="r")
        self.b = np.load(root / "images_b.npy", mmap_mode="r")
        self.env = np.load(root / "env.npy")
        keep = np.load(root / "keep.npy")
        splits = [row["split"] for row in csv.DictReader(open(root / "meta.csv"))]
        self.index = np.array([i for i, s in enumerate(splits) if s == split and keep[i]], dtype=np.int64)
        self.dihedral = dihedral
        self.scale = 255.0 if naip_scale_255 else 1.0

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, k: int) -> dict:
        i = int(self.index[k])
        a = torch.from_numpy(np.asarray(self.a[i], dtype=np.float32) / self.scale)
        b = torch.from_numpy(np.asarray(self.b[i], dtype=np.float32) / self.scale)
        if self.dihedral:
            rot = int(torch.randint(0, 4, (1,)))
            flip = bool(torch.randint(0, 2, (1,)))
            a, b = torch.rot90(a, rot, dims=(1, 2)), torch.rot90(b, rot, dims=(1, 2))
            if flip:
                a, b = torch.flip(a, dims=(2,)), torch.flip(b, dims=(2,))
        if self.harmonization is not None:
            from image_harmonization import harmonize_torch
            a, b = harmonize_torch(a, self.harmonization), harmonize_torch(b, self.harmonization)
        return {"sg_image_a": a, "sg_image_b": b, "sg_env": torch.from_numpy(self.env[i].astype(np.float32))}


class PairAugmentedLoader:
    """Wraps a training DataLoader and adds one batch of same-ground pairs to every training batch."""

    def __init__(self, base_loader, pair_dataset: SameGroundPairDataset, batch_size: int, num_workers: int = 4, seed: int = 42):
        self.base = base_loader
        g = torch.Generator(); g.manual_seed(seed)
        self.pairs = DataLoader(pair_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                                num_workers=num_workers, generator=g, persistent_workers=num_workers > 0)
        self._it = None

    def __len__(self) -> int:
        return len(self.base)

    def _next_pairs(self) -> dict:
        if self._it is None:
            self._it = iter(self.pairs)
        try:
            return next(self._it)
        except StopIteration:
            self._it = iter(self.pairs)
            return next(self._it)

    def __iter__(self):
        for batch in self.base:
            merged = dict(batch)
            merged.update(self._next_pairs())
            yield merged
