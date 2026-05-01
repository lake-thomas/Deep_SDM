"""
Image transforms for Host NAIP SDM models.

The original RandomAugment4Band transform is retained for NAIP-only models.
A paired transform is provided for cases where the same spatial operation must
be applied to both NAIP and topographic chips.

Important topography note
-------------------------
Topographic chips include northness and eastness. These channels encode
absolute directional information. Random rotations/flips of topo chips are not
usually ecologically valid unless the northness/eastness vector components are
also transformed consistently. Therefore, main.py disables spatial augmentation
for any model that uses topographic chips by default.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


@dataclass
class _SpatialParams:
    angle: float
    hflip: bool
    vflip: bool


class RandomAugment4Band:
    """Randomly apply rotation, horizontal flip, and vertical flip to one 4-band tensor."""

    def __init__(
        self,
        rotation_degrees: float = 30,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        interpolation=InterpolationMode.BILINEAR,
        fill: float = 0.0,
    ):
        self.rotation_degrees = rotation_degrees
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.interpolation = interpolation
        self.fill = fill

    def _sample_params(self) -> _SpatialParams:
        return _SpatialParams(
            angle=random.uniform(-self.rotation_degrees, self.rotation_degrees),
            hflip=random.random() < self.hflip_prob,
            vflip=random.random() < self.vflip_prob,
        )

    def _apply(self, img, params: _SpatialParams):
        img = TF.rotate(
            img,
            params.angle,
            interpolation=self.interpolation,
            fill=self.fill,
        )
        if params.hflip:
            img = TF.hflip(img)
        if params.vflip:
            img = TF.vflip(img)
        return img

    def __call__(self, img):
        return self._apply(img, self._sample_params())


class RandomPairedAugment4Band(RandomAugment4Band):
    """
    Apply identical random spatial operations to two aligned 4-band tensors.

    Use this only when spatial augmentation is scientifically valid for both
    inputs. For topo chips containing northness/eastness, keep this disabled by
    default unless you also implement directional-vector correction.
    """

    def __call__(self, img, topo):
        params = self._sample_params()
        return self._apply(img, params), self._apply(topo, params)
