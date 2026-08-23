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

import torch
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


class Compose4Band:
    """Compose tensor-native transforms without converting four-band NAIP to PIL."""

    def __init__(self, transforms):
        self.transforms = [transform for transform in transforms if transform is not None]

    def __call__(self, image):
        for transform in self.transforms:
            image = transform(image)
        return image


class RandomDihedral4Band:
    """Apply one of the eight exact square symmetries without interpolation."""

    def __call__(self, image):
        if image.ndim != 3:
            raise ValueError(f"Expected C,H,W tensor, received {tuple(image.shape)}")
        transform_index = random.randrange(8)
        result = torch.rot90(image, k=transform_index % 4, dims=(-2, -1))
        if transform_index >= 4:
            result = torch.flip(result, dims=(-1,))
        return result


class RandomRadiometricAugment4Band:
    """Apply mild correlated four-band radiometric perturbations.

    Scene-level gain, gamma, and contrast dominate. Small band-specific gain
    and offset terms prevent the augmentation from being perfectly gray-axis
    correlated, while an NIR/red NDVI guard rejects implausibly large changes.
    Input is expected to be scaled to [0, 1].
    """

    def __init__(
        self,
        probability: float = 0.8,
        scene_gain: float = 0.08,
        gamma: float = 0.08,
        contrast: float = 0.08,
        band_gain: float = 0.02,
        band_offset: float = 0.01,
        max_mean_ndvi_shift: float = 0.06,
        attempts: int = 3,
    ):
        ranges = {
            "probability": probability,
            "scene_gain": scene_gain,
            "gamma": gamma,
            "contrast": contrast,
            "band_gain": band_gain,
            "band_offset": band_offset,
            "max_mean_ndvi_shift": max_mean_ndvi_shift,
        }
        if not 0 <= probability <= 1:
            raise ValueError("probability must be in [0, 1]")
        if any(value < 0 for name, value in ranges.items() if name != "probability"):
            raise ValueError("radiometric augmentation ranges must be non-negative")
        if attempts <= 0:
            raise ValueError("attempts must be positive")
        self.probability = probability
        self.scene_gain = scene_gain
        self.gamma = gamma
        self.contrast = contrast
        self.band_gain = band_gain
        self.band_offset = band_offset
        self.max_mean_ndvi_shift = max_mean_ndvi_shift
        self.attempts = attempts

    @staticmethod
    def _mean_ndvi(image: torch.Tensor) -> torch.Tensor:
        red = image[0]
        nir = image[3]
        denominator = nir + red
        valid = denominator.abs() > 1e-6
        if not torch.any(valid):
            return torch.tensor(0.0, dtype=image.dtype, device=image.device)
        ndvi = (nir[valid] - red[valid]) / denominator[valid]
        return ndvi.clamp(-1.0, 1.0).mean()

    @staticmethod
    def _uniform(low: float, high: float) -> float:
        return random.uniform(low, high)

    def _candidate(self, image: torch.Tensor) -> torch.Tensor:
        scene_gain = self._uniform(1.0 - self.scene_gain, 1.0 + self.scene_gain)
        gamma = self._uniform(1.0 - self.gamma, 1.0 + self.gamma)
        contrast = self._uniform(1.0 - self.contrast, 1.0 + self.contrast)
        bands = image.shape[0]
        band_gain = torch.tensor(
            [self._uniform(1.0 - self.band_gain, 1.0 + self.band_gain) for _ in range(bands)],
            dtype=image.dtype,
            device=image.device,
        ).view(-1, 1, 1)
        band_offset = torch.tensor(
            [self._uniform(-self.band_offset, self.band_offset) for _ in range(bands)],
            dtype=image.dtype,
            device=image.device,
        ).view(-1, 1, 1)

        candidate = image.clamp(0.0, 1.0).pow(gamma) * scene_gain
        scene_mean = candidate.mean()
        candidate = (candidate - scene_mean) * contrast + scene_mean
        candidate = candidate * band_gain + band_offset
        return candidate.clamp(0.0, 1.0)

    def __call__(self, image):
        if image.ndim != 3 or image.shape[0] != 4:
            raise ValueError(f"Expected four-band C,H,W tensor, received {tuple(image.shape)}")
        if random.random() >= self.probability:
            return image
        original_ndvi = self._mean_ndvi(image)
        for _ in range(self.attempts):
            candidate = self._candidate(image)
            if not torch.isfinite(candidate).all():
                continue
            ndvi_shift = torch.abs(self._mean_ndvi(candidate) - original_ndvi)
            if float(ndvi_shift) <= self.max_mean_ndvi_shift:
                return candidate
        return image


class PairedIndependentRadiometricViews4Band:
    """Create two radiometric views after one shared spatial transformation.

    The caller applies any exact spatial transform first. This wrapper then
    clones that common tensor and samples the supplied radiometric transform
    independently for each view, isolating the intended consistency axis.
    """

    def __init__(self, radiometric_transform):
        if radiometric_transform is None or not callable(radiometric_transform):
            raise ValueError("A callable radiometric transform is required")
        self.radiometric_transform = radiometric_transform

    def __call__(self, image):
        if image.ndim != 3 or image.shape[0] != 4:
            raise ValueError(
                f"Expected four-band C,H,W tensor, received {tuple(image.shape)}"
            )
        return (
            self.radiometric_transform(image.clone()),
            self.radiometric_transform(image.clone()),
        )
