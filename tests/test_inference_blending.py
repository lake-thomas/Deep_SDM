"""Regression tests for flat and Hann tiled-inference blending."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import rasterio
import torch
from rasterio.transform import from_origin

import inference_multimodal_production as inference


class MeanImageModel(torch.nn.Module):
    def forward_from_batch(self, batch):
        return batch["image"].mean(dim=(1, 2, 3))


class BlendWindowUnitTests(unittest.TestCase):
    def test_flat_window_is_all_ones(self):
        weight = inference.make_blend_weight(5, mode="flat", edge_floor=-1.0)
        np.testing.assert_array_equal(weight, np.ones((5, 5), dtype=np.float32))

    def test_hann_window_is_symmetric_normalized_and_positive(self):
        weight = inference.make_blend_weight(8, mode="hann", edge_floor=0.05)
        self.assertEqual(weight.dtype, np.float32)
        np.testing.assert_array_equal(weight, np.flipud(weight))
        np.testing.assert_array_equal(weight, np.fliplr(weight))
        self.assertAlmostEqual(float(weight.min()), 0.05, places=7)
        self.assertAlmostEqual(float(weight.max()), 1.0, places=7)
        self.assertTrue(np.isfinite(weight).all())
        self.assertTrue((weight > 0.0).all())

    def test_hann_rejects_an_edge_floor_that_can_create_gaps(self):
        for edge_floor in (0.0, -0.1, 1.1, float("nan")):
            with self.subTest(edge_floor=edge_floor):
                with self.assertRaises(ValueError):
                    inference.make_blend_weight(8, mode="hann", edge_floor=edge_floor)

    def test_cli_defaults_to_legacy_flat_mode(self):
        argv = [
            "inference_multimodal_production.py",
            "--checkpoint",
            "model.pt",
            "--output-dir",
            "outputs",
        ]
        with patch.object(sys, "argv", argv):
            args = inference.parse_args()
        self.assertEqual(args.blend_window, "flat")
        self.assertEqual(args.blend_edge_floor, 0.05)


class BlendWindowIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.tile = self.root / "synthetic_naip.tif"
        grid = np.linspace(0.1, 1.0, 36, dtype=np.float32).reshape(6, 6)
        image = np.stack([grid, grid * 0.8, grid * 0.6, grid * 0.4])
        profile = {
            "driver": "GTiff",
            "height": 6,
            "width": 6,
            "count": 4,
            "dtype": "float32",
            "transform": from_origin(0.0, 12.0, 2.0, 2.0),
        }
        with rasterio.open(self.tile, "w", **profile) as destination:
            destination.write(image)
        self.arguments = {
            "tile_fp": self.tile,
            "model": MeanImageModel(),
            "model_type": "image_only",
            "req": {"image": True, "topo": False, "env": False},
            "device": torch.device("cpu"),
            "wc_rasters": None,
            "ghm_raster": None,
            "lat_lon_stats": {},
            "env_vars": [],
            "topo_sources": None,
            "topo_norm_stats": {},
            "chip_size": 4,
            "stride": 2,
            "batch_size": 2,
            "normalize_topography": False,
            "naip_scale_255": False,
            "min_image_valid_frac": 0.0,
            "mc_samples": 1,
            "use_amp": False,
        }

    def tearDown(self):
        self.tempdir.cleanup()

    def _run(self, name, **blend):
        output = self.root / name
        summary = inference.run_inference_on_tile(
            output_dir=output,
            **self.arguments,
            **blend,
        )
        stem = self.tile.stem
        with rasterio.open(output / f"{stem}_prediction.tif") as source:
            prediction = source.read(1)
            tags = source.tags()
        with rasterio.open(output / f"{stem}_uncertainty.tif") as source:
            uncertainty = source.read(1)
        with rasterio.open(output / f"{stem}_count.tif") as source:
            count = source.read(1)
            count_tags = source.tags()
        return summary, prediction, uncertainty, count, tags, count_tags

    def test_flat_default_is_deterministic_and_legacy_tagged(self):
        default = self._run("default")
        explicit = self._run("explicit", blend_window="flat")
        np.testing.assert_array_equal(default[1], explicit[1])
        np.testing.assert_array_equal(default[2], explicit[2])
        np.testing.assert_array_equal(default[3], explicit[3])
        self.assertEqual(default[0]["blend_window"], "flat")
        self.assertIsNone(default[0]["blend_edge_floor"])
        self.assertEqual(default[4]["blend_window"], "flat")
        self.assertNotIn("blend_edge_floor", default[4])

    def test_hann_preserves_support_and_count_geometry(self):
        flat = self._run("flat", blend_window="flat")
        hann = self._run(
            "hann",
            blend_window="hann",
            blend_edge_floor=0.05,
        )
        np.testing.assert_array_equal(hann[3], flat[3])
        np.testing.assert_array_equal(hann[1] != -9999.0, flat[1] != -9999.0)
        np.testing.assert_array_equal(hann[2] != -9999.0, flat[2] != -9999.0)
        self.assertTrue(np.isfinite(hann[1][hann[3] > 0]).all())
        self.assertTrue(np.isfinite(hann[2][hann[3] > 0]).all())
        self.assertFalse(np.array_equal(hann[1], flat[1]))
        self.assertEqual(hann[0]["blend_window"], "hann")
        self.assertEqual(hann[0]["blend_edge_floor"], 0.05)
        self.assertEqual(hann[4]["blend_edge_floor"], "0.05")
        self.assertEqual(hann[5]["blend_window"], "hann")


if __name__ == "__main__":
    unittest.main()
