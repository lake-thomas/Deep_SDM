"""End-to-end CPU smoke tests for main.py on tiny synthetic datasets."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


class TrainingSmokeTests(unittest.TestCase):
    def test_image_tabular_training_records_modality_dropout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "chip.tif"
            image = np.stack([
                np.full((64, 64), value, dtype=np.uint8)
                for value in (40, 80, 120, 160)
            ])
            with rasterio.open(
                image_path,
                "w",
                driver="GTiff",
                height=64,
                width=64,
                count=4,
                dtype="uint8",
                crs="EPSG:5070",
                transform=rasterio.transform.from_origin(0, 64, 1, 1),
            ) as target:
                target.write(image)
            rows = []
            for split in ("train", "val", "test"):
                for index in range(4):
                    rows.append({
                        "sample_id": f"{split}_{index}",
                        "split": split,
                        "presence": index % 2,
                        "lat": 35.0,
                        "lon": -78.0,
                        "chip_path": str(image_path),
                        "bio_1": float(index),
                    })
            csv_path = root / "manifest.csv"
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            config = {
                "experiment": "modality_dropout_smoke",
                "output_dir": str(root / "outputs"),
                "csv_path": str(csv_path),
                "image_dir": str(root),
                "model_type": "image_tabular",
                "env_features": ["bio_1"],
                "include_topo_scalars": False,
                "epochs": 1,
                "batch_size": 2,
                "learning_rate": 0.001,
                "dropout": 0.25,
                "hidden_dim": 8,
                "env_feature_dim": 4,
                "naip_channels": 4,
                "pretrained_image": False,
                "num_workers": 0,
                "use_fp16": False,
                "use_wandb": False,
                "loss_strategy": "ordinary_bce",
                "spatial_augmentation": "none",
                "radiometric_augmentation": False,
                "radiometric_consistency": {"enabled": False, "weight": 0.0},
                "image_modality_dropout_probability": 0.25,
                "postfit_calibration": "none",
                "seed": 13,
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, "main.py", "--config", str(config_path)],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                timeout=180,
            )
            self.assertEqual(
                completed.returncode, 0, msg=completed.stderr + completed.stdout
            )
            output = root / "outputs" / "modality_dropout_smoke"
            design = json.loads((output / "training_design.json").read_text())
            self.assertTrue(design["image_modality_dropout"]["enabled"])
            self.assertEqual(design["image_modality_dropout"]["probability"], 0.25)
            self.assertEqual(
                design["image_modality_dropout"]["validation_test_inference"],
                "disabled_full_image_pathway",
            )

    def test_image_training_records_paired_radiometric_consistency(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "chip.tif"
            image = np.stack([
                np.full((64, 64), value, dtype=np.uint8)
                for value in (40, 80, 120, 160)
            ])
            with rasterio.open(
                image_path,
                "w",
                driver="GTiff",
                height=64,
                width=64,
                count=4,
                dtype="uint8",
                crs="EPSG:5070",
                transform=rasterio.transform.from_origin(0, 64, 1, 1),
            ) as target:
                target.write(image)
            rows = []
            for split in ("train", "val", "test"):
                for index in range(4):
                    rows.append({
                        "sample_id": f"{split}_{index}",
                        "split": split,
                        "presence": index % 2,
                        "lat": 35.0,
                        "lon": -78.0,
                        "chip_path": str(image_path),
                    })
            csv_path = root / "manifest.csv"
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            config = {
                "experiment": "paired_consistency_smoke",
                "output_dir": str(root / "outputs"),
                "csv_path": str(csv_path),
                "image_dir": str(root),
                "model_type": "image_only",
                "epochs": 1,
                "batch_size": 2,
                "learning_rate": 0.001,
                "dropout": 0.25,
                "hidden_dim": 8,
                "naip_channels": 4,
                "pretrained_image": False,
                "num_workers": 0,
                "use_fp16": False,
                "use_wandb": False,
                "loss_strategy": "ordinary_bce",
                "spatial_augmentation": "dihedral",
                "radiometric_augmentation": {
                    "enabled": True,
                    "probability": 1.0,
                    "scene_gain": 0.05,
                    "gamma": 0.03,
                    "contrast": 0.05,
                    "band_gain": 0.01,
                    "band_offset": 0.005,
                    "max_mean_ndvi_shift": 0.05,
                    "attempts": 2,
                },
                "radiometric_consistency": {"enabled": True, "weight": 2.0},
                "postfit_calibration": "none",
                "seed": 11,
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, "main.py", "--config", str(config_path)],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                timeout=180,
            )
            self.assertEqual(
                completed.returncode, 0, msg=completed.stderr + completed.stdout
            )
            output = root / "outputs" / "paired_consistency_smoke"
            history = pd.read_csv(output / "training_history.csv")
            self.assertIn("train_supervised_loss", history)
            self.assertIn("train_radiometric_consistency_loss", history)
            heartbeat = json.loads((output / "training_heartbeat.json").read_text())
            self.assertEqual(heartbeat["status"], "training_complete")
            self.assertEqual(heartbeat["epochs_completed"], 1)
            design = json.loads((output / "training_design.json").read_text())
            self.assertTrue(design["radiometric_consistency"]["enabled"])
            self.assertEqual(design["radiometric_consistency"]["weight"], 2.0)

    def test_training_writes_traceable_best_model_and_protects_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rows = []
            for split_index, split in enumerate(["train", "val", "test"]):
                for index in range(4):
                    label = index % 2
                    rows.append(
                        {
                            "sample_id": f"{split}_{index}",
                            "split": split,
                            "presence": label,
                            "lat": 35.0 + split_index,
                            "lon": -78.0 - index,
                            "wc2.1_30s_bio_1": float(label) + index * 0.01,
                        }
                    )
            csv_path = root / "manifest.csv"
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            config = {
                "experiment": "smoke",
                "output_dir": str(root / "outputs"),
                "csv_path": str(csv_path),
                "image_dir": str(root),
                "model_type": "tabular_only",
                "env_features": ["wc2.1_30s_bio_1"],
                "epochs": 1,
                "batch_size": 4,
                "learning_rate": 0.001,
                "dropout": 0.25,
                "hidden_dim": 8,
                "env_feature_dim": 8,
                "num_workers": 0,
                "use_fp16": False,
                "use_wandb": False,
                "loss_strategy": "ordinary_bce",
                "spatial_augmentation": "none",
                "postfit_calibration": "none",
                "seed": 11,
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            command = [sys.executable, "main.py", "--config", str(config_path)]
            completed = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(completed.returncode, 0, msg=completed.stderr + completed.stdout)
            output = root / "outputs" / "smoke"
            self.assertTrue((output / "checkpoints" / "best.pt").is_file())
            self.assertEqual(
                json.loads((output / "run_status.json").read_text())["status"],
                "completed",
            )
            self.assertTrue((output / "runtime_provenance.json").is_file())
            self.assertTrue((output / "validation_threshold.json").is_file())

            refused = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("Refusing to reuse nonempty experiment directory", refused.stderr)

    def test_same_ground_consistency_with_image_harmonization(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "chip.tif"
            image = np.stack([
                np.full((64, 64), value, dtype=np.uint8)
                for value in (40, 80, 120, 160)
            ])
            with rasterio.open(
                image_path,
                "w",
                driver="GTiff",
                height=64,
                width=64,
                count=4,
                dtype="uint8",
                crs="EPSG:5070",
                transform=rasterio.transform.from_origin(0, 64, 1, 1),
            ) as target:
                target.write(image)
            rows = []
            for split in ("train", "val", "test"):
                for index in range(4):
                    rows.append({
                        "sample_id": f"{split}_{index}",
                        "split": split,
                        "presence": index % 2,
                        "lat": 35.0,
                        "lon": -78.0,
                        "chip_path": str(image_path),
                        "bio_1": float(index),
                    })
            csv_path = root / "manifest.csv"
            pd.DataFrame(rows).to_csv(csv_path, index=False)

            # Six same-ground pairs: four for training and two held out.
            pairs_dir = root / "pairs"
            pairs_dir.mkdir()
            rng = np.random.default_rng(7)
            np.save(pairs_dir / "images_a.npy", rng.integers(0, 256, (6, 4, 64, 64), dtype=np.uint8))
            np.save(pairs_dir / "images_b.npy", rng.integers(0, 256, (6, 4, 64, 64), dtype=np.uint8))
            np.save(pairs_dir / "env.npy", rng.normal(size=(6, 1)).astype(np.float32))
            np.save(pairs_dir / "keep.npy", np.ones(6, dtype=bool))
            pd.DataFrame({"split": ["train"] * 4 + ["eval"] * 2}).to_csv(
                pairs_dir / "meta.csv", index=False
            )

            config = {
                "experiment": "same_ground_smoke",
                "output_dir": str(root / "outputs"),
                "csv_path": str(csv_path),
                "image_dir": str(root),
                "model_type": "image_tabular",
                "env_features": ["bio_1"],
                "include_topo_scalars": False,
                "epochs": 1,
                "batch_size": 2,
                "learning_rate": 0.001,
                "dropout": 0.25,
                "hidden_dim": 8,
                "env_feature_dim": 4,
                "naip_channels": 4,
                "pretrained_image": False,
                "num_workers": 0,
                "use_fp16": False,
                "use_wandb": False,
                "loss_strategy": "ordinary_bce",
                "spatial_augmentation": "dihedral",
                "postfit_calibration": "none",
                "image_harmonization": {
                    "mode": "chip_standardize",
                    "ref_mean": [0.35, 0.38, 0.33, 0.45],
                    "ref_std": [0.08, 0.07, 0.06, 0.10],
                },
                "same_ground_consistency": {
                    "enabled": True,
                    "weight": 1.0,
                    "pairs_dir": str(pairs_dir),
                    "batch_size": 2,
                    "num_workers": 0,
                },
                "seed": 5,
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, "main.py", "--config", str(config_path)],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                timeout=180,
            )
            self.assertEqual(
                completed.returncode, 0, msg=completed.stderr + completed.stdout
            )
            self.assertIn("Same-ground consistency: weight 1.0, 4 training pairs", completed.stdout)
            output = root / "outputs" / "same_ground_smoke"
            self.assertTrue((output / "checkpoints" / "best.pt").is_file())
            design = json.loads((output / "training_design.json").read_text())
            self.assertTrue(design["same_ground_consistency"]["enabled"])
            self.assertEqual(design["same_ground_consistency"]["weight"], 1.0)


if __name__ == "__main__":
    unittest.main()
