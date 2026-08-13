import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from calibration_utils import (
    apply_platt_calibrator,
    calibration_metrics,
    derive_mcc_threshold,
    fit_platt_calibrator,
)
from datasets import BalancedBinaryBatchSampler
from model import HostModelBase
from transforms import RandomDihedral4Band, RandomRadiometricAugment4Band


class _DummyBinaryModel(HostModelBase):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))

    def forward_from_batch(self, batch):
        return batch["logits"] + self.bias


class AugustLossTests(unittest.TestCase):
    def test_weighting_and_smoothing_are_training_only(self):
        model = _DummyBinaryModel()
        model.configure_training_loss(pos_weight=3.0, label_smoothing=0.1)
        batch = {
            "logits": torch.tensor([-1.0, 1.0]),
            "label": torch.tensor([0.0, 1.0]),
        }
        smoothed = batch["label"] * 0.9 + 0.05
        expected_training = F.binary_cross_entropy_with_logits(
            batch["logits"],
            smoothed,
            pos_weight=torch.tensor(3.0),
        )
        self.assertTrue(torch.allclose(model.training_step(batch), expected_training))
        expected_validation = F.binary_cross_entropy_with_logits(
            batch["logits"], batch["label"]
        )
        validation = model.validation_step(batch)
        self.assertTrue(torch.allclose(validation["val_loss"], expected_validation))


class BalancedSamplerTests(unittest.TestCase):
    def test_batches_are_balanced_and_reproducible(self):
        labels = np.array([0] * 12 + [1] * 3)
        first = list(BalancedBinaryBatchSampler(labels, batch_size=6, seed=9))
        second = list(BalancedBinaryBatchSampler(labels, batch_size=6, seed=9))
        self.assertEqual(first, second)
        for batch in first:
            selected = labels[np.asarray(batch)]
            self.assertEqual(int((selected == 0).sum()), 3)
            self.assertEqual(int((selected == 1).sum()), 3)


class AugustAugmentationTests(unittest.TestCase):
    def test_dihedral_transform_introduces_no_interpolated_values(self):
        image = torch.arange(4 * 5 * 5, dtype=torch.float32).reshape(4, 5, 5)
        random.seed(4)
        transformed = RandomDihedral4Band()(image)
        self.assertEqual(tuple(transformed.shape), tuple(image.shape))
        self.assertEqual(
            torch.sort(transformed.flatten()).values.tolist(),
            torch.sort(image.flatten()).values.tolist(),
        )

    def test_radiometric_transform_is_bounded_and_ndvi_guarded(self):
        image = torch.full((4, 32, 32), 0.4)
        image[3] = 0.7
        transform = RandomRadiometricAugment4Band(
            probability=1.0,
            max_mean_ndvi_shift=0.06,
        )
        random.seed(5)
        transformed = transform(image)
        self.assertTrue(torch.isfinite(transformed).all())
        self.assertGreaterEqual(float(transformed.min()), 0.0)
        self.assertLessEqual(float(transformed.max()), 1.0)
        shift = abs(
            float(transform._mean_ndvi(transformed) - transform._mean_ndvi(image))
        )
        self.assertLessEqual(shift, 0.06 + 1e-7)


class CalibrationTests(unittest.TestCase):
    def test_mcc_threshold_comes_from_validation_scores(self):
        labels = [0, 0, 1, 1]
        probabilities = [0.1, 0.4, 0.6, 0.9]
        result = derive_mcc_threshold(labels, probabilities)
        predictions = (np.asarray(probabilities) >= result["threshold"]).astype(int)
        self.assertEqual(predictions.tolist(), labels)
        self.assertAlmostEqual(result["validation_mcc"], 1.0)

    def test_calibration_outputs_are_finite_and_platt_is_bounded(self):
        labels = np.array([0, 1] * 20)
        probabilities = np.linspace(0.1, 0.9, len(labels))
        metrics = calibration_metrics(labels, probabilities)
        self.assertTrue(np.isfinite(metrics["brier_score"]))
        self.assertTrue(np.isfinite(metrics["binary_log_loss"]))
        calibrator = fit_platt_calibrator(labels, probabilities)
        calibrated = apply_platt_calibrator(probabilities, calibrator)
        self.assertTrue(np.isfinite(calibrated).all())
        self.assertTrue(((calibrated > 0) & (calibrated < 1)).all())


class AugustEndToEndSmokeTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
