import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F

from calibration_utils import (
    apply_platt_calibrator,
    calibration_metrics,
    derive_mcc_threshold,
    fit_platt_calibrator,
)
from august2026_experiments import (
    assert_frozen_runtime_identity,
    filter_specs,
    stage_specs,
)
from august2026_compare_artifact_probe import (
    exact_pair_merge,
    paired_comparison,
)
from august2026_batch_artifact_probe import reusable_completed_probe
from august2026_dataset_screen_evidence import pareto_nondominated, synthesize
from august2026_dataset_overlap_audit import build_overlap_audit, prepare_dataset
from august2026_cross_host_pathway_audit import (
    correlation_metrics,
    residualize,
    resolve_chip_path,
)
from august2026_final_model_acceptance import artifact_nonregression_gate
from august2026_aligned_raster_correlation import aligned_raster_metrics
from august2026_candidate_builder_provenance_audit import commit_matches
from august2026_ailanthus_robustness_screen import (
    STRONG_RADIOMETRIC,
    candidate_recipes as ailanthus_robustness_recipes,
    experiment_name as ailanthus_robustness_experiment_name,
)
from august2026_ailanthus_robustness_evidence import (
    EXACT_OVERLAP_MAXIMUM,
    MATCHED_BOUNDARY_MAXIMUM,
    add_advancement_gates,
)
from august2026_ailanthus_consistency_screen import (
    CONSISTENCY_WEIGHTS,
    build_config as build_ailanthus_consistency_config,
    experiment_name as ailanthus_consistency_experiment_name,
)
from august2026_ailanthus_modality_dropout_screen import (
    MODALITY_DROPOUT_PROBABILITIES,
    build_config as build_ailanthus_modality_dropout_config,
    experiment_name as ailanthus_modality_dropout_experiment_name,
)
from august2026_freeze_ailanthus_robustness_selection import (
    freeze_selection as freeze_ailanthus_screen_selection,
)
from august2026_ailanthus_blockcv import replacement_name
from august2026_ailanthus_production_train import (
    replacement_name as replacement_production_name,
)
from august2026_ailanthus_horizon_retry import (
    HORIZON_DESIGN_VERSION,
    HORIZON_EPOCHS,
    retry_name as ailanthus_horizon_retry_name,
)
from august2026_mc_stability_evidence import PairAccumulator, advancement_rule
from august2026_mc_stability_screen import (
    experiment_name as mc_experiment_name,
    run_design as run_mc_design,
)
from august2026_combine_dataset_screen_evidence import HOSTS, combine_evidence
from august2026_inference_output_qa import validate_triplet
from august2026_prepare_mosaic_inputs import prepare_inputs
from august2026_calibrate_radiometric_augmentation import derive_recipe
from august2026_freeze_dataset_selection import freeze_selection
from august2026_loss_screen import build_loss_config, loss_experiment_name
from august2026_exact_overlap_probe import (
    compare_exact_overlap,
    reusable_output as reusable_exact_overlap_output,
)
import august2026_materialization_attrition_audit as attrition_audit
from august2026_materialization_attrition_audit import (
    canonicalize_progress_attempts,
    distribution_shift,
)
from august2026_refinalize_materialized_manifests import read_materialization_records
from august2026_wait_for_dataset_screen import smoke_evidence_ready
from august2026_wait_for_control_screen import control_gates_ready
from august2026_loss_screen_evidence import strict_evidence_gate
from august2026_augmentation_screen import augmentation_combinations, build_config as build_augmentation_config
from august2026_blockcv import FOLDS, blockcv_name
from august2026_blockcv_evidence import calibration_recommendation, summarize_values
from august2026_run_blockcv_acceptance import _reuse_evidence
from august2026_production_train import production_name
from august2026_fixed_env_artifact_probe import pair_metrics, sha256
from august2026_summarize_experiments import summarize_registry
from datasets import BalancedBinaryBatchSampler
from model import HostModelBase
from transforms import (
    PairedIndependentRadiometricViews4Band,
    RandomDihedral4Band,
    RandomRadiometricAugment4Band,
)


class _DummyBinaryModel(HostModelBase):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))

    def forward_from_batch(self, batch):
        return batch["logits"] + self.bias


class _DummyPairedImageModel(HostModelBase):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))

    def forward_from_batch(self, batch):
        return batch["image"] + self.bias


class AugustLossTests(unittest.TestCase):
    def test_image_modality_dropout_masks_whole_embeddings_training_only(self):
        model = _DummyBinaryModel()
        model.configure_training_loss(image_modality_dropout_probability=0.5)
        features = torch.ones((4, 3))
        draws = torch.tensor([[0.1], [0.6], [0.2], [0.9]])
        model.train()
        with patch("model.torch.rand", return_value=draws):
            observed = model.apply_image_modality_dropout(features)
        expected = torch.tensor([
            [0.0, 0.0, 0.0],
            [2.0, 2.0, 2.0],
            [0.0, 0.0, 0.0],
            [2.0, 2.0, 2.0],
        ])
        self.assertTrue(torch.equal(observed, expected))
        model.eval()
        self.assertTrue(torch.equal(model.apply_image_modality_dropout(features), features))
        with self.assertRaisesRegex(ValueError, "in \[0, 1\)"):
            model.configure_training_loss(image_modality_dropout_probability=1.0)

    def test_paired_radiometric_consistency_adds_probability_mse(self):
        model = _DummyPairedImageModel()
        model.configure_training_loss(radiometric_consistency_weight=2.0)
        batch = {
            "image": torch.tensor([-1.0, 1.0]),
            "image_consistency": torch.tensor([-0.5, 0.25]),
            "label": torch.tensor([0.0, 1.0]),
        }
        supervised = F.binary_cross_entropy_with_logits(
            batch["image"], batch["label"]
        )
        consistency = F.mse_loss(
            torch.sigmoid(batch["image"]),
            torch.sigmoid(batch["image_consistency"]),
        )
        observed = model.training_step(batch)
        self.assertTrue(torch.allclose(observed, supervised + 2.0 * consistency))
        components = model.training_loss_components()
        self.assertTrue(torch.allclose(components["supervised"], supervised))
        self.assertTrue(
            torch.allclose(components["radiometric_consistency"], consistency)
        )

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
    def test_paired_radiometric_views_are_independent_clones(self):
        class CountingTransform:
            def __init__(self):
                self.calls = 0

            def __call__(self, image):
                self.calls += 1
                image.add_(float(self.calls))
                return image

        source = torch.zeros((4, 3, 3), dtype=torch.float32)
        paired = PairedIndependentRadiometricViews4Band(CountingTransform())
        first, second = paired(source)
        self.assertTrue(torch.equal(source, torch.zeros_like(source)))
        self.assertTrue(torch.equal(first, torch.ones_like(first)))
        self.assertTrue(torch.equal(second, torch.full_like(second, 2.0)))

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


class AugustExperimentDesignTests(unittest.TestCase):
    def test_sequential_launcher_rechecks_frozen_identity_before_each_fit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            design = Path(temp_dir)
            registry = design / "experiment_registry.tsv"
            registry.touch()
            (design / "design_metadata.json").write_text(
                json.dumps({"code_commit": "frozen123", "code_dirty": False}),
                encoding="utf-8",
            )

            clean = [
                subprocess.CompletedProcess([], 0, stdout="frozen123\n"),
                subprocess.CompletedProcess([], 0, stdout=""),
            ]
            with patch("august2026_experiments.subprocess.run", side_effect=clean):
                assert_frozen_runtime_identity(registry)

            changed = [
                subprocess.CompletedProcess([], 0, stdout="later456\n"),
                subprocess.CompletedProcess([], 0, stdout=""),
            ]
            with patch("august2026_experiments.subprocess.run", side_effect=changed):
                with self.assertRaisesRegex(RuntimeError, "changed runtime identity"):
                    assert_frozen_runtime_identity(registry)

            dirty = [
                subprocess.CompletedProcess([], 0, stdout="frozen123\n"),
                subprocess.CompletedProcess([], 0, stdout=" M main.py\n"),
            ]
            with patch("august2026_experiments.subprocess.run", side_effect=dirty):
                with self.assertRaisesRegex(RuntimeError, "clean=False"):
                    assert_frozen_runtime_identity(registry)

    def test_mc_matrix_checks_frozen_identity_before_marking_row_running(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            registry = root / "experiment_registry.tsv"
            row = {
                "experiment": "mc_identity_test",
                "run_dir": str(root / "output"),
                "status": "pending",
            }
            with patch(
                "august2026_mc_stability_screen.assert_frozen_runtime_identity",
                side_effect=RuntimeError("changed runtime identity"),
            ) as identity:
                with self.assertRaisesRegex(RuntimeError, "changed runtime identity"):
                    run_mc_design(registry, [row], resume_completed=False)
            identity.assert_called_once_with(registry)
            self.assertEqual(row["status"], "pending")

    def test_mc_stability_rule_combines_absolute_and_repeat_noise_floors(self):
        passed = advancement_rule(0.004, 0.002, 0.999, 0.98, 0.001, 0.001)
        self.assertTrue(passed["stability_gate_passed"])
        self.assertEqual(passed["prediction_mae_maximum"], 0.005)
        failed = advancement_rule(0.006, 0.002, 0.999, 0.98, 0.001, 0.001)
        self.assertFalse(failed["stability_gate_passed"])
        noise_scaled = advancement_rule(0.01, 0.005, 0.999, 0.98, 0.01, 0.005)
        self.assertTrue(noise_scaled["stability_gate_passed"])

    def test_mc_pair_accumulator_streams_exact_metrics(self):
        accumulator = PairAccumulator()
        accumulator.add(np.array([1.0, 2.0]), np.array([1.0, 3.0]))
        accumulator.add(np.array([4.0]), np.array([5.0]))
        metrics = accumulator.metrics()
        self.assertEqual(metrics["pixel_count"], 3)
        self.assertAlmostEqual(metrics["mae"], 2 / 3)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(2 / 3))

    def test_mc_stability_names_cover_the_frozen_factorial(self):
        names = {
            mc_experiment_name("ailanthus_altissima", level, seed)
            for level in (2, 3, 5, 10) for seed in (42, 43)
        }
        self.assertEqual(len(names), 8)
        self.assertTrue(all("ailanthus_altissima" in name for name in names))

    def test_ailanthus_replacement_blockcv_name_is_isolated(self):
        selection = {
            "arm": "domain_balanced",
            "background_target": 20000,
            "radiometric_augmentation": {"enabled": True},
            "radiometric_recipe": "rad2",
            "dropout": 0.35,
            "label_smoothing": 0.05,
        }
        names = [replacement_name(selection, fold) for fold in range(1, 6)]
        self.assertEqual(len(set(names)), 5)
        self.assertTrue(all(name.startswith("stage05b_ailanthus_altissima_") for name in names))
        self.assertTrue(all("_rad2_" in name for name in names))

        consistency_selection = {
            **selection,
            "radiometric_consistency": {"enabled": True, "weight": 2.0},
        }
        consistency_names = [
            replacement_name(consistency_selection, fold) for fold in range(1, 6)
        ]
        self.assertTrue(all("_rad2_consw200_" in name for name in consistency_names))
        self.assertTrue(set(names).isdisjoint(consistency_names))
        modality_selection = {
            **selection,
            "image_modality_dropout_probability": 0.25,
        }
        modality_names = [
            replacement_name(modality_selection, fold) for fold in range(1, 6)
        ]
        self.assertTrue(all("_rad2_moddrop25_" in name for name in modality_names))
        self.assertTrue(set(names).isdisjoint(modality_names))
        disabled_consistency_selection = {
            **modality_selection,
            "radiometric_consistency": {"enabled": False, "weight": 0.0},
        }
        disabled_names = [
            replacement_name(disabled_consistency_selection, fold)
            for fold in range(1, 6)
        ]
        self.assertEqual(modality_names, disabled_names)
        self.assertTrue(all("consw" not in name for name in disabled_names))

    def test_ailanthus_replacement_production_name_preserves_lineage(self):
        selection = {
            "arm": "domain_balanced",
            "background_target": 20000,
            "radiometric_recipe": "rad2",
            "dropout": 0.35,
            "label_smoothing": 0.05,
            "production_postfit_calibration": "platt",
        }
        name = replacement_production_name(selection)
        self.assertIn("production_replacement_ailanthus_altissima", name)
        self.assertIn("_rad2_", name)
        self.assertIn("_platt_", name)
        selection["radiometric_consistency"] = {"enabled": True, "weight": 0.5}
        consistency_name = replacement_production_name(selection)
        self.assertIn("_rad2_consw050_", consistency_name)
        self.assertNotEqual(name, consistency_name)
        selection.pop("radiometric_consistency")
        selection["image_modality_dropout_probability"] = 0.10
        modality_name = replacement_production_name(selection)
        self.assertIn("_rad2_moddrop10_", modality_name)
        self.assertNotEqual(name, modality_name)
        selection["radiometric_consistency"] = {"enabled": False, "weight": 0.0}
        disabled_consistency_name = replacement_production_name(selection)
        self.assertEqual(modality_name, disabled_consistency_name)
        self.assertNotIn("consw", disabled_consistency_name)

    def test_ailanthus_horizon_retry_name_freezes_selection_rule(self):
        selection = {
            "background_target": 20000,
            "radiometric_recipe": "rad2",
            "image_modality_dropout_probability": 0.25,
            "dropout": 0.25,
            "label_smoothing": 0.05,
            "production_postfit_calibration": "none",
        }
        name = ailanthus_horizon_retry_name(selection)
        self.assertIn(
            f"_h{HORIZON_EPOCHS:03d}v{HORIZON_DESIGN_VERSION}_", name
        )
        self.assertIn("_moddrop25_", name)
        self.assertTrue(name.startswith("production_retry_ailanthus_altissima_"))

    def test_ailanthus_robustness_advancement_gates_are_conjunctive(self):
        import pandas as pd

        common = {
            "test_roc_auc_all_thresholds": 0.98,
            "test_brier_score": 0.06,
            "exact_overlap_candidate_absolute_shift_mean": 0.15,
            "test_calibration_binary_log_loss": 0.22,
            "test_calibration_calibration_slope": 1.1,
            "test_calibration_calibration_intercept": -1.0,
            "test_calibration_expected_calibration_error": 0.06,
        }
        frame = pd.DataFrame(
            [
                {**common, "artifact_candidate_absolute_did_mean": 0.09},
                {
                    **common,
                    "artifact_candidate_absolute_did_mean":
                        MATCHED_BOUNDARY_MAXIMUM + 0.001,
                },
                {
                    **common,
                    "artifact_candidate_absolute_did_mean": 0.08,
                    "exact_overlap_candidate_absolute_shift_mean":
                        EXACT_OVERLAP_MAXIMUM + 0.001,
                },
            ]
        )
        result = add_advancement_gates(frame)
        self.assertEqual(result["advancement_gate_passed"].tolist(), [True, False, False])

    def test_ailanthus_robustness_screen_is_bounded_and_unique(self):
        mild = {
            "enabled": True,
            "probability": 0.8,
            "scene_gain": 0.08,
            "gamma": 0.06,
            "contrast": 0.0627,
            "band_gain": 0.02,
            "band_offset": 0.01,
            "max_mean_ndvi_shift": 0.0538,
            "attempts": 3,
        }
        recipes = ailanthus_robustness_recipes(mild)
        self.assertEqual(len(recipes), 5)
        names = [ailanthus_robustness_experiment_name(item) for item in recipes]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(sum(item["recipe_label"] == "rad1" for item in recipes), 1)
        self.assertEqual(sum(item["recipe_label"] == "rad2" for item in recipes), 4)
        self.assertLessEqual(STRONG_RADIOMETRIC["scene_gain"], 0.14474023994201524)
        self.assertLessEqual(STRONG_RADIOMETRIC["contrast"], 0.15723992553342825)
        self.assertLessEqual(
            STRONG_RADIOMETRIC["max_mean_ndvi_shift"], 0.0630348114243701
        )

    def test_ailanthus_consistency_screen_is_bounded_and_preserves_recipe(self):
        self.assertEqual(CONSISTENCY_WEIGHTS, (0.5, 2.0, 5.0))
        names = [
            ailanthus_consistency_experiment_name(weight)
            for weight in CONSISTENCY_WEIGHTS
        ]
        self.assertEqual(len(names), len(set(names)))
        source = {
            "seed": 42,
            "radiometric_augmentation": dict(STRONG_RADIOMETRIC),
            "dropout": 0.25,
            "label_smoothing": 0.05,
        }
        base = {
            "best_checkpoint_sha256": "a" * 64,
            "artifact_candidate_absolute_did_mean": 0.105374757,
        }
        config = build_ailanthus_consistency_config(
            source,
            base,
            2.0,
            Path("/tmp/stage"),
            Path("/tmp/evidence"),
            "b" * 64,
        )
        self.assertEqual(config["radiometric_augmentation"], STRONG_RADIOMETRIC)
        self.assertEqual(config["dropout"], 0.25)
        self.assertEqual(config["label_smoothing"], 0.05)
        self.assertEqual(
            config["radiometric_consistency"], {"enabled": True, "weight": 2.0}
        )
        self.assertEqual(
            config["august_experiment_spec"]["stage"],
            "ailanthus-radiometric-consistency-screen",
        )

    def test_ailanthus_modality_dropout_screen_is_bounded_and_isolated(self):
        self.assertEqual(MODALITY_DROPOUT_PROBABILITIES, (0.10, 0.25, 0.50))
        names = [
            ailanthus_modality_dropout_experiment_name(probability)
            for probability in MODALITY_DROPOUT_PROBABILITIES
        ]
        self.assertEqual(len(names), len(set(names)))
        source = {
            "seed": 42,
            "radiometric_augmentation": dict(STRONG_RADIOMETRIC),
            "dropout": 0.25,
            "label_smoothing": 0.05,
        }
        base = {
            "experiment": "stage05a_base",
            "best_checkpoint_sha256": "a" * 64,
            "artifact_candidate_absolute_did_mean": 0.105374757,
        }
        config = build_ailanthus_modality_dropout_config(
            source,
            base,
            0.25,
            Path("/tmp/stage"),
            Path("/tmp/consistency-evidence"),
            "b" * 64,
        )
        self.assertEqual(config["radiometric_augmentation"], STRONG_RADIOMETRIC)
        self.assertEqual(config["image_modality_dropout_probability"], 0.25)
        self.assertEqual(
            config["radiometric_consistency"], {"enabled": False, "weight": 0.0}
        )
        self.assertEqual(
            config["august_experiment_spec"]["stage"],
            "ailanthus-image-modality-dropout-screen",
        )

    def test_consistency_evidence_freezes_explicit_blockcv_lineage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            evidence_dir = root / "evidence"
            run_dir = root / "run"
            (run_dir / "checkpoints").mkdir(parents=True)
            evidence_dir.mkdir()
            dataset = root / "uniform.csv"
            checkpoint = run_dir / "checkpoints" / "best.pt"
            config_path = root / "config.json"
            dataset.write_text("presence,split\n1,train\n", encoding="utf-8")
            checkpoint.write_bytes(b"checkpoint")
            config = {
                "csv_path": str(dataset),
                "image_dir": str(root / "images"),
                "radiometric_augmentation": dict(STRONG_RADIOMETRIC),
                "radiometric_consistency": {"enabled": True, "weight": 2.0},
                "august_experiment_spec": {
                    "radiometric_recipe": "rad2",
                    "seed": 42,
                },
            }
            config_path.write_text(json.dumps(config), encoding="utf-8")
            experiment = "stage06_consistency_test"
            evidence_path = evidence_dir / "candidate.csv"
            pd.DataFrame([{
                "experiment": experiment,
                "host": "ailanthus_altissima",
                "arm": "domain_balanced",
                "background_target": 20000,
                "loss_strategy": "pos_weight_bce",
                "spatial_augmentation": "dihedral",
                "dropout": 0.25,
                "label_smoothing": 0.05,
                "evidence_gate_passed": True,
                "advancement_gate_passed": True,
                "pareto_nondominated_within_screen": True,
                "config_path": str(config_path),
                "config_sha256": sha256(config_path),
                "run_dir": str(run_dir),
                "best_checkpoint_sha256": sha256(checkpoint),
                "training_dataset_csv_sha256": sha256(dataset),
                "artifact_comparison_summary": "matched.json",
                "exact_overlap_summary": "exact.json",
                "test_roc_auc_all_thresholds": 0.98,
                "test_brier_score": 0.06,
            }]).to_csv(evidence_path, index=False)
            summary = {
                "status": "complete",
                "stage": "ailanthus-radiometric-consistency-screen",
                "candidate_evidence": str(evidence_path),
                "candidate_evidence_sha256": sha256(evidence_path),
            }
            (evidence_dir / "summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            (evidence_dir / "COMPLETED").write_text("complete\n", encoding="utf-8")
            output = root / "selection.json"
            frozen = freeze_ailanthus_screen_selection(
                evidence_dir, experiment, "passed frozen gates", output, "c" * 40
            )
            selected = frozen["selections"][0]
            self.assertEqual(
                selected["source_screen_stage"],
                "ailanthus-radiometric-consistency-screen",
            )
            self.assertEqual(
                selected["radiometric_consistency"],
                {"enabled": True, "weight": 2.0},
            )

    def test_current_builder_snapshot_is_recoverable_from_head(self):
        code_root = Path(__file__).resolve().parents[1]
        names = ["candidate_manifests.py", "background.py", "naip.py"]
        expected = {
            name: sha256(code_root / "host_dataset_builder" / name)
            for name in names
        }
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=code_root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        self.assertTrue(commit_matches(code_root, head, expected, names))

    def test_mosaic_input_preparation_requires_exact_outputs_and_groups_zones(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tiles = [
                root / "m_4007501_ne_18_060_20210801.tif",
                root / "m_4007601_ne_17_060_20210801.tif",
            ]
            for path in tiles:
                path.write_bytes(b"input")
            manifest = root / "tiles.txt"
            manifest.write_text(
                "\n".join(str(path) for path in tiles) + "\n", encoding="utf-8"
            )
            outputs = root / "outputs"
            outputs.mkdir()
            for path in tiles:
                (outputs / f"{path.stem}_prediction.tif").write_bytes(b"output")
            design = root / "mosaic_inputs"
            payload = prepare_inputs(manifest, outputs, "prediction", design)
            self.assertEqual(payload["expected_raster_count"], 2)
            self.assertEqual(set(payload["zone_manifests"]), {"17", "18"})
            self.assertEqual(
                payload["zone_manifests"]["17"]["raster_count"], 1
            )

    def test_inference_output_qa_requires_hann_tags_and_zero_deterministic_uncertainty(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            profile = {
                "driver": "GTiff",
                "width": 8,
                "height": 8,
                "count": 1,
                "dtype": "uint8",
                "crs": "EPSG:5070",
                "transform": rasterio.transform.from_origin(0, 8, 1, 1),
                "nodata": 0,
            }
            input_path = root / "tile.tif"
            with rasterio.open(input_path, "w", **profile) as target:
                target.write(np.ones((8, 8), dtype=np.uint8), 1)
            tags = {
                "blend_window": "hann",
                "blend_edge_floor": "0.05",
                "score_space": "raw_sigmoid_probability",
                "inference_seed": "123",
                "mc_samples": "1",
            }
            for suffix, dtype, nodata, values in (
                ("prediction", "float32", -9999.0, np.full((8, 8), 0.4, np.float32)),
                ("uncertainty", "float32", -9999.0, np.zeros((8, 8), np.float32)),
                ("count", "uint16", 0, np.full((8, 8), 4, np.uint16)),
            ):
                path = root / f"tile_{suffix}.tif"
                output_profile = {**profile, "dtype": dtype, "nodata": nodata}
                with rasterio.open(path, "w", **output_profile) as target:
                    target.write(values, 1)
                    target.update_tags(**tags)
            result = validate_triplet(
                input_path,
                root / "tile_prediction.tif",
                root / "tile_uncertainty.tif",
                root / "tile_count.tif",
                "raw_sigmoid_probability",
                None,
                "deterministic",
                123,
                1,
            )
            self.assertTrue(result["passed"], result["failures"])
            self.assertTrue(result["has_sampled_valid_support"])

    def test_inference_output_qa_accepts_consistent_empty_support_triplet(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            profile = {
                "driver": "GTiff",
                "width": 8,
                "height": 8,
                "count": 1,
                "dtype": "uint8",
                "crs": "EPSG:5070",
                "transform": rasterio.transform.from_origin(0, 8, 1, 1),
                "nodata": 0,
            }
            input_path = root / "tile.tif"
            with rasterio.open(input_path, "w", **profile) as target:
                target.write(np.ones((8, 8), dtype=np.uint8), 1)
            tags = {
                "blend_window": "hann",
                "blend_edge_floor": "0.05",
                "score_space": "raw_sigmoid_probability",
                "inference_seed": "123",
                "mc_samples": "1",
            }
            for suffix, dtype, nodata, values in (
                ("prediction", "float32", -9999.0, np.full((8, 8), -9999.0, np.float32)),
                ("uncertainty", "float32", -9999.0, np.full((8, 8), -9999.0, np.float32)),
                ("count", "uint16", 0, np.zeros((8, 8), np.uint16)),
            ):
                path = root / f"tile_{suffix}.tif"
                output_profile = {**profile, "dtype": dtype, "nodata": nodata}
                with rasterio.open(path, "w", **output_profile) as target:
                    target.write(values, 1)
                    target.update_tags(**tags)
            result = validate_triplet(
                input_path,
                root / "tile_prediction.tif",
                root / "tile_uncertainty.tif",
                root / "tile_count.tif",
                "raw_sigmoid_probability",
                None,
                "deterministic",
                123,
                1,
            )
            self.assertTrue(result["passed"], result["failures"])
            self.assertFalse(result["has_sampled_valid_support"])

    def test_aligned_raster_correlation_streams_exact_pearson(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first_path = root / "first.tif"
            second_path = root / "second.tif"
            profile = {
                "driver": "GTiff",
                "width": 4,
                "height": 4,
                "count": 1,
                "dtype": "float32",
                "crs": "EPSG:5070",
                "transform": rasterio.transform.from_origin(0, 4, 1, 1),
                "nodata": -9999.0,
            }
            values = np.arange(16, dtype=np.float32).reshape(4, 4)
            with rasterio.open(first_path, "w", **profile) as target:
                target.write(values, 1)
            with rasterio.open(second_path, "w", **profile) as target:
                target.write(values * 2 + 1, 1)
            metrics = aligned_raster_metrics(
                first_path, second_path, sample_size=16, seed=42
            )
            self.assertEqual(metrics["joint_valid_pixel_count"], 16)
            self.assertAlmostEqual(metrics["pearson"], 1.0)
            self.assertAlmostEqual(metrics["spearman_reservoir_sample"], 1.0)

    def test_final_artifact_gate_is_strict_for_seam_positive_hosts(self):
        self.assertTrue(
            artifact_nonregression_gate("vitis_labrusca", 0.08, 0.08)["passed"]
        )
        self.assertFalse(
            artifact_nonregression_gate("vitis_labrusca", 0.08, 0.081)["passed"]
        )
        control = artifact_nonregression_gate("ailanthus_altissima", 0.01, 0.012)
        self.assertTrue(control["passed"])
        self.assertAlmostEqual(control["allowed_maximum"], 0.0125)

    def test_cross_host_pathway_residual_removes_environment_signal(self):
        environment = np.linspace(-2.0, 2.0, 100)
        outcome = 0.25 + 0.5 * environment
        residual = residualize(outcome, environment)
        self.assertLess(float(np.abs(residual).max()), 1e-12)
        correlated = correlation_metrics(environment, environment * 3.0 + 2.0)
        self.assertEqual(correlated["n"], 100)
        self.assertAlmostEqual(correlated["pearson"], 1.0)
        self.assertAlmostEqual(correlated["spearman"], 1.0)

    def test_cross_host_pathway_resolves_chips_from_configured_image_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "model_manifests" / "uniform_dataset.csv"
            image_root = root / "materialized_host"
            relative = Path("images") / "01" / "sample.tif"
            expected = image_root / relative
            expected.parent.mkdir(parents=True)
            expected.write_bytes(b"chip")
            dataset.parent.mkdir(parents=True)
            dataset.write_text("chip_path\n", encoding="utf-8")
            self.assertEqual(
                resolve_chip_path(image_root, str(relative)).resolve(),
                expected.resolve(),
            )

    def test_cross_host_dataset_overlap_is_class_specific(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first_path = root / "first.csv"
            second_path = root / "second.csv"
            common_columns = {
                "source_domain_id": ["NY|2021|a", "NY|2021|a", "PA|2022|b"],
                "source_archive_state": ["NY", "NY", "PA"],
            }
            pd.DataFrame(
                {
                    "presence": [0, 0, 1],
                    "lat": [40.0, 41.0, 42.0],
                    "lon": [-75.0, -76.0, -77.0],
                    "source_tile_key": ["tile_a", "tile_b", "tile_c"],
                    "source_local_path": ["a.tif", "b.tif", "c.tif"],
                    **common_columns,
                }
            ).to_csv(first_path, index=False)
            pd.DataFrame(
                {
                    "presence": [0, 0, 1],
                    "lat": [40.0, 43.0, 44.0],
                    "lon": [-75.0, -78.0, -79.0],
                    "source_tile_key": ["tile_a", "tile_d", "tile_e"],
                    "source_local_path": ["a.tif", "d.tif", "e.tif"],
                    "source_domain_id": ["NY|2021|a", "VA|2022|c", "OH|2021|d"],
                    "source_archive_state": ["NY", "VA", "OH"],
                }
            ).to_csv(second_path, index=False)
            summary, exposure = build_overlap_audit(
                prepare_dataset(first_path),
                prepare_dataset(second_path),
                "first",
                "second",
            )
            background = summary["background"]["dimensions"]
            self.assertEqual(background["location"]["intersection_unique"], 1)
            self.assertEqual(background["location"]["union_unique"], 3)
            self.assertAlmostEqual(background["location"]["jaccard"], 1 / 3)
            self.assertEqual(
                summary["presence"]["dimensions"]["location"]["intersection_unique"],
                0,
            )
            self.assertEqual(
                background["acquisition_domain"]["intersection_unique"], 1
            )
            self.assertEqual(set(exposure["class"]), {"all", "background", "presence"})

    def test_dataset_screen_is_the_approved_fourteen_run_matrix(self):
        specs = stage_specs("dataset-screen")
        self.assertEqual(len(specs), 14)
        vitis = [spec for spec in specs if spec.host == "vitis_labrusca"]
        tanoak = [
            spec for spec in specs if spec.host == "notholithocarpus_densiflorus"
        ]
        self.assertEqual({spec.background_target for spec in vitis}, {6000, 10000, 15000, 20000})
        self.assertEqual({spec.background_target for spec in tanoak}, {5000, 10000, 15000})
        self.assertEqual({spec.arm for spec in specs}, {"random", "domain_balanced"})
        self.assertTrue(all(spec.loss_strategy == "ordinary_bce" for spec in specs))
        self.assertTrue(all(spec.spatial_augmentation == "dihedral" for spec in specs))

    def test_control_dataset_screen_is_the_approved_six_run_matrix(self):
        specs = stage_specs("control-dataset-screen")
        self.assertEqual(len(specs), 6)
        self.assertEqual(
            {spec.host for spec in specs},
            {"ailanthus_altissima", "juglans_nigra", "prunus_serotina"},
        )
        self.assertEqual({spec.background_target for spec in specs}, {20000})
        self.assertEqual({spec.arm for spec in specs}, {"random", "domain_balanced"})
        self.assertTrue(all(spec.name.startswith("stage01c_") for spec in specs))

    def test_filters_select_without_changing_the_frozen_specs(self):
        specs = filter_specs(
            stage_specs("dataset-screen"),
            hosts=["vitis_labrusca"],
            arms=["domain_balanced"],
            targets=[10000],
        )
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].background_target, 10000)
        self.assertIn("stage01_vitis_labrusca_db_bg10000", specs[0].name)

    def test_incomplete_registry_is_reported_without_selecting_a_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "config.json"
            config.write_text(json.dumps({"csv_path": str(root / "missing.csv")}))
            registry = root / "experiment_registry.tsv"
            pd.DataFrame(
                [
                    {
                        "stage": "dataset-screen",
                        "host": "vitis_labrusca",
                        "arm": "random",
                        "background_target": 6000,
                        "split_design": "uniform",
                        "experiment": "not_run",
                        "config_path": str(config),
                        "run_dir": str(root / "run"),
                    }
                ]
            ).to_csv(registry, sep="\t", index=False)
            status = summarize_registry(registry, allow_incomplete=True)
            self.assertEqual(status["complete_gate_count"], 0)
            self.assertEqual(status["incomplete_gate_count"], 1)
            self.assertFalse(status["all_experiments_evidence_complete"])

    def test_platt_run_without_calibrated_evidence_fails_summary_gate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset.csv"
            dataset.write_text("split,presence\ntrain,1\n", encoding="utf-8")
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {"csv_path": str(dataset), "postfit_calibration": "platt"}
                ),
                encoding="utf-8",
            )
            run = root / "run"
            (run / "checkpoints").mkdir(parents=True)
            (run / "checkpoints" / "best.pt").write_bytes(b"checkpoint")
            (run / "run_status.json").write_text(
                json.dumps({"status": "completed"}), encoding="utf-8"
            )
            (run / "runtime_provenance.json").write_text(
                json.dumps(
                    {
                        "git_dirty": False,
                        "git_commit": "commit",
                        "config_sha256": sha256(config),
                    }
                ),
                encoding="utf-8",
            )
            (run / "training_design.json").write_text(
                json.dumps({"dataset_csv_sha256": sha256(dataset)}),
                encoding="utf-8",
            )
            pd.DataFrame([{"val_loss": 0.2}]).to_csv(
                run / "training_history.csv", index=False
            )
            pd.DataFrame([{"roc_auc_all_thresholds": 0.8}]).to_csv(
                run / "model_summary_metrics.csv", index=False
            )
            for name in (
                "validation_raw_calibration_metrics.json",
                "test_raw_calibration_metrics.json",
                "validation_threshold.json",
            ):
                (run / name).write_text(json.dumps({}), encoding="utf-8")
            registry = root / "experiment_registry.tsv"
            pd.DataFrame(
                [
                    {
                        "stage": "blockcv",
                        "host": "vitis_labrusca",
                        "arm": "domain_balanced",
                        "background_target": 10000,
                        "split_design": "blockcv_1_heldout_fold",
                        "experiment": "platt_missing",
                        "config_path": str(config),
                        "run_dir": str(run),
                    }
                ]
            ).to_csv(registry, sep="\t", index=False)
            (root / "design_metadata.json").write_text(
                json.dumps({"code_commit": "commit"}), encoding="utf-8"
            )
            status = summarize_registry(registry, allow_incomplete=True)
            self.assertEqual(status["complete_gate_count"], 0)
            metrics = pd.read_csv(root / "analysis" / "experiment_metrics.csv")
            self.assertIn("missing_platt", metrics.loc[0, "evidence_gate_failures"])

    def test_platt_json_metrics_are_the_primary_configured_test_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset.csv"
            dataset.write_text("split,presence\ntrain,1\n", encoding="utf-8")
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {"csv_path": str(dataset), "postfit_calibration": "platt"}
                ),
                encoding="utf-8",
            )
            run = root / "run"
            (run / "checkpoints").mkdir(parents=True)
            (run / "checkpoints" / "best.pt").write_bytes(b"checkpoint")
            (run / "run_status.json").write_text(
                json.dumps({"status": "completed"}), encoding="utf-8"
            )
            (run / "runtime_provenance.json").write_text(
                json.dumps(
                    {
                        "git_dirty": False,
                        "git_commit": "commit",
                        "config_sha256": sha256(config),
                    }
                ),
                encoding="utf-8",
            )
            (run / "training_design.json").write_text(
                json.dumps({"dataset_csv_sha256": sha256(dataset)}),
                encoding="utf-8",
            )
            pd.DataFrame([{"val_loss": 0.2}]).to_csv(
                run / "training_history.csv", index=False
            )
            for name, payload in (
                ("validation_raw_calibration_metrics.json", {}),
                ("test_raw_calibration_metrics.json", {}),
                ("validation_threshold.json", {}),
                ("platt_calibrator.json", {"method": "platt"}),
                (
                    "test_metrics_platt.json",
                    {"roc_auc_all_thresholds": 0.81, "brier_score": 0.18},
                ),
                ("test_platt_calibration_metrics.json", {}),
            ):
                (run / name).write_text(json.dumps(payload), encoding="utf-8")
            (run / "prediction_results_platt.csv").write_text(
                "true_label,probability\n1,0.8\n", encoding="utf-8"
            )
            registry = root / "experiment_registry.tsv"
            pd.DataFrame(
                [
                    {
                        "stage": "blockcv",
                        "host": "vitis_labrusca",
                        "arm": "random",
                        "background_target": 20000,
                        "split_design": "blockcv_1_heldout_fold",
                        "experiment": "platt_complete",
                        "config_path": str(config),
                        "run_dir": str(run),
                    }
                ]
            ).to_csv(registry, sep="\t", index=False)
            (root / "design_metadata.json").write_text(
                json.dumps({"code_commit": "commit"}), encoding="utf-8"
            )

            status = summarize_registry(registry, allow_incomplete=False)

            self.assertEqual(status["complete_gate_count"], 1)
            metrics = pd.read_csv(root / "analysis" / "experiment_metrics.csv")
            self.assertAlmostEqual(metrics.loc[0, "test_roc_auc_all_thresholds"], 0.81)
            self.assertAlmostEqual(metrics.loc[0, "test_brier_score"], 0.18)

    def test_fixed_environment_pair_metric_uses_matched_did(self):
        pairs = pd.DataFrame(
            [
                {
                    "pair_id": 7,
                    "tile_i": "cross_a",
                    "tile_j": "cross_b",
                    "control_tile_i": "control_a",
                    "control_tile_j": "control_b",
                    "date_pair": "20210919|20220415",
                    "mid_lon": -74.0,
                    "mid_lat": 43.0,
                }
            ]
        )
        probabilities = {
            "cross_a": 0.2,
            "cross_b": 0.8,
            "control_a": 0.4,
            "control_b": 0.5,
        }
        features = {
            "cross_a": np.array([1.0, 0.0]),
            "cross_b": np.array([0.0, 1.0]),
            "control_a": np.array([1.0, 0.0]),
            "control_b": np.array([1.0, 0.0]),
        }
        raw = {tile: value.copy() for tile, value in features.items()}
        result = pair_metrics("ny", pairs, features, probabilities, raw).iloc[0]
        self.assertAlmostEqual(result["fixed_env_cross_absdiff"], 0.6)
        self.assertAlmostEqual(result["fixed_env_control_absdiff"], 0.1)
        self.assertAlmostEqual(result["fixed_env_prediction_did"], 0.5)
        self.assertAlmostEqual(result["embedding_distance_did"], 1.0)

    def test_artifact_comparison_is_exactly_paired_and_reports_reduction(self):
        baseline = pd.DataFrame(
            {
                "pair_id": [1, 2, 3],
                "date_pair": ["a|b", "a|b", "c|d"],
                "baseline_did": [0.10, 0.20, 0.30],
            }
        )
        candidate = pd.DataFrame(
            {
                "pair_id": [3, 1, 2],
                "date_pair": ["c|d", "a|b", "a|b"],
                "fixed_env_prediction_did": [0.15, 0.05, 0.10],
            }
        )
        merged = exact_pair_merge(baseline, candidate)
        result = paired_comparison(merged, seed=4)
        self.assertEqual(result["pair_count"], 3)
        self.assertAlmostEqual(result["baseline_signed_did_mean"], 0.2)
        self.assertAlmostEqual(result["candidate_signed_did_mean"], 0.1)
        self.assertAlmostEqual(result["relative_signed_did_reduction"], 0.5)
        self.assertEqual(
            result["comparison_status"],
            "statistically_supported_signed_reduction",
        )

    def test_artifact_comparison_rejects_partial_pair_replay(self):
        baseline = pd.DataFrame(
            {
                "pair_id": [1, 2],
                "date_pair": ["a|b", "a|b"],
                "baseline_did": [0.1, 0.2],
            }
        )
        partial = pd.DataFrame(
            {
                "pair_id": [1],
                "date_pair": ["a|b"],
                "fixed_env_prediction_did": [0.05],
            }
        )
        with self.assertRaisesRegex(ValueError, "exact frozen pair set"):
            exact_pair_merge(baseline, partial)

    def test_batch_artifact_resume_requires_matching_complete_inputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "best.pt"
            config = root / "config.json"
            dataset = root / "dataset.csv"
            pairs = root / "pairs.csv"
            for path, content in (
                (checkpoint, b"checkpoint"),
                (config, b"{}"),
                (dataset, b"split,presence\ntrain,1\n"),
                (pairs, b"pair_id\n1\n"),
            ):
                path.write_bytes(content)
            output = root / "probe"
            (output / "june_baseline_comparison").mkdir(parents=True)
            for path in (
                output / "COMPLETED",
                output / "pair_metrics.csv",
                output / "june_baseline_comparison" / "COMPLETED",
            ):
                path.write_text("complete\n", encoding="utf-8")
            (output / "summary.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "checkpoint_sha256": sha256(checkpoint),
                        "config_sha256": sha256(config),
                        "dataset_sha256": sha256(dataset),
                        "pairs_sha256": sha256(pairs),
                        "score_space": "raw_sigmoid_probability",
                        "platt_calibrator_sha256": None,
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(
                reusable_completed_probe(output, checkpoint, config, dataset, pairs)
            )
            dataset.write_text("changed\n", encoding="utf-8")
            self.assertFalse(
                reusable_completed_probe(output, checkpoint, config, dataset, pairs)
            )

    def test_artifact_resume_rejects_score_space_or_calibrator_change(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "best.pt"
            config = root / "config.json"
            dataset = root / "dataset.csv"
            pairs = root / "pairs.csv"
            calibrator = root / "platt_calibrator.json"
            for path, content in (
                (checkpoint, b"checkpoint"),
                (config, b"{}"),
                (dataset, b"split,presence\ntrain,1\n"),
                (pairs, b"pair_id\n1\n"),
                (calibrator, b'{"method":"platt","slope":1,"intercept":0}'),
            ):
                path.write_bytes(content)
            output = root / "probe"
            (output / "june_baseline_comparison").mkdir(parents=True)
            for path in (
                output / "COMPLETED",
                output / "pair_metrics.csv",
                output / "june_baseline_comparison" / "COMPLETED",
            ):
                path.write_text("complete\n", encoding="utf-8")
            calibrator_hash = sha256(calibrator)
            (output / "summary.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "checkpoint_sha256": sha256(checkpoint),
                        "config_sha256": sha256(config),
                        "dataset_sha256": sha256(dataset),
                        "pairs_sha256": sha256(pairs),
                        "score_space": "platt_calibrated_probability",
                        "platt_calibrator_sha256": calibrator_hash,
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(
                reusable_completed_probe(
                    output,
                    checkpoint,
                    config,
                    dataset,
                    pairs,
                    score_space="platt_calibrated_probability",
                    platt_calibrator_sha256=calibrator_hash,
                )
            )
            self.assertFalse(
                reusable_completed_probe(
                    output, checkpoint, config, dataset, pairs
                )
            )

    def test_exact_overlap_resume_rejects_score_space_or_calibrator_change(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "best.pt"
            config = root / "config.json"
            dataset = root / "dataset.csv"
            for path, content in (
                (checkpoint, b"checkpoint"),
                (config, b"{}"),
                (dataset, b"split,presence\ntrain,1\n"),
            ):
                path.write_bytes(content)
            candidate = {
                "checkpoint_path": checkpoint,
                "config_path_object": config,
                "dataset_path": dataset,
            }
            output = root / "exact"
            output.mkdir()
            for name in (
                "COMPLETED",
                "exact_overlap_candidate_metrics.csv",
                "paired_june_comparison.csv",
            ):
                (output / name).write_text("complete\n", encoding="utf-8")
            (output / "summary.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "checkpoint_sha256": sha256(checkpoint),
                        "config_sha256": sha256(config),
                        "dataset_sha256": sha256(dataset),
                        "baseline_metrics_sha256": "baseline-hash",
                        "score_space": "raw_sigmoid_probability",
                        "platt_calibrator_sha256": None,
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(
                reusable_exact_overlap_output(
                    output, candidate, "baseline-hash"
                )
            )
            self.assertFalse(
                reusable_exact_overlap_output(
                    output,
                    candidate,
                    "baseline-hash",
                    score_space="platt_calibrated_probability",
                    platt_calibrator_sha256="calibrator-hash",
                )
            )

    def test_dataset_screen_pareto_flags_do_not_use_an_arbitrary_score(self):
        candidates = pd.DataFrame(
            {
                "test_roc_auc_all_thresholds": [0.90, 0.92, 0.89, 0.91],
                "test_brier_score": [0.12, 0.11, 0.14, 0.09],
                "artifact_candidate_absolute_did_mean": [0.05, 0.04, 0.08, 0.06],
            }
        )
        objectives = {
            "test_roc_auc_all_thresholds": "max",
            "test_brier_score": "min",
            "artifact_candidate_absolute_did_mean": "min",
        }
        self.assertEqual(
            pareto_nondominated(candidates, objectives).tolist(),
            [False, True, False, True],
        )

    def test_dataset_screen_synthesis_requires_and_joins_complete_evidence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            experiments = ["candidate_random", "candidate_domain_balanced"]
            training_rows = []
            for index, experiment in enumerate(experiments):
                training_rows.append(
                    {
                        "experiment": experiment,
                        "evidence_gate_passed": "True",
                        "evidence_gate_failures": "",
                        "stage": "dataset-screen",
                        "host": "vitis_labrusca",
                        "arm": "random" if index == 0 else "domain_balanced",
                        "background_target": 6000,
                        "best_checkpoint_sha256": f"checkpoint-{index}",
                        "training_dataset_csv_sha256": f"dataset-{index}",
                        "test_roc_auc_all_thresholds": 0.90 + index * 0.01,
                        "test_average_precision_all_thresholds": 0.88 + index * 0.01,
                        "test_brier_score": 0.10 - index * 0.01,
                        "test_calibration_binary_log_loss": 0.30 - index * 0.01,
                    }
                )
            training_path = root / "training.csv"
            pd.DataFrame(training_rows).to_csv(training_path, index=False)

            artifact_root = root / "artifacts"
            completed = []
            for index, experiment in enumerate(experiments):
                candidate = artifact_root / experiment
                comparison = candidate / "june_baseline_comparison"
                comparison.mkdir(parents=True)
                probe = {
                    "status": "complete",
                    "candidate_experiment": experiment,
                    "checkpoint_sha256": f"checkpoint-{index}",
                    "dataset_sha256": f"dataset-{index}",
                    "score_space": "raw_sigmoid_probability",
                    "platt_calibrator_sha256": None,
                }
                comparison_summary = {
                    "status": "complete",
                    "diagnostic": "ny",
                    "taxon": "vitis_labrusca",
                    "pair_count": 253,
                    "baseline_signed_did_mean": 0.08,
                    "candidate_signed_did_mean": 0.06 - index * 0.01,
                    "paired_signed_did_reduction_mean": 0.02 + index * 0.01,
                    "paired_signed_did_reduction_bootstrap_95ci": [0.01, 0.03],
                    "relative_signed_did_reduction": 0.25 + index * 0.1,
                    "baseline_absolute_did_mean": 0.09,
                    "candidate_absolute_did_mean": 0.07 - index * 0.01,
                    "paired_absolute_did_reduction_mean": 0.02 + index * 0.01,
                    "paired_absolute_did_reduction_bootstrap_95ci": [0.01, 0.03],
                    "comparison_status": "statistically_supported_signed_reduction",
                    "candidate_checkpoint_sha256": f"checkpoint-{index}",
                    "candidate_dataset_sha256": f"dataset-{index}",
                }
                (candidate / "summary.json").write_text(
                    json.dumps(probe), encoding="utf-8"
                )
                (comparison / "summary.json").write_text(
                    json.dumps(comparison_summary), encoding="utf-8"
                )
                (candidate / "COMPLETED").write_text("complete\n", encoding="utf-8")
                (comparison / "COMPLETED").write_text(
                    "complete\n", encoding="utf-8"
                )
                completed.append(experiment)
            (artifact_root / "batch_status.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "candidate_count": 2,
                        "completed_experiments": completed,
                    }
                ),
                encoding="utf-8",
            )
            (artifact_root / "COMPLETED").write_text("complete\n", encoding="utf-8")
            exact_root = root / "exact_overlap"
            for index, experiment in enumerate(experiments):
                candidate = exact_root / experiment
                candidate.mkdir(parents=True)
                (candidate / "summary.json").write_text(
                    json.dumps(
                        {
                            "status": "complete",
                            "host": "vitis_labrusca",
                            "pair_count": 25,
                            "baseline_pa_minus_ny_probability_mean": -0.5,
                            "candidate_pa_minus_ny_probability_mean": -0.3,
                            "baseline_absolute_probability_shift_mean": 0.5,
                            "candidate_absolute_probability_shift_mean": 0.3,
                            "paired_absolute_probability_shift_reduction_mean": 0.2,
                            "paired_absolute_probability_shift_reduction_bootstrap_95ci": [
                                0.1,
                                0.3,
                            ],
                            "absolute_shift_reduction_supported": True,
                            "checkpoint_sha256": f"checkpoint-{index}",
                            "dataset_sha256": f"dataset-{index}",
                            "score_space": "raw_sigmoid_probability",
                            "platt_calibrator_sha256": None,
                        }
                    ),
                    encoding="utf-8",
                )
                (candidate / "COMPLETED").write_text(
                    "complete\n", encoding="utf-8"
                )
            (exact_root / "batch_status.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "candidate_count": 2,
                        "completed_experiments": experiments,
                    }
                ),
                encoding="utf-8",
            )
            (exact_root / "COMPLETED").write_text("complete\n", encoding="utf-8")
            status = synthesize(
                training_path,
                [artifact_root],
                root / "synthesis",
                exact_overlap_roots=[exact_root],
            )
            self.assertEqual(status["experiment_count"], 2)
            self.assertEqual(len(status["exact_overlap_roots"]), 1)
            self.assertTrue(
                (root / "synthesis" / "combined_candidate_evidence.csv").is_file()
            )

    def test_radiometric_recipe_uses_conservative_empirical_caps(self):
        metrics = {
            "quantiles": {
                "absolute_scene_gain_change": {"q50": 0.12},
                "absolute_contrast_ratio_change": {"q50": 0.06},
                "absolute_band_gain_residual_change": {"q25": 0.03},
                "absolute_mean_ndvi_change": {"q90": 0.05},
            }
        }
        recipe = derive_recipe(metrics)
        self.assertEqual(recipe["scene_gain"], 0.08)
        self.assertEqual(recipe["contrast"], 0.06)
        self.assertEqual(recipe["band_gain"], 0.02)
        self.assertEqual(recipe["max_mean_ndvi_shift"], 0.05)

    def test_dataset_screen_evidence_join_requires_all_five_hosts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dirs = [root / "initial", root / "controls"]
            host_groups = [
                {"vitis_labrusca", "notholithocarpus_densiflorus"},
                HOSTS - {"vitis_labrusca", "notholithocarpus_densiflorus"},
            ]
            for source_dir, hosts in zip(source_dirs, host_groups):
                source_dir.mkdir()
                rows = [
                    {
                        "host": host,
                        "arm": "domain_balanced",
                        "background_target": 6000,
                        "experiment": f"dataset_screen_{host}",
                        "evidence_gate_passed": True,
                        "pareto_nondominated_within_host": True,
                        "sentinel_metric": 0.25,
                    }
                    for host in sorted(hosts)
                ]
                pd.DataFrame(rows).to_csv(
                    source_dir / "combined_candidate_evidence.csv", index=False
                )
                (source_dir / "summary.json").write_text(
                    json.dumps({"status": "complete"}), encoding="utf-8"
                )
                (source_dir / "COMPLETED").write_text(
                    "completed\n", encoding="utf-8"
                )
            output_dir = root / "unified"
            payload = combine_evidence(source_dirs, output_dir)
            combined = pd.read_csv(output_dir / "combined_candidate_evidence.csv")
            self.assertEqual(payload["host_count"], 5)
            self.assertEqual(set(combined["host"]), HOSTS)
            self.assertTrue((combined["sentinel_metric"] == 0.25).all())
            self.assertTrue((output_dir / "COMPLETED").is_file())

    def test_dataset_selection_freezes_exact_artifact_hashes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            evidence = root / "evidence"
            evidence.mkdir()
            rows = []
            choices = []
            rationales = {}
            checkpoint_by_host = {}
            for host in sorted(HOSTS):
                run_dir = root / host
                dataset = run_dir / "dataset.csv"
                checkpoint = run_dir / "checkpoints" / "best.pt"
                config = run_dir / "config.json"
                dataset.parent.mkdir(parents=True)
                dataset.write_text("split,presence\ntrain,1\n", encoding="utf-8")
                checkpoint.parent.mkdir()
                checkpoint.write_bytes(f"checkpoint-{host}".encode())
                source_config = {
                    "csv_path": str(dataset),
                    "image_dir": str(run_dir),
                    "loss_strategy": "ordinary_bce",
                    "spatial_augmentation": "dihedral",
                    "radiometric_augmentation": False,
                    "dropout": 0.25,
                    "label_smoothing": 0.0,
                    "august_experiment_spec": {
                        "host": host,
                        "arm": "domain_balanced",
                        "background_target": 6000,
                    },
                }
                config.write_text(json.dumps(source_config), encoding="utf-8")
                experiment = f"ordinary_baseline_{host}"
                rows.append(
                    {
                        "host": host, "arm": "domain_balanced",
                        "background_target": 6000, "experiment": experiment,
                        "evidence_gate_passed": True,
                        "pareto_nondominated_within_host": True,
                        "config_path": str(config), "config_sha256": sha256(config),
                        "run_dir": str(run_dir),
                        "best_checkpoint_sha256": sha256(checkpoint),
                        "training_dataset_csv_sha256": sha256(dataset),
                        "artifact_comparison_summary": "comparison.json",
                        "artifact_relative_signed_did_reduction": 0.4,
                        "artifact_candidate_absolute_did_mean": 0.04,
                        "test_roc_auc_all_thresholds": 0.95,
                        "test_brier_score": 0.08,
                    }
                )
                choices.append((host, "domain_balanced", 6000))
                rationales[host] = "Best seam/fit Pareto tradeoff."
                checkpoint_by_host[host] = checkpoint
            combined = pd.DataFrame(rows)
            combined.to_csv(evidence / "combined_candidate_evidence.csv", index=False)
            (evidence / "summary.json").write_text(
                json.dumps({"status": "complete"}), encoding="utf-8"
            )
            (evidence / "COMPLETED").write_text("complete\n", encoding="utf-8")
            output = root / "selection.json"
            payload = freeze_selection(
                evidence,
                choices,
                rationales,
                output,
                "test-commit",
            )
            self.assertEqual(payload["selection_count"], 5)
            selected_vitis = next(
                item for item in payload["selections"] if item["host"] == "vitis_labrusca"
            )
            self.assertEqual(
                selected_vitis["source_checkpoint_sha256"],
                sha256(checkpoint_by_host["vitis_labrusca"]),
            )

    def test_loss_screen_reuses_ordinary_baseline_and_isolates_strategy(self):
        selection = {
            "host": "vitis_labrusca",
            "arm": "domain_balanced",
            "background_target": 10000,
            "source_experiment": "ordinary_baseline",
            "source_checkpoint": "/tmp/best.pt",
            "source_checkpoint_sha256": "checkpoint-hash",
        }
        source = {
            "experiment": "ordinary_baseline",
            "output_dir": "/tmp/old",
            "epochs": 15,
            "loss_strategy": "ordinary_bce",
            "spatial_augmentation": "dihedral",
            "radiometric_augmentation": False,
            "dropout": 0.25,
            "label_smoothing": 0.0,
            "seed": 42,
        }
        config = build_loss_config(
            source,
            selection,
            "pos_weight_bce",
            Path("/tmp/stage"),
            Path("/tmp/selection.json"),
            "selection-hash",
        )
        self.assertEqual(config["loss_strategy"], "pos_weight_bce")
        self.assertEqual(config["label_smoothing"], 0.0)
        self.assertFalse(config["radiometric_augmentation"])
        self.assertIn("posw", loss_experiment_name(selection, "pos_weight_bce"))

    def test_exact_overlap_comparison_uses_paired_absolute_shift_reduction(self):
        baseline = pd.DataFrame(
            {
                "chip_id": [0, 1, 2],
                "x": [1.0, 2.0, 3.0],
                "y": [4.0, 5.0, 6.0],
                "pa_minus_ny_probability": [-0.5, -0.3, -0.1],
            }
        )
        candidate = pd.DataFrame(
            {
                "chip_id": [2, 0, 1],
                "x": [3.0, 1.0, 2.0],
                "y": [6.0, 4.0, 5.0],
                "pa_minus_ny_probability": [-0.05, -0.2, -0.2],
            }
        )
        paired, summary = compare_exact_overlap(baseline, candidate)
        self.assertEqual(len(paired), 3)
        self.assertAlmostEqual(
            summary["paired_absolute_probability_shift_reduction_mean"],
            0.15,
        )
        self.assertTrue(summary["absolute_shift_reduction_supported"])

    def test_materialization_attrition_audit_measures_domain_replacement_shift(self):
        total_variation, maximum_shift = distribution_shift(
            pd.Series(["a", "a", "b", "b"]),
            pd.Series(["a", "b", "b", "b"]),
        )
        self.assertAlmostEqual(total_variation, 0.25)
        self.assertAlmostEqual(maximum_shift, 0.25)

    def test_refinalization_preserves_acquisition_dates_as_identifiers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            records = Path(temp_dir) / "materialization_records.csv"
            records.write_text(
                "sample_id,materialization_status,actual_source_acq_date\n"
                "success_1,success,20220902\n"
                "failure_1,failed,\n",
                encoding="utf-8",
            )
            loaded = read_materialization_records(records)
            self.assertEqual(loaded.loc[0, "actual_source_acq_date"], "20220902")
            self.assertTrue(pd.isna(loaded.loc[1, "actual_source_acq_date"]))

    def test_materialization_retry_log_collapses_to_verified_final_state(self):
        attempts = pd.DataFrame(
            [
                {"sample_id": "a", "materialization_status": "failed", "failure_type": "edge"},
                {"sample_id": "b", "materialization_status": "success", "failure_type": None},
                {"sample_id": "a", "materialization_status": "success", "failure_type": None},
                {"sample_id": "c", "materialization_status": "failed", "failure_type": "edge"},
                {"sample_id": "c", "materialization_status": "failed", "failure_type": "edge"},
            ]
        )
        canonical = attempts.iloc[[1, 2, 4]].sort_values("sample_id").reset_index(drop=True)
        final, retry, failures = canonicalize_progress_attempts(attempts, canonical)
        self.assertFalse(failures)
        self.assertEqual(len(final), 3)
        self.assertEqual(retry["retried_sample_ids"], 2)
        self.assertEqual(retry["failed_then_succeeded_sample_ids"], 1)
        self.assertEqual(retry["repeated_terminal_failures"], 1)

    def test_materialization_attrition_audit_passes_exact_synthetic_build(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            candidate_root = root / "candidates"
            materialized_root = root / "materialized"
            host = "vitis_labrusca"
            package = candidate_root / host
            dataset = materialized_root / host
            package.mkdir(parents=True)
            dataset.mkdir(parents=True)
            rows = [
                {
                    "sample_id": "presence_1",
                    "presence": 1,
                    "source_domain_id": "A|2021|20210101",
                    "random_nested_rank": 0,
                    "domain_balanced_nested_rank": 0,
                },
                {
                    "sample_id": "presence_2",
                    "presence": 1,
                    "source_domain_id": "B|2021|20210102",
                    "random_nested_rank": 0,
                    "domain_balanced_nested_rank": 0,
                },
            ]
            for index, domain in enumerate(["A", "B", "C", "D"], start=1):
                rows.append(
                    {
                        "sample_id": f"background_{index}",
                        "presence": 0,
                        "source_domain_id": f"{domain}|2022|20220101",
                        "random_nested_rank": index,
                        "domain_balanced_nested_rank": [2, 1, 4, 3][index - 1],
                    }
                )
            candidates = pd.DataFrame(rows)
            candidates.to_csv(package / "extraction_candidates.csv", index=False)
            (package / "manifest_metadata.json").write_text(
                json.dumps({"targets": [2]}), encoding="utf-8"
            )
            progress = pd.DataFrame(
                {
                    "sample_id": candidates["sample_id"],
                    "materialization_status": "success",
                    "failure_stage": None,
                    "failure_type": None,
                    "source_provenance_match": True,
                    "intended_source_used": True,
                    "actual_multi_acquisition": False,
                    "actual_multi_state": False,
                }
            )
            progress.to_json(
                dataset / "materialization_progress.jsonl",
                orient="records",
                lines=True,
            )
            progress.to_csv(dataset / "materialization_records.csv", index=False)
            (dataset / "materialization_run.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "manifest_status": {"production_ready_manifests": True},
                    }
                ),
                encoding="utf-8",
            )
            for arm, backgrounds in {
                "random": ["background_1", "background_2"],
                "domain_balanced": ["background_2", "background_1"],
            }.items():
                target_root = dataset / "model_manifests" / arm / "background_00002"
                target_root.mkdir(parents=True)
                selected_ids = ["presence_1", "presence_2", *backgrounds]
                candidates[candidates["sample_id"].isin(selected_ids)].to_csv(
                    target_root / "uniform_dataset.csv", index=False
                )
            with (
                patch.object(attrition_audit, "CANDIDATE_ROOT", candidate_root),
                patch.object(attrition_audit, "MATERIALIZED_ROOT", materialized_root),
            ):
                summary = attrition_audit.audit_host(host, root / "audit")
            self.assertTrue(summary["production_ready"])
            self.assertEqual(summary["failed_rows"], 0)

    def test_stage00_smoke_gate_requires_complete_finite_evidence(self):
        metrics = pd.DataFrame(
            [
                {
                    "evidence_gate_passed": True,
                    "epochs_completed": 2,
                    "test_roc_auc_all_thresholds": 0.8,
                    "test_brier_score": 0.2,
                    "test_calibration_binary_log_loss": 0.5,
                }
            ]
        )
        self.assertTrue(smoke_evidence_ready(metrics))
        metrics.loc[0, "test_brier_score"] = np.nan
        self.assertFalse(smoke_evidence_ready(metrics))

    def test_control_screen_waits_for_materialization_stage01_and_all_audits(self):
        audits = {
            "ailanthus_altissima": True,
            "juglans_nigra": True,
            "prunus_serotina": True,
        }
        self.assertTrue(
            control_gates_ready(
                "control_materializations_completed",
                "stage01_dataset_screen_evidence_completed",
                audits,
            )
        )
        audits["prunus_serotina"] = False
        self.assertFalse(
            control_gates_ready(
                "control_materializations_completed",
                "stage01_dataset_screen_evidence_completed",
                audits,
            )
        )

    def test_loss_evidence_gate_rejects_incomplete_candidate(self):
        complete = pd.DataFrame(
            [{"experiment": "a", "evidence_gate_passed": "True"}]
        )
        strict_evidence_gate(complete)
        incomplete = pd.DataFrame(
            [
                {
                    "experiment": "a",
                    "evidence_gate_passed": "False",
                    "evidence_gate_failures": "missing:checkpoint",
                }
            ]
        )
        with self.assertRaisesRegex(RuntimeError, "Loss training evidence"):
            strict_evidence_gate(incomplete)

    def test_augmentation_screen_is_seven_nonbaseline_factorial_arms(self):
        combinations = augmentation_combinations()
        self.assertEqual(len(combinations), 7)
        self.assertNotIn(
            {"radiometric": False, "dropout": 0.25, "label_smoothing": 0.0},
            combinations,
        )
        self.assertIn(
            {"radiometric": True, "dropout": 0.35, "label_smoothing": 0.05},
            combinations,
        )
        source = {
            "seed": 42,
            "loss_strategy": "ordinary_bce",
            "spatial_augmentation": "dihedral",
            "radiometric_augmentation": False,
            "dropout": 0.25,
            "label_smoothing": 0.0,
        }
        selection = {
            "host": "vitis_labrusca",
            "arm": "domain_balanced",
            "background_target": 10000,
            "loss_strategy": "ordinary_bce",
            "source_experiment": "baseline",
            "source_checkpoint_sha256": "checkpoint",
        }
        config = build_augmentation_config(
            source,
            selection,
            {"radiometric": True, "dropout": 0.35, "label_smoothing": 0.05},
            {"enabled": True, "scene_gain": 0.08},
            Path("/tmp/stage"),
            Path("/tmp/selection.json"),
            "selection-hash",
            "calibration-hash",
        )
        self.assertEqual(config["dropout"], 0.35)
        self.assertEqual(config["label_smoothing"], 0.05)
        self.assertTrue(config["radiometric_augmentation"]["enabled"])

    def test_blockcv_design_has_five_named_heldout_folds(self):
        selection = {
            "host": "vitis_labrusca",
            "arm": "domain_balanced",
            "background_target": 10000,
            "loss_strategy": "ordinary_bce",
            "radiometric_augmentation": {"enabled": True},
            "dropout": 0.35,
            "label_smoothing": 0.05,
        }
        self.assertEqual(FOLDS, [1, 2, 3, 4, 5])
        names = [blockcv_name(selection, fold) for fold in FOLDS]
        self.assertEqual(len(set(names)), 5)
        self.assertTrue(names[0].endswith("blockcv1_20260813"))

    def test_blockcv_summary_requires_five_finite_values(self):
        summary = summarize_values(pd.Series([0.7, 0.8, 0.9, 0.85, 0.75]), "auc")
        self.assertAlmostEqual(summary["auc_median"], 0.8)
        with self.assertRaises(ValueError):
            summarize_values(pd.Series([0.7, 0.8]), "auc")

    def test_blockcv_calibration_rule_is_paired_and_conservative(self):
        group = pd.DataFrame(
            {
                "test_calibration_brier_score": [0.20, 0.21, 0.19, 0.22, 0.18],
                "test_platt_calibration_brier_score": [0.18, 0.20, 0.18, 0.23, 0.17],
                "test_calibration_binary_log_loss": [0.60, 0.61, 0.58, 0.63, 0.55],
                "test_platt_calibration_binary_log_loss": [0.56, 0.58, 0.55, 0.64, 0.52],
            }
        )
        recommendation = calibration_recommendation(group)
        self.assertEqual(recommendation["calibration_recommendation"], "platt")
        self.assertEqual(recommendation["folds_platt_nonworse_both"], 4)
        group.loc[1:4, "test_platt_calibration_binary_log_loss"] += 0.2
        recommendation = calibration_recommendation(group)
        self.assertEqual(recommendation["calibration_recommendation"], "raw")

    def test_blockcv_acceptance_reuses_only_complete_hash_matched_evidence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            selection = root / "selection.json"
            training = root / "training.csv"
            selection.write_text("{}\n", encoding="utf-8")
            training.write_text("experiment\nexample\n", encoding="utf-8")
            evidence = root / "evidence"
            evidence.mkdir()
            folds = evidence / "blockcv_fold_metrics.csv"
            hosts = evidence / "blockcv_host_summary.csv"
            folds.write_text("fold\n1\n", encoding="utf-8")
            hosts.write_text("host\nexample\n", encoding="utf-8")
            (evidence / "COMPLETED").write_text("completed\n", encoding="utf-8")
            summary = {
                "status": "complete",
                "production_ready": True,
                "selection_manifest_sha256": sha256(selection),
                "training_metrics_sha256": sha256(training),
                "host_count": 5,
                "expected_fold_count_per_host": 5,
                "fold_metrics": str(folds),
                "host_summary": str(hosts),
            }
            (evidence / "summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            reused = _reuse_evidence(evidence, selection, training)
            self.assertTrue(reused["production_ready"])
            training.write_text("experiment\nchanged\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                _reuse_evidence(evidence, selection, training)

    def test_production_name_records_recipe_and_score_space(self):
        selection = {
            "host": "vitis_labrusca",
            "arm": "domain_balanced",
            "background_target": 10000,
            "loss_strategy": "pos_weight_bce",
            "radiometric_augmentation": {"enabled": True},
            "dropout": 0.35,
            "label_smoothing": 0.05,
            "production_postfit_calibration": "platt",
        }
        name = production_name(selection)
        self.assertIn("posw_rad1_d035_s005_platt", name)
        self.assertTrue(name.endswith("20260813"))


if __name__ == "__main__":
    unittest.main()
