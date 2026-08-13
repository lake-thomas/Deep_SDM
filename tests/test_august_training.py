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
from august2026_experiments import filter_specs, stage_specs
from august2026_compare_artifact_probe import (
    exact_pair_merge,
    paired_comparison,
)
from august2026_batch_artifact_probe import reusable_completed_probe
from august2026_dataset_screen_evidence import pareto_nondominated, synthesize
from august2026_dataset_overlap_audit import build_overlap_audit, prepare_dataset
from august2026_cross_host_pathway_audit import correlation_metrics, residualize
from august2026_final_model_acceptance import artifact_nonregression_gate
from august2026_aligned_raster_correlation import aligned_raster_metrics
from august2026_candidate_builder_provenance_audit import commit_matches
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
from august2026_materialization_attrition_audit import distribution_shift
from august2026_wait_for_dataset_screen import smoke_evidence_ready
from august2026_wait_for_control_screen import control_gates_ready
from august2026_loss_screen_evidence import strict_evidence_gate
from august2026_augmentation_screen import augmentation_combinations, build_config as build_augmentation_config
from august2026_blockcv import FOLDS, blockcv_name
from august2026_blockcv_evidence import calibration_recommendation, summarize_values
from august2026_production_train import production_name
from august2026_fixed_env_artifact_probe import pair_metrics, sha256
from august2026_summarize_experiments import summarize_registry
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


class AugustExperimentDesignTests(unittest.TestCase):
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
            )
            self.assertTrue(result["passed"], result["failures"])

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

    def test_dataset_selection_freezes_exact_artifact_hashes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset.csv"
            checkpoint = root / "checkpoints" / "best.pt"
            config = root / "config.json"
            dataset.write_text("split,presence\ntrain,1\n", encoding="utf-8")
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"checkpoint")
            source_config = {
                "csv_path": str(dataset),
                "image_dir": str(root),
                "loss_strategy": "ordinary_bce",
                "spatial_augmentation": "dihedral",
                "radiometric_augmentation": False,
                "dropout": 0.25,
                "label_smoothing": 0.0,
                "august_experiment_spec": {
                    "host": "vitis_labrusca",
                    "arm": "domain_balanced",
                    "background_target": 6000,
                },
            }
            config.write_text(json.dumps(source_config), encoding="utf-8")
            evidence = root / "evidence"
            evidence.mkdir()
            combined = pd.DataFrame(
                [
                    {
                        "host": "vitis_labrusca",
                        "arm": "domain_balanced",
                        "background_target": 6000,
                        "experiment": "ordinary_baseline",
                        "evidence_gate_passed": True,
                        "pareto_nondominated_within_host": True,
                        "config_path": str(config),
                        "config_sha256": sha256(config),
                        "run_dir": str(root),
                        "best_checkpoint_sha256": sha256(checkpoint),
                        "training_dataset_csv_sha256": sha256(dataset),
                        "artifact_comparison_summary": "comparison.json",
                        "artifact_relative_signed_did_reduction": 0.4,
                        "artifact_candidate_absolute_did_mean": 0.04,
                        "test_roc_auc_all_thresholds": 0.95,
                        "test_brier_score": 0.08,
                    }
                ]
            )
            combined.to_csv(evidence / "combined_candidate_evidence.csv", index=False)
            (evidence / "summary.json").write_text(
                json.dumps({"status": "complete"}), encoding="utf-8"
            )
            (evidence / "COMPLETED").write_text("complete\n", encoding="utf-8")
            output = root / "selection.json"
            payload = freeze_selection(
                evidence,
                [("vitis_labrusca", "domain_balanced", 6000)],
                {"vitis_labrusca": "Best seam/fit Pareto tradeoff."},
                output,
                "test-commit",
            )
            self.assertEqual(payload["selection_count"], 1)
            self.assertEqual(
                payload["selections"][0]["source_checkpoint_sha256"],
                sha256(checkpoint),
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
                "test_brier_score": [0.20, 0.21, 0.19, 0.22, 0.18],
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
