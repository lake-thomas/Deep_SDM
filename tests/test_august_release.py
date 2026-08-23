import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import numpy as np
import rasterio
from rasterio.transform import from_origin

from august2026_augmentation_screen_evidence import (
    EFFECT_METRICS,
    add_augmentation_pareto_flags,
    factorial_effect_contrasts,
    fill_missing_tanoak_coastal_evidence,
)
import august2026_batch_artifact_probe as batch_artifact_probe
from august2026_batch_artifact_probe import add_tanoak_coastal_comparison
from august2026_record_visual_review import HOSTS, REVIEW_TARGETS, record_visual_review
from august2026_final_model_acceptance import resolve_production_selection
from august2026_assemble_ailanthus_replacement_inputs import (
    combine_by_host,
    combine_selections,
)
from august2026_native_mosaic_parity import run_parity, stable_tile_sample
from august2026_render_fixed_quickviews import read_aligned_reference
from august2026_run_augmentation_acceptance import (
    _validate_resume_identity,
    probe_commands,
)
from august2026_run_final_model_acceptance import (
    _resume_status,
    adopt_completed_support_step,
    final_probe_commands,
    record_failed_final_status,
    run_command_step,
)
from august2026_run_ailanthus_replacement_acceptance import (
    probe_commands as replacement_probe_commands,
)
import august2026_run_ailanthus_blockcv_stage as ailanthus_blockcv_stage
import august2026_run_ailanthus_production_stage as ailanthus_production_stage
import august2026_run_ailanthus_five_host_promotion as ailanthus_five_host_promotion
import august2026_run_mc_stability_stage as mc_stability_stage
from august2026_prepare_dgx_inference_bundle import (
    validate_mc_lineage,
    visual_review_template,
)
from august2026_release_evidence import (
    _index_dgx_run_chain,
    assemble_release_evidence,
)
from august2026_summarize_experiments import read_json, sha256
from august2026_validate_mosaics import validate_mosaics
from hazel_august2026.validate_submission_gate import (
    validate_bundle,
    validate_inference_qa,
)
from dgx_august2026.validate_dgx_bundle import (
    validate_inference_qa as validate_dgx_inference_qa,
)
from dgx_august2026.project_full_runtime import synthesize_projection


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_minimal_hazel_bundle(bundle: Path, registry: Path | None = None) -> None:
    bundle.mkdir()
    if registry is None:
        sentinel = bundle / "sentinel.txt"
        sentinel.write_text("frozen\n", encoding="utf-8")
        hashes = {sentinel.name: sha256(sentinel)}
        source_registry_hash = None
    else:
        bundled_registry = bundle / "production_experiment_registry.tsv"
        bundled_registry.write_bytes(registry.read_bytes())
        hashes = {bundled_registry.name: sha256(bundled_registry)}
        source_registry_hash = sha256(registry)
    write_json(
        bundle / "bundle_metadata.json",
        {
            "status": "complete",
            "bundle_type": "accepted_august2026_hazel_hann_inference",
            "source_registry_sha256": source_registry_hash,
            "file_sha256": hashes,
        },
    )
    (bundle / "COMPLETED").write_text("completed\n", encoding="utf-8")


class AugustVisualReviewTests(unittest.TestCase):
    def test_dgx_visual_review_template_is_complete_but_non_approving(self):
        template = visual_review_template()
        self.assertEqual(template["status"], "template_pending_human_review")
        self.assertEqual(template["reviewer"], "")
        self.assertEqual(template["reviewed_at"], "")
        reviews = template["reviews"]
        self.assertEqual({item["host"] for item in reviews}, HOSTS)
        self.assertTrue(all(item["decision"] == "pending" for item in reviews))
        self.assertTrue(all(set(item["checks"]) == REVIEW_TARGETS for item in reviews))
        self.assertTrue(
            all(set(item["checks"].values()) == {"pending"} for item in reviews)
        )

    def test_aligned_june_reference_keeps_nonoverlap_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prediction = root / "august.tif"
            reference = root / "june.tif"
            with rasterio.open(
                prediction, "w", driver="GTiff", width=4, height=2, count=1,
                dtype="float32", crs="EPSG:5070", transform=from_origin(0, 2, 1, 1),
                nodata=-9999,
            ) as target:
                target.write(np.full((2, 4), 0.4, dtype="float32"), 1)
            with rasterio.open(
                reference, "w", driver="GTiff", width=2, height=2, count=1,
                dtype="float32", crs="EPSG:5070", transform=from_origin(0, 2, 1, 1),
                nodata=-9999,
            ) as target:
                target.write(np.full((2, 2), 0.2, dtype="float32"), 1)
            aligned = read_aligned_reference(reference, prediction, 4, 2)
            self.assertTrue(np.allclose(aligned[:, :2], 0.2))
            self.assertTrue(np.isnan(aligned[:, 2:]).all())

    def test_native_mosaic_parity_is_stable_and_checks_pointwise_differences(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tile_dir = root / "tiles"
            prediction_dir = root / "prediction"
            uncertainty_dir = root / "uncertainty"
            tile_dir.mkdir()
            prediction_dir.mkdir()
            uncertainty_dir.mkdir()
            tile_paths = []
            for index, value in enumerate((0.2, 0.7)):
                tile = tile_dir / f"tile_ne_17_{index:03d}.tif"
                tile_paths.append(tile)
                transform = from_origin(index * 4, 4, 1, 1)
                for path, layer_value in (
                    (tile, value),
                    (prediction_dir / f"{tile.stem}_prediction.tif", value),
                    (uncertainty_dir / f"{tile.stem}_uncertainty.tif", 0.03),
                ):
                    with rasterio.open(
                        path, "w", driver="GTiff", width=4, height=4, count=1,
                        dtype="float32", crs="EPSG:5070", transform=transform,
                        nodata=-9999,
                    ) as target:
                        target.write(np.full((4, 4), layer_value, dtype="float32"), 1)
            manifest = root / "tiles.txt"
            manifest.write_text("\n".join(map(str, reversed(tile_paths))) + "\n", encoding="utf-8")
            self.assertEqual(
                stable_tile_sample(tile_paths, 2),
                stable_tile_sample(list(reversed(tile_paths)), 2),
            )
            prediction_mosaic = root / "prediction_mosaic.tif"
            uncertainty_mosaic = root / "uncertainty_mosaic.tif"
            for path, left, right in (
                (prediction_mosaic, 0.2, 0.7),
                (uncertainty_mosaic, 0.03, 0.03),
            ):
                values = np.hstack(
                    [np.full((4, 4), left, dtype="float32"),
                     np.full((4, 4), right, dtype="float32")]
                )
                with rasterio.open(
                    path, "w", driver="GTiff", width=8, height=4, count=1,
                    dtype="float32", crs="EPSG:5070", transform=from_origin(0, 4, 1, 1),
                    nodata=-9999,
                ) as target:
                    target.write(values, 1)
            result = run_parity(
                manifest, prediction_dir, uncertainty_dir,
                prediction_mosaic, uncertainty_mosaic,
                "ailanthus_altissima", "east", root / "parity",
            )
            self.assertTrue(result["production_ready"])
            self.assertEqual(result["failure_count"], 0)
            self.assertEqual(
                result["prediction"]["mean_pointwise_absolute_difference"], 0.0
            )
            self.assertTrue((root / "parity" / "COMPLETED").is_file())

    def test_native_mosaic_parity_rejects_large_mosaic_shift(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tile = root / "tile_ne_17_000.tif"
            prediction_dir = root / "prediction"
            uncertainty_dir = root / "uncertainty"
            prediction_dir.mkdir()
            uncertainty_dir.mkdir()
            manifest = root / "tiles.txt"
            manifest.write_text(f"{tile}\n", encoding="utf-8")
            transform = from_origin(0, 4, 1, 1)
            rasters = (
                (tile, 0.2),
                (prediction_dir / f"{tile.stem}_prediction.tif", 0.2),
                (uncertainty_dir / f"{tile.stem}_uncertainty.tif", 0.03),
                (root / "prediction_mosaic.tif", 0.3),
                (root / "uncertainty_mosaic.tif", 0.03),
            )
            for path, value in rasters:
                with rasterio.open(
                    path, "w", driver="GTiff", width=4, height=4, count=1,
                    dtype="float32", crs="EPSG:5070", transform=transform,
                    nodata=-9999,
                ) as target:
                    target.write(np.full((4, 4), value, dtype="float32"), 1)
            result = run_parity(
                manifest, prediction_dir, uncertainty_dir,
                root / "prediction_mosaic.tif", root / "uncertainty_mosaic.tif",
                "ailanthus_altissima", "east", root / "parity",
            )
            self.assertFalse(result["production_ready"])
            self.assertIn("prediction_mean_absolute_difference", result["failures"])
            self.assertTrue((root / "parity" / "REJECTED").is_file())

    def test_mosaic_validation_checks_status_counts_ranges_and_quickview_hashes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            products = {}
            for label in ("prediction", "uncertainty", "count"):
                path = root / f"host_east_{label}.tif"
                dtype = "uint16" if label == "count" else "float32"
                nodata = 0 if label == "count" else -9999
                value = 4 if label == "count" else (0.02 if label == "uncertainty" else 0.4)
                with rasterio.open(
                    path, "w", driver="GTiff", width=4, height=4, count=1,
                    dtype=dtype, crs="EPSG:5070", transform=from_origin(0, 1024, 256, 256),
                    nodata=nodata,
                ) as target:
                    target.write(np.full((4, 4), value, dtype=dtype), 1)
                write_json(
                    path.with_name(f"{path.stem}_gdalinfo.json"),
                    {"bands": [{"minimum": value, "maximum": value}]},
                )
                write_json(
                    root / "logs" / f"{label}_mosaic_status.json",
                    {
                        "status": "complete", "release_product": label,
                        "input_count": 12, "output": str(path),
                    },
                )
                products[label] = path
            quick_outputs = {}
            for label in (
                "probability_transparent", "probability_framed",
                "uncertainty_transparent", "uncertainty_framed",
                "support_transparent", "support_framed",
                "difference_from_june_transparent", "difference_from_june_framed",
            ):
                path = root / f"{label}.png"
                path.write_bytes(label.encode())
                quick_outputs[label] = str(path)
            quickview = root / "quickview_summary.json"
            reference = root / "june_reference.tif"
            reference.write_bytes(b"reference")
            write_json(
                quickview,
                {
                    "status": "complete", "host": "ailanthus_altissima", "region": "east",
                    "mask_definition": "count > 0 AND finite(product)",
                    "prediction_scale": [0.0, 1.0], "uncertainty_scale": [0.0, 0.1],
                    "support_scale": [0.0, 16.0],
                    "difference_from_june_scale": [-0.5, 0.5],
                    "reference_prediction_sha256": sha256(reference),
                    "prediction_sha256": sha256(products["prediction"]),
                    "uncertainty_sha256": sha256(products["uncertainty"]),
                    "count_sha256": sha256(products["count"]), "outputs": quick_outputs,
                },
            )
            parity_detail = root / "parity.csv"
            parity_detail.write_text("tile,layer\nexample,prediction\n", encoding="utf-8")
            parity = root / "parity.json"
            write_json(
                parity,
                {
                    "status": "complete", "production_ready": True,
                    "host": "ailanthus_altissima", "region": "east",
                    "prediction_mosaic_sha256": sha256(products["prediction"]),
                    "uncertainty_mosaic_sha256": sha256(products["uncertainty"]),
                    "detail": str(parity_detail), "detail_sha256": sha256(parity_detail),
                },
            )
            result = validate_mosaics(
                products["prediction"], products["uncertainty"], products["count"],
                quickview, parity, expected_input_count=12, target_resolution=256,
            )
            self.assertTrue(result["production_ready_for_visual_review"])
            self.assertEqual(result["host"], "ailanthus_altissima")

    def test_visual_review_requires_all_fixed_scale_checks_and_hashes_images(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            qa_paths = []
            reviews = []
            for host in sorted(HOSTS):
                quick_root = root / host / "quickviews"
                outputs = {}
                for label in (
                    "probability_transparent", "probability_framed",
                    "uncertainty_transparent", "uncertainty_framed",
                    "support_transparent", "support_framed",
                    "difference_from_june_transparent", "difference_from_june_framed",
                ):
                    image = quick_root / f"{label}.png"
                    image.parent.mkdir(parents=True, exist_ok=True)
                    image.write_bytes(f"{host}-{label}".encode())
                    outputs[label] = str(image)
                quickview = quick_root / "quickview_summary.json"
                region = "west" if host == "notholithocarpus_densiflorus" else "east"
                write_json(
                    quickview,
                    {"status": "complete", "host": host, "region": region, "outputs": outputs},
                )
                qa = root / host / "mosaic_qa.json"
                write_json(
                    qa,
                    {
                        "status": "complete",
                        "production_ready_for_visual_review": True,
                        "host": host,
                        "region": region,
                        "quickview_summary": str(quickview),
                        "quickview_summary_sha256": sha256(quickview),
                    },
                )
                qa_paths.append(qa)
                reviews.append(
                    {
                        "host": host,
                        "decision": "approved",
                        "checks": {target: "pass" for target in REVIEW_TARGETS},
                        "observations": "Fixed-scale prediction and uncertainty panels inspected.",
                    }
                )
            spec = root / "review_spec.json"
            write_json(
                spec,
                {
                    "reviewer": "Test reviewer",
                    "reviewed_at": "2026-08-13T16:00:00-04:00",
                    "reviews": reviews,
                },
            )
            result = record_visual_review(spec, qa_paths, root / "review")
            self.assertTrue(result["approved_for_release_evidence"])
            self.assertEqual(result["host_count"], 5)
            self.assertTrue((root / "review" / "APPROVED").is_file())


class AugustReleaseEvidenceTests(unittest.TestCase):
    def test_dgx_bundle_requires_mc_evidence_from_same_accepted_registry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            registry = root / "production.tsv"
            acceptance = root / "acceptance.json"
            registry.write_text("accepted registry\n", encoding="utf-8")
            write_json(acceptance, {"status": "complete"})
            manifest = root / "mc_design" / "analysis" / "mc_manifest.csv"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("host\nexample\n", encoding="utf-8")
            write_json(root / "mc_design" / "design_metadata.json", {
                "status": "complete",
                "production_registry_sha256": sha256(registry),
                "final_acceptance_sha256": sha256(acceptance),
            })
            mc = {
                "manifest": str(manifest),
                "manifest_sha256": sha256(manifest),
            }

            observed_manifest, observed_design = validate_mc_lineage(
                mc, registry, acceptance
            )
            self.assertEqual(observed_manifest, manifest)
            self.assertEqual(
                observed_design, root / "mc_design" / "design_metadata.json"
            )

            registry.write_text("different registry\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not generated from this registry"):
                validate_mc_lineage(mc, registry, acceptance)

    def test_mc_stability_stage_runs_matrix_before_synthesis(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            production_registry = root / "production.tsv"
            final_acceptance = root / "acceptance.json"
            production_registry.write_text("registry\n", encoding="utf-8")
            write_json(final_acceptance, {"status": "complete"})
            design_root = root / "design"
            output_root = root / "outputs"
            registry = design_root / "experiment_registry.tsv"
            manifest = design_root / "analysis" / "mc_stability_manifest.csv"
            rows = [{"experiment": "mc2"}]
            calls = []

            def record_design(*args):
                calls.append(("design", args))
                return registry, rows

            def record_runs(*args, **kwargs):
                calls.append(("runs", args, kwargs))

            def record_manifest(*args):
                calls.append(("manifest", args))
                manifest.parent.mkdir(parents=True)
                manifest.write_text("host\nexample\n", encoding="utf-8")
                return manifest

            def record_evidence(*args):
                calls.append(("evidence", args))
                return {"status": "complete", "recommended_mc_samples": 5}

            with (
                patch.object(mc_stability_stage, "DESIGN_ROOT", design_root),
                patch.object(mc_stability_stage, "OUTPUT_ROOT", output_root),
                patch.object(
                    mc_stability_stage, "write_design", side_effect=record_design
                ),
                patch.object(
                    mc_stability_stage, "run_design", side_effect=record_runs
                ),
                patch.object(
                    mc_stability_stage, "materialize_analysis_manifest",
                    side_effect=record_manifest,
                ),
                patch.object(
                    mc_stability_stage, "synthesize_evidence",
                    side_effect=record_evidence,
                ),
            ):
                result = mc_stability_stage.run_stage(
                    production_registry, final_acceptance, False
                )

            self.assertEqual(result["recommended_mc_samples"], 5)
            self.assertEqual(
                [item[0] for item in calls],
                ["design", "runs", "manifest", "evidence"],
            )
            self.assertEqual(calls[1][1], (registry, rows))
            self.assertFalse(calls[1][2]["resume_completed"])
            self.assertEqual(calls[3][1], (manifest, output_root / "evidence"))

    def test_ailanthus_five_host_promotion_assembles_before_final_gate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            original_registry = root / "original_registry.tsv"
            original_acceptance = root / "original_acceptance.json"
            replacement_acceptance = root / "replacement_acceptance.json"
            for path in (
                original_registry, original_acceptance, replacement_acceptance
            ):
                path.write_text("source\n", encoding="utf-8")
            assembly_dir = root / "assembly"
            combined_registry = assembly_dir / "production" / "registry.tsv"
            combined_blockcv = assembly_dir / "blockcv"
            calls = []

            def record_assembly(*args):
                calls.append(("assembly", args))
                assembly_dir.mkdir()
                write_json(assembly_dir / "summary.json", {"status": "complete"})
                return {
                    "production_registry": str(combined_registry),
                    "blockcv_evidence_dir": str(combined_blockcv),
                }

            def record_final(*args, **kwargs):
                calls.append(("final", args, kwargs))
                return {"status": "complete", "production_ready_for_inference": True}

            with (
                patch.object(
                    ailanthus_five_host_promotion, "PRODUCTION_OUTPUT_ROOT",
                    root / "outputs",
                ),
                patch.object(
                    ailanthus_five_host_promotion, "current_clean_commit",
                    return_value="c" * 40,
                ),
                patch.object(
                    ailanthus_five_host_promotion, "assemble_inputs",
                    side_effect=record_assembly,
                ),
                patch.object(
                    ailanthus_five_host_promotion, "run_final_acceptance",
                    side_effect=record_final,
                ),
            ):
                result = ailanthus_five_host_promotion.run_stage(
                    original_registry,
                    original_acceptance,
                    replacement_acceptance,
                    assembly_dir,
                    "cuda",
                    64,
                    512,
                    False,
                )

            self.assertTrue(result["production_ready_for_inference"])
            self.assertEqual([item[0] for item in calls], ["assembly", "final"])
            final_args, final_kwargs = calls[1][1], calls[1][2]
            self.assertEqual(final_args[:2], (combined_registry, combined_blockcv))
            self.assertEqual(final_args[3:], ("cuda", 64, 512))
            self.assertFalse(final_kwargs["resume_completed"])

    def test_five_host_promotion_resume_rejects_original_registry_change(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            original_registry = root / "original_registry.tsv"
            original_registry.write_text("original\n", encoding="utf-8")
            original_acceptance = root / "original_acceptance.json"
            replacement_acceptance = root / "replacement_acceptance.json"
            write_json(original_acceptance, {"status": "complete"})
            write_json(replacement_acceptance, {"status": "complete"})
            assembly = root / "assembly"
            assembly.mkdir()
            (assembly / "COMPLETED").write_text("complete\n", encoding="utf-8")
            write_json(assembly / "summary.json", {
                "status": "complete",
                "original_acceptance_sha256": sha256(original_acceptance),
                "replacement_acceptance_sha256": sha256(replacement_acceptance),
            })
            write_json(assembly / "production" / "design_metadata.json", {
                "source_original_registry": str(original_registry),
                "source_original_registry_sha256": "wrong",
            })

            with self.assertRaisesRegex(ValueError, "original registry changed"):
                ailanthus_five_host_promotion.validate_existing_assembly(
                    assembly,
                    original_registry,
                    original_acceptance,
                    replacement_acceptance,
                )

    def test_ailanthus_production_stage_runs_fit_before_acceptance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            blockcv = root / "blockcv"
            blockcv.mkdir()
            blockcv_summary = blockcv / "summary.json"
            write_json(blockcv_summary, {"status": "complete"})
            selection = root / "production_selection.json"
            write_json(selection, {
                "blockcv_evidence_dir": str(blockcv),
                "blockcv_evidence_summary_sha256": sha256(blockcv_summary),
            })
            registry = root / "design" / "experiment_registry.tsv"
            rows = [{"experiment": "production"}]
            calls = []

            def record_training(observed_registry, observed_rows, max_runs):
                calls.append(("training", observed_registry, observed_rows, max_runs))

            def record_acceptance(*args):
                calls.append(("acceptance", args))
                return {"status": "complete", "production_ready": True}

            with (
                patch.object(
                    ailanthus_production_stage, "PRODUCTION_OUTPUT_ROOT",
                    root / "outputs",
                ),
                patch.object(
                    ailanthus_production_stage, "write_design",
                    return_value=(registry, rows),
                ),
                patch.object(
                    ailanthus_production_stage, "run_experiments",
                    side_effect=record_training,
                ),
                patch.object(
                    ailanthus_production_stage, "run_acceptance",
                    side_effect=record_acceptance,
                ),
            ):
                result = ailanthus_production_stage.run_stage(
                    selection, blockcv, "cuda", 64, True
                )

            self.assertTrue(result["production_ready"])
            self.assertEqual([item[0] for item in calls], ["training", "acceptance"])
            self.assertEqual(calls[0][1:], (registry, rows, None))
            acceptance_args = calls[1][1]
            self.assertEqual(acceptance_args[0:2], (registry, blockcv))
            self.assertEqual(acceptance_args[3:], ("cuda", 64, True))

    def test_ailanthus_blockcv_stage_runs_folds_before_evidence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            selection = root / "selection.json"
            selection.write_text("{}", encoding="utf-8")
            registry = root / "design" / "experiment_registry.tsv"
            metrics = registry.parent / "analysis" / "experiment_metrics.csv"
            metrics.parent.mkdir(parents=True)
            metrics.write_text("experiment\nexample\n", encoding="utf-8")
            rows = [{"experiment": "fold1"}]
            calls = []

            def record_training(observed_registry, observed_rows, max_runs):
                calls.append(("training", observed_registry, observed_rows, max_runs))

            def record_evidence(*args, **kwargs):
                calls.append(("evidence", args, kwargs))
                return {"status": "complete", "production_ready": True}

            with (
                patch.object(ailanthus_blockcv_stage, "OUTPUT_BASE", root / "outputs"),
                patch.object(
                    ailanthus_blockcv_stage, "write_design",
                    return_value=(registry, rows),
                ),
                patch.object(
                    ailanthus_blockcv_stage, "run_experiments",
                    side_effect=record_training,
                ),
                patch.object(
                    ailanthus_blockcv_stage, "summarize_registry",
                    return_value={"status": "complete"},
                ),
                patch.object(
                    ailanthus_blockcv_stage, "sha256",
                    return_value="a" * 64,
                ),
                patch.object(
                    ailanthus_blockcv_stage, "synthesize_blockcv",
                    side_effect=record_evidence,
                ),
            ):
                result = ailanthus_blockcv_stage.run_stage(selection)

            self.assertTrue(result["production_ready"])
            self.assertEqual([item[0] for item in calls], ["training", "evidence"])
            self.assertEqual(calls[0][1:], (registry, rows, None))
            evidence_args, evidence_kwargs = calls[1][1], calls[1][2]
            self.assertEqual(evidence_args[:2], (selection, metrics))
            self.assertEqual(evidence_kwargs["expected_hosts"], {"ailanthus_altissima"})
            self.assertEqual(
                evidence_kwargs["expected_selection_type"],
                "ailanthus_recipe_for_blockcv",
            )

    def test_replacement_acceptance_runs_all_raw_and_configured_probes(self):
        commands = replacement_probe_commands(
            Path("registry.tsv"), Path("output"), "cuda", 64, True
        )
        self.assertEqual(
            [step for step, _, _ in commands],
            [
                "raw:matched", "raw:exact_overlap",
                "configured:matched", "configured:exact_overlap",
            ],
        )
        for step, output, command in commands:
            self.assertIn("--resume-completed", command)
            self.assertEqual(command[command.index("--score-space") + 1], step.split(":")[0])
            self.assertTrue(str(output).endswith(f"{step.split(':')[1]}/ailanthus_altissima"))

    def test_replacement_assembly_substitutes_only_ailanthus(self):
        original = pd.DataFrame([
            {"host": host, "experiment": f"old-{host}", "fold": fold}
            for host in sorted(HOSTS) for fold in range(1, 6)
        ])
        replacement = pd.DataFrame([
            {"host": "ailanthus_altissima", "experiment": "new-ailanthus", "fold": fold}
            for fold in range(1, 6)
        ])
        combined = combine_by_host(
            original, replacement, "test BlockCV", rows_per_host=5
        )
        self.assertEqual(len(combined), 25)
        self.assertEqual(
            set(combined.loc[combined["host"] == "ailanthus_altissima", "experiment"]),
            {"new-ailanthus"},
        )
        self.assertEqual(
            set(combined.loc[combined["host"] == "juglans_nigra", "experiment"]),
            {"old-juglans_nigra"},
        )
        selections = combine_selections(
            {"selections": [{"host": host, "value": "old"} for host in HOSTS]},
            {"selections": [{"host": "ailanthus_altissima", "value": "new"}]},
            "test",
        )
        by_host = {item["host"]: item["value"] for item in selections}
        self.assertEqual(by_host["ailanthus_altissima"], "new")
        self.assertEqual(by_host["juglans_nigra"], "old")

    def test_final_acceptance_resolves_identity_checked_replacement_selection(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            old_selection = root / "old_selection.json"
            new_selection = root / "new_selection.json"
            write_json(old_selection, {"status": "complete"})
            write_json(new_selection, {"status": "complete"})
            config_paths = []
            for host in sorted(HOSTS):
                source = new_selection if host == "ailanthus_altissima" else old_selection
                config = root / "configs" / f"{host}.json"
                write_json(config, {
                    "selection_gate": {
                        "selection_manifest": str(source),
                        "selection_manifest_sha256": sha256(source),
                    }
                })
                config_paths.append({"host": host, "config_path": str(config)})
            blockcv_dir = root / "blockcv"
            combined = root / "combined_production_selection.json"
            summary = blockcv_dir / "summary.json"
            write_json(summary, {
                "status": "complete", "production_ready": True,
                "combined_production_selection_manifest": str(combined),
            })
            write_json(combined, {
                "status": "complete",
                "selection_type": "blockcv_approved_recipe_for_production_fit",
                "selection_count": 5,
                "selections": [{"host": host} for host in sorted(HOSTS)],
                "source_production_selection_manifests": [
                    {"path": str(path), "sha256": sha256(path)}
                    for path in (old_selection, new_selection)
                ],
                "blockcv_evidence_summary_sha256": sha256(summary),
            })
            path, payload = resolve_production_selection(
                pd.DataFrame(config_paths), blockcv_dir, read_json(summary)
            )
            self.assertEqual(path, combined)
            self.assertEqual(payload["selection_count"], 5)
            payload["source_production_selection_manifests"][0]["sha256"] = "bad"
            write_json(combined, payload)
            with self.assertRaises(ValueError):
                resolve_production_selection(
                    pd.DataFrame(config_paths), blockcv_dir, read_json(summary)
                )

    def test_dgx_run_chain_requires_complete_hashed_shard_coverage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            registry = root / "registry.tsv"
            registry.write_text("host\nailanthus_altissima\n", encoding="utf-8")
            bundle = root / "bundle_metadata.json"
            write_json(
                bundle,
                {
                    "status": "complete",
                    "inference_defaults": {
                        "base_seed": 42,
                        "deterministic_mc_samples": 1,
                        "uncertainty_mc_samples": 5,
                    },
                },
            )
            inference_by_key = {}
            status_paths = []
            for host in sorted(HOSTS):
                region = "west" if host == "notholithocarpus_densiflorus" else "east"
                for product in ("deterministic", "uncertainty"):
                    output_root = root / "outputs" / host / product
                    output_dir = output_root / "shard_000"
                    output_dir.mkdir(parents=True)
                    tile_list = root / "tiles" / host / f"{product}.txt"
                    tile_list.parent.mkdir(parents=True, exist_ok=True)
                    tile_list.write_text("/imagery/tile.tif\n", encoding="utf-8")
                    key = (host, product)
                    inference_by_key[key] = {
                        "host": host,
                        "product": product,
                        "region": region,
                        "expected_tile_count": 1,
                        "output_root": str(output_root),
                    }
                    status = root / "status" / host / f"{product}.json"
                    write_json(
                        status,
                        {
                            "status": "completed",
                            "returncode": 0,
                            "host": host,
                            "region": region,
                            "product": product,
                            "shard_id": "000",
                            "tile_list": str(tile_list),
                            "tile_list_sha256": sha256(tile_list),
                            "expected_tile_count": 1,
                            "output_dir": str(output_dir),
                            "prediction_count": 1,
                            "uncertainty_count": 1,
                            "count_count": 1,
                            "elapsed_seconds": 10,
                            "base_seed": 42,
                            "mc_samples": 1 if product == "deterministic" else 5,
                            "production_registry_sha256": sha256(registry),
                            "bundle_metadata_sha256": sha256(bundle),
                        },
                    )
                    status_paths.append(status)
            rows, failures = [], []
            _index_dgx_run_chain(
                status_paths, registry, bundle, inference_by_key, {}, rows, failures
            )
            self.assertEqual(failures, [])
            self.assertEqual(len(rows), 10)
            self.assertEqual({row["category"] for row in rows}, {"dgx_run_status"})

            broken = read_json(status_paths[0])
            broken["prediction_count"] = 0
            write_json(status_paths[0], broken)
            rows, failures = [], []
            _index_dgx_run_chain(
                status_paths, registry, bundle, inference_by_key, {}, rows, failures
            )
            self.assertTrue(any(item.startswith("dgx_run_output_counts:") for item in failures))

    def test_dgx_shard_runner_is_local_seeded_hann_and_resumable(self):
        script = Path(__file__).resolve().parents[1] / "dgx_august2026" / "run_inference_shard.sh"
        text = script.read_text(encoding="utf-8")
        self.assertIn("--blend-window hann", text)
        self.assertIn("--blend-edge-floor 0.05", text)
        self.assertIn("--skip-existing", text)
        self.assertIn("--seed", text)
        self.assertIn("elapsed_seconds", text)
        self.assertNotIn("sbatch", text)
        self.assertNotIn("SLURM_", text)

    def test_dgx_queue_is_serial_local_and_runs_provenance_aware_qa(self):
        script = Path(__file__).resolve().parents[1] / "dgx_august2026" / "run_inference_queue.sh"
        text = script.read_text(encoding="utf-8")
        self.assertIn('bash "${BUNDLE_ROOT}/run_inference_shard.sh"', text)
        self.assertIn("--expected-mc-samples", text)
        self.assertIn("--base-seed", text)
        self.assertIn("tile_manifest_sha256", text)
        self.assertNotIn("sbatch", text)
        self.assertNotIn("SLURM_", text)

    def test_dgx_qa_gate_includes_seed_mc_count_and_output_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "tiles.txt"
            manifest.write_text("/archive/tile.tif\n", encoding="utf-8")
            output_root = root / "outputs"
            summary = root / "qa" / "summary.json"
            write_json(
                summary,
                {
                    "status": "complete",
                    "production_ready": True,
                    "host": "ailanthus_altissima",
                    "region": "east",
                    "product": "uncertainty",
                    "tile_manifest_sha256": sha256(manifest),
                    "expected_tile_count": 1,
                    "base_seed": 42,
                    "expected_mc_samples": 5,
                    "output_root": str(output_root.resolve()),
                },
            )
            (summary.parent / "COMPLETED").write_text("complete\n", encoding="utf-8")
            validate_dgx_inference_qa(
                summary, manifest, "ailanthus_altissima", "east", "uncertainty",
                1, 42, 5, output_root,
            )
            payload = read_json(summary)
            payload["expected_mc_samples"] = 3
            write_json(summary, payload)
            with self.assertRaises(ValueError):
                validate_dgx_inference_qa(
                    summary, manifest, "ailanthus_altissima", "east",
                    "uncertainty", 1, 42, 5, output_root,
                )

    def test_dgx_mosaic_runner_is_local_resumable_and_gated_by_qa_pair(self):
        script = Path(__file__).resolve().parents[1] / "dgx_august2026" / "run_mosaic_host.sh"
        text = script.read_text(encoding="utf-8")
        self.assertIn("--deterministic-summary", text)
        self.assertIn("--uncertainty-summary", text)
        self.assertIn("august2026_native_mosaic_parity.py", text)
        self.assertIn("august2026_render_fixed_quickviews.py", text)
        self.assertIn("august2026_validate_mosaics.py", text)
        self.assertIn(".partial.tif", text)
        self.assertNotIn("sbatch", text)
        self.assertNotIn("SLURM_", text)

    def test_dgx_design_prepares_state_domain_smoke_and_full_shards_locally(self):
        script = Path(__file__).resolve().parents[1] / "dgx_august2026" / "prepare_inference_design.sh"
        text = script.read_text(encoding="utf-8")
        self.assertIn("east_of_mississippi tanoak_ca_or", text)
        self.assertIn("SMOKE_PER_STATE", text)
        self.assertIn("FULL_SHARD_SIZE", text)
        self.assertIn("build_archive_manifests.py", text)
        self.assertIn("make_tile_shards.py", text)
        self.assertNotIn("sbatch", text)
        self.assertNotIn("SLURM_", text)

    def test_dgx_runtime_projection_requires_complete_five_host_smoke(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            east = root / "east.txt"
            west = root / "west.txt"
            east.write_text("/tiles/e1.tif\n/tiles/e2.tif\n", encoding="utf-8")
            west.write_text("/tiles/w1.tif\n", encoding="utf-8")
            statuses = []
            for host in sorted(HOSTS):
                region = "west" if host == "notholithocarpus_densiflorus" else "east"
                for product in ("deterministic", "uncertainty"):
                    tile_list = root / "smoke" / host / f"{product}.txt"
                    tile_list.parent.mkdir(parents=True, exist_ok=True)
                    tile_list.write_text("/tiles/smoke.tif\n", encoding="utf-8")
                    status = root / "status" / host / f"{product}.json"
                    write_json(status, {
                        "status": "completed", "returncode": 0,
                        "host": host, "region": region, "product": product,
                        "shard_id": "00000", "tile_list": str(tile_list),
                        "tile_list_sha256": sha256(tile_list),
                        "expected_tile_count": 1, "elapsed_seconds": 10,
                    })
                    statuses.append(status)
            result = synthesize_projection(
                statuses, {"east": east, "west": west}, root / "projection"
            )
            self.assertEqual(result["projected_serial_seconds"], 180.0)
            self.assertEqual(result["host_count"], 5)
            with self.assertRaises(ValueError):
                synthesize_projection(
                    statuses[:-1], {"east": east, "west": west},
                    root / "incomplete_projection",
                )

    def test_hazel_submission_gate_rejects_changed_bundle_and_qa_identity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            registry = root / "registry.tsv"
            registry.write_text("host\nailanthus_altissima\n", encoding="utf-8")
            bundle = root / "bundle"
            write_minimal_hazel_bundle(bundle, registry)
            validate_bundle(bundle, registry)
            (bundle / "production_experiment_registry.tsv").write_text(
                "host\njuglans_nigra\n", encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                validate_bundle(bundle, registry)

            manifest = root / "tiles.txt"
            manifest.write_text("/archive/tile.tif\n", encoding="utf-8")
            qa_root = root / "qa"
            qa_root.mkdir()
            summary = qa_root / "summary.json"
            write_json(
                summary,
                {
                    "status": "complete", "production_ready": True,
                    "host": "ailanthus_altissima", "region": "east",
                    "product": "deterministic",
                    "tile_manifest_sha256": sha256(manifest),
                    "expected_tile_count": 1,
                },
            )
            (qa_root / "COMPLETED").write_text("completed\n", encoding="utf-8")
            validate_inference_qa(
                summary, manifest, "ailanthus_altissima", "east",
                "deterministic", 1,
            )
            with self.assertRaises(ValueError):
                validate_inference_qa(
                    summary, manifest, "ailanthus_altissima", "east",
                    "uncertainty", 1,
                )

    def test_augmentation_pareto_includes_host_specific_artifact_replay(self):
        common = {
            "test_roc_auc_all_thresholds": [0.90, 0.89],
            "test_brier_score": [0.10, 0.11],
            "artifact_candidate_absolute_did_mean": [0.10, 0.11],
        }
        eastern = pd.DataFrame(
            {
                "host": ["vitis_labrusca", "vitis_labrusca"],
                **common,
                "exact_overlap_candidate_absolute_shift_mean": [0.20, 0.05],
                "artifact_coastal_candidate_absolute_did_mean": [np.nan, np.nan],
            }
        )
        eastern = add_augmentation_pareto_flags(eastern)
        self.assertEqual(
            eastern["pareto_nondominated_within_host"].tolist(), [True, True]
        )
        tanoak = pd.DataFrame(
            {
                "host": [
                    "notholithocarpus_densiflorus",
                    "notholithocarpus_densiflorus",
                ],
                **common,
                "exact_overlap_candidate_absolute_shift_mean": [np.nan, np.nan],
                "artifact_coastal_candidate_absolute_did_mean": [0.20, 0.05],
            }
        )
        tanoak = add_augmentation_pareto_flags(tanoak)
        self.assertEqual(
            tanoak["pareto_nondominated_within_host"].tolist(), [True, True]
        )

    def test_final_acceptance_command_step_runs_and_records_completion(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            marker = root / "COMPLETED"
            status_path = root / "status.json"
            state = {"completed_steps": []}
            run_command_step(
                state,
                status_path,
                "raw:matched:test_host",
                marker,
                ["touch", str(marker)],
                root,
            )
            self.assertTrue(marker.is_file())
            self.assertEqual(
                state["completed_steps"], ["raw:matched:test_host"]
            )
            self.assertEqual(
                json.loads(status_path.read_text(encoding="utf-8"))["completed_steps"],
                ["raw:matched:test_host"],
            )

    def test_final_acceptance_records_completed_failed_gate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            status_path = root / "acceptance_status.json"
            summary_path = root / "final_model_acceptance" / "summary.json"
            write_json(
                summary_path,
                {
                    "status": "complete",
                    "production_ready_for_inference": False,
                    "failures": ["test_host: boundary_regression"],
                },
            )
            status = {
                "status": "running",
                "active_step": "support:pathway_audit",
                "completed_steps": ["probe", "support"],
            }
            record_failed_final_status(status, status_path, root, summary_path)
            recorded = json.loads(status_path.read_text(encoding="utf-8"))
            self.assertEqual(recorded["status"], "complete_with_failures")
            self.assertEqual(recorded["completed_step_count"], 2)
            self.assertFalse(recorded["production_ready_for_inference"])
            self.assertNotIn("active_step", recorded)
            self.assertEqual(
                (root / "COMPLETED").read_text(encoding="utf-8"),
                "completed_with_failures\n",
            )

    def test_augmentation_resume_records_ancestor_commit_upgrade(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            registry = root / "registry.tsv"
            selection = root / "selection.json"
            metrics = root / "metrics.csv"
            for path in (registry, selection, metrics):
                path.write_text(path.name, encoding="utf-8")
            code_root = Path(__file__).resolve().parents[1]
            current = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=code_root, check=True,
                capture_output=True, text=True,
            ).stdout.strip()
            previous = subprocess.run(
                ["git", "rev-parse", "HEAD^"], cwd=code_root, check=True,
                capture_output=True, text=True,
            ).stdout.strip()
            status_path = root / "status.json"
            write_json(
                status_path,
                {
                    "registry_sha256": sha256(registry),
                    "selection_manifest_sha256": sha256(selection),
                    "training_metrics_sha256": sha256(metrics),
                    "code_commit": previous,
                    "device": "cuda",
                    "batch_size": 64,
                },
            )
            status = _validate_resume_identity(
                status_path, registry, selection, metrics, current, "cuda", 64,
                allow_code_commit_upgrade=True, code_root=code_root,
            )
            self.assertEqual(status["code_commit"], current)
            self.assertEqual(status["code_commit_history"], [previous])
            self.assertEqual(
                status["resume_code_commit_upgrades"][0]["new_commit"], current
            )

    def test_final_acceptance_builds_raw_and_configured_probe_matrix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            commands = final_probe_commands(
                root / "registry.tsv", root / "acceptance", "cuda", 64, True
            )
            self.assertEqual(len(commands), 18)
            self.assertEqual(
                sum(score == "raw" for score, _, _, _, _ in commands), 9
            )
            self.assertEqual(
                sum(score == "configured" for score, _, _, _, _ in commands), 9
            )
            self.assertEqual(
                sum(kind == "matched" for _, kind, _, _, _ in commands), 10
            )
            self.assertEqual(
                sum(kind == "exact_overlap" for _, kind, _, _, _ in commands),
                8,
            )
            for score, _, _, _, command in commands:
                self.assertEqual(command[command.index("--score-space") + 1], score)
                self.assertIn("--resume-completed", command)

    def test_batch_probe_imports_configured_score_calibrator_validator(self):
        calibrator = batch_artifact_probe.inference.validate_platt_calibrator(
            {
                "method": "platt_logistic_on_validation_logits",
                "intercept": -0.2,
                "slope": 0.8,
            }
        )
        self.assertEqual(calibrator["slope"], 0.8)

    def test_final_resume_records_ancestor_commit_upgrade(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            registry = root / "registry.tsv"
            metrics = root / "metrics.csv"
            blockcv = root / "blockcv.json"
            for path in (registry, metrics, blockcv):
                path.write_text(path.name, encoding="utf-8")
            code_root = Path(__file__).resolve().parents[1]
            current = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=code_root, check=True,
                capture_output=True, text=True,
            ).stdout.strip()
            previous = subprocess.run(
                ["git", "rev-parse", "HEAD^"], cwd=code_root, check=True,
                capture_output=True, text=True,
            ).stdout.strip()
            status_path = root / "status.json"
            write_json(
                status_path,
                {
                    "registry_sha256": sha256(registry),
                    "training_metrics_sha256": sha256(metrics),
                    "blockcv_summary_sha256": sha256(blockcv),
                    "code_commit": previous,
                    "device": "cuda",
                    "batch_size": 64,
                    "samples_per_origin_class": 512,
                },
            )
            status = _resume_status(
                status_path, registry, metrics, blockcv, current, "cuda", 64, 512,
                allow_code_commit_upgrade=True, code_root=code_root,
            )
            self.assertEqual(status["code_commit"], current)
            self.assertEqual(status["code_commit_history"], [previous])

    def test_final_acceptance_adopts_only_hash_matched_support_audit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            marker = root / "COMPLETED"
            summary = root / "summary.json"
            status_path = root / "status.json"
            marker.write_text("completed\n", encoding="utf-8")
            write_json(summary, {"status": "complete", "registry_sha256": "abc"})
            state = {"completed_steps": []}
            self.assertTrue(
                adopt_completed_support_step(
                    state, status_path, "support:pathway", marker, summary,
                    {"registry_sha256": "abc"},
                )
            )
            self.assertEqual(state["completed_steps"], ["support:pathway"])
            with self.assertRaises(ValueError):
                adopt_completed_support_step(
                    {"completed_steps": []}, status_path, "support:pathway",
                    marker, summary, {"registry_sha256": "changed"},
                )

    def test_tanoak_coastal_comparison_is_a_separate_paired_gate(self):
        frame = pd.DataFrame(
            {
                "mid_lon_x": [-124.0, -123.5, -122.5, -122.0],
                "mid_lon_y": [-124.0, -123.5, -122.5, -122.0],
                "baseline_did": [0.10, -0.20, 0.03, -0.04],
                "candidate_did": [0.05, -0.10, 0.02, -0.03],
            }
        )
        result = add_tanoak_coastal_comparison(frame, {"status": "complete"})
        coastal = result["coastal_west_of_123"]
        self.assertEqual(coastal["pair_count"], 2)
        self.assertAlmostEqual(coastal["baseline_absolute_did_mean"], 0.15)
        self.assertAlmostEqual(coastal["candidate_absolute_did_mean"], 0.075)
        changed = frame.copy()
        changed.loc[0, "mid_lon_y"] = -123.9
        with self.assertRaises(ValueError):
            add_tanoak_coastal_comparison(changed, {})

    def test_augmentation_backfills_reused_tanoak_coastal_evidence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            comparison = root / "june_baseline_comparison"
            comparison.mkdir()
            write_json(
                comparison / "summary.json",
                {"candidate_experiment": "baseline-tanoak"},
            )
            pd.DataFrame(
                {
                    "mid_lon_x": [-124.0, -123.5, -122.5],
                    "mid_lon_y": [-124.0, -123.5, -122.5],
                    "baseline_did": [0.10, -0.20, 0.03],
                    "candidate_did": [0.05, -0.10, 0.02],
                }
            ).to_csv(comparison / "paired_artifact_comparison.csv", index=False)
            frame = pd.DataFrame(
                {
                    "host": ["notholithocarpus_densiflorus"],
                    "experiment": ["baseline-tanoak"],
                    "artifact_comparison_summary": [
                        str(comparison / "summary.json")
                    ],
                    "artifact_coastal_candidate_absolute_did_mean": [np.nan],
                }
            )
            result = fill_missing_tanoak_coastal_evidence(frame).iloc[0]
            self.assertEqual(result["artifact_coastal_pair_count"], 2)
            self.assertAlmostEqual(
                result["artifact_coastal_candidate_absolute_did_mean"], 0.075
            )
            self.assertEqual(
                result["artifact_coastal_evidence_origin"],
                "recomputed_from_frozen_paired_artifact_comparison",
            )
            self.assertEqual(
                result["artifact_coastal_paired_comparison_csv_sha256"],
                sha256(comparison / "paired_artifact_comparison.csv"),
            )

    def test_augmentation_effects_are_exactly_matched_factorial_contrasts(self):
        rows = []
        for radiometric in (False, True):
            for dropout in (0.25, 0.35):
                for smoothing in (0.0, 0.05):
                    row = {
                        "host": "ailanthus_altissima",
                        "experiment": f"r{int(radiometric)}-d{dropout}-s{smoothing}",
                        "radiometric_enabled": radiometric,
                        "dropout": dropout,
                        "label_smoothing": smoothing,
                    }
                    for metric, direction in EFFECT_METRICS.items():
                        baseline = 0.5
                        # The high radiometric level improves every metric by 0.01
                        # after applying its declared scientific direction.
                        row[metric] = baseline + (0.01 * direction if radiometric else 0)
                    rows.append(row)
        pairwise, aggregate = factorial_effect_contrasts(pd.DataFrame(rows))
        self.assertEqual(len(pairwise), 12)
        self.assertEqual(len(aggregate), 3)
        radiometric = pairwise[pairwise["factor"] == "radiometric_enabled"]
        self.assertEqual(len(radiometric), 4)
        for metric in EFFECT_METRICS:
            self.assertTrue(
                np.allclose(radiometric[f"improvement_{metric}"], 0.01)
            )
        dropout = pairwise[pairwise["factor"] == "dropout"]
        self.assertTrue(
            np.allclose(dropout["improvement_test_roc_auc_all_thresholds"], 0)
        )

    def test_stage03_acceptance_builds_complete_raw_probe_matrix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            commands = probe_commands(
                root / "registry.tsv", root / "acceptance", "cuda", 32, True
            )
            self.assertEqual(len(commands), 9)
            self.assertEqual(
                sum(diagnostic == "matched" for diagnostic, _, _, _ in commands),
                5,
            )
            self.assertEqual(
                sum(
                    diagnostic == "exact_overlap"
                    for diagnostic, _, _, _ in commands
                ),
                4,
            )
            for _, _, _, command in commands:
                self.assertIn("--score-space", command)
                self.assertEqual(command[command.index("--score-space") + 1], "raw")
                self.assertIn("--resume-completed", command)

    def test_stage03_acceptance_resume_is_hash_locked(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            registry = root / "registry.tsv"
            selection = root / "selection.json"
            registry.write_text("host\nailanthus_altissima\n", encoding="utf-8")
            write_json(selection, {"status": "complete"})
            status = root / "status.json"
            write_json(
                status,
                {
                    "registry_sha256": sha256(registry),
                    "selection_manifest_sha256": sha256(selection),
                    "training_metrics_sha256": sha256(registry),
                    "code_commit": "abc123",
                    "device": "cuda",
                    "batch_size": 64,
                },
            )
            _validate_resume_identity(
                status, registry, selection, registry, "abc123", "cuda", 64
            )
            registry.write_text("host\njuglans_nigra\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                _validate_resume_identity(
                    status, registry, selection, registry, "abc123", "cuda", 64
                )

    def test_hazel_array_launcher_defaults_to_a_bounded_dry_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            shards = root / "shards"
            shards.mkdir()
            (shards / "shard_00000.txt").write_text("/archive/tile.tif\n", encoding="utf-8")
            tile_manifest = root / "tiles.txt"
            tile_manifest.write_text("/archive/tile.tif\n", encoding="utf-8")
            write_json(
                shards / "shard_metadata.json",
                {
                    "status": "complete", "shard_count": 1,
                    "source_manifest": str(tile_manifest),
                    "source_manifest_sha256": sha256(tile_manifest),
                },
            )
            (shards / "COMPLETED").write_text("completed\n", encoding="utf-8")
            registry = root / "registry.tsv"
            registry.write_text(
                "host\nailanthus_altissima\njuglans_nigra\n", encoding="utf-8"
            )
            bundle = root / "bundle"
            write_minimal_hazel_bundle(bundle, registry)
            log = root / "submission.tsv"
            script = Path(__file__).resolve().parents[1] / "hazel_august2026" / "submit_inference_arrays.sh"
            environment = dict(os.environ)
            environment["OUTPUT_ROOT"] = str(root / "active")
            environment["HPC_ROOT"] = str(root / "durable")
            result = subprocess.run(
                [
                    "bash", str(script), "east_of_mississippi", "deterministic",
                    str(shards), str(registry), str(bundle), str(log),
                    "ailanthus_altissima", "juglans_nigra",
                ],
                check=True, capture_output=True, text=True, env=environment,
            )
            self.assertIn("DRY_RUN inference_array_command=", result.stdout)
            self.assertIn("DRY_RUN inference_qa_command=", result.stdout)
            submission = pd.read_csv(log, sep="\t")
            self.assertEqual(len(submission), 2)
            self.assertEqual(submission.loc[0, "array_job_id"], "DRY_RUN")
            self.assertEqual(submission.loc[0, "qa_job_id"], "DRY_RUN")
            self.assertEqual(submission.loc[1, "dependency_job_id"], "DRY_RUN")
            self.assertEqual(int(submission.loc[0, "dry_run"]), 1)
            self.assertEqual(
                int(submission["max_active_tasks_within_submission"].max()), 2
            )

    def test_legacy_hazel_shard_launcher_routes_to_arrays(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            shards = root / "shards"
            shards.mkdir()
            (shards / "shard_00000.txt").write_text(
                "/archive/tile.tif\n", encoding="utf-8"
            )
            tile_manifest = root / "tiles.txt"
            tile_manifest.write_text("/archive/tile.tif\n", encoding="utf-8")
            write_json(
                shards / "shard_metadata.json",
                {
                    "status": "complete", "shard_count": 1,
                    "source_manifest": str(tile_manifest),
                    "source_manifest_sha256": sha256(tile_manifest),
                },
            )
            (shards / "COMPLETED").write_text("completed\n", encoding="utf-8")
            registry = root / "registry.tsv"
            registry.write_text("host\nailanthus_altissima\n", encoding="utf-8")
            bundle = root / "bundle"
            write_minimal_hazel_bundle(bundle, registry)
            submission_log = root / "legacy_submission.tsv"
            script = (
                Path(__file__).resolve().parents[1]
                / "hazel_august2026"
                / "submit_shards.sh"
            )
            environment = dict(os.environ)
            environment.update(
                {
                    "HPC_ROOT": str(root / "hpc"),
                    "OUTPUT_ROOT": str(root / "active"),
                    "SUBMISSION_LOG": str(submission_log),
                }
            )
            result = subprocess.run(
                [
                    "bash", str(script), "east_of_mississippi", "deterministic",
                    str(shards), str(registry), str(bundle), "ailanthus_altissima",
                ],
                check=True, capture_output=True, text=True, env=environment,
            )
            self.assertIn("routing all shards through bounded Slurm arrays", result.stderr)
            self.assertIn("DRY_RUN inference_array_command=", result.stdout)
            submission = pd.read_csv(submission_log, sep="\t")
            self.assertEqual(submission.loc[0, "array_job_id"], "DRY_RUN")

    def test_hazel_mosaic_launcher_uses_three_task_dry_run_array(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tile_manifest = root / "tiles.csv"
            tile_manifest.write_text("/archive/tile.tif\n", encoding="utf-8")
            bundle = root / "bundle"
            write_minimal_hazel_bundle(bundle)
            june_reference = root / "june_reference.tif"
            june_reference.write_bytes(b"june reference")
            qa_summaries = []
            for product in ("deterministic", "uncertainty"):
                qa_root = root / f"{product}_qa"
                qa_root.mkdir()
                summary = qa_root / "summary.json"
                write_json(
                    summary,
                    {
                        "status": "complete", "production_ready": True,
                        "host": "ailanthus_altissima",
                        "region": "east_of_mississippi", "product": product,
                        "tile_manifest_sha256": sha256(tile_manifest),
                        "expected_tile_count": 1,
                    },
                )
                (qa_root / "COMPLETED").write_text(
                    "completed\n", encoding="utf-8"
                )
                qa_summaries.append(summary)
            submission_log = root / "mosaic_submission.tsv"
            script = (
                Path(__file__).resolve().parents[1]
                / "hazel_august2026"
                / "submit_release_mosaics.sh"
            )
            environment = dict(os.environ)
            environment.update(
                {
                    "HPC_ROOT": str(root / "hpc"),
                    "DURABLE_OUTPUT_ROOT": str(root / "durable"),
                    "ACTIVE_OUTPUT_ROOT": str(root / "active"),
                    "SUBMISSION_LOG": str(submission_log),
                }
            )
            result = subprocess.run(
                [
                    "bash", str(script), "ailanthus_altissima",
                    "east_of_mississippi", str(tile_manifest), str(bundle), "1",
                    str(june_reference), str(qa_summaries[0]), str(qa_summaries[1]),
                ],
                check=True, capture_output=True, text=True, env=environment,
            )
            self.assertIn("DRY_RUN mosaic_array_command=", result.stdout)
            self.assertIn("--array=0-2%1", result.stdout)
            submission = pd.read_csv(submission_log, sep="\t")
            self.assertEqual(submission.loc[0, "mosaic_array_job_id"], "DRY_RUN")
            self.assertEqual(submission.loc[0, "qa_job_id"], "DRY_RUN")

    def test_release_evidence_requires_complete_five_host_chain(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            hosts = sorted(HOSTS)
            experiments = {host: f"production_{host}" for host in hosts}
            training_rows = []
            registry_rows = []
            for host in hosts:
                dataset = root / "datasets" / host / "uniform.csv"
                dataset.parent.mkdir(parents=True, exist_ok=True)
                dataset.write_text("sample_id,presence\nexample,1\n", encoding="utf-8")
                run_dir = root / "runs" / experiments[host]
                checkpoint = run_dir / "checkpoints" / "best.pt"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_bytes(f"checkpoint-{host}".encode())
                config = root / "configs" / f"{experiments[host]}.json"
                write_json(config, {"csv_path": str(dataset), "postfit_calibration": "none"})
                for name in (
                    "run_status.json", "runtime_provenance.json", "training_design.json",
                    "validation_raw_calibration_metrics.json",
                    "test_raw_calibration_metrics.json", "validation_threshold.json",
                ):
                    write_json(run_dir / name, {"status": "complete"})
                for name in (
                    "training_history.csv", "model_summary_metrics.csv",
                    "validation_prediction_results.csv", "prediction_results.csv",
                ):
                    path = run_dir / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("value\n1\n", encoding="utf-8")
                common = {
                    "host": host,
                    "experiment": experiments[host],
                    "config_path": str(config),
                    "config_sha256": sha256(config),
                    "run_dir": str(run_dir),
                }
                registry_rows.append(
                    {**common, "approved_score_space": "raw_sigmoid_probability"}
                )
                training_rows.append(
                    {
                        **common,
                        "training_dataset_csv_sha256": sha256(dataset),
                        "best_checkpoint_sha256": sha256(checkpoint),
                        "postfit_calibration": "none",
                        "stage": "production-fit",
                        "evidence_gate_passed": True,
                    }
                )
            registry = root / "registry.tsv"
            pd.DataFrame(registry_rows).to_csv(registry, sep="\t", index=False)
            training_metrics = root / "accepted" / "training_metrics.csv"
            training_metrics.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(training_rows).to_csv(training_metrics, index=False)

            dataset_evidence_table = root / "selection" / "dataset_evidence.csv"
            dataset_evidence_table.parent.mkdir(parents=True, exist_ok=True)
            dataset_evidence_table.write_text("host,score\nexample,1\n", encoding="utf-8")
            dataset_evidence_summary = root / "selection" / "dataset_summary.json"
            write_json(dataset_evidence_summary, {"status": "complete"})
            dataset_selection = root / "selection" / "dataset_selection.json"
            write_json(
                dataset_selection,
                {
                    "status": "complete",
                    "source_evidence_summary": str(dataset_evidence_summary),
                    "source_evidence_summary_sha256": sha256(dataset_evidence_summary),
                    "source_combined_evidence": str(dataset_evidence_table),
                    "source_combined_evidence_sha256": sha256(dataset_evidence_table),
                },
            )
            loss_table = root / "selection" / "loss_evidence.csv"
            loss_table.write_text("host,score\nexample,1\n", encoding="utf-8")
            loss_summary = root / "selection" / "loss_summary.json"
            write_json(
                loss_summary,
                {
                    "status": "complete",
                    "selection_manifest": str(dataset_selection),
                    "selection_manifest_sha256": sha256(dataset_selection),
                    "source_dataset_evidence": str(dataset_evidence_table),
                    "source_dataset_evidence_sha256": sha256(dataset_evidence_table),
                },
            )
            loss_selection = root / "selection" / "loss_selection.json"
            write_json(
                loss_selection,
                {
                    "status": "complete",
                    "source_evidence_summary": str(loss_summary),
                    "source_evidence_summary_sha256": sha256(loss_summary),
                    "source_combined_evidence": str(loss_table),
                    "source_combined_evidence_sha256": sha256(loss_table),
                },
            )
            augmentation_table = root / "selection" / "augmentation_evidence.csv"
            augmentation_table.write_text("host,score\nexample,1\n", encoding="utf-8")
            augmentation_summary = root / "selection" / "augmentation_summary.json"
            write_json(
                augmentation_summary,
                {
                    "status": "complete",
                    "selection_manifest": str(loss_selection),
                    "selection_manifest_sha256": sha256(loss_selection),
                    "source_loss_evidence": str(loss_table),
                    "source_loss_evidence_sha256": sha256(loss_table),
                },
            )
            blockcv_selection = root / "selection" / "blockcv_selection.json"
            write_json(
                blockcv_selection,
                {
                    "status": "complete",
                    "source_evidence_summary": str(augmentation_summary),
                    "source_evidence_summary_sha256": sha256(augmentation_summary),
                    "source_combined_evidence": str(augmentation_table),
                    "source_combined_evidence_sha256": sha256(augmentation_table),
                },
            )
            blockcv_dir = root / "accepted" / "blockcv"
            blockcv_training = blockcv_dir / "training_metrics.csv"
            blockcv_folds = blockcv_dir / "blockcv_fold_metrics.csv"
            blockcv_hosts = blockcv_dir / "blockcv_host_summary.csv"
            for path in (blockcv_training, blockcv_folds, blockcv_hosts):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("host,value\nexample,1\n", encoding="utf-8")
            blockcv_summary = blockcv_dir / "summary.json"
            write_json(
                blockcv_summary,
                {
                    "status": "complete",
                    "selection_manifest": str(blockcv_selection),
                    "selection_manifest_sha256": sha256(blockcv_selection),
                    "training_metrics": str(blockcv_training),
                    "training_metrics_sha256": sha256(blockcv_training),
                    "fold_metrics": str(blockcv_folds),
                    "host_summary": str(blockcv_hosts),
                },
            )
            production_selection = root / "accepted" / "production_selection.json"
            write_json(
                production_selection,
                {
                    "status": "complete",
                    "blockcv_evidence_summary": str(blockcv_summary),
                    "blockcv_evidence_summary_sha256": sha256(blockcv_summary),
                    "blockcv_host_summary": str(blockcv_hosts),
                    "blockcv_host_summary_sha256": sha256(blockcv_hosts),
                    "blockcv_fold_metrics": str(blockcv_folds),
                    "blockcv_fold_metrics_sha256": sha256(blockcv_folds),
                    "blockcv_selection_manifest": str(blockcv_selection),
                    "blockcv_selection_manifest_sha256": sha256(blockcv_selection),
                },
            )

            exposure = root / "accepted" / "shared_exposure_counts.csv"
            exposure.write_text("domain,count\nexample,1\n", encoding="utf-8")
            overlap_summary = root / "accepted" / "dataset_overlap_summary.json"
            write_json(
                overlap_summary,
                {
                    "status": "complete", "shared_exposure_counts": str(exposure),
                    "shared_exposure_counts_sha256": sha256(exposure),
                },
            )
            support_predictions = root / "accepted" / "common_support_predictions.csv"
            pathway_correlations = root / "accepted" / "pathway_correlations.csv"
            support_predictions.write_text("sample,prediction\n1,0.5\n", encoding="utf-8")
            pathway_correlations.write_text("pathway,correlation\nfull,0.5\n", encoding="utf-8")
            pathway_summary = root / "accepted" / "pathway_summary.json"
            write_json(
                pathway_summary,
                {
                    "status": "complete",
                    "common_support_predictions": str(support_predictions),
                    "common_support_predictions_sha256": sha256(support_predictions),
                    "cross_host_pathway_correlations": str(pathway_correlations),
                    "cross_host_pathway_correlations_sha256": sha256(pathway_correlations),
                },
            )
            host_acceptance = root / "accepted" / "host_acceptance.csv"
            host_acceptance.write_text("host,passed\nexample,true\n", encoding="utf-8")

            def make_artifact_batch(name, selected_hosts, exact=False):
                batch_root = root / "artifacts" / name
                selected_experiments = [experiments[host] for host in selected_hosts]
                write_json(
                    batch_root / "batch_status.json",
                    {
                        "status": "complete", "candidate_count": len(selected_experiments),
                        "completed_experiments": selected_experiments,
                    },
                )
                for experiment in selected_experiments:
                    candidate = batch_root / experiment
                    write_json(candidate / "summary.json", {"status": "complete"})
                    if exact:
                        (candidate / "exact_overlap_candidate_metrics.csv").write_text(
                            "value\n1\n", encoding="utf-8"
                        )
                        (candidate / "paired_june_comparison.csv").write_text(
                            "value\n1\n", encoding="utf-8"
                        )
                    else:
                        (candidate / "pair_metrics.csv").write_text("value\n1\n", encoding="utf-8")
                        comparison = candidate / "june_baseline_comparison"
                        write_json(comparison / "summary.json", {"status": "complete"})
                        (comparison / "paired_artifact_comparison.csv").write_text(
                            "value\n1\n", encoding="utf-8"
                        )
                return batch_root

            eastern_hosts = [host for host in hosts if host != "notholithocarpus_densiflorus"]
            raw_artifact = make_artifact_batch("raw_matched", hosts)
            configured_artifact = make_artifact_batch("configured_matched", hosts)
            raw_exact = make_artifact_batch("raw_exact", eastern_hosts, exact=True)
            configured_exact = make_artifact_batch("configured_exact", eastern_hosts, exact=True)

            acceptance = root / "acceptance.json"
            write_json(
                acceptance,
                {
                    "status": "complete",
                    "production_ready_for_inference": True,
                    "training_metrics": str(training_metrics),
                    "training_metrics_sha256": sha256(training_metrics),
                    "blockcv_evidence_dir": str(blockcv_dir),
                    "blockcv_summary_sha256": sha256(blockcv_summary),
                    "production_selection_manifest": str(production_selection),
                    "production_selection_manifest_sha256": sha256(production_selection),
                    "dataset_overlap_summary": str(overlap_summary),
                    "dataset_overlap_summary_sha256": sha256(overlap_summary),
                    "pathway_summary": str(pathway_summary),
                    "pathway_summary_sha256": sha256(pathway_summary),
                    "host_acceptance": str(host_acceptance),
                    "host_acceptance_sha256": sha256(host_acceptance),
                    "raw_artifact_roots": [str(raw_artifact)],
                    "configured_artifact_roots": [str(configured_artifact)],
                    "raw_exact_overlap_roots": [str(raw_exact)],
                    "configured_exact_overlap_roots": [str(configured_exact)],
                },
            )
            bundle = root / "bundle.json"
            bundle_files = {
                "code_snapshot/inference_multimodal_production.py": "inference code\n",
                "code_snapshot/model.py": "model code\n",
                "production_experiment_registry.tsv": registry.read_text(encoding="utf-8"),
                "final_model_acceptance_summary.json": acceptance.read_text(encoding="utf-8"),
            }
            bundle_hashes = {}
            for relative, content in bundle_files.items():
                bundled_path = root / relative
                bundled_path.parent.mkdir(parents=True, exist_ok=True)
                bundled_path.write_text(content, encoding="utf-8")
                bundle_hashes[relative] = sha256(bundled_path)
            write_json(
                bundle,
                {
                    "status": "complete",
                    "source_registry_sha256": sha256(registry),
                    "acceptance_summary_sha256": sha256(acceptance),
                    "code_commit": "deadbeef",
                    "file_sha256": bundle_hashes,
                    "inference_defaults": {
                        "stride": 64,
                        "blend_window": "hann",
                        "blend_edge_floor": 0.05,
                    },
                },
            )

            attrition = []
            provenance = []
            inference = []
            mosaics = []
            mosaic_payloads = {}
            visual_reviews = []
            for host in hosts:
                region = "west" if host == "notholithocarpus_densiflorus" else "east"
                candidate_dir = root / "candidates" / host
                candidate_metadata = candidate_dir / "manifest_metadata.json"
                candidate_csv = candidate_dir / "extraction_candidates.csv"
                candidate_environment = candidate_dir / "extraction_environment_raw.csv"
                candidate_csv.parent.mkdir(parents=True, exist_ok=True)
                candidate_csv.write_text("sample_id\nexample\n", encoding="utf-8")
                candidate_environment.write_text("sample_id,value\nexample,1\n", encoding="utf-8")
                write_json(candidate_metadata, {"status": "complete"})
                materialization_state = root / "materialized" / host / "materialization_run.json"
                write_json(
                    materialization_state,
                    {
                        "status": "complete",
                        "identity": {
                            "extraction_candidates_sha256": sha256(candidate_csv),
                            "extraction_environment_raw_sha256": sha256(candidate_environment),
                        },
                    },
                )
                path = root / "provenance" / host / "summary.json"
                write_json(
                    path,
                    {
                        "status": "complete", "production_ready": True, "host": host,
                        "candidate_metadata": str(candidate_metadata),
                        "candidate_metadata_sha256": sha256(candidate_metadata),
                        "materialization_state": str(materialization_state),
                        "materialization_state_sha256": sha256(materialization_state),
                    },
                )
                provenance.append(path)
                progress = root / "materialized" / host / "progress.jsonl"
                canonical = root / "materialized" / host / "canonical_records.jsonl"
                progress.write_text('{"status":"success"}\n', encoding="utf-8")
                canonical.write_text('{"status":"success"}\n', encoding="utf-8")
                qa_outputs = {}
                for label in (
                    "class_attrition", "source_domain_attrition", "failure_types",
                    "final_manifest_domain_shift",
                ):
                    qa_path = root / "attrition" / host / f"{label}.csv"
                    qa_path.parent.mkdir(parents=True, exist_ok=True)
                    qa_path.write_text("value\n1\n", encoding="utf-8")
                    qa_outputs[label] = str(qa_path)
                path = root / "attrition" / host / "summary.json"
                attrition_inputs = {
                    "candidate_csv": str(candidate_csv),
                    "candidate_metadata": str(candidate_metadata),
                    "progress_jsonl": str(progress),
                    "canonical_records": str(canonical),
                    "materialization_state": str(materialization_state),
                }
                attrition_inputs.update(
                    {f"{key}_sha256": sha256(Path(value)) for key, value in list(attrition_inputs.items())}
                )
                write_json(
                    path,
                    {
                        "status": "complete", "production_ready": True, "host": host,
                        "inputs": attrition_inputs, "outputs": qa_outputs,
                    },
                )
                attrition.append(path)
                for product in ("deterministic", "uncertainty"):
                    tile_manifest = root / "inference" / host / product / "tiles.csv"
                    tile_qa = root / "inference" / host / product / "tile_output_qa.csv"
                    tile_manifest.parent.mkdir(parents=True, exist_ok=True)
                    tile_manifest.write_text("path\nexample.tif\n", encoding="utf-8")
                    tile_qa.write_text("tile,passed\nexample,true\n", encoding="utf-8")
                    path = root / "inference" / host / product / "summary.json"
                    write_json(
                        path,
                        {
                            "status": "complete", "production_ready": True,
                            "host": host, "region": region, "product": product,
                            "expected_score_space": "raw_sigmoid_probability",
                            "expected_tile_count": 1,
                            "tile_manifest": str(tile_manifest),
                            "tile_manifest_sha256": sha256(tile_manifest),
                            "tile_output_qa": str(tile_qa),
                            "tile_output_qa_sha256": sha256(tile_qa),
                        },
                    )
                    inference.append(path)
                products = {}
                for product in ("prediction", "uncertainty", "count"):
                    raster = root / "mosaics" / host / f"{product}.tif"
                    raster.parent.mkdir(parents=True, exist_ok=True)
                    raster.write_bytes(f"{host}-{product}".encode())
                    products[product] = {"path": str(raster), "sha256": sha256(raster)}
                quickview_outputs = {}
                for label in (
                    "probability_transparent", "probability_framed",
                    "uncertainty_transparent", "uncertainty_framed",
                    "support_transparent", "support_framed",
                    "difference_from_june_transparent", "difference_from_june_framed",
                ):
                    quickview_image = root / "mosaics" / host / f"{label}.png"
                    quickview_image.write_bytes(f"{host}-{label}".encode())
                    quickview_outputs[label] = str(quickview_image)
                quickview_summary = root / "mosaics" / host / "quickview_summary.json"
                write_json(quickview_summary, {"status": "complete", "outputs": quickview_outputs})
                parity_detail = root / "mosaics" / host / "native_mosaic_parity.csv"
                parity_detail.write_text("tile,layer\nexample,prediction\n", encoding="utf-8")
                parity_summary = root / "mosaics" / host / "native_mosaic_parity.json"
                write_json(
                    parity_summary,
                    {
                        "status": "complete", "production_ready": True,
                        "host": host, "region": region,
                        "detail": str(parity_detail),
                        "detail_sha256": sha256(parity_detail),
                    },
                )
                path = root / "mosaics" / host / "qa.json"
                payload = {
                    "status": "complete",
                    "production_ready_for_visual_review": True,
                    "host": host, "region": region, "products": products,
                    "quickview_summary": str(quickview_summary),
                    "quickview_summary_sha256": sha256(quickview_summary),
                    "native_mosaic_parity_summary": str(parity_summary),
                    "native_mosaic_parity_summary_sha256": sha256(parity_summary),
                }
                write_json(path, payload)
                mosaics.append(path)
                mosaic_payloads[host] = payload
                visual_reviews.append(
                    {
                        "host": host, "passed": True,
                        "mosaic_qa_summary_sha256": sha256(path),
                    }
                )
            inference_by_key = {
                (read_json(path)["host"], read_json(path)["product"]): path
                for path in inference
            }
            inference_submission = root / "submissions" / "inference.tsv"
            inference_submission.parent.mkdir(parents=True)
            inference_rows = []
            for index, ((host, product), path) in enumerate(
                sorted(inference_by_key.items())
            ):
                payload = read_json(path)
                inference_rows.append(
                    {
                        "host": host, "region": payload["region"],
                        "product": product, "array_job_id": 1000 + index,
                        "qa_job_id": 2000 + index,
                        "tile_manifest_sha256": payload["tile_manifest_sha256"],
                        "production_registry_sha256": sha256(registry),
                        "bundle_metadata_sha256": sha256(bundle), "dry_run": 0,
                    }
                )
            pd.DataFrame(inference_rows).to_csv(
                inference_submission, sep="\t", index=False
            )
            mosaic_submission = root / "submissions" / "mosaics.tsv"
            mosaic_rows = []
            for index, host in enumerate(hosts):
                deterministic = inference_by_key[(host, "deterministic")]
                uncertainty = inference_by_key[(host, "uncertainty")]
                deterministic_payload = read_json(deterministic)
                mosaic_rows.append(
                    {
                        "host": host, "region": deterministic_payload["region"],
                        "mosaic_array_job_id": 3000 + index,
                        "qa_job_id": 4000 + index, "expected_input_count": 1,
                        "deterministic_qa_summary": str(deterministic),
                        "deterministic_qa_sha256": sha256(deterministic),
                        "uncertainty_qa_summary": str(uncertainty),
                        "uncertainty_qa_sha256": sha256(uncertainty),
                        "dry_run": 0,
                    }
                )
            pd.DataFrame(mosaic_rows).to_csv(
                mosaic_submission, sep="\t", index=False
            )
            review_spec = root / "review_spec.json"
            write_json(review_spec, {"reviewer": "test"})
            visual = root / "visual.json"
            write_json(
                visual,
                {
                    "status": "complete",
                    "approved_for_release_evidence": True,
                    "reviews": visual_reviews,
                    "review_spec": str(review_spec),
                    "review_spec_sha256": sha256(review_spec),
                },
            )
            correlation = root / "correlation.json"
            write_json(
                correlation,
                {
                    "status": "complete",
                    "first_label": "ailanthus_altissima",
                    "second_label": "juglans_nigra",
                    "first_raster_sha256": mosaic_payloads["ailanthus_altissima"]["products"]["prediction"]["sha256"],
                    "second_raster_sha256": mosaic_payloads["juglans_nigra"]["products"]["prediction"]["sha256"],
                },
            )
            hann = root / "HANN_RESULTS.md"
            hann.write_text("validated Hann results\n", encoding="utf-8")
            result = assemble_release_evidence(
                acceptance, registry, bundle,
                [inference_submission, mosaic_submission],
                provenance, attrition, inference, mosaics,
                visual, correlation, hann, root / "release",
            )
            self.assertTrue(result["production_ready_for_collaborator_review"])
            self.assertTrue((root / "release" / "PRODUCTION_READY").is_file())
            self.assertGreater(result["indexed_artifact_count"], 150)
            artifact_index = pd.read_csv(result["artifact_index"])
            self.assertTrue(
                {
                    "selected_training_dataset", "production_config",
                    "production_checkpoint", "production_run_evidence",
                    "selection_manifest", "blockcv_evidence",
                    "raw_matched_boundary_candidate_evidence",
                    "configured_exact_overlap_candidate_evidence",
                    "hazel_bundle_file", "fixed_scale_quickview",
                    "native_mosaic_parity_summary", "native_mosaic_parity_detail",
                    "hazel_submission_log",
                }.issubset(set(artifact_index["category"]))
            )
            self.assertEqual(
                int((artifact_index["category"] == "selected_training_dataset").sum()),
                5,
            )
            self.assertEqual(
                int((artifact_index["category"] == "production_checkpoint").sum()),
                5,
            )
            canonical = artifact_index[
                (artifact_index["category"] == "materialization_input")
                & artifact_index["path"].str.endswith("canonical_records.jsonl")
            ]
            self.assertEqual(len(canonical), 5)


if __name__ == "__main__":
    unittest.main()
