import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import numpy as np
import rasterio
from rasterio.transform import from_origin

from august2026_record_visual_review import HOSTS, REVIEW_TARGETS, record_visual_review
from august2026_release_evidence import assemble_release_evidence
from august2026_summarize_experiments import sha256
from august2026_validate_mosaics import validate_mosaics


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class AugustVisualReviewTests(unittest.TestCase):
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
            ):
                path = root / f"{label}.png"
                path.write_bytes(label.encode())
                quick_outputs[label] = str(path)
            quickview = root / "quickview_summary.json"
            write_json(
                quickview,
                {
                    "status": "complete", "host": "ailanthus_altissima", "region": "east",
                    "mask_definition": "count > 0 AND finite(product)",
                    "prediction_scale": [0.0, 1.0], "uncertainty_scale": [0.0, 0.1],
                    "prediction_sha256": sha256(products["prediction"]),
                    "uncertainty_sha256": sha256(products["uncertainty"]),
                    "count_sha256": sha256(products["count"]), "outputs": quick_outputs,
                },
            )
            result = validate_mosaics(
                products["prediction"], products["uncertainty"], products["count"],
                quickview, expected_input_count=12, target_resolution=256,
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
    def test_release_evidence_requires_complete_five_host_chain(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            hosts = sorted(HOSTS)
            registry = root / "registry.tsv"
            pd.DataFrame(
                [
                    {
                        "host": host,
                        "experiment": f"production_{host}",
                        "approved_score_space": "raw_sigmoid_probability",
                    }
                    for host in hosts
                ]
            ).to_csv(registry, sep="\t", index=False)

            pointer_paths = {}
            for key in (
                "training_metrics", "production_selection_manifest",
                "dataset_overlap_summary", "pathway_summary", "host_acceptance",
            ):
                path = root / "accepted" / f"{key}.json"
                write_json(path, {"status": "complete"})
                pointer_paths[key] = str(path)
            blockcv_dir = root / "accepted" / "blockcv"
            write_json(blockcv_dir / "summary.json", {"status": "complete"})
            acceptance = root / "acceptance.json"
            write_json(
                acceptance,
                {
                    "status": "complete",
                    "production_ready_for_inference": True,
                    **pointer_paths,
                    "blockcv_evidence_dir": str(blockcv_dir),
                },
            )
            bundle = root / "bundle.json"
            write_json(
                bundle,
                {
                    "status": "complete",
                    "source_registry_sha256": sha256(registry),
                    "acceptance_summary_sha256": sha256(acceptance),
                    "code_commit": "deadbeef",
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
                path = root / "provenance" / host / "summary.json"
                write_json(path, {"status": "complete", "production_ready": True, "host": host})
                provenance.append(path)
                path = root / "attrition" / host / "summary.json"
                write_json(path, {"status": "complete", "production_ready": True, "host": host})
                attrition.append(path)
                for product in ("deterministic", "uncertainty"):
                    path = root / "inference" / host / product / "summary.json"
                    write_json(
                        path,
                        {
                            "status": "complete", "production_ready": True,
                            "host": host, "region": region, "product": product,
                            "expected_score_space": "raw_sigmoid_probability",
                        },
                    )
                    inference.append(path)
                products = {}
                for product in ("prediction", "uncertainty", "count"):
                    raster = root / "mosaics" / host / f"{product}.tif"
                    raster.parent.mkdir(parents=True, exist_ok=True)
                    raster.write_bytes(f"{host}-{product}".encode())
                    products[product] = {"path": str(raster), "sha256": sha256(raster)}
                path = root / "mosaics" / host / "qa.json"
                payload = {
                    "status": "complete",
                    "production_ready_for_visual_review": True,
                    "host": host, "region": region, "products": products,
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
            visual = root / "visual.json"
            write_json(
                visual,
                {
                    "status": "complete",
                    "approved_for_release_evidence": True,
                    "reviews": visual_reviews,
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
                acceptance, registry, bundle, provenance, attrition, inference, mosaics,
                visual, correlation, hann, root / "release",
            )
            self.assertTrue(result["production_ready_for_collaborator_review"])
            self.assertTrue((root / "release" / "PRODUCTION_READY").is_file())
            self.assertGreater(result["indexed_artifact_count"], 20)


if __name__ == "__main__":
    unittest.main()
