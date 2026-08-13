#!/usr/bin/env python3
"""Assemble the hashed, five-host August production release evidence index."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from august2026_summarize_experiments import read_json, sha256, write_json_atomic


HOSTS = {
    "ailanthus_altissima",
    "juglans_nigra",
    "prunus_serotina",
    "vitis_labrusca",
    "notholithocarpus_densiflorus",
}
PRODUCTS = {"deterministic", "uncertainty"}


def _write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    partial = path.with_name(f"{path.name}.partial")
    frame.to_csv(partial, index=False)
    os.replace(partial, path)


def _index(rows: list[dict], category: str, path: Path, **identity) -> None:
    rows.append(
        {
            "category": category,
            "host": identity.get("host"),
            "region": identity.get("region"),
            "product": identity.get("product"),
            "path": str(path.resolve()),
            "sha256": sha256(path),
        }
    )


def assemble_release_evidence(
    final_acceptance_path: Path,
    production_registry_path: Path,
    hazel_bundle_metadata_path: Path,
    attrition_paths: list[Path],
    inference_qa_paths: list[Path],
    mosaic_qa_paths: list[Path],
    visual_review_path: Path,
    aligned_correlation_path: Path,
    hann_results_path: Path,
    output_dir: Path,
) -> dict:
    failures = []
    index_rows: list[dict] = []

    acceptance = read_json(final_acceptance_path)
    if acceptance.get("status") != "complete" or not acceptance.get(
        "production_ready_for_inference"
    ):
        failures.append("final_model_acceptance")
    _index(index_rows, "final_model_acceptance", final_acceptance_path)

    registry = pd.read_csv(production_registry_path, sep="\t", low_memory=False)
    if len(registry) != 5 or set(registry["host"]) != HOSTS:
        failures.append("production_registry_host_set")
    if registry["experiment"].duplicated().any():
        failures.append("production_registry_duplicate_experiment")
    _index(index_rows, "production_registry", production_registry_path)
    registry_by_host = registry.set_index("host") if set(registry["host"]) == HOSTS else None

    bundle = read_json(hazel_bundle_metadata_path)
    if bundle.get("status") != "complete":
        failures.append("hazel_bundle_status")
    if bundle.get("source_registry_sha256") != sha256(production_registry_path):
        failures.append("hazel_bundle_registry_hash")
    if bundle.get("acceptance_summary_sha256") != sha256(final_acceptance_path):
        failures.append("hazel_bundle_acceptance_hash")
    defaults = bundle.get("inference_defaults") or {}
    if (
        defaults.get("stride") != 64
        or defaults.get("blend_window") != "hann"
        or float(defaults.get("blend_edge_floor", -1)) != 0.05
    ):
        failures.append("hazel_bundle_hann_defaults")
    _index(index_rows, "hazel_inference_bundle", hazel_bundle_metadata_path)

    attrition_by_host = {}
    for path in attrition_paths:
        payload = read_json(path)
        host = payload.get("host")
        if host in attrition_by_host:
            failures.append(f"duplicate_attrition:{host}")
        attrition_by_host[host] = payload
        if payload.get("status") != "complete" or not payload.get("production_ready"):
            failures.append(f"attrition_not_ready:{host}")
        _index(index_rows, "materialization_attrition", path, host=host)
    if set(attrition_by_host) != HOSTS:
        failures.append("attrition_host_set")

    inference_by_key = {}
    for path in inference_qa_paths:
        payload = read_json(path)
        key = (payload.get("host"), payload.get("product"))
        if key in inference_by_key:
            failures.append(f"duplicate_inference_qa:{key}")
        inference_by_key[key] = payload
        host, product = key
        if payload.get("status") != "complete" or not payload.get("production_ready"):
            failures.append(f"inference_qa_not_ready:{host}:{product}")
        if registry_by_host is not None and host in registry_by_host.index:
            if payload.get("expected_score_space") != registry_by_host.loc[
                host, "approved_score_space"
            ]:
                failures.append(f"inference_score_space:{host}:{product}")
        _index(
            index_rows, "tiled_inference_qa", path,
            host=host, region=payload.get("region"), product=product,
        )
    expected_inference = {(host, product) for host in HOSTS for product in PRODUCTS}
    if set(inference_by_key) != expected_inference:
        failures.append("inference_qa_host_product_set")

    mosaic_by_host = {}
    for path in mosaic_qa_paths:
        payload = read_json(path)
        host = payload.get("host")
        if host in mosaic_by_host:
            failures.append(f"duplicate_mosaic_qa:{host}")
        mosaic_by_host[host] = payload
        if payload.get("status") != "complete" or not payload.get(
            "production_ready_for_visual_review"
        ):
            failures.append(f"mosaic_qa_not_ready:{host}")
        _index(index_rows, "mosaic_qa", path, host=host, region=payload.get("region"))
        for product, record in (payload.get("products") or {}).items():
            raster_path = Path(record.get("path", ""))
            if not raster_path.is_file() or sha256(raster_path) != record.get("sha256"):
                failures.append(f"mosaic_raster_hash:{host}:{product}")
            else:
                _index(
                    index_rows, "release_mosaic", raster_path,
                    host=host, region=payload.get("region"), product=product,
                )
    if set(mosaic_by_host) != HOSTS:
        failures.append("mosaic_qa_host_set")

    visual = read_json(visual_review_path)
    if visual.get("status") != "complete" or not visual.get(
        "approved_for_release_evidence"
    ):
        failures.append("visual_review")
    visual_reviews = visual.get("reviews") or []
    if {item.get("host") for item in visual_reviews} != HOSTS:
        failures.append("visual_review_host_set")
    for item in visual_reviews:
        host = item.get("host")
        if not item.get("passed"):
            failures.append(f"visual_review_host:{host}")
        if host in mosaic_by_host and item.get("mosaic_qa_summary_sha256") != next(
            (
                row["sha256"] for row in index_rows
                if row["category"] == "mosaic_qa" and row["host"] == host
            ),
            None,
        ):
            failures.append(f"visual_review_mosaic_hash:{host}")
    _index(index_rows, "visual_review", visual_review_path)

    correlation = read_json(aligned_correlation_path)
    if correlation.get("status") != "complete" or {
        correlation.get("first_label"), correlation.get("second_label")
    } != {"ailanthus_altissima", "juglans_nigra"}:
        failures.append("aligned_correlation_identity")
    expected_correlation_hashes = {
        (mosaic_by_host.get(host, {}).get("products") or {}).get("prediction", {}).get(
            "sha256"
        )
        for host in ("ailanthus_altissima", "juglans_nigra")
    }
    if {
        correlation.get("first_raster_sha256"),
        correlation.get("second_raster_sha256"),
    } != expected_correlation_hashes:
        failures.append("aligned_correlation_raster_hashes")
    _index(index_rows, "aligned_cross_host_correlation", aligned_correlation_path)

    if not hann_results_path.is_file():
        failures.append("hann_validation_results")
    else:
        _index(index_rows, "hann_validation_results", hann_results_path)

    for key in (
        "training_metrics", "blockcv_evidence_dir", "production_selection_manifest",
        "dataset_overlap_summary", "pathway_summary", "host_acceptance",
    ):
        raw = acceptance.get(key)
        if not raw:
            failures.append(f"acceptance_pointer:{key}")
            continue
        pointer = Path(raw)
        if key == "blockcv_evidence_dir":
            pointer = pointer / "summary.json"
        if not pointer.is_file():
            failures.append(f"acceptance_pointer_missing:{key}")
        else:
            _index(index_rows, f"accepted_{key}", pointer)

    output_dir.mkdir(parents=True, exist_ok=False)
    index_path = output_dir / "artifact_index.csv"
    _write_csv_atomic(index_path, pd.DataFrame(index_rows))
    payload = {
        "status": "complete",
        "production_ready_for_collaborator_review": not failures,
        "host_count": len(HOSTS),
        "indexed_artifact_count": len(index_rows),
        "failure_count": len(failures),
        "failures": failures,
        "artifact_index": str(index_path.resolve()),
        "artifact_index_sha256": sha256(index_path),
        "code_commit": bundle.get("code_commit"),
        "release_score_interpretation": (
            "Outputs are sampling-design-conditional suitability scores, not literal "
            "occurrence probabilities. Candidate-specific calibration is preserved in "
            "the tiled raster tags and QA evidence."
        ),
        "known_limitations": [
            "Hann blending mitigates the fine stride lattice but is not a correction for broad acquisition-domain sensitivity.",
            "Cross-host map similarity requires interpretation alongside dataset-overlap and pathway-isolation audits.",
            "Visual approval is limited to the fixed-scale products and regions recorded in the review manifest.",
        ],
    }
    write_json_atomic(output_dir / "summary.json", payload)
    marker = "PRODUCTION_READY" if not failures else "NOT_PRODUCTION_READY"
    (output_dir / marker).write_text(marker.lower() + "\n", encoding="utf-8")
    if failures:
        raise RuntimeError(f"August release evidence failed: {failures}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-acceptance", type=Path, required=True)
    parser.add_argument("--production-registry", type=Path, required=True)
    parser.add_argument("--hazel-bundle-metadata", type=Path, required=True)
    parser.add_argument("--attrition-summary", type=Path, action="append", required=True)
    parser.add_argument("--inference-qa-summary", type=Path, action="append", required=True)
    parser.add_argument("--mosaic-qa-summary", type=Path, action="append", required=True)
    parser.add_argument("--visual-review", type=Path, required=True)
    parser.add_argument("--aligned-correlation", type=Path, required=True)
    parser.add_argument("--hann-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required = [
        args.final_acceptance, args.production_registry, args.hazel_bundle_metadata,
        args.visual_review, args.aligned_correlation, args.hann_results,
        *args.attrition_summary, *args.inference_qa_summary, *args.mosaic_qa_summary,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing release evidence inputs: {missing}")
    payload = assemble_release_evidence(
        args.final_acceptance, args.production_registry, args.hazel_bundle_metadata,
        args.attrition_summary, args.inference_qa_summary, args.mosaic_qa_summary,
        args.visual_review, args.aligned_correlation, args.hann_results,
        args.output_dir,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
