#!/usr/bin/env python3
"""Validate three aligned EPSG:5070 production mosaics and their quickviews."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import rasterio

from august2026_summarize_experiments import read_json, sha256, write_json_atomic


def close(first, second, tolerance=1e-6) -> bool:
    return math.isclose(first, second, rel_tol=0.0, abs_tol=tolerance)


def validate_mosaics(
    prediction: Path,
    uncertainty: Path,
    count: Path,
    quickview_summary: Path,
    parity_summary: Path,
    expected_input_count: int,
    target_resolution: float,
) -> dict:
    failures = []
    products = {}
    reference = None
    quickview = None
    parity = None
    for label, path in (
        ("prediction", prediction),
        ("uncertainty", uncertainty),
        ("count", count),
    ):
        info_path = path.with_name(f"{path.stem}_gdalinfo.json")
        status_path = path.parent / "logs" / f"{label}_mosaic_status.json"
        if not path.is_file() or not info_path.is_file():
            failures.append(f"missing_{label}_or_gdalinfo")
            continue
        with rasterio.open(path) as source:
            record = {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "crs": str(source.crs),
                "width": source.width,
                "height": source.height,
                "bounds": list(source.bounds),
                "transform": list(source.transform),
                "nodata": source.nodata,
                "dtype": source.dtypes[0],
                "statistics_valid_percent": source.tags(1).get(
                    "STATISTICS_VALID_PERCENT"
                ),
            }
            if source.crs is None or source.crs.to_epsg() != 5070:
                failures.append(f"{label}_crs")
            if not close(source.transform.a, target_resolution) or not close(
                abs(source.transform.e), target_resolution
            ):
                failures.append(f"{label}_resolution")
            if not close(source.transform.c / target_resolution, round(source.transform.c / target_resolution)):
                failures.append(f"{label}_x_alignment")
            if not close(source.transform.f / target_resolution, round(source.transform.f / target_resolution)):
                failures.append(f"{label}_y_alignment")
            expected_dtype = "uint16" if label == "count" else "float32"
            if source.dtypes[0] != expected_dtype:
                failures.append(f"{label}_dtype")
            expected_nodata = 0.0 if label == "count" else -9999.0
            if source.nodata is None or not close(float(source.nodata), expected_nodata):
                failures.append(f"{label}_nodata")
            geometry = (
                source.width, source.height, source.crs,
                tuple(source.transform), tuple(source.bounds),
            )
            if reference is None:
                reference = geometry
            elif geometry != reference:
                failures.append(f"{label}_grid")
            products[label] = record
        info = read_json(info_path)
        if not info.get("bands"):
            failures.append(f"{label}_gdalinfo_bands")
        else:
            band = info["bands"][0]
            minimum = band.get("minimum")
            maximum = band.get("maximum")
            if minimum is None or maximum is None:
                failures.append(f"{label}_gdalinfo_statistics")
            elif label == "prediction" and (minimum < 0 or maximum > 1):
                failures.append("prediction_range")
            elif label == "uncertainty" and (minimum < 0 or maximum > 0.5 + 1e-6):
                failures.append("uncertainty_range")
            elif label == "count" and (minimum < 0 or maximum > 16):
                failures.append("count_range")
        if not status_path.is_file():
            failures.append(f"missing_{label}_mosaic_status")
        else:
            status = read_json(status_path)
            if status.get("status") != "complete":
                failures.append(f"{label}_mosaic_status")
            if status.get("release_product") != label:
                failures.append(f"{label}_mosaic_product")
            try:
                observed_input_count = int(status.get("input_count", -1))
            except (TypeError, ValueError):
                observed_input_count = -1
            if observed_input_count != expected_input_count:
                failures.append(f"{label}_mosaic_input_count")
            try:
                status_output = Path(status.get("output", "")).resolve()
            except (OSError, RuntimeError, TypeError):
                status_output = None
            if status_output != path.resolve():
                failures.append(f"{label}_mosaic_output")
            record["mosaic_status"] = str(status_path.resolve())
            record["mosaic_status_sha256"] = sha256(status_path)
    if not quickview_summary.is_file():
        failures.append("missing_quickview_summary")
    else:
        quickview = read_json(quickview_summary)
        if quickview.get("status") != "complete":
            failures.append("quickview_status")
        if quickview.get("mask_definition") != "count > 0 AND finite(product)":
            failures.append("quickview_mask")
        if quickview.get("prediction_scale") != [0.0, 1.0]:
            failures.append("quickview_prediction_scale")
        if quickview.get("uncertainty_scale") != [0.0, 0.1]:
            failures.append("quickview_uncertainty_scale")
        if quickview.get("support_scale") != [0.0, 16.0]:
            failures.append("quickview_support_scale")
        if quickview.get("difference_from_june_scale") != [-0.5, 0.5]:
            failures.append("quickview_difference_scale")
        if not quickview.get("reference_prediction_sha256"):
            failures.append("quickview_reference_prediction")
        expected_outputs = {
            f"{label}_{frame}"
            for label in (
                "probability",
                "uncertainty",
                "support",
                "difference_from_june",
            )
            for frame in ("transparent", "framed")
        }
        if set((quickview.get("outputs") or {})) != expected_outputs:
            failures.append("quickview_output_set")
        for path in (quickview.get("outputs") or {}).values():
            if not Path(path).is_file():
                failures.append("missing_quickview_png")
        for label in ("prediction", "uncertainty", "count"):
            if label in products and quickview.get(f"{label}_sha256") != products[label]["sha256"]:
                failures.append(f"quickview_{label}_hash")
    if not parity_summary.is_file():
        failures.append("missing_native_mosaic_parity")
    else:
        parity = read_json(parity_summary)
        if parity.get("status") != "complete" or not parity.get("production_ready"):
            failures.append("native_mosaic_parity_status")
        if quickview and (
            parity.get("host") != quickview.get("host")
            or parity.get("region") != quickview.get("region")
        ):
            failures.append("native_mosaic_parity_identity")
        for label in ("prediction", "uncertainty"):
            if label in products and parity.get(f"{label}_mosaic_sha256") != products[label]["sha256"]:
                failures.append(f"native_mosaic_parity_{label}_hash")
    return {
        "status": "complete",
        "production_ready_for_visual_review": not failures,
        "host": quickview.get("host") if quickview else None,
        "region": quickview.get("region") if quickview else None,
        "expected_input_count": expected_input_count,
        "target_epsg": 5070,
        "target_resolution_m": target_resolution,
        "products": products,
        "quickview_summary": str(quickview_summary.resolve()),
        "quickview_summary_sha256": (
            sha256(quickview_summary) if quickview_summary.is_file() else None
        ),
        "native_mosaic_parity_summary": str(parity_summary.resolve()),
        "native_mosaic_parity_summary_sha256": (
            sha256(parity_summary) if parity_summary.is_file() else None
        ),
        "failure_count": len(failures),
        "failures": failures,
        "visual_review_required": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--uncertainty", type=Path, required=True)
    parser.add_argument("--count", type=Path, required=True)
    parser.add_argument("--quickview-summary", type=Path, required=True)
    parser.add_argument("--parity-summary", type=Path, required=True)
    parser.add_argument("--expected-input-count", type=int, required=True)
    parser.add_argument("--target-resolution", type=float, default=256.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace mosaic QA: {args.output}")
    payload = validate_mosaics(
        args.prediction, args.uncertainty, args.count, args.quickview_summary,
        args.parity_summary,
        args.expected_input_count, args.target_resolution,
    )
    write_json_atomic(args.output, payload)
    if payload["failures"]:
        raise RuntimeError(f"Mosaic QA failed: {payload['failures']}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
