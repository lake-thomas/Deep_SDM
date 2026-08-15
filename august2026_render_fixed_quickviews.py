#!/usr/bin/env python3
"""Render fixed-scale probability, uncertainty, support, and change quickviews."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from august2026_summarize_experiments import sha256, write_json_atomic


PROBABILITY_STOPS = np.asarray([0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
PROBABILITY_COLORS = [
    (49, 54, 149), (69, 117, 180), (116, 173, 209),
    (254, 224, 144), (244, 109, 67), (165, 0, 38),
]
UNCERTAINTY_STOPS = np.asarray([0.0, 0.02, 0.04, 0.06, 0.08, 0.10])
UNCERTAINTY_COLORS = [
    (255, 255, 255), (224, 243, 219), (168, 221, 181),
    (67, 162, 202), (8, 104, 172), (3, 19, 43),
]
SUPPORT_STOPS = np.asarray([0.0, 1.0, 2.0, 4.0, 8.0, 16.0])
SUPPORT_COLORS = [
    (255, 255, 255), (240, 249, 232), (186, 228, 188),
    (123, 204, 196), (67, 162, 202), (8, 104, 172),
]
DIFFERENCE_STOPS = np.asarray([-0.50, -0.25, -0.10, 0.0, 0.10, 0.25, 0.50])
DIFFERENCE_COLORS = [
    (103, 0, 31), (178, 24, 43), (239, 138, 98), (247, 247, 247),
    (103, 169, 207), (33, 102, 172), (5, 48, 97),
]


def read_preview(path: Path, width: int, height: int, resampling) -> np.ndarray:
    with rasterio.open(path) as source:
        values = source.read(
            1, out_shape=(height, width), resampling=resampling
        ).astype(np.float32)
        if source.nodata is not None:
            values[values == source.nodata] = np.nan
    return values


def read_aligned_reference(
    path: Path,
    prediction_path: Path,
    width: int,
    height: int,
) -> np.ndarray:
    with rasterio.open(prediction_path) as prediction, rasterio.open(path) as source:
        with WarpedVRT(
            source,
            crs=prediction.crs,
            transform=prediction.transform,
            width=prediction.width,
            height=prediction.height,
            resampling=Resampling.bilinear,
            src_nodata=source.nodata,
            nodata=np.nan,
            dtype="float32",
            init_dest_nodata=True,
        ) as aligned:
            values = aligned.read(
                1,
                out_shape=(height, width),
                resampling=Resampling.bilinear,
            ).astype(np.float32)
            values[~np.isfinite(values)] = np.nan
    return values


def rgba(values, stops, colors, valid) -> np.ndarray:
    output = np.zeros(values.shape + (4,), dtype=np.uint8)
    for channel in range(3):
        output[..., channel] = np.interp(
            values, stops, [color[channel] for color in colors]
        ).astype(np.uint8)
    output[..., 3] = np.where(valid, 255, 0).astype(np.uint8)
    return output


def save(image: np.ndarray, path: Path, framed: bool) -> None:
    rendered = Image.fromarray(image, mode="RGBA")
    if framed:
        background = Image.new("RGBA", rendered.size, (242, 242, 242, 255))
        background.alpha_composite(rendered)
        rendered = background
    rendered.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--uncertainty", type=Path, required=True)
    parser.add_argument("--count", type=Path, required=True)
    parser.add_argument("--reference-prediction", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-width", type=int, default=3000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (
        args.prediction,
        args.uncertainty,
        args.count,
        args.reference_prediction,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to replace quickviews: {args.output_dir}")
    with rasterio.open(args.count) as source:
        scale = min(1.0, args.max_width / source.width)
        width = max(1, int(round(source.width * scale)))
        height = max(1, int(round(source.height * scale)))
        grid = {
            "crs": str(source.crs),
            "bounds": list(source.bounds),
            "source_width": source.width,
            "source_height": source.height,
            "preview_width": width,
            "preview_height": height,
            "transform": list(source.transform),
        }
    prediction = read_preview(args.prediction, width, height, Resampling.bilinear)
    uncertainty = read_preview(args.uncertainty, width, height, Resampling.bilinear)
    count = read_preview(args.count, width, height, Resampling.nearest)
    reference = read_aligned_reference(
        args.reference_prediction, args.prediction, width, height
    )
    support = np.isfinite(count) & (count > 0)
    prediction_valid = support & np.isfinite(prediction)
    uncertainty_valid = support & np.isfinite(uncertainty)
    reference_valid = support & np.isfinite(prediction) & np.isfinite(reference)
    difference = prediction - reference
    prediction_image = rgba(
        np.clip(np.nan_to_num(prediction, nan=0.0), 0, 1),
        PROBABILITY_STOPS,
        PROBABILITY_COLORS,
        prediction_valid,
    )
    uncertainty_image = rgba(
        np.clip(np.nan_to_num(uncertainty, nan=0.0), 0, 0.10),
        UNCERTAINTY_STOPS,
        UNCERTAINTY_COLORS,
        uncertainty_valid,
    )
    support_image = rgba(
        np.clip(np.nan_to_num(count, nan=0.0), 0, 16),
        SUPPORT_STOPS,
        SUPPORT_COLORS,
        support,
    )
    difference_image = rgba(
        np.clip(np.nan_to_num(difference, nan=0.0), -0.50, 0.50),
        DIFFERENCE_STOPS,
        DIFFERENCE_COLORS,
        reference_valid,
    )
    args.output_dir.mkdir(parents=True)
    stem = f"{args.host}_{args.region}"
    outputs = {}
    for label, image in (
        ("probability", prediction_image),
        ("uncertainty", uncertainty_image),
        ("support", support_image),
        ("difference_from_june", difference_image),
    ):
        for frame in ("transparent", "framed"):
            path = args.output_dir / f"{stem}_{label}_fixed_scale_{frame}.png"
            save(image, path, framed=frame == "framed")
            outputs[f"{label}_{frame}"] = str(path.resolve())
    finite_uncertainty = uncertainty[uncertainty_valid]
    payload = {
        "status": "complete",
        "host": args.host,
        "region": args.region,
        "prediction": str(args.prediction.resolve()),
        "prediction_sha256": sha256(args.prediction),
        "uncertainty": str(args.uncertainty.resolve()),
        "uncertainty_sha256": sha256(args.uncertainty),
        "count": str(args.count.resolve()),
        "count_sha256": sha256(args.count),
        "reference_prediction": str(args.reference_prediction.resolve()),
        "reference_prediction_sha256": sha256(args.reference_prediction),
        "grid": grid,
        "mask_definition": "count > 0 AND finite(product)",
        "prediction_scale": [0.0, 1.0],
        "uncertainty_scale": [0.0, 0.10],
        "support_scale": [0.0, 16.0],
        "difference_from_june_scale": [-0.50, 0.50],
        "preview_supported_pixels": int(support.sum()),
        "preview_prediction_valid_pixels": int(prediction_valid.sum()),
        "preview_uncertainty_valid_pixels": int(uncertainty_valid.sum()),
        "preview_difference_valid_pixels": int(reference_valid.sum()),
        "preview_uncertainty_above_display_max_fraction": (
            float((finite_uncertainty > 0.10).mean())
            if len(finite_uncertainty)
            else None
        ),
        "outputs": outputs,
        "visual_review_targets": [
            "UTM seams", "state/acquisition boundaries", "east-west striations",
            "NAIP tile footprints", "unexpected empty coverage", "ecological plausibility",
            "support plausibility", "August-minus-June change plausibility",
            "known NY, NY-PA, CA-OR, southern Oregon, and Gold Beach domains",
        ],
    }
    write_json_atomic(args.output_dir / "quickview_summary.json", payload)
    (args.output_dir / "COMPLETED").write_text("completed\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
