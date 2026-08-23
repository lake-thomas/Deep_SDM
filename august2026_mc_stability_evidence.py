#!/usr/bin/env python3
"""Evaluate reproducible MC-dropout pass counts against a 10-pass reference."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from august2026_combine_dataset_screen_evidence import HOSTS
from august2026_inference_provenance import derive_tile_seed
from august2026_summarize_experiments import sha256, write_json_atomic


PASS_LEVELS = (2, 3, 5, 10)
REPLICATE_SEEDS = (42, 43)
REFERENCE_LEVEL = 10
MIN_TILES_PER_HOST = 2

PREDICTION_MAE_FLOOR = 0.005
UNCERTAINTY_MAE_FLOOR = 0.003
REFERENCE_NOISE_MULTIPLIER = 1.5
MIN_PREDICTION_PEARSON = 0.995
MIN_UNCERTAINTY_PEARSON = 0.95


@dataclass
class PairAccumulator:
    count: int = 0
    sum_abs: float = 0.0
    sum_sq_error: float = 0.0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_x2: float = 0.0
    sum_y2: float = 0.0
    sum_xy: float = 0.0

    def add(self, candidate: np.ndarray, reference: np.ndarray) -> None:
        candidate = np.asarray(candidate, dtype=np.float64).reshape(-1)
        reference = np.asarray(reference, dtype=np.float64).reshape(-1)
        valid = np.isfinite(candidate) & np.isfinite(reference)
        x, y = candidate[valid], reference[valid]
        if not len(x):
            return
        difference = x - y
        self.count += len(x)
        self.sum_abs += float(np.abs(difference).sum())
        self.sum_sq_error += float(np.square(difference).sum())
        self.sum_x += float(x.sum())
        self.sum_y += float(y.sum())
        self.sum_x2 += float(np.square(x).sum())
        self.sum_y2 += float(np.square(y).sum())
        self.sum_xy += float((x * y).sum())

    def metrics(self) -> dict:
        if self.count == 0:
            return {"pixel_count": 0, "mae": None, "rmse": None, "pearson": None}
        n = float(self.count)
        covariance = self.sum_xy - self.sum_x * self.sum_y / n
        variance_x = self.sum_x2 - self.sum_x**2 / n
        variance_y = self.sum_y2 - self.sum_y**2 / n
        denominator = math.sqrt(max(0.0, variance_x) * max(0.0, variance_y))
        return {
            "pixel_count": self.count,
            "mae": self.sum_abs / n,
            "rmse": math.sqrt(self.sum_sq_error / n),
            "pearson": covariance / denominator if denominator > 0 else None,
        }


def advancement_rule(
    prediction_mae: float,
    uncertainty_mae: float,
    prediction_pearson: float,
    uncertainty_pearson: float,
    reference_prediction_repeat_mae: float,
    reference_uncertainty_repeat_mae: float,
) -> dict:
    prediction_maximum = max(
        PREDICTION_MAE_FLOOR,
        REFERENCE_NOISE_MULTIPLIER * reference_prediction_repeat_mae,
    )
    uncertainty_maximum = max(
        UNCERTAINTY_MAE_FLOOR,
        REFERENCE_NOISE_MULTIPLIER * reference_uncertainty_repeat_mae,
    )
    components = {
        "prediction_mae_passed": prediction_mae <= prediction_maximum,
        "uncertainty_mae_passed": uncertainty_mae <= uncertainty_maximum,
        "prediction_pearson_passed": prediction_pearson >= MIN_PREDICTION_PEARSON,
        "uncertainty_pearson_passed": uncertainty_pearson >= MIN_UNCERTAINTY_PEARSON,
    }
    return {
        "prediction_mae_maximum": prediction_maximum,
        "uncertainty_mae_maximum": uncertainty_maximum,
        **components,
        "stability_gate_passed": all(components.values()),
    }


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    frame.to_csv(partial, index=False)
    os.replace(partial, path)


def _read_layer(path: Path, expected_seed: int) -> tuple[np.ndarray, tuple]:
    with rasterio.open(path) as source:
        tags = source.tags()
        expected_tags = {
            "blend_window": "hann",
            "blend_edge_floor": "0.05",
            "inference_seed": str(expected_seed),
        }
        mismatches = {
            key: {"expected": value, "found": tags.get(key)}
            for key, value in expected_tags.items()
            if tags.get(key) != value
        }
        if mismatches:
            raise ValueError(f"MC stability raster tags changed: {path}: {mismatches}")
        data = source.read(1, masked=True).filled(np.nan).astype(np.float32)
        geometry = (source.width, source.height, str(source.crs), source.transform)
    return data, geometry


def _validated_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    required = {
        "host", "tile_id", "tile_path", "mc_samples", "replicate_seed",
        "prediction", "uncertainty", "elapsed_seconds",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"MC stability manifest lacks columns: {sorted(missing)}")
    if frame.duplicated(["host", "tile_id", "mc_samples", "replicate_seed"]).any():
        raise ValueError("MC stability manifest contains duplicate runs")
    if set(frame["host"]) != HOSTS:
        raise ValueError("MC stability manifest must cover all five production hosts")
    if set(pd.to_numeric(frame["mc_samples"]).astype(int)) != set(PASS_LEVELS):
        raise ValueError("MC stability manifest pass levels changed")
    if set(pd.to_numeric(frame["replicate_seed"]).astype(int)) != set(REPLICATE_SEEDS):
        raise ValueError("MC stability replicate seeds changed")
    frame["mc_samples"] = pd.to_numeric(frame["mc_samples"]).astype(int)
    frame["replicate_seed"] = pd.to_numeric(frame["replicate_seed"]).astype(int)
    for host, group in frame.groupby("host"):
        tiles = set(group["tile_id"])
        if len(tiles) < MIN_TILES_PER_HOST:
            raise ValueError(f"MC stability requires at least two tiles for {host}")
        expected = {
            (tile, level, seed)
            for tile in tiles for level in PASS_LEVELS for seed in REPLICATE_SEEDS
        }
        observed = set(zip(group["tile_id"], group["mc_samples"], group["replicate_seed"]))
        if observed != expected:
            raise ValueError(f"MC stability factorial is incomplete for {host}")
    for column in ("tile_path", "prediction", "uncertainty"):
        missing_paths = [raw for raw in frame[column] if not Path(raw).is_file()]
        if missing_paths:
            raise FileNotFoundError(f"Missing MC stability {column}: {missing_paths[:5]}")
    return frame


def synthesize_evidence(manifest: Path, output_dir: Path) -> dict:
    frame = _validated_manifest(manifest)
    rows = []
    for host, host_frame in frame.groupby("host", sort=True):
        candidate_accumulators = {
            level: {"prediction": PairAccumulator(), "uncertainty": PairAccumulator()}
            for level in PASS_LEVELS if level != REFERENCE_LEVEL
        }
        repeat_accumulators = {
            "prediction": PairAccumulator(), "uncertainty": PairAccumulator()
        }
        tile_count = 0
        for tile_id, tile_frame in host_frame.groupby("tile_id", sort=True):
            tile_count += 1
            tile_paths = set(tile_frame["tile_path"])
            if len(tile_paths) != 1:
                raise ValueError(f"Tile path differs within factorial: {host}:{tile_id}")
            tile_path = Path(next(iter(tile_paths)))
            arrays = {}
            geometry = None
            for record in tile_frame.to_dict("records"):
                expected_seed = derive_tile_seed(record["replicate_seed"], tile_path)
                prediction, prediction_geometry = _read_layer(
                    Path(record["prediction"]), expected_seed
                )
                uncertainty, uncertainty_geometry = _read_layer(
                    Path(record["uncertainty"]), expected_seed
                )
                if prediction_geometry != uncertainty_geometry:
                    raise ValueError(f"Prediction/uncertainty geometry differs: {host}:{tile_id}")
                if geometry is None:
                    geometry = prediction_geometry
                elif geometry != prediction_geometry:
                    raise ValueError(f"MC factorial geometry differs: {host}:{tile_id}")
                arrays[(record["mc_samples"], record["replicate_seed"])] = (
                    prediction, uncertainty
                )
            reference_replicates = [
                arrays[(REFERENCE_LEVEL, seed)] for seed in REPLICATE_SEEDS
            ]
            for layer_index, label in enumerate(("prediction", "uncertainty")):
                repeat_accumulators[label].add(
                    reference_replicates[0][layer_index],
                    reference_replicates[1][layer_index],
                )
            reference = tuple(
                np.nanmean(
                    np.stack([item[index] for item in reference_replicates]), axis=0
                )
                for index in (0, 1)
            )
            for level, accumulators in candidate_accumulators.items():
                candidate_replicates = [arrays[(level, seed)] for seed in REPLICATE_SEEDS]
                candidate = tuple(
                    np.nanmean(
                        np.stack([item[index] for item in candidate_replicates]), axis=0
                    )
                    for index in (0, 1)
                )
                accumulators["prediction"].add(candidate[0], reference[0])
                accumulators["uncertainty"].add(candidate[1], reference[1])

        repeat = {key: accumulator.metrics() for key, accumulator in repeat_accumulators.items()}
        for level, accumulators in candidate_accumulators.items():
            prediction = accumulators["prediction"].metrics()
            uncertainty = accumulators["uncertainty"].metrics()
            if prediction["pearson"] is None or uncertainty["pearson"] is None:
                raise RuntimeError(f"Undefined MC stability correlation: {host}:MC{level}")
            gate = advancement_rule(
                prediction["mae"], uncertainty["mae"],
                prediction["pearson"], uncertainty["pearson"],
                repeat["prediction"]["mae"], repeat["uncertainty"]["mae"],
            )
            elapsed = pd.to_numeric(
                host_frame.loc[host_frame["mc_samples"] == level, "elapsed_seconds"],
                errors="coerce",
            )
            rows.append({
                "host": host,
                "mc_samples": level,
                "tile_count": tile_count,
                "prediction_pixel_count": prediction["pixel_count"],
                "prediction_mae_vs_mc10": prediction["mae"],
                "prediction_rmse_vs_mc10": prediction["rmse"],
                "prediction_pearson_vs_mc10": prediction["pearson"],
                "uncertainty_pixel_count": uncertainty["pixel_count"],
                "uncertainty_mae_vs_mc10": uncertainty["mae"],
                "uncertainty_rmse_vs_mc10": uncertainty["rmse"],
                "uncertainty_pearson_vs_mc10": uncertainty["pearson"],
                "mc10_prediction_repeat_mae": repeat["prediction"]["mae"],
                "mc10_uncertainty_repeat_mae": repeat["uncertainty"]["mae"],
                "median_elapsed_seconds": float(elapsed.median()),
                **gate,
            })
    evidence = pd.DataFrame(rows).sort_values(["mc_samples", "host"])
    passing_levels = [
        level for level in PASS_LEVELS if level != REFERENCE_LEVEL
        and len(evidence[evidence["mc_samples"] == level]) == len(HOSTS)
        and evidence.loc[
            evidence["mc_samples"] == level, "stability_gate_passed"
        ].all()
    ]
    recommendation = min(passing_levels) if passing_levels else REFERENCE_LEVEL
    output_dir.mkdir(parents=True, exist_ok=False)
    evidence_path = output_dir / "mc_stability_by_host.csv"
    write_csv_atomic(evidence_path, evidence)
    status = {
        "status": "complete",
        "stage": "mc-dropout-pass-stability",
        "manifest": str(manifest.resolve()),
        "manifest_sha256": sha256(manifest),
        "host_count": len(HOSTS),
        "minimum_tiles_per_host": MIN_TILES_PER_HOST,
        "pass_levels": list(PASS_LEVELS),
        "replicate_base_seeds": list(REPLICATE_SEEDS),
        "reference_mc_samples": REFERENCE_LEVEL,
        "recommended_mc_samples": recommendation,
        "all_host_passing_candidate_levels": passing_levels,
        "evidence": str(evidence_path.resolve()),
        "evidence_sha256": sha256(evidence_path),
        "thresholds": {
            "prediction_mae_floor": PREDICTION_MAE_FLOOR,
            "uncertainty_mae_floor": UNCERTAINTY_MAE_FLOOR,
            "reference_repeat_noise_multiplier": REFERENCE_NOISE_MULTIPLIER,
            "minimum_prediction_pearson": MIN_PREDICTION_PEARSON,
            "minimum_uncertainty_pearson": MIN_UNCERTAINTY_PEARSON,
        },
        "guardrail": (
            "The recommendation is the smallest pass count clearing every host's "
            "two-replicate probability and uncertainty stability gates. If none of "
            "MC2/3/5 passes, retain MC10. Full Hann inference remains separately gated."
        ),
    }
    write_json_atomic(output_dir / "summary.json", status)
    (output_dir / "COMPLETED").write_text("completed\n", encoding="utf-8")
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to replace MC evidence: {args.output_dir}")
    print(json.dumps(synthesize_evidence(args.manifest, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
