#!/usr/bin/env python3
"""Audit shared sampling and acquisition exposure between two host datasets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from august2026_summarize_experiments import sha256, write_json_atomic


IDENTITY_DIMENSIONS = {
    "location": ["latitude_key", "longitude_key"],
    "location_and_source": ["latitude_key", "longitude_key", "source_identity"],
    "source_tile": ["source_tile_key"],
    "source_acquisition": ["source_identity"],
    "acquisition_domain": ["source_domain_id"],
    "state": ["source_archive_state"],
}


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    partial = path.with_name(f"{path.name}.partial")
    frame.to_csv(partial, index=False)
    os.replace(partial, path)


def prepare_dataset(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    required = {"presence", "lat", "lon"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns {sorted(missing)}: {path}")
    output = frame.copy()
    output["presence"] = pd.to_numeric(output["presence"], errors="raise").astype(int)
    if not set(output["presence"]).issubset({0, 1}):
        raise ValueError(f"Presence labels must be binary: {path}")
    output["latitude_key"] = pd.to_numeric(output["lat"], errors="raise").round(7)
    output["longitude_key"] = pd.to_numeric(output["lon"], errors="raise").round(7)
    source_column = next(
        (
            column
            for column in (
                "actual_source_paths",
                "actual_source_path",
                "source_local_path",
                "local_path",
                "filename",
            )
            if column in output.columns
        ),
        None,
    )
    if source_column is None:
        raise ValueError(f"Dataset has no source-acquisition identity column: {path}")
    output["source_identity"] = output[source_column].fillna("").astype(str)
    source_tile_column = (
        "actual_source_tile_keys"
        if "actual_source_tile_keys" in output.columns
        else "source_tile_key"
    )
    output["source_tile_key"] = (
        output[source_tile_column].fillna("").astype(str)
        if source_tile_column in output.columns
        else ""
    )
    actual_domain_columns = {
        "actual_source_archive_state",
        "actual_source_acq_date",
    }
    if actual_domain_columns.issubset(output.columns):
        output["source_domain_id"] = (
            output["actual_source_archive_state"].fillna("").astype(str)
            + "|"
            + output["actual_source_acq_date"].fillna("").astype(str)
        )
        output["source_archive_state"] = (
            output["actual_source_archive_state"].fillna("").astype(str)
        )
    else:
        for column in ("source_domain_id", "source_archive_state"):
            if column not in output.columns:
                output[column] = ""
            output[column] = output[column].fillna("").astype(str)
    return output


def key_series(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    values = frame[columns].astype(str)
    return values.agg("|".join, axis=1)


def exposure_metrics(first: pd.Series, second: pd.Series) -> tuple[dict, pd.DataFrame]:
    first_counts = first.value_counts().sort_index()
    second_counts = second.value_counts().sort_index()
    keys = first_counts.index.union(second_counts.index)
    table = pd.DataFrame(
        {
            "key": keys,
            "first_count": first_counts.reindex(keys, fill_value=0).to_numpy(int),
            "second_count": second_counts.reindex(keys, fill_value=0).to_numpy(int),
        }
    )
    table["minimum_count"] = table[["first_count", "second_count"]].min(axis=1)
    table["maximum_count"] = table[["first_count", "second_count"]].max(axis=1)
    first_keys = set(first_counts.index)
    second_keys = set(second_counts.index)
    intersection = first_keys & second_keys
    union = first_keys | second_keys
    minimum_total = min(int(first_counts.sum()), int(second_counts.sum()))
    denominator = float(
        np.linalg.norm(table["first_count"].to_numpy(float))
        * np.linalg.norm(table["second_count"].to_numpy(float))
    )
    metrics = {
        "first_unique": len(first_keys),
        "second_unique": len(second_keys),
        "intersection_unique": len(intersection),
        "union_unique": len(union),
        "jaccard": len(intersection) / len(union) if union else 1.0,
        "first_containment": len(intersection) / len(first_keys) if first_keys else 1.0,
        "second_containment": len(intersection) / len(second_keys) if second_keys else 1.0,
        "weighted_overlap_coefficient": (
            float(table["minimum_count"].sum()) / minimum_total
            if minimum_total
            else 1.0
        ),
        "count_vector_cosine_similarity": (
            float(
                np.dot(
                    table["first_count"].to_numpy(float),
                    table["second_count"].to_numpy(float),
                )
                / denominator
            )
            if denominator
            else None
        ),
    }
    return metrics, table


def build_overlap_audit(
    first: pd.DataFrame,
    second: pd.DataFrame,
    first_label: str,
    second_label: str,
) -> tuple[dict, pd.DataFrame]:
    rows = []
    summaries = {}
    for class_value, class_label in ((None, "all"), (0, "background"), (1, "presence")):
        first_class = first if class_value is None else first[first["presence"] == class_value]
        second_class = second if class_value is None else second[second["presence"] == class_value]
        summaries[class_label] = {
            first_label: {"rows": len(first_class)},
            second_label: {"rows": len(second_class)},
            "dimensions": {},
        }
        for dimension, columns in IDENTITY_DIMENSIONS.items():
            metrics, exposure = exposure_metrics(
                key_series(first_class, columns), key_series(second_class, columns)
            )
            summaries[class_label]["dimensions"][dimension] = metrics
            exposure.insert(0, "dimension", dimension)
            exposure.insert(0, "class", class_label)
            rows.append(exposure)
    return summaries, pd.concat(rows, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-dataset", type=Path, required=True)
    parser.add_argument("--second-dataset", type=Path, required=True)
    parser.add_argument("--first-label", default="ailanthus_altissima")
    parser.add_argument("--second-label", default="juglans_nigra")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.first_dataset, args.second_dataset):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to replace overlap audit: {args.output_dir}")
    first = prepare_dataset(args.first_dataset)
    second = prepare_dataset(args.second_dataset)
    summary, exposure = build_overlap_audit(
        first, second, args.first_label, args.second_label
    )
    args.output_dir.mkdir(parents=True)
    exposure_path = args.output_dir / "shared_exposure_counts.csv"
    write_csv_atomic(exposure_path, exposure)
    payload = {
        "status": "complete",
        "audit": "cross_host_dataset_sampling_and_acquisition_overlap",
        "first_label": args.first_label,
        "second_label": args.second_label,
        "first_dataset": str(args.first_dataset.resolve()),
        "first_dataset_sha256": sha256(args.first_dataset),
        "second_dataset": str(args.second_dataset.resolve()),
        "second_dataset_sha256": sha256(args.second_dataset),
        "rounding_decimal_places_for_location_identity": 7,
        "summary": summary,
        "shared_exposure_counts": str(exposure_path.resolve()),
        "shared_exposure_counts_sha256": sha256(exposure_path),
        "interpretation": (
            "Exact sample overlap and shared source/domain exposure can identify a "
            "sampler-mediated contribution to cross-host similarity. High overlap is "
            "descriptive, not proof of leakage or ecological invalidity."
        ),
    }
    write_json_atomic(args.output_dir / "summary.json", payload)
    (args.output_dir / "COMPLETED").write_text("completed\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
