from .common import *


MANIFEST_REQUIRED_COLUMNS = [
    "sample_id",
    "presence",
    "lat",
    "lon",
    "split",
    "chip_path",
    "source_tile_key",
    "source_archive_state",
    "source_acq_date",
]


def _truthy(series: pd.Series) -> pd.Series:
    """Interpret booleans serialized through CSV without treating nonempty text as true."""
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.isin({"true", "1", "yes"})


def _partition_columns(df: pd.DataFrame) -> list[str]:
    return ["cv_round"] if "cv_round" in df.columns else []


def audit_dataset_manifest(
    manifest: pd.DataFrame,
    dataset_root: str | Path | None = None,
    require_actual_provenance: bool = True,
    enforce_source_tile_isolation: bool = True,
    enforce_spatial_block_isolation: bool = True,
    require_all_split_classes: bool = True,
) -> dict[str, pd.DataFrame]:
    """Build machine-readable pre-fit QA tables without changing the manifest.

    Hard findings describe conditions that can invalidate an August training or
    evaluation dataset. Domain counts are diagnostic and intentionally do not
    impose a universal minimum depth.
    """
    df = manifest.copy()
    findings: list[dict] = []
    detail_tables: dict[str, pd.DataFrame] = {}

    def add_finding(code: str, severity: str, count: int, details: str) -> None:
        findings.append(
            {
                "check": code,
                "severity": severity,
                "status": "fail" if count else "pass",
                "count": int(count),
                "details": details,
            }
        )

    missing_required = [column for column in MANIFEST_REQUIRED_COLUMNS if column not in df.columns]
    add_finding(
        "missing_required_columns",
        "hard",
        len(missing_required),
        ", ".join(missing_required) if missing_required else "all required columns present",
    )
    if missing_required:
        return {
            "findings": pd.DataFrame(findings),
            "class_split_counts": pd.DataFrame(),
            "missing_split_classes": pd.DataFrame(),
            "domain_counts": pd.DataFrame(),
            "background_domain_depth": pd.DataFrame(),
            "background_domain_depth_summary": pd.DataFrame(),
            "domain_prevalence": pd.DataFrame(),
            "source_tile_split_leakage": pd.DataFrame(),
            "actual_source_tile_split_leakage": pd.DataFrame(),
            "spatial_block_split_leakage": pd.DataFrame(),
            "coordinate_conflicts": pd.DataFrame(),
            "path_failures": pd.DataFrame(),
            "provenance_failures": pd.DataFrame(),
        }

    add_finding("empty_manifest", "hard", int(df.empty), "manifest must contain rows")
    partition = _partition_columns(df)

    required_nulls = df[MANIFEST_REQUIRED_COLUMNS].isna().any(axis=1)
    detail_tables["required_null_rows"] = df.loc[required_nulls].copy()
    add_finding(
        "null_required_values",
        "hard",
        int(required_nulls.sum()),
        "rows with a null value in a required manifest column",
    )

    valid_presence = pd.to_numeric(df["presence"], errors="coerce").isin([0, 1])
    add_finding(
        "invalid_presence_labels",
        "hard",
        int((~valid_presence).sum()),
        "presence labels must be binary 0/1",
    )
    valid_split = df["split"].isin(["train", "val", "test"])
    add_finding(
        "invalid_split_labels",
        "hard",
        int((~valid_split).sum()),
        "split labels must be train, val, or test",
    )

    duplicate_sample_mask = df.duplicated(partition + ["sample_id"], keep=False)
    detail_tables["duplicate_sample_ids"] = df.loc[duplicate_sample_mask].copy()
    add_finding(
        "duplicate_sample_ids",
        "hard",
        int(duplicate_sample_mask.sum()),
        "rows participating in duplicate sample IDs within an evaluation round",
    )

    duplicate_path_mask = df.duplicated(partition + ["chip_path"], keep=False)
    detail_tables["duplicate_chip_paths"] = df.loc[duplicate_path_mask].copy()
    add_finding(
        "duplicate_chip_paths",
        "hard",
        int(duplicate_path_mask.sum()),
        "rows participating in duplicate NAIP chip paths within an evaluation round",
    )

    coordinate_group = partition + ["lat", "lon"]
    coordinate_summary = (
        df.groupby(coordinate_group, dropna=False)
        .agg(
            row_count=("presence", "size"),
            label_count=("presence", "nunique"),
            split_count=("split", "nunique"),
        )
        .reset_index()
    )
    coordinate_conflicts = coordinate_summary[
        (coordinate_summary["row_count"] > 1)
        | (coordinate_summary["label_count"] > 1)
        | (coordinate_summary["split_count"] > 1)
    ].copy()
    detail_tables["coordinate_conflicts"] = coordinate_conflicts
    add_finding(
        "duplicate_coordinates",
        "hard",
        int((coordinate_summary["row_count"] > 1).sum()),
        "exact coordinate groups containing more than one row",
    )
    add_finding(
        "conflicting_coordinate_labels",
        "hard",
        int((coordinate_summary["label_count"] > 1).sum()),
        "exact coordinate groups containing both classes",
    )
    add_finding(
        "coordinate_split_leakage",
        "hard",
        int((coordinate_summary["split_count"] > 1).sum()),
        "exact coordinate groups spanning evaluation partitions",
    )

    tile_group = partition + ["source_tile_key"]
    tile_split_summary = (
        df.dropna(subset=["source_tile_key"])
        .groupby(tile_group, dropna=False)
        .agg(split_count=("split", "nunique"), row_count=("split", "size"))
        .reset_index()
    )
    tile_leakage = tile_split_summary[tile_split_summary["split_count"] > 1].copy()
    detail_tables["source_tile_split_leakage"] = tile_leakage
    add_finding(
        "source_tile_split_leakage",
        "hard" if enforce_source_tile_isolation else "diagnostic",
        len(tile_leakage),
        "date-independent source tiles spanning evaluation partitions",
    )

    if "actual_source_tile_keys" in df.columns:
        actual_tiles = df[partition + ["split", "sample_id", "actual_source_tile_keys"]].copy()
        actual_tiles["actual_source_tile_key"] = (
            actual_tiles["actual_source_tile_keys"].fillna("").astype(str).str.split(";")
        )
        actual_tiles = actual_tiles.explode("actual_source_tile_key")
        actual_tiles["actual_source_tile_key"] = actual_tiles["actual_source_tile_key"].str.strip()
        actual_tiles = actual_tiles[actual_tiles["actual_source_tile_key"] != ""]
        actual_tile_summary = (
            actual_tiles.groupby(partition + ["actual_source_tile_key"], dropna=False)
            .agg(split_count=("split", "nunique"), sample_count=("sample_id", "nunique"))
            .reset_index()
        )
        actual_tile_leakage = actual_tile_summary[
            actual_tile_summary["split_count"] > 1
        ].copy()
    else:
        actual_tile_leakage = pd.DataFrame()
    detail_tables["actual_source_tile_split_leakage"] = actual_tile_leakage
    add_finding(
        "actual_source_tile_split_leakage",
        "hard" if enforce_source_tile_isolation else "diagnostic",
        len(actual_tile_leakage),
        "actual intended or boundary-constituent tiles spanning evaluation partitions",
    )

    if "block_id" in df.columns:
        block_group = partition + ["block_id"]
        block_split_summary = (
            df.dropna(subset=["block_id"])
            .groupby(block_group, dropna=False)
            .agg(split_count=("split", "nunique"), row_count=("split", "size"))
            .reset_index()
        )
        block_leakage = block_split_summary[block_split_summary["split_count"] > 1].copy()
    else:
        block_leakage = pd.DataFrame()
    detail_tables["spatial_block_split_leakage"] = block_leakage
    add_finding(
        "spatial_block_split_leakage",
        "hard" if enforce_spatial_block_isolation else "diagnostic",
        len(block_leakage),
        "spatial blocks spanning evaluation partitions",
    )

    provenance_failure = pd.Series(False, index=df.index)
    provenance_checks_available = {
        "actual_source_lock_mode",
        "source_provenance_match",
        "intended_source_used",
        "actual_multi_state",
        "actual_multi_acquisition",
        "actual_source_archive_state",
        "actual_source_acq_date",
        "actual_source_tile_keys",
    }.issubset(df.columns)
    if provenance_checks_available:
        strict = df["actual_source_lock_mode"].eq("strict")
        provenance_failure |= strict & ~_truthy(df["source_provenance_match"])
        provenance_failure |= strict & ~_truthy(df["intended_source_used"])
        provenance_failure |= strict & _truthy(df["actual_multi_state"])
        provenance_failure |= strict & _truthy(df["actual_multi_acquisition"])
        provenance_failure |= strict & (
            df["source_archive_state"].astype("string")
            != df["actual_source_archive_state"].astype("string")
        )
        provenance_failure |= strict & (
            df["source_acq_date"].astype("string")
            != df["actual_source_acq_date"].astype("string")
        )
        if require_actual_provenance:
            provenance_failure |= ~strict
    elif require_actual_provenance:
        provenance_failure[:] = True
    provenance_failures = df.loc[provenance_failure].copy()
    detail_tables["provenance_failures"] = provenance_failures
    add_finding(
        "source_provenance_failures",
        "hard",
        int(provenance_failure.sum()),
        "rows lacking matching strict intended/actual acquisition provenance",
    )

    path_failure_rows = []
    if dataset_root is not None:
        root = Path(dataset_root)
        path_columns = ["chip_path"]
        if "topo_chip_path" in df.columns and df["topo_chip_path"].notna().any():
            path_columns.append("topo_chip_path")
        for row_index, row in df.iterrows():
            for column in path_columns:
                value = row.get(column)
                path = Path(str(value)) if pd.notna(value) else None
                if path is not None and not path.is_absolute():
                    path = root / path
                if path is None or not path.is_file():
                    path_failure_rows.append(
                        {
                            "row_index": row_index,
                            "sample_id": row.get("sample_id"),
                            "path_column": column,
                            "path": str(path) if path is not None else None,
                        }
                    )
    path_failures = pd.DataFrame(path_failure_rows)
    detail_tables["path_failures"] = path_failures
    add_finding(
        "missing_or_unreadable_paths",
        "hard",
        len(path_failures),
        "required chip paths that are not files under the dataset root",
    )

    predictor_columns = [
        column
        for column in df.columns
        if column.startswith("wc2.1_30s_bio_")
        or column in {"ghm", "lat_norm", "lon_norm"}
        or column.startswith(("elev_", "slope_", "northness_", "eastness_"))
    ]
    if predictor_columns:
        numeric = df[predictor_columns].apply(pd.to_numeric, errors="coerce")
        nonfinite = ~np.isfinite(numeric.to_numpy(dtype=float))
        nonfinite_rows = pd.Series(nonfinite.any(axis=1), index=df.index)
    else:
        nonfinite_rows = pd.Series(False, index=df.index)
    detail_tables["nonfinite_predictor_rows"] = df.loc[nonfinite_rows].copy()
    add_finding(
        "nonfinite_predictors",
        "hard",
        int(nonfinite_rows.sum()),
        "rows with a missing or non-finite model predictor",
    )

    class_split_counts = (
        df.groupby(partition + ["split", "presence"], dropna=False)
        .size()
        .rename("row_count")
        .reset_index()
    )
    if require_all_split_classes:
        partition_values = (
            df[partition].drop_duplicates().to_dict("records") if partition else [{}]
        )
        expected_rows = []
        for values in partition_values:
            for split_name in ["train", "val", "test"]:
                for label in [0, 1]:
                    expected_rows.append({**values, "split": split_name, "presence": label})
        expected_split_classes = pd.DataFrame(expected_rows)
        observed_split_classes = class_split_counts[partition + ["split", "presence"]].copy()
        observed_split_classes["presence"] = pd.to_numeric(
            observed_split_classes["presence"], errors="coerce"
        )
        missing_split_classes = expected_split_classes.merge(
            observed_split_classes,
            on=partition + ["split", "presence"],
            how="left",
            indicator=True,
        )
        missing_split_classes = missing_split_classes[
            missing_split_classes["_merge"] == "left_only"
        ].drop(columns="_merge")
    else:
        missing_split_classes = pd.DataFrame()
    add_finding(
        "missing_split_classes",
        "hard",
        len(missing_split_classes),
        "required train/val/test × presence/background combinations absent from a round",
    )
    domain_columns = [
        column
        for column in [
            "source_archive_state",
            "source_archive_year",
            "source_acq_year",
            "source_acq_month",
            "source_acq_date",
        ]
        if column in df.columns
    ]
    domain_counts = (
        df.groupby(partition + ["split", "presence", *domain_columns], dropna=False)
        .size()
        .rename("row_count")
        .reset_index()
    )

    depth_domain_columns = [
        column for column in ["source_archive_state", "source_acq_date"] if column in df.columns
    ]
    backgrounds = df[pd.to_numeric(df["presence"], errors="coerce") == 0].copy()
    if depth_domain_columns and not backgrounds.empty:
        background_domain_depth = (
            backgrounds.groupby(partition + ["split", *depth_domain_columns], dropna=False)
            .size()
            .rename("background_count")
            .reset_index()
        )
        background_domain_depth_summary = (
            background_domain_depth.groupby(partition + ["split"], dropna=False)["background_count"]
            .agg(
                background_count="sum",
                domain_count="size",
                median_backgrounds_per_domain="median",
                min_backgrounds_per_domain="min",
                max_backgrounds_per_domain="max",
            )
            .reset_index()
        )
        depth_thresholds = (
            background_domain_depth.assign(
                domain_ge_5=background_domain_depth["background_count"] >= 5,
                domain_ge_10=background_domain_depth["background_count"] >= 10,
            )
            .groupby(partition + ["split"], dropna=False)[["domain_ge_5", "domain_ge_10"]]
            .sum()
            .reset_index()
        )
        background_domain_depth_summary = background_domain_depth_summary.merge(
            depth_thresholds,
            on=partition + ["split"],
            how="left",
        )
    else:
        background_domain_depth = pd.DataFrame()
        background_domain_depth_summary = pd.DataFrame()

    if depth_domain_columns:
        prevalence_frame = df.copy()
        prevalence_frame["_presence_numeric"] = pd.to_numeric(
            prevalence_frame["presence"],
            errors="coerce",
        )
        domain_prevalence = (
            prevalence_frame.groupby(partition + [*depth_domain_columns], dropna=False)["_presence_numeric"]
            .agg(row_count="size", presence_count="sum", prevalence="mean")
            .reset_index()
        )
        extreme_domain_prevalence = (
            (domain_prevalence["row_count"] >= 10)
            & (
                (domain_prevalence["prevalence"] <= 0.02)
                | (domain_prevalence["prevalence"] >= 0.98)
            )
        )
    else:
        domain_prevalence = pd.DataFrame()
        extreme_domain_prevalence = pd.Series(dtype=bool)
    add_finding(
        "extreme_domain_prevalence",
        "diagnostic",
        int(extreme_domain_prevalence.sum()),
        "domains with at least ten rows and prevalence <=0.02 or >=0.98",
    )

    findings_df = pd.DataFrame(findings)
    result = {
        "findings": findings_df,
        "class_split_counts": class_split_counts,
        "missing_split_classes": missing_split_classes,
        "domain_counts": domain_counts,
        "background_domain_depth": background_domain_depth,
        "background_domain_depth_summary": background_domain_depth_summary,
        "domain_prevalence": domain_prevalence,
    }
    result.update(detail_tables)
    return result


def manifest_has_hard_failures(qa_tables: dict[str, pd.DataFrame]) -> bool:
    findings = qa_tables["findings"]
    return bool(((findings["severity"] == "hard") & (findings["status"] == "fail")).any())


def write_manifest_qa_package(
    qa_tables: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> Path:
    """Write one traceable CSV-per-table QA package plus a compact JSON status."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, table in qa_tables.items():
        table.to_csv(output / f"{name}.csv", index=False)

    findings = qa_tables["findings"]
    hard_failures = findings[
        (findings["severity"] == "hard") & (findings["status"] == "fail")
    ]
    status = {
        "production_ready": hard_failures.empty,
        "hard_failure_checks": hard_failures["check"].tolist(),
        "hard_failure_count": int(len(hard_failures)),
        "qa_table_files": sorted(f"{name}.csv" for name in qa_tables),
    }
    status_path = output / "qa_status.json"
    with open(status_path, "w", encoding="utf-8") as stream:
        json.dump(status, stream, indent=2)
    return status_path
