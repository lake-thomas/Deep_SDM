"""Materialize one reviewed August candidate package exactly once.

The candidate package contains presence rows plus the union of primary and
ranked-reserve backgrounds for both sampling arms. This module extracts one
strict, source-locked NAIP/topography pair per unique sample, reconciles split
groups using the *actual* constituent source tiles, and writes exact nested
uniform and five-fold BlockCV manifests that all reference the shared chips.
"""

from .common import *

import argparse
import hashlib
import json
import os
import re
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from .candidate_manifests import _builder_source_hashes, _git_state, _sha256
from .env import raw_environment_valid_mask
from .io_utils import add_canonical_naip_source_metadata
from .naip import extract_naip_chip_for_point
from .qa import audit_dataset_manifest, manifest_has_hard_failures, write_manifest_qa_package
from .splits import assign_spatial_blocks, make_spatial_cv_rounds, stratified_train_val_test_split
from .topo import (
    close_dataset_dict,
    extract_topo_for_naip_chip,
    load_topo_normalization_stats,
)


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(f"{path}.partial")
    with open(partial, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    os.replace(partial, path)


def _file_identity(path: str | Path) -> dict:
    file_path = Path(path).resolve()
    stat = file_path.stat()
    return {
        "path": str(file_path),
        "size_bytes": stat.st_size,
        "mtime": pd.Timestamp(stat.st_mtime, unit="s", tz="UTC").isoformat(),
    }


def _load_worldclim_stats(path: str | Path) -> dict[str, dict[str, float]]:
    with open(path, "r", encoding="utf-8") as stream:
        payload = json.load(stream)
    stats = payload.get("worldclim_stats", payload)
    missing = [variable for variable in WC_VARS if variable not in stats]
    if missing:
        raise ValueError(f"WorldClim statistics lack variables: {missing}")
    for variable in WC_VARS:
        mean = float(stats[variable]["mean"])
        std = float(stats[variable]["std"])
        if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
            raise ValueError(f"Invalid normalization statistics for {variable}")
    return stats


def _sample_shard(sample_id: str) -> str:
    match = re.search(r"(\d+)$", sample_id)
    if match:
        return f"{int(match.group(1)) % 100:02d}"
    return hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:2]


def _safe_sample_id(value: object) -> str:
    sample_id = str(value)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", sample_id):
        raise ValueError(f"Unsafe sample ID for a file path: {sample_id!r}")
    return sample_id


def _jsonable(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _validate_chip(path: Path, size: int, count: int) -> None:
    if not path.is_file():
        raise RuntimeError(f"Expected chip was not written: {path}")
    with rasterio.open(path) as dataset:
        if dataset.width != size or dataset.height != size or dataset.count != count:
            raise RuntimeError(
                f"Unexpected chip structure {dataset.count}x{dataset.height}x{dataset.width}: {path}"
            )
        if dataset.crs is None or not np.isfinite(np.asarray(dataset.transform)[:6]).all():
            raise RuntimeError(f"Chip lacks valid CRS/transform: {path}")


@dataclass(frozen=True)
class MaterializationSettings:
    output_root: Path
    chip_size: int
    topo_chip_size: int
    naip_min_valid_frac: float
    topo_min_valid_frac: float
    dem_raster: Path
    slope_raster: Path
    northness_raster: Path
    eastness_raster: Path


class MaterializationWorker:
    """Thread-safe dispatcher with one set of topography handles per thread."""

    def __init__(
        self,
        tileindex: gpd.GeoDataFrame,
        worldclim_stats: dict,
        topo_stats: dict,
        settings: MaterializationSettings,
    ) -> None:
        self.tileindex = tileindex
        # Build the read-only spatial index before workers query it.
        _ = self.tileindex.sindex
        self.worldclim_stats = worldclim_stats
        self.topo_stats = topo_stats
        self.settings = settings
        self.local = threading.local()
        self.opened_topo: list[dict] = []
        self.lock = threading.Lock()

    def _topo_sources(self) -> dict:
        if not hasattr(self.local, "topo_sources"):
            sources = {
                "elevation": rasterio.open(self.settings.dem_raster),
                "slope": rasterio.open(self.settings.slope_raster),
                "northness": rasterio.open(self.settings.northness_raster),
                "eastness": rasterio.open(self.settings.eastness_raster),
            }
            self.local.topo_sources = sources
            with self.lock:
                self.opened_topo.append(sources)
        return self.local.topo_sources

    def close(self) -> None:
        for sources in self.opened_topo:
            close_dataset_dict(sources)

    def _reject_written(self, sample_id: str, reason: str, paths: list[Path]) -> None:
        rejected = self.settings.output_root / "rejected_artifacts" / reason
        rejected.mkdir(parents=True, exist_ok=True)
        for path in paths:
            for candidate in [path, Path(f"{path}.partial")]:
                if candidate.is_file():
                    destination = rejected / f"{sample_id}_{candidate.name}"
                    os.replace(candidate, destination)

    def materialize(self, row: dict) -> dict:
        sample_id = _safe_sample_id(row["sample_id"])
        shard = _sample_shard(sample_id)
        chip_path = self.settings.output_root / "images" / shard / f"{sample_id}.tif"
        topo_path = self.settings.output_root / "topo_chips" / shard / f"{sample_id}.tif"
        record = {
            "sample_id": sample_id,
            "materialization_status": "failed",
            "failure_stage": None,
            "failure_type": None,
            "failure_message": None,
        }
        try:
            chip_path.parent.mkdir(parents=True, exist_ok=True)
            provenance = extract_naip_chip_for_point(
                lon=float(row["lon"]),
                lat=float(row["lat"]),
                out_fp=str(chip_path),
                chip_size=self.settings.chip_size,
                tileindex_gdf=self.tileindex,
                naip_file_index={},
                source_record=row,
                source_lock_mode="strict",
                min_valid_frac=self.settings.naip_min_valid_frac,
            )
            if not provenance:
                record["failure_stage"] = "naip"
                record["failure_type"] = "strict_source_extraction_rejected"
                return record
            _validate_chip(chip_path, self.settings.chip_size, 4)

            environment = {variable: float(row[variable]) for variable in ALL_ENV_VARS}
            if not raw_environment_valid_mask(pd.DataFrame([environment])).iloc[0]:
                record["failure_stage"] = "environment"
                record["failure_type"] = "invalid_retained_environment"
                self._reject_written(sample_id, "environment", [chip_path])
                return record
            normalized = {
                variable: (
                    environment[variable] - float(self.worldclim_stats[variable]["mean"])
                )
                / float(self.worldclim_stats[variable]["std"])
                for variable in WC_VARS
            }
            normalized["ghm"] = environment["ghm"]

            topo_path.parent.mkdir(parents=True, exist_ok=True)
            topo = extract_topo_for_naip_chip(
                naip_chip_fp=str(chip_path),
                topo_sources=self._topo_sources(),
                out_fp=str(topo_path),
                topo_chip_size=self.settings.topo_chip_size,
                topo_norm_stats=self.topo_stats,
                normalize_topo=True,
            )
            if float(topo.get("topo_valid_frac", 0.0)) < self.settings.topo_min_valid_frac:
                record["failure_stage"] = "topography"
                record["failure_type"] = "insufficient_topography_valid_fraction"
                record["failure_message"] = str(topo.get("topo_valid_frac"))
                self._reject_written(sample_id, "topography", [chip_path, topo_path])
                return record
            _validate_chip(topo_path, self.settings.topo_chip_size, 4)

            record.update(
                {
                    "materialization_status": "success",
                    "chip_path": chip_path.relative_to(self.settings.output_root).as_posix(),
                    "topo_chip_path": topo_path.relative_to(
                        self.settings.output_root
                    ).as_posix(),
                    **{key: _jsonable(value) for key, value in provenance.items()},
                    **normalized,
                    **{key: _jsonable(value) for key, value in topo.items()},
                }
            )
            return record
        except Exception as error:
            record["failure_stage"] = record["failure_stage"] or "exception"
            record["failure_type"] = type(error).__name__
            record["failure_message"] = str(error)[:2000]
            record["failure_traceback"] = traceback.format_exc(limit=8)
            self._reject_written(sample_id, "exception", [chip_path, topo_path])
            return record


class _UnionFind:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        while parent != self.parent[parent]:
            parent = self.parent[parent]
        while value != parent:
            next_value = self.parent[value]
            self.parent[value] = parent
            value = next_value
        return parent

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            smaller, larger = sorted([left_root, right_root])
            self.parent[larger] = smaller


def reconcile_actual_source_groups(
    points: pd.DataFrame,
    seed: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    block_size_m: float,
    n_folds: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Freeze leakage-safe splits after exact constituent provenance is known."""
    work = points.copy()
    union = _UnionFind(work["sample_id"].astype(str).tolist())
    tile_owner: dict[str, str] = {}
    for row in work.itertuples(index=False):
        sample_id = str(row.sample_id)
        tile_keys = [
            value.strip()
            for value in str(row.actual_source_tile_keys).split(";")
            if value.strip()
        ]
        if not tile_keys:
            raise RuntimeError(f"Successful sample lacks actual source tiles: {sample_id}")
        for tile_key in tile_keys:
            owner = tile_owner.setdefault(tile_key, sample_id)
            union.union(sample_id, owner)
    work["actual_source_component"] = work["sample_id"].astype(str).map(union.find)
    component_labels = {
        root: "actual_" + hashlib.sha1(root.encode("utf-8")).hexdigest()[:12]
        for root in sorted(work["actual_source_component"].unique())
    }
    work["actual_source_component"] = work["actual_source_component"].map(component_labels)

    work["candidate_split"] = work.get("split")
    work["candidate_block_id"] = work.get("block_id")
    work["candidate_fold"] = work.get("fold")
    uniform = stratified_train_val_test_split(
        work.drop(columns=["split"], errors="ignore"),
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
        group_column="actual_source_component",
    )
    spatial = assign_spatial_blocks(
        gpd.GeoDataFrame(
            work.drop(columns=["block_id", "fold"], errors="ignore"),
            geometry=gpd.points_from_xy(work["lon"], work["lat"]),
            crs="EPSG:4326",
        ),
        block_size_m=block_size_m,
        n_folds=n_folds,
        seed=seed,
        group_column="actual_source_component",
    ).drop(columns="geometry")
    assignments = uniform[["sample_id", "split"]].merge(
        spatial[["sample_id", "block_id", "fold"]],
        on="sample_id",
        validate="one_to_one",
    )
    audit = work[
        [
            "sample_id",
            "actual_source_component",
            "candidate_split",
            "candidate_block_id",
            "candidate_fold",
        ]
    ].merge(
        assignments,
        on="sample_id",
        validate="one_to_one",
    )
    audit["split_changed"] = audit["candidate_split"] != audit["split"]
    audit["block_changed"] = audit["candidate_block_id"].astype(str) != audit[
        "block_id"
    ].astype(str)
    audit["fold_changed"] = pd.to_numeric(audit["candidate_fold"], errors="coerce") != pd.to_numeric(
        audit["fold"], errors="coerce"
    )
    return assignments, audit


def _apply_train_lat_lon_normalization(manifest: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = manifest.copy()
    train = out[out["split"] == "train"]
    if train.empty:
        raise RuntimeError("Cannot normalize coordinates without training rows.")
    stats = {}
    for column in ["lat", "lon"]:
        mean = float(train[column].mean())
        std = float(train[column].std())
        if not np.isfinite(std) or std <= 0:
            raise RuntimeError(f"Invalid training-only coordinate std for {column}: {std}")
        out[f"{column}_norm"] = (out[column] - mean) / std
        stats[column] = {"mean": mean, "std": std, "training_count": len(train)}
    return out, stats


def _merge_materialized(candidate: pd.DataFrame, records: pd.DataFrame) -> pd.DataFrame:
    record_columns = [
        column
        for column in records.columns
        if column == "sample_id" or column not in candidate.columns
    ]
    return candidate.merge(
        records[record_columns],
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )


def build_model_manifests(
    candidate_package: Path,
    output_root: Path,
    extraction_candidates: pd.DataFrame,
    successful_records: pd.DataFrame,
    assignments: pd.DataFrame,
    metadata: dict,
    n_folds: int,
    seed: int,
) -> dict:
    targets = [int(value) for value in metadata["targets"]]
    presence = pd.read_csv(candidate_package / "presence_manifest.csv", low_memory=False)
    presence = presence.drop(columns=["split", "block_id", "fold"], errors="ignore").merge(
        assignments,
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    presence = _merge_materialized(presence, successful_records)
    actual_components = extraction_candidates[["sample_id", "actual_source_component"]]
    presence = presence.merge(
        actual_components,
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    expected_presence_count = len(pd.read_csv(candidate_package / "presence_manifest.csv"))
    summary_rows = []
    hard_failures = []

    manifest_root = output_root / "model_manifests"
    for arm in ["random", "domain_balanced"]:
        reserve_paths = sorted((candidate_package / "ranked_reserves").glob(f"*_{arm}_ranked_to_*.csv"))
        if len(reserve_paths) != 1:
            raise RuntimeError(f"Expected one ranked reserve table for {arm}, found {reserve_paths}")
        reserve = pd.read_csv(reserve_paths[0], low_memory=False).sort_values("nested_rank")
        reserve = reserve.drop(columns=["split", "block_id", "fold"], errors="ignore").merge(
            assignments,
            on="sample_id",
            how="left",
            validate="one_to_one",
        )
        reserve = _merge_materialized(reserve, successful_records)
        reserve = reserve.merge(
            actual_components,
            on="sample_id",
            how="left",
            validate="one_to_one",
        )

        for target in targets:
            backgrounds = reserve.head(target).copy()
            if len(backgrounds) != target:
                hard_failures.append(
                    f"{arm}/{target}: only {len(backgrounds)} successful ranked backgrounds"
                )
                continue
            manifest = pd.concat([presence.copy(), backgrounds], ignore_index=True, sort=False)
            manifest["arm"] = arm
            manifest["background_target"] = target
            manifest["manifest_id"] = f"{metadata['host_slug']}_{arm}_{target:05d}_materialized"
            manifest, uniform_lat_lon = _apply_train_lat_lon_normalization(manifest)
            target_root = manifest_root / arm / f"background_{target:05d}"
            target_root.mkdir(parents=True, exist_ok=True)
            uniform_path = target_root / "uniform_dataset.csv"
            manifest.to_csv(uniform_path, index=False)
            _atomic_json(uniform_lat_lon, target_root / "uniform_lat_lon_stats.json")
            uniform_qa = audit_dataset_manifest(
                manifest,
                dataset_root=output_root,
                require_actual_provenance=True,
                enforce_source_tile_isolation=True,
                enforce_spatial_block_isolation=False,
                require_all_split_classes=True,
            )
            write_manifest_qa_package(uniform_qa, target_root / "qa_uniform")
            if manifest_has_hard_failures(uniform_qa):
                hard_failures.append(f"{arm}/{target}: uniform manifest QA failed")

            rounds = make_spatial_cv_rounds(
                manifest.drop(columns=["split", "lat_norm", "lon_norm"], errors="ignore"),
                n_folds=n_folds,
                seed=seed,
                validation_mode="heldout_fold",
            )
            for cv_round, round_frame in rounds.groupby("cv_round", sort=True):
                round_frame, round_lat_lon = _apply_train_lat_lon_normalization(round_frame)
                round_path = target_root / f"blockcv_{int(cv_round)}_dataset.csv"
                round_frame.to_csv(round_path, index=False)
                _atomic_json(
                    round_lat_lon,
                    target_root / f"blockcv_{int(cv_round)}_lat_lon_stats.json",
                )
                round_qa = audit_dataset_manifest(
                    round_frame,
                    dataset_root=output_root,
                    require_actual_provenance=True,
                    enforce_source_tile_isolation=True,
                    enforce_spatial_block_isolation=True,
                    require_all_split_classes=True,
                )
                write_manifest_qa_package(
                    round_qa,
                    target_root / f"qa_blockcv_{int(cv_round)}",
                )
                if manifest_has_hard_failures(round_qa):
                    hard_failures.append(
                        f"{arm}/{target}/CV{int(cv_round)}: manifest QA failed"
                    )

            summary_rows.append(
                {
                    "arm": arm,
                    "background_target": target,
                    "presence_candidates": expected_presence_count,
                    "retained_presences": len(presence),
                    "presence_attrition": expected_presence_count - len(presence),
                    "backgrounds": len(backgrounds),
                    "maximum_rank_used": int(backgrounds["nested_rank"].max()),
                    "reserve_replacements_used": int(
                        (backgrounds["nested_rank"] > target).sum()
                    ),
                    "rows": len(manifest),
                }
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_root / "model_manifest_summary.csv", index=False)
    if hard_failures:
        raise RuntimeError("Final model-manifest gate failed: " + "; ".join(hard_failures))
    return {
        "production_ready_manifests": True,
        "manifest_count": len(summary) * (n_folds + 1),
        "uniform_design_count": len(summary),
        "summary": summary.to_dict("records"),
    }


def _read_progress(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    records = {}
    with open(path, "r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if line_number == sum(1 for _ in open(path, "r", encoding="utf-8")):
                    break
                raise
            records[str(record["sample_id"])] = record
    return records


def run(args: argparse.Namespace) -> dict:
    package = Path(args.candidate_package).resolve()
    output = Path(args.output_root).resolve()
    required = [
        package / "manifest_metadata.json",
        package / "extraction_candidates.csv",
        package / "extraction_environment_raw.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Candidate package lacks required files: {missing}")
    with open(package / "manifest_metadata.json", "r", encoding="utf-8") as stream:
        candidate_metadata = json.load(stream)
    with open(package / "qa" / "status.json", "r", encoding="utf-8") as stream:
        candidate_status = json.load(stream)
    if not candidate_status.get("production_ready_for_extraction_review"):
        raise RuntimeError("Candidate package did not pass its structural QA gate.")

    identity = {
        "candidate_package": str(package),
        "candidate_metadata_sha256": _sha256(package / "manifest_metadata.json"),
        "extraction_candidates_sha256": _sha256(package / "extraction_candidates.csv"),
        "extraction_environment_raw_sha256": _sha256(
            package / "extraction_environment_raw.csv"
        ),
        "tileindex": str(Path(args.tileindex).resolve()),
        "tileindex_sha256": _sha256(args.tileindex),
        "worldclim_stats": str(Path(args.worldclim_stats_json).resolve()),
        "worldclim_stats_sha256": _sha256(args.worldclim_stats_json),
        "topo_normalization_stats": str(Path(args.topo_normalization_stats).resolve()),
        "topo_normalization_stats_sha256": _sha256(args.topo_normalization_stats),
        "topography_sources": {
            "elevation": _file_identity(args.dem_raster),
            "slope": _file_identity(args.slope_raster),
            "northness": _file_identity(args.northness_raster),
            "eastness": _file_identity(args.eastness_raster),
        },
        "chip_size": args.chip_size,
        "topo_chip_size": args.topo_chip_size,
        "naip_min_valid_frac": args.naip_min_valid_frac,
        "topo_min_valid_frac": args.topo_min_valid_frac,
        "seed": args.seed,
        "builder_git": _git_state(Path(__file__).resolve().parents[1]),
        "builder_source_sha256": _builder_source_hashes(Path(__file__).resolve().parent),
    }
    state_path = output / "materialization_run.json"
    if output.exists() and not args.resume:
        raise FileExistsError(f"Refusing existing output without --resume: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if state_path.is_file():
        with open(state_path, "r", encoding="utf-8") as stream:
            previous_state = json.load(stream)
        previous_identity = previous_state.get("identity")
        if previous_identity != identity:
            raise RuntimeError("Resume identity/configuration does not match the existing run.")
    else:
        _atomic_json(
            {
                "status": "running",
                "started_at": pd.Timestamp.now(tz="America/New_York").isoformat(),
                "identity": identity,
            },
            state_path,
        )

    candidates = pd.read_csv(package / "extraction_candidates.csv", low_memory=False)
    environment = pd.read_csv(package / "extraction_environment_raw.csv")
    candidates = candidates.merge(
        environment,
        on=["sample_id", "lat", "lon"],
        how="left",
        validate="one_to_one",
    )
    if candidates["sample_id"].duplicated().any():
        raise RuntimeError("Extraction candidates contain duplicate sample IDs.")
    if not raw_environment_valid_mask(candidates).all():
        raise RuntimeError("Candidate package contains invalid environmental values.")

    tileindex = add_canonical_naip_source_metadata(
        gpd.read_file(args.tileindex).to_crs("EPSG:4326")
    )
    worldclim_stats = _load_worldclim_stats(args.worldclim_stats_json)
    topo_stats = load_topo_normalization_stats(args.topo_normalization_stats)
    settings = MaterializationSettings(
        output_root=output,
        chip_size=args.chip_size,
        topo_chip_size=args.topo_chip_size,
        naip_min_valid_frac=args.naip_min_valid_frac,
        topo_min_valid_frac=args.topo_min_valid_frac,
        dem_raster=Path(args.dem_raster),
        slope_raster=Path(args.slope_raster),
        northness_raster=Path(args.northness_raster),
        eastness_raster=Path(args.eastness_raster),
    )
    worker = MaterializationWorker(tileindex, worldclim_stats, topo_stats, settings)
    progress_path = output / "materialization_progress.jsonl"
    progress = _read_progress(progress_path)
    successful_ids = {
        sample_id
        for sample_id, record in progress.items()
        if record.get("materialization_status") == "success"
    }
    pending = candidates[~candidates["sample_id"].astype(str).isin(successful_ids)].copy()
    print(
        f"Materialization candidates={len(candidates):,}; resumed successes={len(successful_ids):,}; "
        f"pending={len(pending):,}; workers={args.workers}"
    )
    completed_now = 0
    with open(progress_path, "a", encoding="utf-8") as progress_stream:
        try:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(worker.materialize, row): str(row["sample_id"])
                    for row in pending.to_dict("records")
                }
                for future in as_completed(futures):
                    record = future.result()
                    progress[str(record["sample_id"])] = record
                    progress_stream.write(json.dumps(record, allow_nan=False) + "\n")
                    progress_stream.flush()
                    completed_now += 1
                    if completed_now % 100 == 0 or completed_now == len(pending):
                        successes = sum(
                            item.get("materialization_status") == "success"
                            for item in progress.values()
                        )
                        print(
                            f"Completed this run {completed_now:,}/{len(pending):,}; "
                            f"total successes={successes:,}",
                            flush=True,
                        )
        finally:
            worker.close()

    records = pd.DataFrame(progress.values())
    records.to_csv(output / "materialization_records.csv", index=False)
    successes = records[records["materialization_status"] == "success"].copy()
    failures = records[records["materialization_status"] != "success"].copy()
    failures.to_csv(output / "materialization_failures.csv", index=False)
    successful_points = candidates.drop(columns=ALL_ENV_VARS).merge(
        successes[["sample_id", "actual_source_tile_keys"]],
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )
    assignments, assignment_audit = reconcile_actual_source_groups(
        successful_points,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        block_size_m=args.block_size_m,
        n_folds=args.n_folds,
    )
    successful_points = successful_points.merge(
        assignment_audit[["sample_id", "actual_source_component"]],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    assignments.to_csv(output / "actual_source_safe_split_assignments.csv", index=False)
    assignment_audit.to_csv(output / "candidate_to_actual_split_audit.csv", index=False)
    manifest_status = build_model_manifests(
        candidate_package=package,
        output_root=output,
        extraction_candidates=successful_points,
        successful_records=successes,
        assignments=assignments,
        metadata=candidate_metadata,
        n_folds=args.n_folds,
        seed=args.seed,
    )
    final = {
        "status": "complete",
        "completed_at": pd.Timestamp.now(tz="America/New_York").isoformat(),
        "identity": identity,
        "candidate_count": len(candidates),
        "success_count": len(successes),
        "failure_count": len(failures),
        "success_fraction": len(successes) / len(candidates),
        "failure_stage_counts": failures["failure_stage"].value_counts(dropna=False).to_dict(),
        "split_change_count": int(assignment_audit["split_changed"].sum()),
        "manifest_status": manifest_status,
    }
    _atomic_json(final, state_path)
    return final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-package", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--tileindex", required=True)
    parser.add_argument("--worldclim-stats-json", required=True)
    parser.add_argument("--topo-normalization-stats", required=True)
    parser.add_argument("--dem-raster", required=True)
    parser.add_argument("--slope-raster", required=True)
    parser.add_argument("--northness-raster", required=True)
    parser.add_argument("--eastness-raster", required=True)
    parser.add_argument("--chip-size", type=int, default=256)
    parser.add_argument("--topo-chip-size", type=int, default=64)
    parser.add_argument("--naip-min-valid-frac", type=float, default=1.0)
    parser.add_argument("--topo-min-valid-frac", type=float, default=0.90)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--block-size-m", type=float, default=200000.0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.chip_size <= 0 or args.topo_chip_size <= 0:
        parser.error("chip sizes must be positive")
    if not 0 <= args.naip_min_valid_frac <= 1 or not 0 <= args.topo_min_valid_frac <= 1:
        parser.error("valid-fraction thresholds must be in [0, 1]")
    if not np.isclose(args.train_frac + args.val_frac + args.test_frac, 1.0):
        parser.error("train/val/test fractions must sum to one")
    if args.n_folds < 3:
        parser.error("n-folds must be at least three")
    return args


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
