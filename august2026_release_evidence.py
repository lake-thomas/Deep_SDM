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


def _verified_index(
    rows: list[dict],
    failures: list[str],
    category: str,
    path: Path,
    *,
    expected_sha256: str | None = None,
    failure_label: str | None = None,
    **identity,
) -> bool:
    """Index one file only after checking its existence and recorded identity."""
    label = failure_label or category
    if not path.is_file():
        failures.append(f"missing:{label}")
        return False
    actual = sha256(path)
    if expected_sha256 and actual != expected_sha256:
        failures.append(f"hash:{label}")
        return False
    _index(rows, category, path, **identity)
    return True


def _payload_pointer(
    rows: list[dict],
    failures: list[str],
    payload: dict,
    owner_path: Path,
    path_key: str,
    category: str,
    *,
    hash_key: str | None = None,
    required: bool = True,
    **identity,
) -> Path | None:
    raw = payload.get(path_key)
    label = f"{category}:{path_key}"
    if not raw:
        if required:
            failures.append(f"pointer:{label}")
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = owner_path.parent / path
    expected = payload.get(hash_key or f"{path_key}_sha256")
    if not expected and required:
        failures.append(f"pointer_hash:{label}")
        return None
    if _verified_index(
        rows,
        failures,
        category,
        path,
        expected_sha256=expected,
        failure_label=label,
        **identity,
    ):
        return path
    return None


def _index_selection_chain(
    production_selection_path: Path,
    expected_sha256: str | None,
    rows: list[dict],
    failures: list[str],
) -> None:
    """Follow the frozen production -> BlockCV -> screen selection chain."""
    queue: list[tuple[Path, str | None]] = [
        (production_selection_path, expected_sha256)
    ]
    seen: set[Path] = set()
    evidence_pointer_keys = (
        "source_evidence_summary",
        "source_combined_evidence",
        "blockcv_evidence_summary",
        "blockcv_host_summary",
        "blockcv_fold_metrics",
    )
    evidence_payload_keys = (
        "source_dataset_evidence",
        "source_loss_evidence",
        "loss_training_metrics",
        "augmentation_training_metrics",
        "training_metrics",
        "combined_loss_candidate_evidence",
        "combined_augmentation_candidate_evidence",
        "matched_factorial_effects",
        "factorial_effect_summary",
        "fold_metrics",
        "host_summary",
    )
    while queue:
        manifest_path, expected = queue.pop(0)
        manifest_path = manifest_path.resolve()
        if manifest_path in seen:
            continue
        seen.add(manifest_path)
        if not _verified_index(
            rows,
            failures,
            "selection_manifest",
            manifest_path,
            expected_sha256=expected,
            failure_label=f"selection_manifest:{manifest_path.name}",
        ):
            continue
        payload = read_json(manifest_path)
        for source_key in (
            "source_blockcv_selection_manifests",
            "source_production_selection_manifests",
        ):
            for source in payload.get(source_key) or []:
                raw = source.get("path")
                if not raw:
                    failures.append(f"selection_source_pointer:{source_key}")
                    continue
                source_path = Path(raw)
                if not source_path.is_absolute():
                    source_path = manifest_path.parent / source_path
                queue.append((source_path, source.get("sha256")))
        for key in evidence_pointer_keys:
            pointer = _payload_pointer(
                rows,
                failures,
                payload,
                manifest_path,
                key,
                "selection_evidence",
                required=False,
            )
            if pointer is not None and pointer.suffix.lower() == ".json":
                evidence = read_json(pointer)
                prior = evidence.get("selection_manifest")
                if prior:
                    prior_path = Path(prior)
                    if not prior_path.is_absolute():
                        prior_path = pointer.parent / prior_path
                    queue.append(
                        (prior_path, evidence.get("selection_manifest_sha256"))
                    )
                for evidence_key in evidence_payload_keys:
                    _payload_pointer(
                        rows,
                        failures,
                        evidence,
                        pointer,
                        evidence_key,
                        "experiment_evidence",
                        required=False,
                    )
        direct_prior = payload.get("blockcv_selection_manifest")
        if direct_prior:
            direct_path = Path(direct_prior)
            if not direct_path.is_absolute():
                direct_path = manifest_path.parent / direct_path
            queue.append(
                (direct_path, payload.get("blockcv_selection_manifest_sha256"))
            )


def _index_production_training(
    training_metrics_path: Path,
    registry: pd.DataFrame,
    rows: list[dict],
    failures: list[str],
) -> None:
    """Verify and index the five selected datasets, configs, fits, and calibration."""
    training = pd.read_csv(training_metrics_path, low_memory=False)
    required_columns = {
        "host", "experiment", "config_path", "config_sha256", "run_dir",
        "training_dataset_csv_sha256", "best_checkpoint_sha256",
        "postfit_calibration", "stage", "evidence_gate_passed",
    }
    if not required_columns.issubset(training.columns):
        failures.append("production_training_columns")
        return
    if (
        len(training) != 5
        or set(training["host"]) != HOSTS
        or training["experiment"].duplicated().any()
    ):
        failures.append("production_training_host_set")
        return
    if set(training["experiment"]) != set(registry["experiment"]):
        failures.append("production_training_registry_experiments")
    if set(training["stage"]) != {"production-fit"}:
        failures.append("production_training_stage")
    gate_values = training["evidence_gate_passed"].astype(str).str.lower()
    if not gate_values.isin({"true"}).all():
        failures.append("production_training_evidence_gate")
    registry_by_experiment = registry.set_index("experiment")
    base_run_files = (
        "run_status.json",
        "runtime_provenance.json",
        "training_design.json",
        "training_history.csv",
        "model_summary_metrics.csv",
        "validation_prediction_results.csv",
        "validation_raw_calibration_metrics.json",
        "test_raw_calibration_metrics.json",
        "validation_threshold.json",
        "prediction_results.csv",
    )
    platt_run_files = (
        "platt_calibrator.json",
        "test_metrics_platt.json",
        "test_platt_calibration_metrics.json",
        "prediction_results_platt.csv",
    )
    for record in training.to_dict("records"):
        host = record["host"]
        experiment = record["experiment"]
        config_path = Path(record["config_path"])
        run_dir = Path(record["run_dir"])
        if experiment in registry_by_experiment.index:
            registry_row = registry_by_experiment.loc[experiment]
            if str(registry_row["host"]) != host:
                failures.append(f"production_registry_host:{experiment}")
            if "config_path" in registry.columns and Path(
                registry_row["config_path"]
            ).resolve() != config_path.resolve():
                failures.append(f"production_registry_config:{experiment}")
            if "run_dir" in registry.columns and Path(
                registry_row["run_dir"]
            ).resolve() != run_dir.resolve():
                failures.append(f"production_registry_run_dir:{experiment}")
            if (
                "config_sha256" in registry.columns
                and str(registry_row["config_sha256"]) != str(record["config_sha256"])
            ):
                failures.append(f"production_registry_config_hash:{experiment}")
        if not _verified_index(
            rows, failures, "production_config", config_path,
            expected_sha256=str(record["config_sha256"]),
            failure_label=f"production_config:{experiment}", host=host,
        ):
            continue
        config = read_json(config_path)
        dataset_path = Path(config.get("csv_path", ""))
        if not dataset_path.is_absolute():
            dataset_path = config_path.parent / dataset_path
        _verified_index(
            rows, failures, "selected_training_dataset", dataset_path,
            expected_sha256=str(record["training_dataset_csv_sha256"]),
            failure_label=f"selected_training_dataset:{experiment}", host=host,
        )
        checkpoint = run_dir / "checkpoints" / "best.pt"
        _verified_index(
            rows, failures, "production_checkpoint", checkpoint,
            expected_sha256=str(record["best_checkpoint_sha256"]),
            failure_label=f"production_checkpoint:{experiment}", host=host,
        )
        for name in base_run_files:
            _verified_index(
                rows, failures, "production_run_evidence", run_dir / name,
                failure_label=f"production_run_evidence:{experiment}:{name}",
                host=host,
            )
        calibration = str(record["postfit_calibration"]).lower()
        if calibration not in {"none", "platt"}:
            failures.append(f"production_calibration:{experiment}")
        expected_score_space = (
            "platt_calibrated_probability"
            if calibration == "platt"
            else "raw_sigmoid_probability"
        )
        if experiment in registry_by_experiment.index and str(
            registry_by_experiment.loc[experiment, "approved_score_space"]
        ) != expected_score_space:
            failures.append(f"production_score_space:{experiment}")
        if calibration == "platt":
            for name in platt_run_files:
                expected = None
                if name == "platt_calibrator.json":
                    expected = record.get("platt_calibrator_sha256")
                elif name == "prediction_results_platt.csv":
                    expected = record.get("test_platt_predictions_sha256")
                if pd.isna(expected):
                    expected = None
                if name in {"platt_calibrator.json", "prediction_results_platt.csv"} and not expected:
                    failures.append(
                        f"production_calibration_hash_missing:{experiment}:{name}"
                    )
                _verified_index(
                    rows, failures, "production_calibration_evidence", run_dir / name,
                    expected_sha256=expected,
                    failure_label=f"production_calibration_evidence:{experiment}:{name}",
                    host=host,
                )


def _index_artifact_batch_roots(
    roots: list[Path],
    category_prefix: str,
    exact_overlap: bool,
    rows: list[dict],
    failures: list[str],
) -> None:
    for root in roots:
        status_path = root / "batch_status.json"
        if not _verified_index(
            rows, failures, f"{category_prefix}_batch_status", status_path,
            failure_label=f"{category_prefix}_batch_status:{root}",
        ):
            continue
        status = read_json(status_path)
        experiments = status.get("completed_experiments") or []
        if status.get("status") != "complete" or len(experiments) != status.get(
            "candidate_count"
        ):
            failures.append(f"{category_prefix}_batch_identity:{root}")
        for experiment in experiments:
            candidate_root = root / experiment
            names = (
                ("summary.json", "exact_overlap_candidate_metrics.csv", "paired_june_comparison.csv")
                if exact_overlap
                else (
                    "summary.json", "pair_metrics.csv",
                    "june_baseline_comparison/summary.json",
                    "june_baseline_comparison/paired_artifact_comparison.csv",
                )
            )
            for name in names:
                _verified_index(
                    rows, failures, f"{category_prefix}_candidate_evidence",
                    candidate_root / name,
                    failure_label=f"{category_prefix}:{experiment}:{name}",
                )


def _job_id_is_real(value) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip()
    return bool(text) and text not in {
        "DRY_RUN", "SUBMISSION_FAILED", "MOSAIC_SUBMISSION_FAILED",
        "QA_SUBMISSION_FAILED", "NOT_SUBMITTED",
    }


def _index_hazel_submission_chain(
    submission_log_paths: list[Path],
    production_registry_path: Path,
    hazel_bundle_metadata_path: Path,
    inference_by_key: dict,
    inference_paths_by_key: dict,
    rows: list[dict],
    failures: list[str],
) -> None:
    expected_inference = {(host, product) for host in HOSTS for product in PRODUCTS}
    expected_mosaics = set(HOSTS)
    observed_inference = set()
    observed_mosaics = set()
    registry_hash = sha256(production_registry_path)
    bundle_hash = sha256(hazel_bundle_metadata_path)
    for path in submission_log_paths:
        if not _verified_index(
            rows, failures, "hazel_submission_log", path,
            failure_label=f"hazel_submission_log:{path.name}",
        ):
            continue
        frame = pd.read_csv(path, sep="\t", low_memory=False)
        if frame.empty:
            failures.append(f"hazel_submission_empty:{path.name}")
            continue
        if {"array_job_id", "qa_job_id", "product"}.issubset(frame.columns):
            required = {
                "host", "region", "product", "array_job_id", "qa_job_id",
                "tile_manifest_sha256", "production_registry_sha256",
                "bundle_metadata_sha256", "dry_run",
            }
            if not required.issubset(frame.columns):
                failures.append(f"hazel_inference_submission_columns:{path.name}")
                continue
            for record in frame.to_dict("records"):
                key = (record.get("host"), record.get("product"))
                if key in observed_inference:
                    failures.append(f"duplicate_hazel_inference_submission:{key}")
                observed_inference.add(key)
                evidence = inference_by_key.get(key) or {}
                if int(record.get("dry_run", 1)) != 0:
                    failures.append(f"hazel_inference_dry_run:{key}")
                if not _job_id_is_real(record.get("array_job_id")) or not _job_id_is_real(
                    record.get("qa_job_id")
                ):
                    failures.append(f"hazel_inference_job_id:{key}")
                if record.get("region") != evidence.get("region"):
                    failures.append(f"hazel_inference_region:{key}")
                if record.get("tile_manifest_sha256") != evidence.get(
                    "tile_manifest_sha256"
                ):
                    failures.append(f"hazel_inference_manifest_hash:{key}")
                if record.get("production_registry_sha256") != registry_hash:
                    failures.append(f"hazel_inference_registry_hash:{key}")
                if record.get("bundle_metadata_sha256") != bundle_hash:
                    failures.append(f"hazel_inference_bundle_hash:{key}")
        elif {"mosaic_array_job_id", "qa_job_id"}.issubset(frame.columns):
            required = {
                "host", "region", "mosaic_array_job_id", "qa_job_id",
                "expected_input_count", "deterministic_qa_summary",
                "deterministic_qa_sha256", "uncertainty_qa_summary",
                "uncertainty_qa_sha256", "dry_run",
            }
            if not required.issubset(frame.columns):
                failures.append(f"hazel_mosaic_submission_columns:{path.name}")
                continue
            for record in frame.to_dict("records"):
                host = record.get("host")
                if host in observed_mosaics:
                    failures.append(f"duplicate_hazel_mosaic_submission:{host}")
                observed_mosaics.add(host)
                if int(record.get("dry_run", 1)) != 0:
                    failures.append(f"hazel_mosaic_dry_run:{host}")
                if not _job_id_is_real(
                    record.get("mosaic_array_job_id")
                ) or not _job_id_is_real(record.get("qa_job_id")):
                    failures.append(f"hazel_mosaic_job_id:{host}")
                expected_count = None
                for product, path_key, hash_key in (
                    ("deterministic", "deterministic_qa_summary", "deterministic_qa_sha256"),
                    ("uncertainty", "uncertainty_qa_summary", "uncertainty_qa_sha256"),
                ):
                    key = (host, product)
                    expected_path = inference_paths_by_key.get(key)
                    supplied_path = Path(str(record.get(path_key, "")))
                    if expected_path is None or supplied_path.resolve() != expected_path.resolve():
                        failures.append(f"hazel_mosaic_qa_path:{host}:{product}")
                    elif record.get(hash_key) != sha256(expected_path):
                        failures.append(f"hazel_mosaic_qa_hash:{host}:{product}")
                    evidence = inference_by_key.get(key) or {}
                    if record.get("region") != evidence.get("region"):
                        failures.append(f"hazel_mosaic_region:{host}:{product}")
                    count = evidence.get("expected_tile_count")
                    if expected_count is None:
                        expected_count = count
                    elif count != expected_count:
                        failures.append(f"hazel_mosaic_product_count:{host}")
                if int(record.get("expected_input_count", -1)) != expected_count:
                    failures.append(f"hazel_mosaic_expected_count:{host}")
        else:
            failures.append(f"hazel_submission_schema:{path.name}")
    if observed_inference != expected_inference:
        failures.append("hazel_inference_submission_host_product_set")
    if observed_mosaics != expected_mosaics:
        failures.append("hazel_mosaic_submission_host_set")


def _index_dgx_run_chain(
    status_paths: list[Path],
    production_registry_path: Path,
    bundle_metadata_path: Path,
    inference_by_key: dict,
    inference_paths_by_key: dict,
    rows: list[dict],
    failures: list[str],
) -> None:
    """Verify completed local-DGX shards against their frozen bundle and QA."""
    expected = {(host, product) for host in HOSTS for product in PRODUCTS}
    registry_hash = sha256(production_registry_path)
    bundle_hash = sha256(bundle_metadata_path)
    bundle = read_json(bundle_metadata_path)
    defaults = bundle.get("inference_defaults") or {}
    base_seed = defaults.get("base_seed")
    expected_mc = {
        "deterministic": defaults.get("deterministic_mc_samples"),
        "uncertainty": defaults.get("uncertainty_mc_samples"),
    }
    observed_shards: set[tuple[str, str, str]] = set()
    counts_by_key: dict[tuple[str, str], int] = {}
    observed_keys: set[tuple[str, str]] = set()
    for path in status_paths:
        if not _verified_index(
            rows, failures, "dgx_run_status", path,
            failure_label=f"dgx_run_status:{path.name}",
        ):
            continue
        payload = read_json(path)
        host = payload.get("host")
        product = payload.get("product")
        key = (host, product)
        shard_id = str(payload.get("shard_id", ""))
        shard_key = (host, product, shard_id)
        if shard_key in observed_shards:
            failures.append(f"duplicate_dgx_run_status:{shard_key}")
        observed_shards.add(shard_key)
        observed_keys.add(key)
        evidence = inference_by_key.get(key) or {}
        if key not in expected:
            failures.append(f"dgx_run_identity:{key}")
            continue
        if payload.get("status") != "completed" or int(payload.get("returncode", -1)) != 0:
            failures.append(f"dgx_run_not_complete:{shard_key}")
        try:
            elapsed_seconds = float(payload.get("elapsed_seconds", -1))
        except (TypeError, ValueError):
            elapsed_seconds = -1
        if elapsed_seconds < 0:
            failures.append(f"dgx_run_elapsed_seconds:{shard_key}")
        if payload.get("production_registry_sha256") != registry_hash:
            failures.append(f"dgx_run_registry_hash:{shard_key}")
        if payload.get("bundle_metadata_sha256") != bundle_hash:
            failures.append(f"dgx_run_bundle_hash:{shard_key}")
        if payload.get("region") != evidence.get("region"):
            failures.append(f"dgx_run_region:{shard_key}")
        tile_list = Path(str(payload.get("tile_list", "")))
        if not tile_list.is_file() or payload.get("tile_list_sha256") != sha256(tile_list):
            failures.append(f"dgx_run_tile_list:{shard_key}")
        if payload.get("base_seed") != base_seed:
            failures.append(f"dgx_run_base_seed:{shard_key}")
        if payload.get("mc_samples") != expected_mc.get(product):
            failures.append(f"dgx_run_mc_samples:{shard_key}")
        try:
            expected_count = int(payload.get("expected_tile_count", -1))
            counts = {
                int(payload.get("prediction_count", -1)),
                int(payload.get("uncertainty_count", -1)),
                int(payload.get("count_count", -1)),
            }
        except (TypeError, ValueError):
            expected_count, counts = -1, {-1}
        if expected_count < 1 or counts != {expected_count}:
            failures.append(f"dgx_run_output_counts:{shard_key}")
        counts_by_key[key] = counts_by_key.get(key, 0) + max(expected_count, 0)
        output_root = evidence.get("output_root")
        output_dir = payload.get("output_dir")
        if not output_root or not output_dir:
            failures.append(f"dgx_run_output_root:{shard_key}")
        else:
            try:
                Path(output_dir).resolve().relative_to(Path(output_root).resolve())
            except ValueError:
                failures.append(f"dgx_run_output_root:{shard_key}")
        experiment = payload.get("experiment")
        if experiment and "experiment" in evidence and experiment != evidence.get("experiment"):
            failures.append(f"dgx_run_experiment:{shard_key}")
    if observed_keys != expected:
        failures.append("dgx_run_status_host_product_set")
    for key in expected:
        evidence = inference_by_key.get(key) or {}
        if counts_by_key.get(key, 0) != evidence.get("expected_tile_count"):
            failures.append(f"dgx_run_expected_count:{key}")


def _index_compacted_dgx_queue_chain(
    queue_summary_paths: list[Path],
    production_registry_path: Path,
    bundle_metadata_path: Path,
    inference_by_key: dict,
    rows: list[dict],
    failures: list[str],
) -> None:
    """Verify complete compacted DGX queues and their per-shard evidence."""
    expected_regions = {
        "east_of_mississippi": HOSTS - {"notholithocarpus_densiflorus"},
        "tanoak_ca_or": {"notholithocarpus_densiflorus"},
    }
    registry_hash = sha256(production_registry_path)
    bundle_hash = sha256(bundle_metadata_path)
    bundle = read_json(bundle_metadata_path)
    defaults = bundle.get("inference_defaults") or {}
    active_root = Path(defaults.get("active_tiled_output_root", "")).resolve()
    storage = defaults.get("storage_compaction") or {}
    if (
        storage.get("enabled") is not True
        or storage.get("float_compression") != "LERC_ZSTD"
        or float(storage.get("max_z_error", -1)) != 1e-5
    ):
        failures.append("compacted_dgx_bundle_storage_defaults")
    observed_regions = set()
    for summary_path in queue_summary_paths:
        if not _verified_index(
            rows, failures, "compacted_dgx_queue_summary", summary_path,
            failure_label=f"compacted_dgx_queue_summary:{summary_path.name}",
        ):
            continue
        if not summary_path.with_name("COMPLETED").is_file():
            failures.append(f"compacted_dgx_queue_marker:{summary_path}")
        summary = read_json(summary_path)
        region = summary.get("region")
        if region in observed_regions:
            failures.append(f"duplicate_compacted_dgx_queue:{region}")
        observed_regions.add(region)
        hosts = expected_regions.get(region)
        if hosts is None:
            failures.append(f"compacted_dgx_queue_region:{region}")
            continue
        if (
            summary.get("status") != "complete"
            or summary.get("stage") != "local-dgx-compacted-inference-queue"
            or summary.get("complete_manifest") is not True
            or summary.get("all_compact_qa_passed") is not True
            or summary.get("all_global_qa_passed") is not True
            or summary.get("all_staging_cleaned") is not True
            or int(summary.get("global_qa_count", -1)) != len(hosts) * 2
        ):
            failures.append(f"compacted_dgx_queue_not_ready:{region}")
        design_path = _payload_pointer(
            rows, failures, summary, summary_path, "design",
            "compacted_dgx_queue_design", region=region,
        )
        details_path = _payload_pointer(
            rows, failures, summary, summary_path, "shard_summaries",
            "compacted_dgx_shard_summaries", region=region,
        )
        if design_path is None or details_path is None:
            continue
        design = read_json(design_path)
        selected = design.get("selected_shards") or []
        if (
            design.get("status") != "frozen"
            or design.get("region") != region
            or design.get("bundle_metadata_sha256") != bundle_hash
            or design.get("registry_sha256") != registry_hash
            or set(design.get("hosts") or []) != hosts
            or design.get("cleanup_staging") is not True
            or design.get("full_manifest_authorized") is not True
            or len(selected) != int(summary.get("selected_shard_count", -1))
        ):
            failures.append(f"compacted_dgx_queue_design_identity:{region}")
        shard_dir = Path(str(design.get("shard_dir", "")))
        shard_metadata_path = shard_dir / "shard_metadata.json"
        if not _verified_index(
            rows, failures, "compacted_dgx_shard_metadata", shard_metadata_path,
            expected_sha256=design.get("shard_metadata_sha256"),
            failure_label=f"compacted_dgx_shard_metadata:{region}", region=region,
        ):
            continue
        shard_metadata = read_json(shard_metadata_path)
        source_manifest = Path(str(shard_metadata.get("source_manifest", "")))
        if not _verified_index(
            rows, failures, "compacted_dgx_source_manifest", source_manifest,
            expected_sha256=design.get("source_manifest_sha256"),
            failure_label=f"compacted_dgx_source_manifest:{region}", region=region,
        ):
            continue
        if (
            shard_metadata.get("status") != "complete"
            or int(shard_metadata.get("shard_count", -1)) != len(selected)
            or selected != [f"{index:05d}" for index in range(len(selected))]
        ):
            failures.append(f"compacted_dgx_shard_coverage:{region}")
        shard_summaries = read_json(details_path)
        if not isinstance(shard_summaries, list) or len(shard_summaries) != len(selected):
            failures.append(f"compacted_dgx_shard_summary_count:{region}")
            continue
        observed_shards = set()
        summed_tiles = 0
        for shard in shard_summaries:
            shard_id = str(shard.get("shard_id", ""))
            if shard_id in observed_shards:
                failures.append(f"duplicate_compacted_dgx_shard:{region}:{shard_id}")
            observed_shards.add(shard_id)
            if (
                shard.get("status") != "complete"
                or shard.get("region") != region
                or shard.get("all_compact_qa_passed") is not True
                or shard.get("staging_cleaned") is not True
                or int(shard.get("compaction_count", -1)) != len(hosts) * 2
                or int(shard.get("compact_qa_count", -1)) != len(hosts) * 2
                or int(shard.get("source_bytes", 0)) <= int(shard.get("compact_bytes", -1))
            ):
                failures.append(f"compacted_dgx_shard_not_ready:{region}:{shard_id}")
            summed_tiles += int(shard.get("tile_count", 0))
            shard_design_path = _payload_pointer(
                rows, failures, shard, details_path, "design",
                "compacted_dgx_shard_design", region=region,
            )
            compactions_path = _payload_pointer(
                rows, failures, shard, details_path, "compactions",
                "compacted_dgx_compactions", region=region,
            )
            if shard_design_path is None or compactions_path is None:
                continue
            shard_design = read_json(shard_design_path)
            tile_list = Path(str(shard_design.get("tile_list", "")))
            if (
                shard_design.get("status") != "frozen"
                or shard_design.get("region") != region
                or shard_design.get("shard_id") != shard_id
                or shard_design.get("bundle_metadata_sha256") != bundle_hash
                or set(shard_design.get("hosts") or []) != hosts
                or not tile_list.is_file()
                or shard_design.get("tile_list_sha256") != sha256(tile_list)
            ):
                failures.append(f"compacted_dgx_shard_design_identity:{region}:{shard_id}")
            compactions = read_json(compactions_path)
            if not isinstance(compactions, list) or len(compactions) != len(hosts) * 2:
                failures.append(f"compacted_dgx_compaction_count:{region}:{shard_id}")
                continue
            for compaction in compactions:
                if (
                    compaction.get("status") != "complete"
                    or compaction.get("stage") != "native-inference-shard-storage-compaction"
                    or compaction.get("all_outputs_passed") is not True
                    or compaction.get("float_compression") != "LERC_ZSTD"
                    or float(compaction.get("max_z_error", -1)) != 1e-5
                    or compaction.get("tile_manifest_sha256")
                    != shard_design.get("tile_list_sha256")
                    or int(compaction.get("expected_tile_count", -1))
                    != int(shard.get("tile_count", -2))
                ):
                    failures.append(f"compacted_dgx_compaction_identity:{region}:{shard_id}")
        if observed_shards != set(selected) or summed_tiles != int(summary.get("tile_count", -1)):
            failures.append(f"compacted_dgx_queue_tile_coverage:{region}")
        expected_tile_count = len([
            line for line in source_manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ])
        if summed_tiles != expected_tile_count:
            failures.append(f"compacted_dgx_source_tile_count:{region}")
        for host in hosts:
            for product in PRODUCTS:
                evidence = inference_by_key.get((host, product)) or {}
                expected_output = active_root / region / host / product
                if (
                    evidence.get("region") != region
                    or evidence.get("expected_tile_count") != expected_tile_count
                    or Path(str(evidence.get("output_root", ""))).resolve()
                    != expected_output.resolve()
                ):
                    failures.append(f"compacted_dgx_global_qa_identity:{host}:{product}")
    if observed_regions != set(expected_regions):
        failures.append("compacted_dgx_queue_region_set")


def assemble_release_evidence(
    final_acceptance_path: Path,
    production_registry_path: Path,
    hazel_bundle_metadata_path: Path,
    hazel_submission_log_paths: list[Path],
    candidate_provenance_paths: list[Path],
    attrition_paths: list[Path],
    inference_qa_paths: list[Path],
    mosaic_qa_paths: list[Path],
    visual_review_path: Path,
    aligned_correlation_path: Path,
    hann_results_path: Path,
    output_dir: Path,
    dgx_run_status_paths: list[Path] | None = None,
    dgx_compacted_queue_summary_paths: list[Path] | None = None,
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
    bundle_type = bundle.get("bundle_type") or "accepted_august2026_hazel_hann_inference"
    is_dgx = bundle_type == "accepted_august2026_dgx_hann_inference"
    is_hazel = bundle_type == "accepted_august2026_hazel_hann_inference"
    platform = "dgx" if is_dgx else "hazel"
    if not (is_dgx or is_hazel):
        failures.append(f"inference_bundle_type:{bundle_type}")
    if bundle.get("status") != "complete":
        failures.append(f"{platform}_bundle_status")
    if bundle.get("source_registry_sha256") != sha256(production_registry_path):
        failures.append(f"{platform}_bundle_registry_hash")
    if bundle.get("acceptance_summary_sha256") != sha256(final_acceptance_path):
        failures.append(f"{platform}_bundle_acceptance_hash")
    defaults = bundle.get("inference_defaults") or {}
    if (
        defaults.get("stride") != 64
        or defaults.get("blend_window") != "hann"
        or float(defaults.get("blend_edge_floor", -1)) != 0.05
    ):
        failures.append(f"{platform}_bundle_hann_defaults")
    _index(index_rows, f"{platform}_inference_bundle", hazel_bundle_metadata_path)
    bundle_hashes = bundle.get("file_sha256") or {}
    required_bundle_files = {
        "code_snapshot/inference_multimodal_production.py",
        "code_snapshot/model.py",
        "production_experiment_registry.tsv",
        "final_model_acceptance_summary.json",
    }
    if is_dgx:
        required_bundle_files.update({
            "code_snapshot/august2026_inference_provenance.py",
            "mc_stability_summary.json",
        })
    if not isinstance(bundle_hashes, dict) or not required_bundle_files.issubset(
        bundle_hashes
    ):
        failures.append(f"{platform}_bundle_file_manifest")
    else:
        for relative, expected in sorted(bundle_hashes.items()):
            _verified_index(
                index_rows, failures, f"{platform}_bundle_file",
                hazel_bundle_metadata_path.parent / relative,
                expected_sha256=expected,
                failure_label=f"{platform}_bundle_file:{relative}",
            )
    if is_dgx:
        mc_path = hazel_bundle_metadata_path.parent / "mc_stability_summary.json"
        if mc_path.is_file():
            mc = read_json(mc_path)
            if (
                bundle.get("mc_stability_summary_sha256") != sha256(mc_path)
                or mc.get("status") != "complete"
                or mc.get("recommended_mc_samples")
                != defaults.get("uncertainty_mc_samples")
            ):
                failures.append("dgx_bundle_mc_stability")

    provenance_by_host = {}
    for path in candidate_provenance_paths:
        payload = read_json(path)
        host = payload.get("host")
        if host in provenance_by_host:
            failures.append(f"duplicate_candidate_provenance:{host}")
        provenance_by_host[host] = payload
        if payload.get("status") != "complete" or not payload.get("production_ready"):
            failures.append(f"candidate_provenance_not_ready:{host}")
        _index(index_rows, "candidate_builder_provenance", path, host=host)
        metadata_path = _payload_pointer(
            index_rows, failures, payload, path, "candidate_metadata",
            "candidate_manifest_metadata", host=host,
        )
        state_path = _payload_pointer(
            index_rows, failures, payload, path, "materialization_state",
            "materialization_state", host=host,
        )
        if metadata_path is not None and state_path is not None:
            state = read_json(state_path)
            identity = state.get("identity") or {}
            _verified_index(
                index_rows, failures, "candidate_extraction_manifest",
                metadata_path.parent / "extraction_candidates.csv",
                expected_sha256=identity.get("extraction_candidates_sha256"),
                failure_label=f"candidate_extraction_manifest:{host}", host=host,
            )
            _verified_index(
                index_rows, failures, "candidate_environment_manifest",
                metadata_path.parent / "extraction_environment_raw.csv",
                expected_sha256=identity.get("extraction_environment_raw_sha256"),
                failure_label=f"candidate_environment_manifest:{host}", host=host,
            )
    if set(provenance_by_host) != HOSTS:
        failures.append("candidate_provenance_host_set")

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
        for key in (
            "candidate_csv", "candidate_metadata", "progress_jsonl",
            "canonical_records", "materialization_state",
        ):
            _payload_pointer(
                index_rows, failures, payload.get("inputs") or {}, path, key,
                "materialization_input", host=host,
            )
        for key, raw in (payload.get("outputs") or {}).items():
            output_path = Path(raw)
            if not output_path.is_absolute():
                output_path = path.parent / output_path
            _verified_index(
                index_rows, failures, "materialization_qa_table", output_path,
                failure_label=f"materialization_qa_table:{host}:{key}", host=host,
            )
    if set(attrition_by_host) != HOSTS:
        failures.append("attrition_host_set")

    inference_by_key = {}
    inference_paths_by_key = {}
    for path in inference_qa_paths:
        payload = read_json(path)
        key = (payload.get("host"), payload.get("product"))
        if key in inference_by_key:
            failures.append(f"duplicate_inference_qa:{key}")
        inference_by_key[key] = payload
        inference_paths_by_key[key] = path
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
        for key in ("tile_manifest", "tile_output_qa"):
            _payload_pointer(
                index_rows, failures, payload, path, key, "tiled_inference_evidence",
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
            status_raw = record.get("mosaic_status")
            if status_raw:
                if is_dgx and not record.get("mosaic_status_sha256"):
                    failures.append(f"mosaic_execution_status_hash:{host}:{product}")
                status_path = Path(status_raw)
                if not status_path.is_absolute():
                    status_path = path.parent / status_path
                if _verified_index(
                    index_rows, failures, "mosaic_execution_status", status_path,
                    expected_sha256=record.get("mosaic_status_sha256"),
                    failure_label=f"mosaic_execution_status:{host}:{product}",
                    host=host, region=payload.get("region"), product=product,
                ):
                    status = read_json(status_path)
                    if is_dgx and not status.get("output_sha256"):
                        failures.append(f"mosaic_execution_raster_hash:{host}:{product}")
                    if (
                        status.get("output_sha256")
                        and status.get("output_sha256") != record.get("sha256")
                    ):
                        failures.append(f"mosaic_execution_raster_hash:{host}:{product}")
            elif is_dgx:
                failures.append(f"mosaic_execution_status:{host}:{product}")
            info_path = raster_path.with_name(f"{raster_path.stem}_gdalinfo.json")
            if info_path.is_file():
                _index(
                    index_rows, "mosaic_gdalinfo", info_path,
                    host=host, region=payload.get("region"), product=product,
                )
            elif is_dgx:
                failures.append(f"mosaic_gdalinfo:{host}:{product}")
        quickview_path = _payload_pointer(
            index_rows, failures, payload, path, "quickview_summary",
            "fixed_scale_quickview_summary", host=host, region=payload.get("region"),
        )
        if quickview_path is not None:
            quickview = read_json(quickview_path)
            for label, raw in (quickview.get("outputs") or {}).items():
                image_path = Path(raw)
                if not image_path.is_absolute():
                    image_path = quickview_path.parent / image_path
                _verified_index(
                    index_rows, failures, "fixed_scale_quickview", image_path,
                    failure_label=f"fixed_scale_quickview:{host}:{label}", host=host,
                    region=payload.get("region"),
                )
        parity_path = _payload_pointer(
            index_rows, failures, payload, path, "native_mosaic_parity_summary",
            "native_mosaic_parity_summary", host=host,
            region=payload.get("region"),
        )
        if parity_path is not None:
            parity = read_json(parity_path)
            if parity.get("status") != "complete" or not parity.get(
                "production_ready"
            ):
                failures.append(f"native_mosaic_parity_not_ready:{host}")
            if (
                parity.get("host") != host
                or parity.get("region") != payload.get("region")
            ):
                failures.append(f"native_mosaic_parity_identity:{host}")
            _payload_pointer(
                index_rows, failures, parity, parity_path, "detail",
                "native_mosaic_parity_detail", host=host,
                region=payload.get("region"),
            )
    if set(mosaic_by_host) != HOSTS:
        failures.append("mosaic_qa_host_set")

    if is_dgx:
        if dgx_compacted_queue_summary_paths:
            if dgx_run_status_paths:
                failures.append("multiple_dgx_execution_evidence_modes")
            _index_compacted_dgx_queue_chain(
                dgx_compacted_queue_summary_paths, production_registry_path,
                hazel_bundle_metadata_path, inference_by_key,
                index_rows, failures,
            )
        else:
            _index_dgx_run_chain(
                dgx_run_status_paths or [], production_registry_path,
                hazel_bundle_metadata_path, inference_by_key, inference_paths_by_key,
                index_rows, failures,
            )
    else:
        _index_hazel_submission_chain(
            hazel_submission_log_paths, production_registry_path,
            hazel_bundle_metadata_path, inference_by_key, inference_paths_by_key,
            index_rows, failures,
        )

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
    _payload_pointer(
        index_rows, failures, visual, visual_review_path, "review_spec",
        "visual_review_spec",
    )

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

    acceptance_pointers = {}
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
            expected = acceptance.get("blockcv_summary_sha256")
        else:
            expected = acceptance.get(f"{key}_sha256")
        if not expected:
            failures.append(f"acceptance_pointer_hash:{key}")
            continue
        if _verified_index(
            index_rows, failures, f"accepted_{key}", pointer,
            expected_sha256=expected,
            failure_label=f"acceptance_pointer:{key}",
        ):
            acceptance_pointers[key] = pointer

    training_metrics_path = acceptance_pointers.get("training_metrics")
    if training_metrics_path is not None:
        _index_production_training(
            training_metrics_path, registry, index_rows, failures
        )

    blockcv_summary_path = acceptance_pointers.get("blockcv_evidence_dir")
    if blockcv_summary_path is not None:
        blockcv_payload = read_json(blockcv_summary_path)
        for key in ("selection_manifest", "training_metrics", "fold_metrics", "host_summary"):
            _payload_pointer(
                index_rows, failures, blockcv_payload, blockcv_summary_path, key,
                "blockcv_evidence", required=key in {"selection_manifest", "training_metrics"},
            )

    production_selection_path = acceptance_pointers.get(
        "production_selection_manifest"
    )
    if production_selection_path is not None:
        _index_selection_chain(
            production_selection_path,
            acceptance.get("production_selection_manifest_sha256"),
            index_rows,
            failures,
        )

    for key, pointer_key_pairs in (
        (
            "dataset_overlap_summary",
            (("shared_exposure_counts", "cross_host_dataset_overlap_table"),),
        ),
        (
            "pathway_summary",
            (
                ("common_support_predictions", "cross_host_common_support_predictions"),
                ("cross_host_pathway_correlations", "cross_host_pathway_correlations"),
            ),
        ),
    ):
        pointer = acceptance_pointers.get(key)
        if pointer is None:
            continue
        payload = read_json(pointer)
        for payload_key, category in pointer_key_pairs:
            _payload_pointer(
                index_rows, failures, payload, pointer, payload_key, category
            )

    for root_key, prefix, exact in (
        ("raw_artifact_roots", "raw_matched_boundary", False),
        ("configured_artifact_roots", "configured_matched_boundary", False),
        ("raw_exact_overlap_roots", "raw_exact_overlap", True),
        ("configured_exact_overlap_roots", "configured_exact_overlap", True),
    ):
        raw_roots = acceptance.get(root_key)
        if not isinstance(raw_roots, list) or not raw_roots:
            failures.append(f"acceptance_pointer:{root_key}")
            continue
        _index_artifact_batch_roots(
            [Path(raw) for raw in raw_roots], prefix, exact, index_rows, failures
        )

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
    bundle = parser.add_mutually_exclusive_group(required=True)
    bundle.add_argument("--hazel-bundle-metadata", type=Path)
    bundle.add_argument("--dgx-bundle-metadata", type=Path)
    parser.add_argument("--hazel-submission-log", type=Path, action="append", default=[])
    parser.add_argument("--dgx-run-status", type=Path, action="append", default=[])
    parser.add_argument(
        "--dgx-compacted-queue-summary", type=Path, action="append", default=[]
    )
    parser.add_argument("--candidate-provenance-summary", type=Path, action="append", required=True)
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
    bundle_metadata = args.dgx_bundle_metadata or args.hazel_bundle_metadata
    if args.dgx_bundle_metadata and args.hazel_submission_log:
        raise ValueError("DGX release evidence cannot use Hazel submission logs")
    if args.hazel_bundle_metadata and (
        args.dgx_run_status or args.dgx_compacted_queue_summary
    ):
        raise ValueError("Hazel release evidence cannot use DGX execution evidence")
    if args.hazel_bundle_metadata and not args.hazel_submission_log:
        raise ValueError("Hazel release evidence requires --hazel-submission-log")
    if args.dgx_bundle_metadata and bool(args.dgx_run_status) == bool(
        args.dgx_compacted_queue_summary
    ):
        raise ValueError(
            "DGX release evidence requires exactly one of --dgx-run-status or "
            "--dgx-compacted-queue-summary"
        )
    required = [
        args.final_acceptance, args.production_registry, bundle_metadata,
        *args.hazel_submission_log,
        *args.dgx_run_status,
        *args.dgx_compacted_queue_summary,
        args.visual_review, args.aligned_correlation, args.hann_results,
        *args.candidate_provenance_summary, *args.attrition_summary,
        *args.inference_qa_summary, *args.mosaic_qa_summary,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing release evidence inputs: {missing}")
    payload = assemble_release_evidence(
        args.final_acceptance, args.production_registry, bundle_metadata,
        args.hazel_submission_log,
        args.candidate_provenance_summary, args.attrition_summary,
        args.inference_qa_summary, args.mosaic_qa_summary,
        args.visual_review, args.aligned_correlation, args.hann_results,
        args.output_dir,
        dgx_run_status_paths=args.dgx_run_status,
        dgx_compacted_queue_summary_paths=args.dgx_compacted_queue_summary,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
