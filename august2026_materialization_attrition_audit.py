#!/usr/bin/env python3
"""Audit source-locked materialization attrition before August model training."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from august2026_summarize_experiments import read_json, sha256, write_json_atomic


CANDIDATE_ROOT = Path(
    "/mnt/rsstu_dsfas_gsv_naip/Host_NAIP_SDM/Host_Datasets_USBg_August2026/"
    "candidate_manifests_20260813"
)
MATERIALIZED_ROOT = Path(
    "/mnt/rsstu_dsfas_gsv_naip/Host_NAIP_SDM/Host_Datasets_USBg_August2026/"
    "materialized_source_locked_20260813"
)
HOSTS = {
    "ailanthus_altissima",
    "juglans_nigra",
    "prunus_serotina",
    "vitis_labrusca",
    "notholithocarpus_densiflorus",
}
MAX_OVERALL_FAILURE_FRACTION = 0.20
MAX_CLASS_FAILURE_FRACTION = 0.10
MAX_LARGE_DOMAIN_FAILURE_FRACTION = 0.50
LARGE_DOMAIN_MINIMUM_N = 20
MAX_TARGET_DOMAIN_TOTAL_VARIATION = 0.05
MAX_TARGET_DOMAIN_SHARE_SHIFT = 0.02


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    frame.to_csv(partial, index=False)
    os.replace(partial, path)


def distribution_shift(
    intended: pd.Series,
    actual: pd.Series,
) -> tuple[float, float]:
    domains = sorted(set(intended.dropna()) | set(actual.dropna()))
    intended_share = intended.value_counts(normalize=True).reindex(domains, fill_value=0.0)
    actual_share = actual.value_counts(normalize=True).reindex(domains, fill_value=0.0)
    absolute = (actual_share - intended_share).abs()
    return float(0.5 * absolute.sum()), float(absolute.max()) if len(absolute) else 0.0


def canonicalize_progress_attempts(
    progress: pd.DataFrame,
    canonical_records: pd.DataFrame,
) -> tuple[pd.DataFrame, dict, list[str]]:
    """Validate the append-only retry log against the unique final-state table."""
    failures = []
    if "sample_id" not in progress or "sample_id" not in canonical_records:
        raise ValueError("Materialization progress tables require sample_id")
    attempts = progress.copy()
    attempts["_attempt_order"] = np.arange(len(attempts))
    attempt_counts = attempts["sample_id"].value_counts()
    duplicate_ids = attempt_counts[attempt_counts > 1]
    success_counts = (
        attempts["materialization_status"].eq("success").groupby(attempts["sample_id"]).sum()
    )
    if (success_counts > 1).any():
        failures.append("multiple_success_attempts_for_sample")
    final_attempts = (
        attempts.sort_values("_attempt_order", kind="mergesort")
        .drop_duplicates("sample_id", keep="last")
        .drop(columns="_attempt_order")
        .sort_values("sample_id", kind="mergesort")
        .reset_index(drop=True)
    )
    canonical = (
        canonical_records.sort_values("sample_id", kind="mergesort")
        .reset_index(drop=True)
    )
    if canonical["sample_id"].duplicated().any():
        failures.append("canonical_record_sample_ids_not_unique")
    if set(final_attempts["sample_id"]) != set(canonical["sample_id"]):
        failures.append("canonical_record_sample_set_mismatch")
    else:
        shared_columns = sorted(set(final_attempts.columns) & set(canonical.columns))
        try:
            pd.testing.assert_frame_equal(
                final_attempts[shared_columns],
                canonical[shared_columns],
                check_dtype=False,
                check_exact=False,
                rtol=1e-10,
                atol=1e-12,
            )
        except AssertionError:
            failures.append("canonical_records_differ_from_final_attempts")
    histories = attempts.groupby("sample_id")["materialization_status"].agg(list)
    retry_summary = {
        "attempt_rows": len(attempts),
        "canonical_rows": len(canonical),
        "retried_sample_ids": int(len(duplicate_ids)),
        "extra_attempt_rows": int((attempt_counts - 1).clip(lower=0).sum()),
        "maximum_attempts_per_sample": int(attempt_counts.max()) if len(attempt_counts) else 0,
        "failed_then_succeeded_sample_ids": int(
            sum("failed" in states[:-1] and states[-1] == "success" for states in histories)
        ),
        "repeated_terminal_failures": int(
            sum(len(states) > 1 and states[-1] == "failed" for states in histories)
        ),
    }
    return canonical, retry_summary, failures


def audit_host(host: str, output_dir: Path) -> dict:
    candidate_root = CANDIDATE_ROOT / host
    materialized_root = MATERIALIZED_ROOT / host
    candidates_path = candidate_root / "extraction_candidates.csv"
    metadata_path = candidate_root / "manifest_metadata.json"
    progress_path = materialized_root / "materialization_progress.jsonl"
    canonical_records_path = materialized_root / "materialization_records.csv"
    state_path = materialized_root / "materialization_run.json"
    required = [
        candidates_path, metadata_path, progress_path, canonical_records_path, state_path
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Materialization attrition inputs are missing: {missing}")
    state = read_json(state_path)
    if state.get("status") != "complete" or not state.get("manifest_status", {}).get(
        "production_ready_manifests"
    ):
        raise RuntimeError(f"Materialization is not production-ready: {state_path}")
    metadata = read_json(metadata_path)
    candidates = pd.read_csv(candidates_path, low_memory=False)
    attempt_progress = pd.read_json(progress_path, lines=True)
    canonical_records = pd.read_csv(canonical_records_path, low_memory=False)
    hard_failures = []
    if candidates["sample_id"].duplicated().any():
        hard_failures.append("candidate_sample_ids_not_unique")
    progress, retry_summary, progress_failures = canonicalize_progress_attempts(
        attempt_progress, canonical_records
    )
    hard_failures.extend(progress_failures)
    candidate_ids = set(candidates["sample_id"])
    progress_ids = set(progress["sample_id"])
    if candidate_ids != progress_ids:
        hard_failures.append(
            "candidate_progress_sample_set_mismatch:"
            f"missing={len(candidate_ids - progress_ids)},"
            f"unexpected={len(progress_ids - candidate_ids)}"
        )
    merged = candidates.merge(
        progress,
        on="sample_id",
        how="left",
        validate="one_to_one",
        suffixes=("_intended", "_actual"),
    )
    merged["materialization_success"] = merged["materialization_status"] == "success"
    allowed_status = merged["materialization_status"].isin({"success", "failed"})
    if not allowed_status.all():
        hard_failures.append("progress_contains_nonterminal_status")
    successful = merged[merged["materialization_success"]].copy()
    if "source_provenance_match" not in successful or not successful[
        "source_provenance_match"
    ].fillna(False).astype(bool).all():
        hard_failures.append("successful_rows_without_source_provenance_match")
    if "intended_source_used" not in successful or not successful[
        "intended_source_used"
    ].fillna(False).astype(bool).all():
        hard_failures.append("successful_rows_without_intended_source")
    if "actual_multi_acquisition" in successful and successful[
        "actual_multi_acquisition"
    ].fillna(True).astype(bool).any():
        hard_failures.append("successful_multi_acquisition_rows")
    if "actual_multi_state" in successful and successful[
        "actual_multi_state"
    ].fillna(True).astype(bool).any():
        hard_failures.append("successful_multi_state_rows")

    overall_failure_fraction = float((~merged["materialization_success"]).mean())
    if overall_failure_fraction > MAX_OVERALL_FAILURE_FRACTION:
        hard_failures.append(
            f"overall_failure_fraction={overall_failure_fraction:.6f}"
        )
    class_rows = []
    for presence, group in merged.groupby("presence"):
        failure_fraction = float((~group["materialization_success"]).mean())
        class_rows.append(
            {
                "presence": int(presence),
                "attempted": len(group),
                "success": int(group["materialization_success"].sum()),
                "failed": int((~group["materialization_success"]).sum()),
                "failure_fraction": failure_fraction,
            }
        )
        if failure_fraction > MAX_CLASS_FAILURE_FRACTION:
            hard_failures.append(
                f"class_{int(presence)}_failure_fraction={failure_fraction:.6f}"
            )
    class_summary = pd.DataFrame(class_rows)

    domain_rows = []
    for (domain, presence), group in merged.groupby(
        ["source_domain_id", "presence"], dropna=False
    ):
        failed = ~group["materialization_success"]
        failure_fraction = float(failed.mean())
        domain_rows.append(
            {
                "source_domain_id": domain,
                "presence": int(presence),
                "attempted": len(group),
                "success": int((~failed).sum()),
                "failed": int(failed.sum()),
                "failure_fraction": failure_fraction,
            }
        )
        if len(group) >= LARGE_DOMAIN_MINIMUM_N and failure_fraction > MAX_LARGE_DOMAIN_FAILURE_FRACTION:
            hard_failures.append(
                f"large_domain_high_attrition:{domain}:class={int(presence)}:"
                f"n={len(group)}:fraction={failure_fraction:.6f}"
            )
        if len(group) >= LARGE_DOMAIN_MINIMUM_N and int((~failed).sum()) == 0:
            hard_failures.append(
                f"large_domain_fully_lost:{domain}:class={int(presence)}"
            )
    domain_summary = pd.DataFrame(domain_rows).sort_values(
        ["failure_fraction", "attempted"], ascending=[False, False]
    )
    failure_summary = (
        merged[~merged["materialization_success"]]
        .groupby(["failure_stage", "failure_type"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
    )

    manifest_rows = []
    manifest_root = materialized_root / "model_manifests"
    targets = [int(value) for value in metadata["targets"]]
    arms = metadata.get("arms") or ["random", "domain_balanced"]
    for arm in arms:
        rank_column = f"{arm}_nested_rank"
        for target in targets:
            target_root = manifest_root / arm / f"background_{target:05d}"
            manifest_path = target_root / "uniform_dataset.csv"
            if not manifest_path.is_file():
                hard_failures.append(f"missing_uniform_manifest:{arm}:{target}")
                continue
            manifest = pd.read_csv(manifest_path, low_memory=False)
            manifest_ids = set(manifest["sample_id"])
            failed_manifest_ids = manifest_ids - set(successful["sample_id"])
            if failed_manifest_ids:
                hard_failures.append(
                    f"manifest_contains_failed_samples:{arm}:{target}:"
                    f"n={len(failed_manifest_ids)}"
                )
            backgrounds = manifest[manifest["presence"] == 0]
            presences = manifest[manifest["presence"] == 1]
            if len(backgrounds) != target:
                hard_failures.append(
                    f"background_target_shortfall:{arm}:{target}:n={len(backgrounds)}"
                )
            expected_backgrounds = (
                candidates[candidates["presence"] == 0]
                .dropna(subset=[rank_column])
                .sort_values(rank_column)
                .head(target)
            )
            total_variation, max_share_shift = distribution_shift(
                expected_backgrounds["source_domain_id"],
                backgrounds["source_domain_id"],
            )
            if total_variation > MAX_TARGET_DOMAIN_TOTAL_VARIATION:
                hard_failures.append(
                    f"target_domain_total_variation:{arm}:{target}:"
                    f"value={total_variation:.6f}"
                )
            if max_share_shift > MAX_TARGET_DOMAIN_SHARE_SHIFT:
                hard_failures.append(
                    f"target_domain_max_share_shift:{arm}:{target}:"
                    f"value={max_share_shift:.6f}"
                )
            manifest_rows.append(
                {
                    "arm": arm,
                    "background_target": target,
                    "manifest_rows": len(manifest),
                    "presence_rows": len(presences),
                    "background_rows": len(backgrounds),
                    "failed_sample_ids": len(failed_manifest_ids),
                    "intended_vs_actual_background_domain_total_variation": total_variation,
                    "intended_vs_actual_background_domain_max_share_shift": max_share_shift,
                    "manifest": str(manifest_path.resolve()),
                    "manifest_sha256": sha256(manifest_path),
                }
            )
    manifest_summary = pd.DataFrame(manifest_rows)

    output_dir.mkdir(parents=True, exist_ok=False)
    write_csv_atomic(output_dir / "class_attrition.csv", class_summary)
    write_csv_atomic(output_dir / "source_domain_attrition.csv", domain_summary)
    write_csv_atomic(output_dir / "failure_types.csv", failure_summary)
    write_csv_atomic(output_dir / "final_manifest_domain_shift.csv", manifest_summary)
    summary = {
        "status": "complete",
        "host": host,
        "production_ready": not hard_failures,
        "candidate_rows": len(candidates),
        "progress_rows": len(progress),
        "progress_attempt_rows": len(attempt_progress),
        "retry_summary": retry_summary,
        "success_rows": int(merged["materialization_success"].sum()),
        "failed_rows": int((~merged["materialization_success"]).sum()),
        "overall_failure_fraction": overall_failure_fraction,
        "hard_failure_count": len(hard_failures),
        "hard_failures": hard_failures,
        "thresholds": {
            "max_overall_failure_fraction": MAX_OVERALL_FAILURE_FRACTION,
            "max_class_failure_fraction": MAX_CLASS_FAILURE_FRACTION,
            "large_domain_minimum_n": LARGE_DOMAIN_MINIMUM_N,
            "max_large_domain_failure_fraction": MAX_LARGE_DOMAIN_FAILURE_FRACTION,
            "max_target_domain_total_variation": MAX_TARGET_DOMAIN_TOTAL_VARIATION,
            "max_target_domain_share_shift": MAX_TARGET_DOMAIN_SHARE_SHIFT,
        },
        "inputs": {
            "candidate_csv": str(candidates_path.resolve()),
            "candidate_csv_sha256": sha256(candidates_path),
            "candidate_metadata": str(metadata_path.resolve()),
            "candidate_metadata_sha256": sha256(metadata_path),
            "progress_jsonl": str(progress_path.resolve()),
            "progress_jsonl_sha256": sha256(progress_path),
            "canonical_records": str(canonical_records_path.resolve()),
            "canonical_records_sha256": sha256(canonical_records_path),
            "materialization_state": str(state_path.resolve()),
            "materialization_state_sha256": sha256(state_path),
        },
        "outputs": {
            "class_attrition": str(output_dir / "class_attrition.csv"),
            "source_domain_attrition": str(output_dir / "source_domain_attrition.csv"),
            "failure_types": str(output_dir / "failure_types.csv"),
            "final_manifest_domain_shift": str(
                output_dir / "final_manifest_domain_shift.csv"
            ),
        },
        "guardrail": (
            "This audit tests extraction attrition and reserve-replacement drift; it "
            "does not establish ecological representativeness or acquisition-domain "
            "generalization by itself."
        ),
    }
    write_json_atomic(output_dir / "summary.json", summary)
    (output_dir / "COMPLETED").write_text("completed\n", encoding="utf-8")
    if hard_failures:
        raise RuntimeError(
            f"Materialization attrition audit failed for {host}: {hard_failures}"
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=sorted(HOSTS), required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (
        MATERIALIZED_ROOT / args.host / "qa" / "materialization_attrition_audit"
    )
    if output_dir.exists():
        raise FileExistsError(f"Refusing to reuse attrition-audit output: {output_dir}")
    summary = audit_host(args.host, output_dir)
    provenance_output = output_dir.parent / "candidate_builder_provenance"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parent / "august2026_candidate_builder_provenance_audit.py"),
            "--host", args.host,
            "--candidate-root", str(CANDIDATE_ROOT),
            "--materialized-root", str(MATERIALIZED_ROOT),
            "--output-dir", str(provenance_output),
        ],
        check=True,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
