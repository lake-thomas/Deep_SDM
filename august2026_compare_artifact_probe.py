#!/usr/bin/env python3
"""Compare an August fixed-environment artifact replay to its frozen June baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


BASELINE_ROOT = Path(
    "/home/ncsucnr/projects/research_agent/agent_outputs/"
    "host_sdm_inference_artifacts_aug122026"
)
BASELINE_METRICS = {
    "ny": BASELINE_ROOT
    / "results_embeddings_expanded"
    / "matched_embedding_pair_metrics.csv",
    "tanoak": BASELINE_ROOT
    / "results_tanoak_embeddings"
    / "tanoak_embedding_pair_metrics.csv",
}
BASELINE_SHA256 = {
    "ny": "6c9fb2f85d3c6add68c8f1641a473f06e4781edbde415aa6aff8c2a642407b2f",
    "tanoak": "7380b398e0676fb87a4ec43bc9f609928123ea35a1ed8beebf5b01867cf713ca",
}
NY_TAXA = {
    "ailanthus_altissima",
    "juglans_nigra",
    "prunus_serotina",
    "vitis_labrusca",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload) -> None:
    partial = path.with_name(f"{path.name}.partial")
    with partial.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    os.replace(partial, path)


def bootstrap_mean_ci(values, seed: int, draws: int = 10000) -> list[float]:
    array = np.asarray(values, dtype=float)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("Bootstrap values must be nonempty and finite")
    random = np.random.default_rng(seed)
    means = np.empty(draws)
    for start in range(0, draws, 500):
        stop = min(draws, start + 500)
        indices = random.integers(0, len(array), size=(stop - start, len(array)))
        means[start:stop] = array[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def paired_comparison(frame: pd.DataFrame, seed: int = 20260813) -> dict:
    required = {"baseline_did", "candidate_did"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing paired comparison columns: {sorted(missing)}")
    values = frame[list(required)].apply(pd.to_numeric, errors="coerce")
    if values.empty or not np.isfinite(values.to_numpy()).all():
        raise ValueError("Paired comparison values must be nonempty and finite")

    baseline = values["baseline_did"].to_numpy(float)
    candidate = values["candidate_did"].to_numpy(float)
    signed_reduction = baseline - candidate
    absolute_reduction = np.abs(baseline) - np.abs(candidate)
    baseline_mean = float(baseline.mean())
    candidate_mean = float(candidate.mean())
    signed_ci = bootstrap_mean_ci(signed_reduction, seed=seed)
    absolute_ci = bootstrap_mean_ci(absolute_reduction, seed=seed + 1)

    if signed_ci[0] > 0:
        status = "statistically_supported_signed_reduction"
    elif candidate_mean < baseline_mean:
        status = "directional_signed_reduction"
    else:
        status = "no_signed_reduction"
    relative = (
        float(signed_reduction.mean() / baseline_mean)
        if baseline_mean != 0
        else None
    )
    return {
        "pair_count": int(len(values)),
        "baseline_signed_did_mean": baseline_mean,
        "candidate_signed_did_mean": candidate_mean,
        "paired_signed_did_reduction_mean": float(signed_reduction.mean()),
        "paired_signed_did_reduction_bootstrap_95ci": signed_ci,
        "relative_signed_did_reduction": relative,
        "baseline_absolute_did_mean": float(np.abs(baseline).mean()),
        "candidate_absolute_did_mean": float(np.abs(candidate).mean()),
        "paired_absolute_did_reduction_mean": float(absolute_reduction.mean()),
        "paired_absolute_did_reduction_bootstrap_95ci": absolute_ci,
        "comparison_status": status,
    }


def load_baseline(diagnostic: str, taxon: str | None, path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    if diagnostic == "ny":
        frame = frame[frame["taxon"] == taxon].copy()
        frame = frame.rename(
            columns={"constant_env_probability_absdiff_did": "baseline_did"}
        )
        required = ["pair_id", "date_pair", "baseline_did"]
    else:
        frame = frame.rename(
            columns={
                "selected_pair_id": "pair_id",
                "constant_env_prediction_did": "baseline_did",
            }
        )
        required = ["pair_id", "date_pair", "mid_lon", "baseline_did"]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Frozen baseline is missing columns: {sorted(missing)}")
    return frame[required]


def exact_pair_merge(baseline: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    required = {"pair_id", "date_pair", "fixed_env_prediction_did"}
    missing = required - set(candidate.columns)
    if missing:
        raise ValueError(f"Candidate pair metrics are missing columns: {sorted(missing)}")
    if baseline["pair_id"].duplicated().any() or candidate["pair_id"].duplicated().any():
        raise ValueError("Pair IDs must be unique in baseline and candidate metrics")
    baseline_ids = set(baseline["pair_id"])
    candidate_ids = set(candidate["pair_id"])
    if baseline_ids != candidate_ids:
        raise ValueError(
            "Candidate must replay the exact frozen pair set; "
            f"missing={len(baseline_ids - candidate_ids)}, "
            f"unexpected={len(candidate_ids - baseline_ids)}"
        )
    candidate = candidate.rename(
        columns={
            "date_pair": "candidate_date_pair",
            "fixed_env_prediction_did": "candidate_did",
        }
    )
    merged = baseline.merge(candidate, on="pair_id", how="inner", validate="one_to_one")
    if not (merged["date_pair"] == merged["candidate_date_pair"]).all():
        raise ValueError("Candidate and baseline date-pair labels disagree")
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", choices=["ny", "tanoak"], required=True)
    parser.add_argument("--taxon", choices=sorted(NY_TAXA), default=None)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-metrics", type=Path, default=None)
    args = parser.parse_args()
    if args.diagnostic == "ny" and args.taxon is None:
        parser.error("--taxon is required for the NY diagnostic")
    if args.diagnostic == "tanoak" and args.taxon is not None:
        parser.error("--taxon is not used for the Tanoak diagnostic")
    return args


def main() -> None:
    args = parse_args()
    baseline_path = args.baseline_metrics or BASELINE_METRICS[args.diagnostic]
    candidate_summary_path = args.candidate_dir / "summary.json"
    candidate_metrics_path = args.candidate_dir / "pair_metrics.csv"
    required = [
        baseline_path,
        candidate_summary_path,
        candidate_metrics_path,
        args.candidate_dir / "COMPLETED",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing artifact comparison inputs: {missing}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to reuse nonempty output: {args.output_dir}")

    candidate_summary = json.loads(candidate_summary_path.read_text(encoding="utf-8"))
    if candidate_summary.get("status") != "complete":
        raise ValueError("Candidate probe summary is not complete")
    if candidate_summary.get("diagnostic") != args.diagnostic:
        raise ValueError("Candidate probe diagnostic does not match requested diagnostic")
    expected_host = args.taxon or "notholithocarpus_densiflorus"
    candidate_spec = candidate_summary.get("candidate_august_experiment_spec")
    if candidate_spec is not None and candidate_spec.get("host") != expected_host:
        raise ValueError(
            "Candidate model host does not match the requested comparison: "
            f"expected {expected_host}, found {candidate_spec.get('host')}"
        )
    baseline_hash = sha256(baseline_path)
    if args.baseline_metrics is None and baseline_hash != BASELINE_SHA256[args.diagnostic]:
        raise ValueError(
            "Frozen June baseline hash changed: "
            f"expected {BASELINE_SHA256[args.diagnostic]}, found {baseline_hash}"
        )

    baseline = load_baseline(args.diagnostic, args.taxon, baseline_path)
    candidate = pd.read_csv(candidate_metrics_path, low_memory=False)
    merged = exact_pair_merge(baseline, candidate)
    result = paired_comparison(merged)
    result.update(
        {
            "status": "complete",
            "diagnostic": args.diagnostic,
            "taxon": expected_host,
            "candidate_experiment": candidate_summary.get("candidate_experiment"),
            "candidate_august_experiment_spec": candidate_spec,
            "baseline_metrics": str(baseline_path.resolve()),
            "baseline_metrics_sha256": baseline_hash,
            "frozen_default_baseline_sha256": BASELINE_SHA256[args.diagnostic],
            "candidate_probe_dir": str(args.candidate_dir.resolve()),
            "candidate_summary_sha256": sha256(candidate_summary_path),
            "candidate_pair_metrics_sha256": sha256(candidate_metrics_path),
            "candidate_checkpoint_sha256": candidate_summary.get("checkpoint_sha256"),
            "candidate_dataset_sha256": candidate_summary.get("dataset_sha256"),
            "code_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parent,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "interpretation": (
                "Positive reductions favor the August candidate. The comparison is paired "
                "on the exact frozen diagnostic cases and is one acceptance-suite component; "
                "it does not by itself select or approve a production model."
            ),
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_dir / "paired_artifact_comparison.csv", index=False)
    write_json_atomic(args.output_dir / "summary.json", result)
    (args.output_dir / "COMPLETED").write_text("completed\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
