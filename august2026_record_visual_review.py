#!/usr/bin/env python3
"""Freeze an explicit five-host visual review of fixed-scale August mosaics."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from august2026_summarize_experiments import read_json, sha256, write_json_atomic


HOSTS = {
    "ailanthus_altissima",
    "juglans_nigra",
    "prunus_serotina",
    "vitis_labrusca",
    "notholithocarpus_densiflorus",
}
REVIEW_TARGETS = {
    "utm_seams",
    "state_acquisition_boundaries",
    "east_west_striations",
    "naip_tile_footprints",
    "unexpected_empty_coverage",
    "ecological_plausibility",
    "uncertainty_plausibility",
}
PASS_STATES = {"pass", "pass_with_note"}


def _parse_review_time(value: str) -> str:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("reviewed_at must include an explicit UTC offset")
    return parsed.isoformat()


def record_visual_review(
    review_spec_path: Path,
    mosaic_qa_paths: list[Path],
    output_dir: Path,
) -> dict:
    spec = read_json(review_spec_path)
    reviewer = str(spec.get("reviewer", "")).strip()
    if not reviewer:
        raise ValueError("Visual review requires a named reviewer")
    reviewed_at = _parse_review_time(str(spec.get("reviewed_at", "")))
    reviews = spec.get("reviews") or []
    if len(reviews) != 5 or {item.get("host") for item in reviews} != HOSTS:
        raise ValueError("Visual review spec must contain exactly the five hosts")
    review_by_host = {item["host"]: item for item in reviews}

    qa_by_host = {}
    for path in mosaic_qa_paths:
        qa = read_json(path)
        host = qa.get("host")
        if host in qa_by_host:
            raise ValueError(f"Duplicate mosaic QA for {host}")
        qa_by_host[host] = (path, qa)
    if set(qa_by_host) != HOSTS:
        raise ValueError("Visual review requires one mosaic QA summary per host")

    failures = []
    records = []
    for host in sorted(HOSTS):
        qa_path, qa = qa_by_host[host]
        review = review_by_host[host]
        host_failures = []
        if qa.get("status") != "complete" or not qa.get(
            "production_ready_for_visual_review"
        ):
            host_failures.append("automated_mosaic_qa")
        quickview_path = Path(qa.get("quickview_summary", ""))
        if not quickview_path.is_file():
            host_failures.append("quickview_summary_missing")
            quickview = {}
        else:
            quickview = read_json(quickview_path)
            if sha256(quickview_path) != qa.get("quickview_summary_sha256"):
                host_failures.append("quickview_summary_hash")
            if quickview.get("host") != host or quickview.get("region") != qa.get("region"):
                host_failures.append("quickview_identity")
        checks = review.get("checks") or {}
        if set(checks) != REVIEW_TARGETS:
            host_failures.append("review_target_set")
        failed_checks = sorted(
            target for target, decision in checks.items() if decision not in PASS_STATES
        )
        if failed_checks:
            host_failures.extend(f"visual_{target}" for target in failed_checks)
        if review.get("decision") != "approved":
            host_failures.append("review_decision")
        observations = str(review.get("observations", "")).strip()
        if not observations:
            host_failures.append("missing_observations")
        image_records = {}
        for label, raw_path in (quickview.get("outputs") or {}).items():
            image_path = Path(raw_path)
            if not image_path.is_file():
                host_failures.append(f"missing_image_{label}")
            else:
                image_records[label] = {
                    "path": str(image_path.resolve()),
                    "sha256": sha256(image_path),
                }
        if set(image_records) != {
            "probability_transparent", "probability_framed",
            "uncertainty_transparent", "uncertainty_framed",
        }:
            host_failures.append("quickview_image_set")
        records.append(
            {
                "host": host,
                "region": qa.get("region"),
                "decision": review.get("decision"),
                "checks": checks,
                "observations": observations,
                "mosaic_qa_summary": str(qa_path.resolve()),
                "mosaic_qa_summary_sha256": sha256(qa_path),
                "quickview_summary": str(quickview_path.resolve()),
                "quickview_summary_sha256": (
                    sha256(quickview_path) if quickview_path.is_file() else None
                ),
                "images": image_records,
                "passed": not host_failures,
                "failures": host_failures,
            }
        )
        failures.extend(f"{host}: {failure}" for failure in host_failures)

    output_dir.mkdir(parents=True, exist_ok=False)
    payload = {
        "status": "complete",
        "reviewer": reviewer,
        "reviewed_at": reviewed_at,
        "review_spec": str(review_spec_path.resolve()),
        "review_spec_sha256": sha256(review_spec_path),
        "host_count": len(records),
        "approved_for_release_evidence": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "reviews": records,
        "interpretation": (
            "Visual approval records inspection of fixed-scale probability and uncertainty "
            "quickviews. It complements, but does not replace, tiled and mosaic raster QA."
        ),
    }
    write_json_atomic(output_dir / "summary.json", payload)
    marker = "APPROVED" if not failures else "REJECTED"
    (output_dir / marker).write_text(f"{marker.lower()}\n", encoding="utf-8")
    if failures:
        raise RuntimeError(f"Visual review failed: {failures}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-spec", type=Path, required=True)
    parser.add_argument("--mosaic-qa-summary", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.review_spec.is_file():
        raise FileNotFoundError(args.review_spec)
    payload = record_visual_review(
        args.review_spec, args.mosaic_qa_summary, args.output_dir
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
