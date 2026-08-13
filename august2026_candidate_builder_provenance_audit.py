#!/usr/bin/env python3
"""Recover and verify exact committed builder provenance for one August dataset."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from august2026_summarize_experiments import read_json, sha256, write_json_atomic


CANDIDATE_DEPENDENCIES = [
    "candidate_manifests.py",
    "background.py",
    "env.py",
    "io_utils.py",
    "naip.py",
    "splits.py",
    "common.py",
    "candidate_environment_qa.py",
]


def git_file_bytes(code_root: Path, commit: str, name: str) -> bytes | None:
    result = subprocess.run(
        ["git", "show", f"{commit}:host_dataset_builder/{name}"],
        cwd=code_root,
        capture_output=True,
    )
    return result.stdout if result.returncode == 0 else None


def bytes_sha256(content: bytes) -> str:
    import hashlib

    return hashlib.sha256(content).hexdigest()


def commit_matches(
    code_root: Path,
    commit: str,
    expected_hashes: dict[str, str],
    names: list[str],
) -> bool:
    for name in names:
        content = git_file_bytes(code_root, commit, name)
        if content is None or bytes_sha256(content) != expected_hashes.get(name):
            return False
    return True


def recover_commit(
    code_root: Path,
    expected_hashes: dict[str, str],
    names: list[str],
) -> str | None:
    commits = subprocess.run(
        ["git", "rev-list", "--all", "--", "host_dataset_builder"],
        cwd=code_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return next(
        (
            commit
            for commit in commits
            if commit_matches(code_root, commit, expected_hashes, names)
        ),
        None,
    )


def audit_provenance(
    host: str,
    candidate_root: Path,
    materialized_root: Path,
    code_root: Path,
    output_dir: Path,
) -> dict:
    candidate_package = candidate_root / host
    dataset_root = materialized_root / host
    candidate_metadata_path = candidate_package / "manifest_metadata.json"
    materialization_path = dataset_root / "materialization_run.json"
    candidate_csv = candidate_package / "extraction_candidates.csv"
    raw_environment = candidate_package / "extraction_environment_raw.csv"
    for path in (
        candidate_metadata_path, materialization_path, candidate_csv, raw_environment
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    candidate = read_json(candidate_metadata_path)
    materialization = read_json(materialization_path)
    failures = []
    if materialization.get("status") != "complete":
        failures.append("materialization_incomplete")
    identity = materialization.get("identity") or {}
    if identity.get("candidate_metadata_sha256") != sha256(candidate_metadata_path):
        failures.append("candidate_metadata_hash")
    if identity.get("extraction_candidates_sha256") != sha256(candidate_csv):
        failures.append("candidate_csv_hash")
    if identity.get("extraction_environment_raw_sha256") != sha256(raw_environment):
        failures.append("candidate_environment_hash")

    candidate_hashes = candidate.get("builder_source_sha256") or {}
    missing_dependencies = [
        name for name in CANDIDATE_DEPENDENCIES if name not in candidate_hashes
    ]
    if missing_dependencies:
        failures.append("candidate_dependency_hashes_missing")
        recovered_candidate_commit = None
    else:
        candidate_git = candidate.get("builder_git") or {}
        recorded_commit = candidate_git.get("commit")
        if recorded_commit and commit_matches(
            code_root, recorded_commit, candidate_hashes, CANDIDATE_DEPENDENCIES
        ):
            recovered_candidate_commit = recorded_commit
        else:
            recovered_candidate_commit = recover_commit(
                code_root, candidate_hashes, CANDIDATE_DEPENDENCIES
            )
        if recovered_candidate_commit is None:
            failures.append("candidate_dependency_snapshot_not_recoverable")

    materializer_git = identity.get("builder_git") or {}
    materializer_hashes = identity.get("builder_source_sha256") or {}
    materializer_commit = materializer_git.get("commit")
    materializer_names = sorted(materializer_hashes)
    if materializer_git.get("dirty") is not False:
        failures.append("materializer_code_dirty")
    if not materializer_commit or not materializer_names or not commit_matches(
        code_root, materializer_commit, materializer_hashes, materializer_names
    ):
        failures.append("materializer_commit_hashes")

    output_dir.mkdir(parents=True, exist_ok=False)
    payload = {
        "status": "complete",
        "host": host,
        "production_ready": not failures,
        "candidate_metadata": str(candidate_metadata_path.resolve()),
        "candidate_metadata_sha256": sha256(candidate_metadata_path),
        "candidate_recorded_git": candidate.get("builder_git"),
        "candidate_dependency_scope": CANDIDATE_DEPENDENCIES,
        "candidate_dependency_snapshot_commit": recovered_candidate_commit,
        "candidate_dependency_snapshot_recovered": recovered_candidate_commit is not None,
        "candidate_dirty_flag_interpretation": (
            "A dirty candidate record is accepted only when every transitive source used "
            "by candidate generation exactly matches one recoverable committed snapshot."
        ),
        "materialization_state": str(materialization_path.resolve()),
        "materialization_state_sha256": sha256(materialization_path),
        "materializer_git": materializer_git,
        "materializer_source_count": len(materializer_names),
        "materializer_clean_commit_verified": (
            materializer_git.get("dirty") is False
            and "materializer_commit_hashes" not in failures
        ),
        "failure_count": len(failures),
        "failures": failures,
    }
    write_json_atomic(output_dir / "summary.json", payload)
    marker = "COMPLETED" if not failures else "FAILED"
    (output_dir / marker).write_text(marker.lower() + "\n", encoding="utf-8")
    if failures:
        raise RuntimeError(f"Candidate/materializer provenance failed: {failures}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--materialized-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = audit_provenance(
        args.host,
        args.candidate_root,
        args.materialized_root,
        Path(__file__).resolve().parent,
        args.output_dir,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
