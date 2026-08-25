#!/usr/bin/env python3
"""Run a bounded, resumable sequence of compacted local-DGX inference shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

from august2026_summarize_experiments import sha256, write_json_atomic
try:
    from .run_compacted_shard_group import EAST_HOSTS, TANOAK_HOST
except ImportError:  # Frozen bundle copies both entry points to one directory.
    from run_compacted_shard_group import EAST_HOSTS, TANOAK_HOST


def selected_shards(
    shard_dir: Path,
    start_index: int,
    end_index: int | None,
    authorize_full_manifest: bool,
) -> list[tuple[str, Path]]:
    paths = sorted(shard_dir.glob("shard_*.txt"))
    if not paths:
        raise ValueError(f"No shard manifests found: {shard_dir}")
    expected = [f"shard_{index:05d}.txt" for index in range(len(paths))]
    if [path.name for path in paths] != expected:
        raise ValueError("Shard manifests are not a contiguous zero-based sequence")
    final = len(paths) - 1 if end_index is None else end_index
    if start_index < 0 or final < start_index or final >= len(paths):
        raise ValueError("Invalid shard index range")
    selected = [(f"{index:05d}", paths[index]) for index in range(start_index, final + 1)]
    if len(selected) == len(paths) and not authorize_full_manifest:
        raise PermissionError(
            "Processing the complete manifest requires --authorize-full-manifest"
        )
    return selected


def validate_shard_metadata(shard_dir: Path) -> dict:
    metadata_path = shard_dir / "shard_metadata.json"
    completed = shard_dir / "COMPLETED"
    if not metadata_path.is_file() or not completed.is_file():
        raise FileNotFoundError(f"Incomplete shard design: {shard_dir}")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise ValueError("Shard design status is not complete")
    source = Path(payload["source_manifest"])
    if not source.is_file() or sha256(source) != payload.get("source_manifest_sha256"):
        raise ValueError("Shard source manifest changed")
    if int(payload.get("shard_count", -1)) != len(list(shard_dir.glob("shard_*.txt"))):
        raise ValueError("Shard count differs from metadata")
    return payload


def execute(args: argparse.Namespace) -> dict:
    shard_metadata = validate_shard_metadata(args.shard_dir)
    selected = selected_shards(
        args.shard_dir, args.start_index, args.end_index,
        args.authorize_full_manifest,
    )
    bundle_metadata_path = args.bundle_root / "bundle_metadata.json"
    bundle_metadata = json.loads(bundle_metadata_path.read_text(encoding="utf-8"))
    output_root = Path(
        bundle_metadata["inference_defaults"]["active_tiled_output_root"]
    ).resolve()
    hosts = EAST_HOSTS if args.region == "east_of_mississippi" else (TANOAK_HOST,)
    queue_root = (
        output_root / "_run_metadata/compacted_queues" / args.region
        / f"shards_{selected[0][0]}_{selected[-1][0]}"
    )
    queue_root.mkdir(parents=True, exist_ok=True)
    design = {
        "status": "frozen", "region": args.region,
        "shard_dir": str(args.shard_dir.resolve()),
        "shard_metadata_sha256": sha256(args.shard_dir / "shard_metadata.json"),
        "source_manifest_sha256": shard_metadata["source_manifest_sha256"],
        "selected_shards": [shard_id for shard_id, _ in selected],
        "hosts": list(hosts), "bundle_metadata_sha256": sha256(bundle_metadata_path),
        "registry_sha256": sha256(args.registry),
        "cleanup_staging": bool(args.cleanup_staging),
        "full_manifest_authorized": bool(args.authorize_full_manifest),
    }
    design_path = queue_root / "design.json"
    if design_path.is_file():
        if json.loads(design_path.read_text(encoding="utf-8")) != design:
            raise RuntimeError("Pre-existing compacted queue design differs")
    else:
        write_json_atomic(design_path, design)
    completed = queue_root / "COMPLETED"
    summary_path = queue_root / "summary.json"
    if completed.is_file() and summary_path.is_file():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    started = time.monotonic()
    summaries = []
    for shard_id, tile_list in selected:
        command = [
            str(args.python_bin), str(args.bundle_root / "run_compacted_shard_group.py"),
            "--region", args.region, "--shard-id", shard_id,
            "--tile-list", str(tile_list), "--registry", str(args.registry),
            "--bundle-root", str(args.bundle_root),
            "--project-root", str(args.project_root),
            "--staging-base", str(args.staging_base),
            "--python-bin", str(args.python_bin),
            "--compaction-workers", str(args.compaction_workers),
        ]
        for host in hosts:
            command.extend(["--host", host])
        if args.cleanup_staging:
            command.append("--cleanup-staging")
        log = queue_root / f"shard_{shard_id}.log"
        with log.open("a", encoding="utf-8") as stream:
            stream.write("command=" + " ".join(command) + "\n")
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
        shard_summary = (
            output_root / "_run_metadata/compacted_shards" / args.region
            / f"shard_{shard_id}" / "summary.json"
        )
        if result.returncode != 0 or not shard_summary.is_file():
            raise RuntimeError(f"Compacted shard failed: {shard_id}; see {log}")
        payload = json.loads(shard_summary.read_text(encoding="utf-8"))
        if payload.get("status") != "complete" or payload.get("all_compact_qa_passed") is not True:
            raise RuntimeError(f"Compacted shard evidence is incomplete: {shard_id}")
        summaries.append(payload)
        progress = {
            "status": "running", "completed_shard_count": len(summaries),
            "selected_shard_count": len(selected), "last_completed_shard": shard_id,
        }
        write_json_atomic(queue_root / "progress.json", progress)
    detail = queue_root / "shard_summaries.json"
    write_json_atomic(detail, summaries)
    summary = {
        "status": "complete", "stage": "local-dgx-compacted-inference-queue",
        "region": args.region, "selected_shard_count": len(selected),
        "first_shard": selected[0][0], "last_shard": selected[-1][0],
        "tile_count": sum(int(row["tile_count"]) for row in summaries),
        "source_bytes": sum(int(row["source_bytes"]) for row in summaries),
        "compact_bytes": sum(int(row["compact_bytes"]) for row in summaries),
        "elapsed_seconds": time.monotonic() - started,
        "all_compact_qa_passed": all(row["all_compact_qa_passed"] for row in summaries),
        "all_staging_cleaned": all(
            row.get("staging_cleaned") is True for row in summaries
        ) if args.cleanup_staging else False,
        "design": str(design_path.resolve()), "design_sha256": sha256(design_path),
        "shard_summaries": str(detail.resolve()),
        "shard_summaries_sha256": sha256(detail),
        "guardrail": "Only the explicitly selected shard range was processed.",
    }
    write_json_atomic(summary_path, summary)
    completed.write_text("complete\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", choices=("east_of_mississippi", "tanoak_ca_or"), required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--staging-base", type=Path, default=Path("/home/ncsucnr/host_sdm_august2026_staging"))
    parser.add_argument("--python-bin", type=Path, default=Path("/home/ncsucnr/miniforge3/envs/host-naip-sdm/bin/python"))
    parser.add_argument("--compaction-workers", type=int, default=4)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--cleanup-staging", action="store_true")
    parser.add_argument("--authorize-full-manifest", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for required in (args.registry, args.bundle_root / "COMPLETED"):
        if not required.is_file():
            raise FileNotFoundError(required)
    payload = execute(args)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
