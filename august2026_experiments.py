#!/usr/bin/env python3
"""Generate and run gated August 2026 Host SDM screening experiments.

This launcher is intentionally limited to the approved dataset-screen stages:

* ``smoke``: one two-epoch Vitis execution-path check.
* ``dataset-screen``: the nested random versus domain-balanced background-count
  screen for Vitis and Tanoak on frozen uniform manifests.
* ``control-dataset-screen``: the equal-depth random versus domain-balanced
  screen for Ailanthus, Juglans, and Prunus after their separately gated builds.

It refuses to generate configs until the source-locked materialization and the
specific model manifest both pass their hard QA gates. Runs are sequential on
the local GPU, use versioned names, and never reuse a partial output directory.
Later loss, augmentation, BlockCV, and production stages are generated only
after the preceding evidence gate has selected their inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path


DATE_TAG = "20260813"
PYTHON = Path("/home/ncsucnr/miniforge3/envs/host-naip-sdm/bin/python")
CODE_ROOT = Path(__file__).resolve().parent
TRAIN_ENTRYPOINT = CODE_ROOT / "main.py"
HOST_DATA_ROOT = Path(
    "/mnt/rsstu_dsfas_gsv_naip/Host_NAIP_SDM/"
    "Host_Datasets_USBg_August2026/materialized_source_locked_20260813"
)
OUTPUT_BASE = Path(
    "/mnt/rsstu_dsfas_gsv_naip/Host_NAIP_SDM/Outputs_Host_NAIP_SDMs/"
    "experimental_model_runs_august2026"
)
DESIGN_BASE = Path(
    "/mnt/rsstu_dsfas_gsv_naip/Host_NAIP_SDM/Host_Datasets_USBg_August2026/"
    "experiment_designs_20260813"
)
WORLDCLIM_STATS = Path(
    "/mnt/rsstu_dsfas_gsv_naip/Host_NAIP_SDM/Outputs_Host_NAIP_SDMs/"
    "production_model_runs_june222026/pre_fit_qa_june222026/"
    "worldclim_stats_inferred_from_usbg_csv_june222026.json"
)
TOPO_STATS = Path(
    "/mnt/rsstu_dsfas_gsv_naip/Host_NAIP_SDM/Env_Data/Topography_3DEP/"
    "mosaic/topo_norm_stats_3dep_30m.json"
)

HOST_TARGETS = {
    "vitis_labrusca": [6000, 10000, 15000, 20000],
    "notholithocarpus_densiflorus": [5000, 10000, 15000],
}
CONTROL_HOST_TARGETS = {
    "ailanthus_altissima": [20000],
    "juglans_nigra": [20000],
    "prunus_serotina": [20000],
}
ALL_HOST_TARGETS = {**HOST_TARGETS, **CONTROL_HOST_TARGETS}
ARMS = ["random", "domain_balanced"]


@dataclass(frozen=True)
class ExperimentSpec:
    stage: str
    host: str
    arm: str
    background_target: int
    epochs: int
    loss_strategy: str = "ordinary_bce"
    spatial_augmentation: str = "dihedral"
    radiometric_augmentation: bool = False
    dropout: float = 0.25
    label_smoothing: float = 0.0
    split_design: str = "uniform"
    seed: int = 42

    @property
    def name(self) -> str:
        stage_label = {
            "smoke": "stage00",
            "dataset-screen": "stage01",
            "control-dataset-screen": "stage01c",
        }[self.stage]
        arm_label = "db" if self.arm == "domain_balanced" else "rnd"
        return (
            f"{stage_label}_{self.host}_{arm_label}_bg{self.background_target:05d}_"
            f"bce_dih_d025_s000_uniform_{DATE_TAG}"
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    with partial.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    os.replace(partial, path)


def write_tsv_atomic(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    with partial.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(partial, path)


def stage_specs(stage: str) -> list[ExperimentSpec]:
    if stage == "smoke":
        return [
            ExperimentSpec(
                stage=stage,
                host="vitis_labrusca",
                arm="domain_balanced",
                background_target=6000,
                epochs=2,
            )
        ]
    if stage == "dataset-screen":
        return [
            ExperimentSpec(
                stage=stage,
                host=host,
                arm=arm,
                background_target=target,
                epochs=15,
            )
            for host, targets in HOST_TARGETS.items()
            for target in targets
            for arm in ARMS
        ]
    if stage == "control-dataset-screen":
        return [
            ExperimentSpec(
                stage=stage,
                host=host,
                arm=arm,
                background_target=target,
                epochs=15,
            )
            for host, targets in CONTROL_HOST_TARGETS.items()
            for target in targets
            for arm in ARMS
        ]
    raise ValueError(f"Unknown stage: {stage}")


def filter_specs(
    specs: list[ExperimentSpec],
    hosts: list[str] | None,
    arms: list[str] | None,
    targets: list[int] | None,
) -> list[ExperimentSpec]:
    selected = specs
    if hosts:
        selected = [spec for spec in selected if spec.host in hosts]
    if arms:
        selected = [spec for spec in selected if spec.arm in arms]
    if targets:
        selected = [spec for spec in selected if spec.background_target in targets]
    if not selected:
        raise ValueError("The requested filters selected no experiments")
    return selected


def manifest_paths(spec: ExperimentSpec) -> tuple[Path, Path, Path, Path]:
    dataset_root = HOST_DATA_ROOT / spec.host
    target_root = (
        dataset_root
        / "model_manifests"
        / spec.arm
        / f"background_{spec.background_target:05d}"
    )
    manifest = target_root / "uniform_dataset.csv"
    lat_lon_stats = target_root / "uniform_lat_lon_stats.json"
    manifest_qa = target_root / "qa_uniform" / "qa_status.json"
    return dataset_root, manifest, lat_lon_stats, manifest_qa


def validate_source_gate(spec: ExperimentSpec) -> dict:
    dataset_root, manifest, lat_lon_stats, manifest_qa = manifest_paths(spec)
    materialization_state = dataset_root / "materialization_run.json"
    required = [
        materialization_state,
        manifest,
        lat_lon_stats,
        manifest_qa,
        WORLDCLIM_STATS,
        TOPO_STATS,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Experiment source gate is incomplete: {missing}")
    materialization = read_json(materialization_state)
    if materialization.get("status") != "complete":
        raise RuntimeError(f"Materialization is not complete: {materialization_state}")
    if not materialization.get("manifest_status", {}).get("production_ready_manifests"):
        raise RuntimeError(f"Materialized manifest package did not pass: {materialization_state}")
    qa = read_json(manifest_qa)
    if not qa.get("production_ready"):
        raise RuntimeError(f"Specific model manifest did not pass QA: {manifest_qa}")
    return {
        "dataset_root": str(dataset_root),
        "materialization_state": str(materialization_state),
        "materialization_identity": materialization.get("identity"),
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "manifest_qa": str(manifest_qa),
        "manifest_qa_sha256": sha256(manifest_qa),
        "lat_lon_stats": str(lat_lon_stats),
        "lat_lon_stats_sha256": sha256(lat_lon_stats),
    }


def _worldclim_payload() -> dict:
    payload = read_json(WORLDCLIM_STATS)
    return payload.get("worldclim_stats", payload)


def build_config(spec: ExperimentSpec, stage_root: Path) -> dict:
    gate = validate_source_gate(spec)
    dataset_root, manifest, lat_lon_stats, _ = manifest_paths(spec)
    return {
        "experiment": spec.name,
        "output_dir": str(stage_root),
        "csv_path": str(manifest),
        "image_dir": str(dataset_root),
        "model_type": "image_tabular",
        "env_features": "auto",
        "include_topo_scalars": True,
        "epochs": spec.epochs,
        "learning_rate": 1e-5,
        "batch_size": 128,
        "dropout": spec.dropout,
        "label_smoothing": spec.label_smoothing,
        "loss_strategy": spec.loss_strategy,
        "lr_patience": 4,
        "es_patience": 7,
        "spatial_augmentation": spec.spatial_augmentation,
        "radiometric_augmentation": spec.radiometric_augmentation,
        "hidden_dim": 192,
        "env_feature_dim": 128,
        "naip_channels": 4,
        "pretrained_image": True,
        "num_workers": 8,
        "persistent_workers": True,
        "pin_memory": True,
        "use_fp16": True,
        "naip_scale_255": True,
        "strict_dataset": True,
        "seed": spec.seed,
        "deterministic_training": True,
        "restore_best_model": True,
        "save_epoch_checkpoints": False,
        "postfit_calibration": "none",
        "use_wandb": False,
        "worldclim_stats": _worldclim_payload(),
        "lat_lon_stats": read_json(lat_lon_stats),
        "topo_normalization_stats": read_json(TOPO_STATS),
        "august_experiment_spec": asdict(spec),
        "dataset_gate": gate,
        "scientific_note": (
            "August staged screen. Scores are conditional suitability under the "
            "documented background and optimization design, not literal occurrence "
            "probabilities. This uniform split screen does not replace five-fold "
            "held-out BlockCV or acquisition-boundary artifact acceptance tests."
        ),
    }


def write_design(stage: str, specs: list[ExperimentSpec]) -> tuple[Path, list[dict]]:
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=CODE_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    git_dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=CODE_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    if git_dirty:
        raise RuntimeError("Refusing to freeze an experiment design from a dirty code tree")
    stage_label = {
        "smoke": "stage00_smoke",
        "dataset-screen": "stage01_dataset_screen",
        "control-dataset-screen": "stage01_control_dataset_screen",
    }[stage]
    stage_root = OUTPUT_BASE / f"{stage_label}_{DATE_TAG}"
    design_root = DESIGN_BASE / f"{stage_label}_{DATE_TAG}"
    config_root = design_root / "configs"
    rows = []
    for spec in specs:
        config = build_config(spec, stage_root)
        config_path = config_root / f"{spec.name}.json"
        if config_path.exists():
            existing = read_json(config_path)
            if existing != config:
                raise FileExistsError(f"Refusing to replace changed config: {config_path}")
        else:
            write_json_atomic(config_path, config)
        rows.append(
            {
                **asdict(spec),
                "experiment": spec.name,
                "config_path": str(config_path),
                "config_sha256": sha256(config_path),
                "run_dir": str(stage_root / spec.name),
                "status": "pending",
                "note": "",
            }
        )
    columns = list(rows[0])
    registry = design_root / "experiment_registry.tsv"
    write_tsv_atomic(registry, rows, columns)
    write_json_atomic(
        design_root / "design_metadata.json",
        {
            "date_tag": DATE_TAG,
            "stage": stage,
            "code_root": str(CODE_ROOT),
            "code_commit": git_commit,
            "code_dirty": git_dirty,
            "stage_output_root": str(stage_root),
            "experiment_count": len(rows),
            "registry": str(registry),
        },
    )
    return registry, rows


def run_experiments(registry: Path, rows: list[dict], max_runs: int | None) -> None:
    columns = list(rows[0])
    launched = 0
    for row in rows:
        if max_runs is not None and launched >= max_runs:
            break
        run_dir = Path(row["run_dir"])
        status_path = run_dir / "run_status.json"
        best_path = run_dir / "checkpoints" / "best.pt"
        if status_path.is_file() and read_json(status_path).get("status") == "completed":
            if not best_path.is_file():
                raise RuntimeError(f"Completed run lacks best checkpoint: {run_dir}")
            row["status"] = "skipped_completed"
            continue
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"Refusing to reuse partial run directory: {run_dir}")

        log_path = registry.parent / "logs" / f"{row['experiment']}.log"
        if log_path.exists():
            raise FileExistsError(f"Refusing to replace existing launcher log: {log_path}")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        row["status"] = "running"
        row["note"] = str(log_path)
        write_tsv_atomic(registry, rows, columns)
        started = time.time()
        command = [str(PYTHON), str(TRAIN_ENTRYPOINT), "--config", row["config_path"]]
        try:
            with log_path.open("x", encoding="utf-8") as log_stream:
                subprocess.run(
                    command,
                    cwd=CODE_ROOT,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        except Exception as error:
            row["status"] = "failed"
            row["note"] = f"{type(error).__name__}: {error}; log={log_path}"
            write_tsv_atomic(registry, rows, columns)
            raise
        row["status"] = "completed"
        row["note"] = f"elapsed_seconds={time.time() - started:.3f}; log={log_path}"
        write_tsv_atomic(registry, rows, columns)
        launched += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["smoke", "dataset-screen", "control-dataset-screen"],
        required=True,
    )
    parser.add_argument("--host", action="append", choices=sorted(ALL_HOST_TARGETS))
    parser.add_argument("--arm", action="append", choices=ARMS)
    parser.add_argument("--target", action="append", type=int)
    parser.add_argument("--run", action="store_true", help="Run sequentially after writing configs")
    parser.add_argument("--max-runs", type=int, default=None)
    args = parser.parse_args()
    if args.max_runs is not None and args.max_runs <= 0:
        parser.error("--max-runs must be positive")
    return args


def main() -> None:
    args = parse_args()
    specs = filter_specs(stage_specs(args.stage), args.host, args.arm, args.target)
    registry, rows = write_design(args.stage, specs)
    print(f"Wrote gated experiment design: {registry}")
    if args.run:
        run_experiments(registry, rows, max_runs=args.max_runs)


if __name__ == "__main__":
    main()
