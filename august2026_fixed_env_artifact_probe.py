#!/usr/bin/env python3
"""Replay frozen boundary pairs with a candidate model and fixed environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.windows import Window

import inference_multimodal_production as inference


DEFAULT_PAIRS = {
    "ny": Path(
        "/home/ncsucnr/projects/research_agent/agent_outputs/"
        "host_sdm_inference_artifacts_aug122026/results_embeddings_expanded/"
        "selected_matched_pairs.csv"
    ),
    "tanoak": Path(
        "/home/ncsucnr/projects/research_agent/agent_outputs/"
        "host_sdm_inference_artifacts_aug122026/results_tanoak_embeddings/"
        "selected_tanoak_ca_or_pairs.csv"
    ),
}
NY_RAW = Path(
    "/mnt/rsstu_dsfas_gsv_naip/Host_NAIP_SDM/NAIP_Archive/states/NY/2021/raw"
)
TANOAK_RAW = {
    "CA": Path(
        "/mnt/rsstu_dsfas_gsv_naip/Host_NAIP_SDM/NAIP_Archive/states/CA/2022/raw"
    ),
    "OR": Path(
        "/mnt/rsstu_dsfas_gsv_naip/Host_NAIP_SDM/NAIP_Archive/states/OR/2020/raw"
    ),
}
DESCRIPTOR_COLUMNS = [
    *(f"band_{index}_mean" for index in range(1, 5)),
    *(f"band_{index}_within_chip_sd" for index in range(1, 5)),
    "rgb_edge_energy",
]


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


def chip_windows(source, chip_size: int = 256):
    for fraction_y in (0.25, 0.5, 0.75):
        for fraction_x in (0.25, 0.5, 0.75):
            column = int(round(fraction_x * (source.width - chip_size)))
            row = int(round(fraction_y * (source.height - chip_size)))
            yield Window(column, row, chip_size, chip_size)


def read_chips(path: Path) -> tuple[np.ndarray, dict]:
    chips, means, standard_deviations, edges = [], [], [], []
    with rasterio.open(path) as source:
        for window in chip_windows(source):
            image = source.read([1, 2, 3, 4], window=window).astype(np.float32) / 255.0
            valid = np.isfinite(image).all(axis=0) & np.any(image != 0, axis=0)
            if valid.mean() < 0.25:
                continue
            image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
            chips.append(image)
            means.append([float(image[band][valid].mean()) for band in range(4)])
            standard_deviations.append(
                [float(image[band][valid].std()) for band in range(4)]
            )
            gray = image[:3].mean(axis=0)
            edges.append(
                float(
                    np.abs(np.diff(gray, axis=0)).mean()
                    + np.abs(np.diff(gray, axis=1)).mean()
                )
            )
    if not chips:
        raise RuntimeError(f"No valid diagnostic chips found in {path}")
    mean_array = np.asarray(means)
    standard_deviation_array = np.asarray(standard_deviations)
    descriptors = {
        **{
            f"band_{index + 1}_mean": float(mean_array[:, index].mean())
            for index in range(4)
        },
        **{
            f"band_{index + 1}_within_chip_sd": float(
                standard_deviation_array[:, index].mean()
            )
            for index in range(4)
        },
        "rgb_edge_energy": float(np.mean(edges)),
        "valid_chip_n": len(chips),
    }
    return np.stack(chips), descriptors


def tile_state_map(diagnostic: str, pairs: pd.DataFrame) -> dict[str, str]:
    if diagnostic == "ny":
        tiles = set(pairs["tile_i"]) | set(pairs["tile_j"])
        tiles |= set(pairs["control_tile_i"]) | set(pairs["control_tile_j"])
        return {tile: "NY" for tile in tiles}
    states = {}
    for row in pairs.itertuples(index=False):
        states[row.tile_i] = row.state_i
        states[row.tile_j] = row.state_j
        for tile in row.ca_control_tiles.split("|"):
            states[tile] = "CA"
        for tile in row.or_control_tiles.split("|"):
            states[tile] = "OR"
    return states


def tile_path(diagnostic: str, tile: str, state: str) -> Path:
    root = NY_RAW if diagnostic == "ny" else TANOAK_RAW[state]
    return root / f"{tile}.tif"


def load_candidate(
    checkpoint_path: Path,
    config_path: Path,
    dataset_path: Path,
    device: torch.device,
):
    config = inference.load_json(config_path) or {}
    checkpoint = inference.load_checkpoint_state(str(checkpoint_path), torch.device("cpu"))
    model_type = inference.infer_model_type_from_checkpoint(checkpoint, config, None)
    requirements = inference.model_requires(model_type)
    if model_type != "image_tabular":
        raise ValueError(f"Fixed-environment probe requires image_tabular, got {model_type}")
    env_vars = inference.resolve_env_features_for_inference(
        config=config,
        req=requirements,
        cli_env_features=None,
        include_topo_scalars=bool(config.get("include_topo_scalars", True)),
    )
    model = inference.build_and_load_model(
        checkpoint_fp=str(checkpoint_path),
        config=config,
        model_type=model_type,
        num_env_features=len(env_vars),
        device=device,
        strict=True,
    )
    columns = ["split", *env_vars]
    training = pd.read_csv(dataset_path, usecols=columns, low_memory=False)
    training = training[training["split"] == "train"]
    if training.empty:
        raise ValueError("Dataset has no training rows")
    mean_environment = (
        training[env_vars]
        .apply(pd.to_numeric, errors="coerce")
        .mean(axis=0)
        .reindex(env_vars)
        .to_numpy(dtype=np.float32)
    )
    if not np.isfinite(mean_environment).all():
        raise ValueError("Training-mean fixed environment contains nonfinite values")
    return model, env_vars, mean_environment


def embed_and_predict(model, chips, mean_environment, device, batch_size):
    features, probabilities = {}, {}
    model.eval()
    with torch.inference_mode():
        for tile, tile_chips in chips.items():
            feature_batches, probability_batches = [], []
            for start in range(0, len(tile_chips), batch_size):
                image = torch.from_numpy(tile_chips[start : start + batch_size]).to(device)
                environment = (
                    torch.from_numpy(mean_environment)
                    .to(device)
                    .unsqueeze(0)
                    .expand(len(image), -1)
                )
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    image_features = model.image_encoder(image)
                    environment_features = model.env_encoder(environment)
                    logits = model.classifier(
                        torch.cat([image_features, environment_features], dim=1)
                    )
                feature_batches.append(image_features.float().cpu().numpy())
                probability_batches.append(torch.sigmoid(logits).float().cpu().numpy())
            features[tile] = np.concatenate(feature_batches).mean(axis=0)
            probabilities[tile] = float(np.concatenate(probability_batches).mean())
    return features, probabilities


def cosine_distance(first: np.ndarray, second: np.ndarray) -> float:
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    return float(1.0 - np.dot(first, second) / denominator) if denominator > 0 else np.nan


def bootstrap_ci(values, seed: int, draws: int = 10000) -> list[float]:
    array = pd.Series(values).dropna().to_numpy(dtype=float)
    if not len(array):
        return [float("nan"), float("nan")]
    random = np.random.default_rng(seed)
    means = np.empty(draws)
    for start in range(0, draws, 500):
        stop = min(draws, start + 500)
        indices = random.integers(0, len(array), size=(stop - start, len(array)))
        means[start:stop] = array[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def pair_metrics(
    diagnostic: str,
    pairs: pd.DataFrame,
    features: dict,
    probabilities: dict,
    raw_vectors: dict,
) -> pd.DataFrame:
    rows = []
    for row in pairs.itertuples(index=False):
        if diagnostic == "ny":
            cross_tiles = (row.tile_i, row.tile_j)
            control_pairs = [(row.control_tile_i, row.control_tile_j)]
            identifier = row.pair_id
        else:
            cross_tiles = (row.tile_i, row.tile_j)
            ca_first, ca_second = row.ca_control_tiles.split("|")
            or_first, or_second = row.or_control_tiles.split("|")
            control_pairs = [(ca_first, ca_second), (or_first, or_second)]
            identifier = row.selected_pair_id

        cross_probability = abs(
            probabilities[cross_tiles[0]] - probabilities[cross_tiles[1]]
        )
        control_probability = float(
            np.mean(
                [abs(probabilities[first] - probabilities[second]) for first, second in control_pairs]
            )
        )
        cross_embedding = cosine_distance(
            features[cross_tiles[0]], features[cross_tiles[1]]
        )
        control_embedding = float(
            np.mean(
                [cosine_distance(features[first], features[second]) for first, second in control_pairs]
            )
        )
        cross_raw = float(
            np.linalg.norm(raw_vectors[cross_tiles[0]] - raw_vectors[cross_tiles[1]])
        )
        control_raw = float(
            np.mean(
                [np.linalg.norm(raw_vectors[first] - raw_vectors[second]) for first, second in control_pairs]
            )
        )
        output = {
            "pair_id": identifier,
            "date_pair": row.date_pair,
            "mid_lon": row.mid_lon,
            "mid_lat": row.mid_lat,
            "fixed_env_cross_absdiff": cross_probability,
            "fixed_env_control_absdiff": control_probability,
            "fixed_env_prediction_did": cross_probability - control_probability,
            "embedding_cross_distance": cross_embedding,
            "embedding_control_distance": control_embedding,
            "embedding_distance_did": cross_embedding - control_embedding,
            "raw_cross_distance": cross_raw,
            "raw_control_distance": control_raw,
            "raw_descriptor_distance_did": cross_raw - control_raw,
        }
        if diagnostic == "tanoak":
            if row.state_i == "CA":
                output["fixed_env_or_minus_ca"] = (
                    probabilities[row.tile_j] - probabilities[row.tile_i]
                )
            else:
                output["fixed_env_or_minus_ca"] = (
                    probabilities[row.tile_i] - probabilities[row.tile_j]
                )
            output["longitude_bin_west"] = row.longitude_bin_west
        rows.append(output)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", choices=["ny", "tanoak"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, default=None)
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional deterministic prefix limit for bounded smoke tests only.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.max_pairs is not None and args.max_pairs <= 0:
        parser.error("--max-pairs must be positive")
    return args


def main() -> None:
    args = parse_args()
    pairs_path = args.pairs or DEFAULT_PAIRS[args.diagnostic]
    required = [args.checkpoint, args.config, args.dataset, pairs_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing artifact-probe inputs: {missing}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to reuse nonempty probe output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_csv(pairs_path, low_memory=False)
    if args.max_pairs is not None:
        pairs = pairs.head(args.max_pairs).copy()
    states = tile_state_map(args.diagnostic, pairs)
    chips, raw_descriptors = {}, {}
    for tile, state in sorted(states.items()):
        path = tile_path(args.diagnostic, tile, state)
        if not path.is_file():
            raise FileNotFoundError(path)
        chips[tile], raw_descriptors[tile] = read_chips(path)
    raw_frame = pd.DataFrame.from_dict(raw_descriptors, orient="index")
    raw_mean = raw_frame[DESCRIPTOR_COLUMNS].mean(axis=0)
    raw_sd = raw_frame[DESCRIPTOR_COLUMNS].std(axis=0).replace(0, 1)
    raw_vectors = {
        tile: ((raw_frame.loc[tile, DESCRIPTOR_COLUMNS] - raw_mean) / raw_sd).to_numpy(float)
        for tile in raw_frame.index
    }

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    model, env_vars, mean_environment = load_candidate(
        args.checkpoint, args.config, args.dataset, device
    )
    candidate_config = inference.load_json(args.config) or {}
    features, probabilities = embed_and_predict(
        model, chips, mean_environment, device, args.batch_size
    )
    metrics = pair_metrics(
        args.diagnostic, pairs, features, probabilities, raw_vectors
    )
    summary = {
        "status": "complete",
        "diagnostic": args.diagnostic,
        "device": str(device),
        "pair_count": len(metrics),
        "unique_tile_count": len(states),
        "fixed_env_prediction_did_mean": float(metrics["fixed_env_prediction_did"].mean()),
        "fixed_env_prediction_did_bootstrap_95ci": bootstrap_ci(
            metrics["fixed_env_prediction_did"], seed=20260813
        ),
        "fixed_env_cross_absdiff_mean": float(metrics["fixed_env_cross_absdiff"].mean()),
        "fixed_env_control_absdiff_mean": float(metrics["fixed_env_control_absdiff"].mean()),
        "embedding_distance_did_mean": float(metrics["embedding_distance_did"].mean()),
        "raw_descriptor_distance_did_mean": float(
            metrics["raw_descriptor_distance_did"].mean()
        ),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "config": str(args.config.resolve()),
        "config_sha256": sha256(args.config),
        "candidate_experiment": candidate_config.get("experiment"),
        "candidate_august_experiment_spec": candidate_config.get(
            "august_experiment_spec"
        ),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256(args.dataset),
        "pairs": str(pairs_path.resolve()),
        "pairs_sha256": sha256(pairs_path),
        "environment_features": env_vars,
        "fixed_environment_source": "training_split_feature_means",
        "code_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "guardrail": (
            "Matched observational acquisition-boundary replay with environmental "
            "predictors held fixed; acquisition, date, phenology, and geography remain confounded."
        ),
    }
    if args.diagnostic == "tanoak":
        coastal = metrics[metrics["mid_lon"] <= -123]
        summary["coastal_west_of_123"] = {
            "pair_count": len(coastal),
            "fixed_env_prediction_did_mean": float(
                coastal["fixed_env_prediction_did"].mean()
            ),
            "fixed_env_or_minus_ca_mean": float(coastal["fixed_env_or_minus_ca"].mean()),
        }

    raw_frame.rename_axis("tile").reset_index().to_csv(
        args.output_dir / "tile_raw_descriptors.csv", index=False
    )
    pd.DataFrame(
        {"tile": list(probabilities), "fixed_env_probability": list(probabilities.values())}
    ).to_csv(args.output_dir / "tile_fixed_env_probabilities.csv", index=False)
    metrics.to_csv(args.output_dir / "pair_metrics.csv", index=False)
    np.savez_compressed(
        args.output_dir / "tile_mean_embeddings.npz",
        tile_names=np.asarray(sorted(features)),
        embeddings=np.stack([features[tile] for tile in sorted(features)]),
    )
    write_json_atomic(args.output_dir / "summary.json", summary)
    (args.output_dir / "COMPLETED").write_text("completed\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
