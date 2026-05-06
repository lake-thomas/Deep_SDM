#!/usr/bin/env python
"""Summarize and plot Deep_SDM autoresearch results.tsv."""
from __future__ import annotations

import argparse
import json
import math
import numbers
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


KEY_FIELDS = [
    "model_type",
    "optimizer",
    "scheduler",
    "learning_rate",
    "weight_decay",
    "batch_size",
    "seed",
    "dropout",
    "hidden_dim",
    "env_feature_dim",
    "topo_feature_dim",
    "pretrained_image",
    "lr_patience",
    "lr_factor",
]

PLOT_STYLE = {
    "figure.figsize": (11, 6),
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def _safe_json_loads(value: Any) -> dict[str, Any]:
    if not isinstance(value, str) or not value.strip():
        return {}
    return json.loads(value)


def _format_value(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = float(value)
        if math.isnan(value):
            return "nan"
        if value != 0 and (abs(value) < 0.001 or abs(value) >= 10000):
            return f"{value:.2g}"
        return f"{value:g}"
    return str(value)


def _values_equal(left: Any, right: Any) -> bool:
    try:
        left_float = float(left)
        right_float = float(right)
        if math.isnan(left_float) and math.isnan(right_float):
            return True
        return math.isclose(left_float, right_float, rel_tol=1e-12, abs_tol=1e-12)
    except (TypeError, ValueError):
        return str(left) == str(right)


def _parse_run_time(run_id: str) -> pd.Timestamp:
    return pd.to_datetime(str(run_id)[:15], format="%Y%m%d_%H%M%S", errors="coerce")


def _key_changes(row: pd.Series, reference: dict[str, Any]) -> str:
    changes: list[str] = []
    for field in KEY_FIELDS:
        current = row.get(field)
        prior = reference.get(field)
        if pd.isna(current):
            continue
        if not _values_equal(current, prior):
            changes.append(f"{field}: {_format_value(prior)} -> {_format_value(current)}")
    return "; ".join(changes) if changes else "baseline / repeat"


def _one_line_config(row: pd.Series) -> str:
    parts = [
        f"{row['model_type']}",
        f"opt={row['optimizer']}",
        f"lr={_format_value(row['learning_rate'])}",
        f"wd={_format_value(row['weight_decay'])}",
        f"drop={_format_value(row['dropout'])}",
        f"pat={_format_value(row['lr_patience'])}",
        f"seed={_format_value(row['seed'])}",
    ]
    if not bool(row.get("pretrained_image", True)):
        parts.append("no_pretrain")
    return ", ".join(parts)


def _dataframe_to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    headers = [str(c) for c in ["model_type", *df.columns]]
    rows = []
    for idx, row in df.iterrows():
        values = [idx]
        values.extend(row.tolist())
        formatted = []
        for value in values:
            if isinstance(value, numbers.Integral) and not isinstance(value, bool):
                formatted.append(str(int(value)))
            elif isinstance(value, numbers.Real) and not isinstance(value, bool):
                value_float = float(value)
                if value_float.is_integer():
                    formatted.append(str(int(value_float)))
                else:
                    formatted.append(format(value_float, floatfmt))
            else:
                formatted.append(str(value))
        rows.append(formatted)

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def load_enriched_results(results_tsv: Path) -> pd.DataFrame:
    df = pd.read_csv(results_tsv, sep="\t")
    if df.empty:
        raise ValueError(f"No rows found in {results_tsv}")

    search_space = df["search_space_json"].map(_safe_json_loads)
    for field in KEY_FIELDS:
        if field in df.columns:
            df[field] = df[field]
        else:
            df[field] = search_space.map(lambda x, f=field: x.get(f))

    for field in KEY_FIELDS:
        from_search = search_space.map(lambda x, f=field: x.get(f))
        df[field] = df[field].where(df[field].notna(), from_search)

    numeric_fields = [
        "primary_value",
        "mean_val_loss",
        "mean_val_mcc",
        "mean_val_auc",
        "mean_val_sensitivity",
        "mean_val_specificity",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "seed",
        "dropout",
        "hidden_dim",
        "env_feature_dim",
        "topo_feature_dim",
        "lr_patience",
        "lr_factor",
        "elapsed_seconds",
    ]
    for field in numeric_fields:
        if field in df.columns:
            df[field] = pd.to_numeric(df[field], errors="coerce")

    df["trial"] = range(1, len(df) + 1)
    df["run_time"] = df["run_id"].map(_parse_run_time)
    df["best_so_far_mcc"] = df["mean_val_mcc"].cummax()
    df["is_new_best"] = df["mean_val_mcc"].eq(df["best_so_far_mcc"])
    df["is_repeat_best"] = df.duplicated("best_so_far_mcc", keep=False) & df["is_new_best"]
    df["config_summary"] = df.apply(_one_line_config, axis=1)

    baseline = search_space.iloc[0]
    best = df.loc[df["mean_val_mcc"].idxmax()]
    best_cfg = {field: best[field] for field in KEY_FIELDS}
    df["changes_vs_baseline"] = df.apply(_key_changes, axis=1, reference=baseline)
    df["changes_vs_best"] = df.apply(_key_changes, axis=1, reference=best_cfg)
    previous_refs = []
    previous = baseline
    for cfg in search_space:
        previous_refs.append(previous)
        previous = cfg
    df["changes_vs_previous"] = [
        _key_changes(row, reference=previous_refs[i])
        for i, (_, row) in enumerate(df.iterrows())
    ]
    df["run_folder"] = df["run_dir"].map(lambda x: Path(str(x)).name)
    return df


def write_tables(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = [
        "trial",
        "run_id",
        "run_folder",
        "model_type",
        "mean_val_mcc",
        "mean_val_auc",
        "mean_val_loss",
        "mean_val_sensitivity",
        "mean_val_specificity",
        *KEY_FIELDS[1:],
        "changes_vs_previous",
        "changes_vs_baseline",
        "changes_vs_best",
    ]
    df[cols].to_csv(out_dir / "results_enriched.csv", index=False)

    top = df.sort_values(
        ["mean_val_mcc", "mean_val_auc", "mean_val_sensitivity", "mean_val_specificity", "mean_val_loss"],
        ascending=[False, False, False, False, True],
    ).head(20)
    top[cols].to_csv(out_dir / "top_20_runs.csv", index=False)

    family = (
        df.groupby("model_type")
        .agg(
            n=("run_id", "count"),
            best_mcc=("mean_val_mcc", "max"),
            median_mcc=("mean_val_mcc", "median"),
            best_auc=("mean_val_auc", "max"),
            best_loss=("mean_val_loss", "min"),
        )
        .sort_values("best_mcc", ascending=False)
    )
    family.to_csv(out_dir / "model_family_summary.csv")

    seed_rows = df[df["changes_vs_best"].str.contains("seed:", regex=False) | df["seed"].eq(42)]
    seed_rows[cols].to_csv(out_dir / "seed_and_best_config_runs.csv", index=False)

    md_lines = [
        "# Autoresearch Results Summary",
        "",
        f"Total valid trials: {len(df)}",
        "",
        "## Best Run",
        "",
    ]
    best = top.iloc[0]
    md_lines.extend(
        [
            f"- Trial: {int(best['trial'])}",
            f"- Run folder: `{best['run_folder']}`",
            f"- MCC: {best['mean_val_mcc']:.4f}",
            f"- AUC: {best['mean_val_auc']:.4f}",
            f"- Loss: {best['mean_val_loss']:.4f}",
            f"- Config: {best['config_summary']}",
            "",
            "## Main Findings",
            "",
            "- Full `image_topo_tabular` remained the strongest model family.",
            "- `learning_rate=1e-4`, `dropout=0.45`, `lr_patience=2`, and no weight decay produced the top seed-42 endpoint.",
            "- Removing the tabular branch, disabling ImageNet pretraining, or increasing batch size degraded MCC.",
            "- The best endpoint was seed-sensitive: alternate seeds were materially lower, so treat the peak result as promising but not fully robust.",
            "",
            "## Top 10 Runs",
            "",
            "| Trial | Run folder | Model | MCC | AUC | Loss | Targeted change vs previous | Key changes vs baseline |",
            "|---:|---|---|---:|---:|---:|---|---|",
        ]
    )
    for _, row in top.head(10).iterrows():
        md_lines.append(
            f"| {int(row['trial'])} | `{row['run_folder']}` | {row['model_type']} | "
            f"{row['mean_val_mcc']:.4f} | {row['mean_val_auc']:.4f} | {row['mean_val_loss']:.4f} | "
            f"{row['changes_vs_previous']} | "
            f"{row['changes_vs_baseline']} |"
        )
    md_lines.extend(
        [
            "",
            "## Model Family Summary",
            "",
            _dataframe_to_markdown(family, floatfmt=".4f"),
            "",
            "## Generated Files",
            "",
            "- `results_enriched.csv`: all rows with parsed hyperparameters and change labels.",
            "- `top_20_runs.csv`: top runs sorted by benchmark tie-breakers.",
            "- `model_family_summary.csv`: best and median metrics by model family.",
            "- `plots/*.png`: visual summaries.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(md_lines), encoding="utf-8")


def plot_results(df: pd.DataFrame, out_dir: Path) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(PLOT_STYLE)

    fig, ax = plt.subplots()
    ax.plot(df["trial"], df["mean_val_mcc"], marker="o", linewidth=1.2, label="Trial MCC")
    ax.plot(df["trial"], df["best_so_far_mcc"], linewidth=2.4, label="Best so far")
    ax.set_title("Validation MCC Across Autoresearch Trials")
    ax.set_xlabel("Trial")
    ax.set_ylabel("mean_val_mcc")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "mcc_over_trials.png", dpi=180)
    plt.close(fig)

    top = df.sort_values("mean_val_mcc", ascending=False).head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(12, 7))
    labels = [f"{int(r.trial)} | {r.run_folder[:17]}" for r in top.itertuples()]
    ax.barh(labels, top["mean_val_mcc"], color="#377eb8")
    ax.set_title("Top 15 Runs by Validation MCC")
    ax.set_xlabel("mean_val_mcc")
    ax.set_xlim(max(0, top["mean_val_mcc"].min() - 0.03), min(1, top["mean_val_mcc"].max() + 0.01))
    fig.tight_layout()
    fig.savefig(plot_dir / "top_runs_mcc.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots()
    for model_type, group in df.groupby("model_type"):
        ax.scatter(group["mean_val_auc"], group["mean_val_mcc"], label=model_type, s=45, alpha=0.8)
    ax.set_title("AUC vs MCC by Model Family")
    ax.set_xlabel("mean_val_auc")
    ax.set_ylabel("mean_val_mcc")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "auc_vs_mcc_by_model.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    order = df.groupby("model_type")["mean_val_mcc"].max().sort_values(ascending=False).index
    data = [df.loc[df["model_type"].eq(model), "mean_val_mcc"].dropna() for model in order]
    ax.boxplot(data, tick_labels=order, vert=True, patch_artist=True)
    ax.set_title("MCC Distribution by Model Family")
    ax.set_ylabel("mean_val_mcc")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(plot_dir / "mcc_by_model_family.png", dpi=180)
    plt.close(fig)

    tuned = df[df["model_type"].eq("image_topo_tabular")].copy()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, field, title in [
        (axes[0, 0], "learning_rate", "Learning Rate"),
        (axes[0, 1], "dropout", "Dropout"),
        (axes[1, 0], "lr_patience", "Plateau Patience"),
        (axes[1, 1], "seed", "Seed"),
    ]:
        grouped = tuned.groupby(field)["mean_val_mcc"]
        values = grouped.max().sort_index()
        ax.plot(values.index.astype(str), values.values, marker="o")
        ax.set_title(f"Best Full-Model MCC by {title}")
        ax.set_xlabel(field)
        ax.set_ylabel("best mean_val_mcc")
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(plot_dir / "full_model_hyperparameter_slices.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.scatter(df["mean_val_sensitivity"], df["mean_val_specificity"], c=df["mean_val_mcc"], cmap="viridis", s=55)
    ax.set_title("Sensitivity vs Specificity Colored by MCC")
    ax.set_xlabel("mean_val_sensitivity")
    ax.set_ylabel("mean_val_specificity")
    cbar = fig.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("mean_val_mcc")
    fig.tight_layout()
    fig.savefig(plot_dir / "sensitivity_vs_specificity.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="autoresearch_runs/results.tsv", type=Path)
    parser.add_argument("--out-dir", default="autoresearch_runs/analysis", type=Path)
    args = parser.parse_args()

    df = load_enriched_results(args.results)
    write_tables(df, args.out_dir)
    plot_results(df, args.out_dir)
    print(f"Wrote analysis to {args.out_dir.resolve()}")
    print(f"Rows summarized: {len(df)}")
    best = df.loc[df["mean_val_mcc"].idxmax()]
    print(
        "Best: "
        f"trial={int(best['trial'])} run={best['run_folder']} "
        f"mcc={best['mean_val_mcc']:.4f} auc={best['mean_val_auc']:.4f}"
    )


if __name__ == "__main__":
    main()
