"""Add environmental and geographic support QA to a candidate package."""

from .common import *

import argparse
from .env import raw_environment_valid_mask, sample_raw_environment_for_points


def run_candidate_environment_qa(
    package_dir: str | Path,
    worldclim_folder: str | Path,
    ghm_raster: str | Path,
    workers: int = 4,
) -> dict:
    package = Path(package_dir)
    manifest_paths = sorted((package / "manifests").glob("*.csv"))
    if not manifest_paths:
        raise FileNotFoundError(f"No candidate manifests found under {package / 'manifests'}")

    manifest_parts = []
    for path in manifest_paths:
        frame = pd.read_csv(path)
        required = {"sample_id", "presence", "lat", "lon", "arm", "background_target"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path.name} lacks required columns: {sorted(missing)}")
        manifest_parts.append(frame)
    all_manifest_rows = pd.concat(manifest_parts, ignore_index=True, sort=False)

    coordinate_consistency = (
        all_manifest_rows.groupby("sample_id")[["lat", "lon"]].nunique(dropna=False).max(axis=1)
    )
    if (coordinate_consistency > 1).any():
        raise RuntimeError("Candidate sample IDs map to inconsistent coordinates across manifests.")
    points = all_manifest_rows.drop_duplicates("sample_id")[["sample_id", "lat", "lon"]].copy()
    points = points.sort_values("sample_id", kind="mergesort").reset_index(drop=True)

    points, raster_metadata = sample_raw_environment_for_points(
        points,
        worldclim_folder=worldclim_folder,
        ghm_raster=ghm_raster,
        workers=workers,
    )
    variables = ALL_ENV_VARS
    finite = np.isfinite(points[variables].to_numpy(dtype=float))
    points["nonfinite_variable_count"] = (~finite).sum(axis=1)
    valid = raw_environment_valid_mask(points)
    nonfinite = points.loc[~valid].copy()

    enriched = all_manifest_rows.merge(points, on=["sample_id", "lat", "lon"], validate="many_to_one")
    summary_rows = []
    range_rows = []
    for (arm, target, label), group in enriched.groupby(
        ["arm", "background_target", "presence"], dropna=False
    ):
        for variable in variables:
            values = pd.to_numeric(group[variable], errors="coerce")
            summary_rows.append(
                {
                    "arm": arm,
                    "target": int(target),
                    "presence": int(label),
                    "variable": variable,
                    "count": len(values),
                    "finite_count": int(np.isfinite(values).sum()),
                    "mean": values.mean(),
                    "std": values.std(),
                    "min": values.min(),
                    "q05": values.quantile(0.05),
                    "median": values.median(),
                    "q95": values.quantile(0.95),
                    "max": values.max(),
                }
            )

    presence_values = points[
        points["sample_id"].isin(
            set(all_manifest_rows.loc[all_manifest_rows["presence"] == 1, "sample_id"])
        )
    ]
    for (arm, target), backgrounds in enriched[enriched["presence"] == 0].groupby(
        ["arm", "background_target"]
    ):
        for variable in variables:
            lower = presence_values[variable].min()
            upper = presence_values[variable].max()
            values = backgrounds[variable]
            outside = (values < lower) | (values > upper)
            range_rows.append(
                {
                    "arm": arm,
                    "target": int(target),
                    "variable": variable,
                    "presence_min": lower,
                    "presence_max": upper,
                    "background_outside_presence_range_count": int(outside.sum()),
                    "background_outside_presence_range_fraction": float(outside.mean()),
                }
            )

    geographic = (
        enriched.groupby(["arm", "background_target", "presence"], dropna=False)
        .agg(
            row_count=("sample_id", "size"),
            unique_source_tiles=("source_tile_key", "nunique"),
            unique_source_states=("source_archive_state", "nunique"),
            lat_min=("lat", "min"),
            lat_max=("lat", "max"),
            lon_min=("lon", "min"),
            lon_max=("lon", "max"),
            nearest_presence_km_min=("nearest_presence_km", "min"),
            nearest_presence_km_median=("nearest_presence_km", "median"),
            nearest_presence_km_max=("nearest_presence_km", "max"),
        )
        .reset_index()
    )

    qa_dir = package / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    points.to_csv(qa_dir / "candidate_environment_values.csv", index=False)
    nonfinite.to_csv(qa_dir / "candidate_environment_nonfinite.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(qa_dir / "environment_support_summary.csv", index=False)
    pd.DataFrame(range_rows).to_csv(qa_dir / "environment_range_overlap.csv", index=False)
    geographic.to_csv(qa_dir / "geographic_support_summary.csv", index=False)
    status = {
        "production_ready_environment": nonfinite.empty,
        "unique_samples_checked": len(points),
        "variables_checked": variables,
        "nonfinite_sample_count": len(nonfinite),
        "invalid_count_by_variable": {
            variable: int((~np.isfinite(points[variable])).sum()) for variable in variables
        },
        "raster_metadata": sorted(raster_metadata, key=lambda item: item["column"]),
    }
    with open(qa_dir / "environment_status.json", "w", encoding="utf-8") as stream:
        json.dump(status, stream, indent=2)
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", required=True)
    parser.add_argument("--worldclim-folder", required=True)
    parser.add_argument("--ghm-raster", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive.")
    return args


def main() -> None:
    args = parse_args()
    status = run_candidate_environment_qa(
        package_dir=args.package_dir,
        worldclim_folder=args.worldclim_folder,
        ghm_raster=args.ghm_raster,
        workers=args.workers,
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
