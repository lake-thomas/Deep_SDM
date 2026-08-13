from .common import *

def stratified_train_val_test_split(
    df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
    group_column: str | None = None,
) -> pd.DataFrame:
    """Stratified split by presence class so class balance is maintained."""
    if group_column is not None:
        return _group_stratified_train_val_test_split(
            df=df,
            train_frac=train_frac,
            val_frac=val_frac,
            test_frac=test_frac,
            seed=seed,
            group_column=group_column,
        )

    rng = np.random.default_rng(seed)
    chunks = []

    for _, sub in df.groupby("presence"):
        idx = np.array(sub.index)
        rng.shuffle(idx)

        n = len(idx)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        n_test = n - n_train - n_val

        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:n_train + n_val + n_test]

        tmp = df.loc[train_idx].copy()
        tmp["split"] = "train"
        chunks.append(tmp)

        tmp = df.loc[val_idx].copy()
        tmp["split"] = "val"
        chunks.append(tmp)

        if len(test_idx) > 0:
            tmp = df.loc[test_idx].copy()
            tmp["split"] = "test"
            chunks.append(tmp)

    out = pd.concat(chunks, ignore_index=False).sort_index().reset_index(drop=True)
    return out


def _group_stratified_train_val_test_split(
    df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
    group_column: str,
) -> pd.DataFrame:
    """Greedily balance binary-class counts while keeping source groups intact."""
    if group_column not in df.columns:
        raise ValueError(f"Uniform split grouping column is missing: {group_column}")
    if "presence" not in df.columns:
        raise ValueError("Grouped uniform splitting requires a presence column.")
    if df.empty:
        return df.assign(split=pd.Series(dtype=object))

    work = df.copy()
    group_key = work[group_column].astype("string")
    missing = group_key.isna()
    if missing.any():
        group_key.loc[missing] = [f"__missing_group_row_{idx}" for idx in work.index[missing]]
    work["_split_group_key"] = group_key

    labels = pd.to_numeric(work["presence"], errors="raise").astype(int)
    if not labels.isin([0, 1]).all():
        raise ValueError("Grouped uniform splitting requires binary 0/1 presence labels.")
    work["_split_label"] = labels

    group_counts = (
        work.groupby(["_split_group_key", "_split_label"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[0, 1], fill_value=0)
    )
    group_counts.columns = ["class_0", "class_1"]
    group_counts["total"] = group_counts.sum(axis=1)

    rng = np.random.default_rng(seed)
    group_counts["_tie_order"] = rng.random(len(group_counts))
    group_counts = group_counts.sort_values(
        ["total", "_tie_order"],
        ascending=[False, True],
        kind="mergesort",
    )

    split_names = ["train", "val", "test"]
    fractions = np.array([train_frac, val_frac, test_frac], dtype=float)
    target = np.outer(
        fractions,
        np.array([group_counts["class_0"].sum(), group_counts["class_1"].sum()], dtype=float),
    )
    assigned = np.zeros_like(target)
    split_group_counts = np.zeros(3, dtype=int)
    assignment: dict[str, str] = {}

    for group, row in group_counts.iterrows():
        group_vector = np.array([row["class_0"], row["class_1"]], dtype=float)
        candidate_costs = []
        for split_index in range(3):
            proposed = assigned.copy()
            proposed[split_index] += group_vector
            scale = np.maximum(target, 1.0)
            balance_cost = float(np.square((proposed - target) / scale).sum())
            overfill_cost = float(
                (4.0 * np.square(np.maximum(proposed - target, 0.0) / scale)).sum()
            )
            empty_split_bonus = -0.05 if split_group_counts[split_index] == 0 else 0.0
            candidate_costs.append(balance_cost + overfill_cost + empty_split_bonus)
        chosen = min(range(3), key=lambda index: (candidate_costs[index], index))
        assignment[str(group)] = split_names[chosen]
        assigned[chosen] += group_vector
        split_group_counts[chosen] += 1

    if len(group_counts) >= 3 and (split_group_counts == 0).any():
        raise RuntimeError(
            "Grouped uniform split could not populate all three partitions; "
            "review the grouping distribution."
        )

    work["split"] = work["_split_group_key"].astype(str).map(assignment)
    return work.drop(columns=["_split_group_key", "_split_label"])

def assign_spatial_blocks(
    points_gdf: gpd.GeoDataFrame,
    block_size_m: float,
    n_folds: int,
    seed: int,
    group_column: str | None = None,
) -> gpd.GeoDataFrame:
    """
    Assign spatial blocks and folds from projected coordinates.

    Uses EPSG:5070, CONUS Albers Equal Area, for U.S.-scale blocking.
    """
    gdf = points_gdf.copy().to_crs("EPSG:5070")
    minx, miny, _, _ = gdf.total_bounds

    gdf["block_x"] = ((gdf.geometry.x - minx) // block_size_m).astype(int)
    gdf["block_y"] = ((gdf.geometry.y - miny) // block_size_m).astype(int)
    gdf["block_id"] = gdf["block_x"].astype(str) + "_" + gdf["block_y"].astype(str)

    if group_column is not None:
        if group_column not in gdf.columns:
            raise ValueError(f"Spatial block grouping column is missing: {group_column}")
        grouped = gdf[gdf[group_column].notna()].copy()
        if not grouped.empty:
            # Choose the modal center-derived block per source group. Stable
            # block-id ordering makes tied boundary cases deterministic.
            group_block_counts = (
                grouped.groupby([group_column, "block_id"], dropna=False)
                .size()
                .rename("row_count")
                .reset_index()
                .sort_values(
                    [group_column, "row_count", "block_id"],
                    ascending=[True, False, True],
                    kind="mergesort",
                )
                .drop_duplicates(group_column, keep="first")
            )
            group_to_block = group_block_counts.set_index(group_column)["block_id"]
            grouped_mask = gdf[group_column].notna()
            gdf.loc[grouped_mask, "block_id"] = gdf.loc[grouped_mask, group_column].map(group_to_block)
            block_parts = gdf.loc[grouped_mask, "block_id"].str.split("_", n=1, expand=True)
            gdf.loc[grouped_mask, "block_x"] = block_parts[0].astype(int).to_numpy()
            gdf.loc[grouped_mask, "block_y"] = block_parts[1].astype(int).to_numpy()

    unique_blocks = sorted(gdf["block_id"].unique())
    rng = np.random.default_rng(seed)
    shuffled = np.array(unique_blocks, dtype=object)
    rng.shuffle(shuffled)

    block_to_fold = {block_id: (i % n_folds) + 1 for i, block_id in enumerate(shuffled)}
    gdf["fold"] = gdf["block_id"].map(block_to_fold).astype(int)

    return gdf.to_crs("EPSG:4326")

def make_spatial_cv_rounds(
    points_df: pd.DataFrame,
    n_folds: int,
    seed: int,
    validation_mode: str = "row_stratified",
) -> pd.DataFrame:
    """
    For each held-out fold:
      - test = held-out fold
      - train/val = remaining folds, split stratified by presence
    """
    if validation_mode not in {"row_stratified", "heldout_fold"}:
        raise ValueError(f"Unsupported BlockCV validation mode: {validation_mode}")
    if validation_mode == "heldout_fold" and n_folds < 3:
        raise ValueError("heldout_fold validation requires at least three folds.")

    rounds = []

    for heldout in range(1, n_folds + 1):
        if validation_mode == "heldout_fold":
            validation_fold = (heldout % n_folds) + 1
            round_df = points_df.copy()
            round_df["split"] = "train"
            round_df.loc[round_df["fold"] == validation_fold, "split"] = "val"
            round_df.loc[round_df["fold"] == heldout, "split"] = "test"
            round_df["validation_fold"] = validation_fold
        else:
            train_val = points_df[points_df["fold"] != heldout].copy()
            test = points_df[points_df["fold"] == heldout].copy()

            split_train_val = stratified_train_val_test_split(
                train_val.drop(columns=["fold"], errors="ignore"),
                train_frac=0.70,
                val_frac=0.30,
                test_frac=0.0,
                seed=seed,
            )

            test = test.copy()
            test["split"] = "test"

            split_train_val = split_train_val.merge(
                train_val[["lat", "lon", "fold", "block_id"]].drop_duplicates(),
                on=["lat", "lon"],
                how="left",
            )

            round_df = pd.concat([split_train_val, test], ignore_index=True)
        round_df["cv_round"] = heldout
        rounds.append(round_df)

    return pd.concat(rounds, ignore_index=True)

def add_normalized_lat_lon(df: pd.DataFrame) -> pd.DataFrame:
    """Add dataset-level normalized latitude and longitude columns."""
    df = df.copy()
    lat_std = df["lat"].std()
    lon_std = df["lon"].std()

    if lat_std == 0 or not np.isfinite(lat_std):
        df["lat_norm"] = 0.0
    else:
        df["lat_norm"] = (df["lat"] - df["lat"].mean()) / lat_std

    if lon_std == 0 or not np.isfinite(lon_std):
        df["lon_norm"] = 0.0
    else:
        df["lon_norm"] = (df["lon"] - df["lon"].mean()) / lon_std

    return df
