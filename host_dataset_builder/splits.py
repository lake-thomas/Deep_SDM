from .common import *

def stratified_train_val_test_split(
    df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> pd.DataFrame:
    """Stratified split by presence class so class balance is maintained."""
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

def assign_spatial_blocks(
    points_gdf: gpd.GeoDataFrame,
    block_size_m: float,
    n_folds: int,
    seed: int,
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
) -> pd.DataFrame:
    """
    For each held-out fold:
      - test = held-out fold
      - train/val = remaining folds, split stratified by presence
    """
    rounds = []

    for heldout in range(1, n_folds + 1):
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

