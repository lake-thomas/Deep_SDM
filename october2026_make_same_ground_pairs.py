#!/usr/bin/env python3
"""Extract same-ground NAIP chip pairs for the October 2026 Ailanthus retraining and evaluation.

Neighbouring NAIP quarter-quad tiles overlap by 430-820 m (median about 640 m) along shared edges, and
some quarter-quads appear in two state archives ("same_footprint"). A pair is one 512 m x 512 m chip
footprint placed at random inside the overlap rectangle and read from both tiles: identical ground seen in
two acquisitions. Pairs whose overlap is narrower than MIN_OVERLAP_M are skipped.

Each pair gets one environmental vector, computed for chip A's footprint exactly as the production
inference does (WorldClim and gHM sampled at the chip centre and normalized with the model's stats,
lat/lon normalized with the model's lat_lon_stats, 3DEP topo scalars from the fast tile-level cache).

Split: pairs whose tile-A centre falls in a 1-degree block with (floor(lon) + floor(lat)) % 4 == 0 are
held out for evaluation ("eval"); all other blocks are "train". Tiles from the tile table built in
experimental_model_runs_october2026/ailanthus_tile_artifact/01_diagnostics/tables/.

Outputs (in --out-dir): images_a.u8, images_b.u8 (N x 4 x 256 x 256 memmaps), env.f32 (N x F), meta.csv,
manifest.json (feature order, counts, sources, checksums of inputs).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("PROJ_NETWORK", "OFF")
os.environ.setdefault("GTIFF_SRS_SOURCE", "EPSG")

import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from rasterio.windows import Window  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import inference_multimodal_production as inf  # noqa: E402

PR = Path("/mnt/rsstu_dsfas_gsv_naip/Host_NAIP_SDM")
DIAG = PR / "Outputs_Host_NAIP_SDMs/experimental_model_runs_october2026/ailanthus_tile_artifact/01_diagnostics/tables"
AUDIT = PR / "NAIP_Imagery_Tile_Indices/NAIP_US_Local_Archive_Tile_Index_ByTileKey.audit.csv"
TOPO = {
    "dem": PR / "Env_Data/Topography_3DEP/mosaic/dem_3dep_13_epsg5070_30m.tif",
    "slope": PR / "Env_Data/Topography_3DEP/mosaic/slope_degrees_3dep_13_epsg5070_30m.tif",
    "northness": PR / "Env_Data/Topography_3DEP/mosaic/northness_3dep_13_epsg5070_30m.tif",
    "eastness": PR / "Env_Data/Topography_3DEP/mosaic/eastness_3dep_13_epsg5070_30m.tif",
}
CHIP = 256
MIN_OVERLAP_M = 520.0  # overlap must hold a whole 512 m chip (with margin)
_TOPO_SOURCES = None
_TOPO_STATS = None


def _init_worker(topo_stats):
    global _TOPO_SOURCES, _TOPO_STATS
    args = argparse.Namespace(dem_raster=str(TOPO["dem"]), slope_raster=str(TOPO["slope"]),
                              northness_raster=str(TOPO["northness"]), eastness_raster=str(TOPO["eastness"]),
                              aspect_raster=None)
    _TOPO_SOURCES = inf.open_topo_sources(args)
    _TOPO_STATS = topo_stats


def _windows(spec):
    """Pixel windows (col_off, row_off) of the pair's chip in tile A and tile B."""
    return (spec["a_col"], spec["a_row"]), (spec["b_col"], spec["b_row"])


def extract(spec):
    (ac, ar), (bc, br) = _windows(spec)
    out = {"id": spec["id"]}
    try:
        with rasterio.open(spec["naip_a"]) as sa:
            a = sa.read(window=Window(ac, ar, CHIP, CHIP))
            cache = inf.build_tile_topo_scalar_cache(
                naip_src=sa, rows=[ar], cols=[ac], topo_sources=_TOPO_SOURCES, topo_norm_stats=_TOPO_STATS,
                chip_size=CHIP, topo_chip_size=64, topo_min_valid_frac=0.90, normalize_topography=True)
            cx, cy = sa.transform * (ac + CHIP / 2.0, ar + CHIP / 2.0)
            lon, lat = inf.transform_coords(sa.crs, "EPSG:4326", [cx], [cy])
        with rasterio.open(spec["naip_b"]) as sb:
            b = sb.read(window=Window(bc, br, CHIP, CHIP))
    except Exception as exc:  # noqa: BLE001
        out["error"] = repr(exc)[:200]
        return out, None, None
    if a.shape != (4, CHIP, CHIP) or b.shape != (4, CHIP, CHIP):
        out["error"] = f"shape {a.shape} {b.shape}"
        return out, None, None
    va = (a.sum(axis=0) > 0).mean(); vb = (b.sum(axis=0) > 0).mean()
    topo = cache.get((ar, ac))
    if va < 0.99 or vb < 0.99 or topo is None:
        out["error"] = f"valid {va:.3f} {vb:.3f} topo {topo is not None}"
        return out, None, None
    out.update({"lon": float(lon[0]), "lat": float(lat[0]), "topo": topo})
    return out, a.astype(np.uint8), b.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="model config.input.json (feature order and stats)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--train-pairs", type=int, default=24000, help="tile pairs sampled for training")
    ap.add_argument("--eval-pairs", type=int, default=3000, help="tile pairs sampled for evaluation")
    ap.add_argument("--positions", type=int, default=2, help="chip positions per tile pair")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20261006)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(args.config).read_text())
    req = inf.model_requires(inf.normalize_model_type(config.get("model_type", "full")))
    env_vars = inf.resolve_env_features_for_inference(config, req, None, bool(config.get("include_topo_scalars", True)))

    tiles = {r["stem"]: r for r in csv.DictReader(open(DIAG / "tiles.csv"))}
    cen = {r["stem"]: (float(r["lon"]), float(r["lat"])) for r in csv.DictReader(open(DIAG / "tile_centres.csv"))}
    naip = {r["filename"][:-4]: r["local_path"] for r in csv.DictReader(open(AUDIT))}
    pairs = list(csv.DictReader(open(DIAG / "pairs.csv")))
    rng = random.Random(args.seed)
    def split_of(stem):
        x, y = cen[stem]
        return "eval" if (math.floor(x) + math.floor(y)) % 4 == 0 else "train"
    by_split = {"train": [], "eval": []}
    for p in pairs:
        if p["stem_a"] in naip and p["stem_b"] in naip:
            by_split[split_of(p["stem_a"])].append(p)
    for k in by_split:
        rng.shuffle(by_split[k])
    chosen = [("train", p) for p in by_split["train"][: args.train_pairs]] + [("eval", p) for p in by_split["eval"][: args.eval_pairs]]

    specs = []
    for split, p in chosen:
        A, B = tiles[p["stem_a"]], tiles[p["stem_b"]]
        fa = {k: float(A[k]) for k in ("left", "bottom", "right", "top")}
        fb = {k: float(B[k]) for k in ("left", "bottom", "right", "top")}
        wa, ha, wb, hb = int(A["width"]), int(A["height"]), int(B["width"]), int(B["height"])
        for _ in range(args.positions):
            s = {"split": split, "kind": p["kind"], "stem_a": p["stem_a"], "stem_b": p["stem_b"],
                 "naip_a": naip[p["stem_a"]], "naip_b": naip[p["stem_b"]],
                 "archive_a": p["archive_a"], "archive_b": p["archive_b"], "date_a": p["date_a"], "date_b": p["date_b"]}
            # Identical ground: one 512 m x 512 m chip footprint inside the overlap rectangle of both tiles.
            ol, ob, orr, ot = (float(p[k]) for k in ("ov_left", "ov_bottom", "ov_right", "ov_top"))
            if orr - ol < MIN_OVERLAP_M or ot - ob < MIN_OVERLAP_M:
                continue
            x_left = rng.uniform(ol, orr - 2 * CHIP)
            y_top = rng.uniform(ob + 2 * CHIP, ot)
            s["a_col"], s["a_row"] = int(round((x_left - fa["left"]) / 2.0)), int(round((fa["top"] - y_top) / 2.0))
            s["b_col"], s["b_row"] = int(round((x_left - fb["left"]) / 2.0)), int(round((fb["top"] - y_top) / 2.0))
            if min(s["a_col"], s["a_row"], s["b_col"], s["b_row"]) < 0 or s["a_col"] + CHIP > wa or s["a_row"] + CHIP > ha \
                    or s["b_col"] + CHIP > wb or s["b_row"] + CHIP > hb:
                continue
            s["id"] = len(specs)
            specs.append(s)
    print("pair chips planned:", len(specs), flush=True)

    # Same topo normalisation file the production inference was given (--topo-normalization-stats).
    topo_stats = inf.coerce_stats_dict(inf.load_json(str(PR / "Env_Data/Topography_3DEP/mosaic/topo_norm_stats_3dep_30m.json")),
                                       inf.DEFAULT_TOPO_NORM_STATS_3DEP_30M)
    n = len(specs)
    img_a = np.lib.format.open_memmap(out / "images_a.npy", mode="w+", dtype=np.uint8, shape=(n, 4, CHIP, CHIP))
    img_b = np.lib.format.open_memmap(out / "images_b.npy", mode="w+", dtype=np.uint8, shape=(n, 4, CHIP, CHIP))
    results = [None] * n
    with ProcessPoolExecutor(args.workers, initializer=_init_worker, initargs=(topo_stats,)) as pool:
        for k, (res, a, b) in enumerate(pool.map(extract, specs, chunksize=16), 1):
            results[res["id"]] = res
            if a is not None:
                img_a[res["id"]], img_b[res["id"]] = a, b
            if k % 2000 == 0:
                print("extracted", k, flush=True)
    img_a.flush(); img_b.flush()

    # Environmental vectors (WorldClim/gHM/lat-lon exactly as inference; topo scalars from the workers).
    wc_stats = inf.coerce_stats_dict(config.get("worldclim_stats") or inf.DEFAULT_WORLDCLIM_STATS, inf.DEFAULT_WORLDCLIM_STATS)
    wc_rasters, ghm = inf.preload_and_normalize_env_rasters(str(PR / "Env_Data/Worldclim"),
                                                            str(PR / "Env_Data/Global_Human_Modification/gHM_WGS84.tif"),
                                                            wc_stats, bounds_wgs84=(-125.0, 24.0, -66.0, 50.0))
    lat_lon_stats = inf.coerce_stats_dict(config.get("lat_lon_stats") or inf.DEFAULT_LAT_LON_STATS, inf.DEFAULT_LAT_LON_STATS)
    ok = [r for r in results if r and "error" not in r]
    lons = np.array([r["lon"] for r in ok]); lats = np.array([r["lat"] for r in ok])
    sampled = {var: inf.sample_preloaded_raster_points(lons, lats, wc_rasters[var]) for var in inf.WC_VARS}
    sampled["ghm"] = inf.sample_preloaded_raster_points(lons, lats, ghm)
    env = np.full((n, len(env_vars)), np.nan, dtype=np.float32)
    keep = np.zeros(n, dtype=bool)
    for i, r in enumerate(ok):
        fmap = {var: float(sampled[var][i]) for var in sampled}
        fmap["lat_norm"] = inf.normalize_coord(r["lat"], "lat", lat_lon_stats)
        fmap["lon_norm"] = inf.normalize_coord(r["lon"], "lon", lat_lon_stats)
        fmap.update(r["topo"])
        vec = inf.vector_from_feature_map(fmap, env_vars)
        if vec is not None:
            env[r["id"]] = vec; keep[r["id"]] = True
    np.save(out / "env.npy", env)
    np.save(out / "keep.npy", keep)
    with open(out / "meta.csv", "w", newline="") as f:
        cols = ["id", "split", "kind", "stem_a", "stem_b", "archive_a", "archive_b", "date_a", "date_b",
                "a_col", "a_row", "b_col", "b_row", "lon", "lat", "keep", "error"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
        for s in specs:
            r = results[s["id"]] or {}
            w.writerow({**s, "lon": r.get("lon", ""), "lat": r.get("lat", ""), "keep": int(keep[s["id"]]), "error": r.get("error", "")})
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    manifest = {"env_vars": env_vars, "n": n, "kept": int(keep.sum()),
                "kept_by_split": {k: int(sum(1 for s in specs if s["split"] == k and keep[s["id"]])) for k in ("train", "eval")},
                "config": args.config, "config_sha256": sha(args.config), "seed": args.seed,
                "pairs_csv_sha256": sha(DIAG / "pairs.csv"), "tiles_csv_sha256": sha(DIAG / "tiles.csv"),
                "split_rule": "eval if (floor(lon)+floor(lat)) % 4 == 0 of tile-A centre, else train"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: manifest[k] for k in ("n", "kept", "kept_by_split")}), flush=True)


if __name__ == "__main__":
    main()
