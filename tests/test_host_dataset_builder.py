import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, box

from host_dataset_builder.background import (
    _allocate_domain_counts,
    sample_background_points_us_naip,
    spatially_thin_points,
)
from host_dataset_builder.candidate_manifests import (
    _allocate_with_capacity,
    assign_balanced_order,
    assign_weighted_random_order,
    spatially_filter_order,
)
from host_dataset_builder.env import (
    extract_ghm_for_point,
    raw_environment_valid_mask,
    sample_raw_environment_for_points,
)
from host_dataset_builder.io_utils import add_canonical_naip_source_metadata
from host_dataset_builder.materialize_candidates import (
    reconcile_actual_source_groups,
    select_domain_replenished_backgrounds,
)
from host_dataset_builder.naip import (
    attach_naip_filename_or_url,
    extract_naip_chip_for_point,
)
from host_dataset_builder.qa import (
    audit_dataset_manifest,
    manifest_has_hard_failures,
    write_manifest_qa_package,
)
from host_dataset_builder.splits import (
    assign_spatial_blocks,
    make_spatial_cv_rounds,
    stratified_train_val_test_split,
)


def _write_four_band_raster(
    path: Path,
    value: int,
    left: float,
    top: float,
    width: int = 4,
    height: int = 8,
    pixel_size: float = 1.0,
):
    data = np.full((4, height, width), value, dtype=np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=4,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_origin(left, top, pixel_size, pixel_size),
        nodata=0,
    ) as dst:
        dst.write(data)


def _write_one_band_raster(path: Path, data: np.ndarray, nodata: float = -9999.0):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(0.0, 2.0, 1.0, 1.0),
        nodata=nodata,
    ) as dst:
        dst.write(data.astype(np.float32), 1)


def _tile_row(path: Path, tile_key: str, state: str, date: str, geometry):
    return {
        "tile_key": tile_key,
        "filename": f"{tile_key}_{date}.tif",
        "local_filename": f"{tile_key}_{date}.tif",
        "local_path": str(path),
        "local_acq_date": date,
        "local_year": int(date[:4]),
        "archive_state": state,
        "archive_folder_year": date[:4],
        "geometry": geometry,
    }


class CanonicalMetadataTests(unittest.TestCase):
    def test_canonical_metadata_uses_exact_local_acquisition(self):
        tiles = gpd.GeoDataFrame(
            [
                _tile_row(
                    Path("/archive/CA/2022/m_1_20220703.tif"),
                    "m_1",
                    "ca",
                    "20220703",
                    box(0, 0, 1, 1),
                )
            ],
            crs="EPSG:4326",
        )
        result = add_canonical_naip_source_metadata(tiles).iloc[0]
        self.assertEqual(result["source_archive_state"], "CA")
        self.assertEqual(result["source_acq_date"], "20220703")
        self.assertEqual(int(result["source_acq_year"]), 2022)
        self.assertEqual(int(result["source_acq_month"]), 7)
        self.assertEqual(result["source_domain_id"], "CA|2022|20220703")

    def test_domain_allocation_has_exact_total_and_minimum(self):
        weights = np.array([1.0, 2.0, 7.0])
        allocation = _allocate_domain_counts(
            domain_weights=pd.Series(weights, index=["a", "b", "c"]),
            total=20,
            minimum_per_domain=3,
        )
        self.assertEqual(int(allocation.sum()), 20)
        self.assertTrue((allocation >= 3).all())
        self.assertGreater(allocation["c"], allocation["a"])


class EnvironmentSupportTests(unittest.TestCase):
    def test_ghm_nodata_is_not_clipped_to_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ghm.tif"
            _write_one_band_raster(path, np.array([[-9999.0, 0.25], [0.5, 0.75]]))
            with rasterio.open(path) as dataset:
                self.assertTrue(np.isnan(extract_ghm_for_point(0.5, 1.5, dataset)))
                self.assertAlmostEqual(extract_ghm_for_point(1.5, 1.5, dataset), 0.25)

    def test_raw_environment_gate_rejects_any_missing_covariate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            worldclim = root / "worldclim"
            worldclim.mkdir()
            valid_data = np.array([[1.0, 2.0], [3.0, 4.0]])
            for index in range(1, 20):
                data = valid_data.copy()
                if index == 7:
                    data[0, 0] = -9999.0
                _write_one_band_raster(worldclim / f"wc2.1_30s_bio_{index}.tif", data)
            ghm = root / "ghm.tif"
            _write_one_band_raster(ghm, np.array([[0.1, 0.2], [0.3, 0.4]]))
            points = pd.DataFrame(
                {"sample_id": ["missing", "valid"], "lat": [1.5, 1.5], "lon": [0.5, 1.5]}
            )
            values, metadata = sample_raw_environment_for_points(
                points,
                worldclim_folder=worldclim,
                ghm_raster=ghm,
                workers=2,
            )
            self.assertEqual(len(metadata), 20)
            self.assertEqual(raw_environment_valid_mask(values).tolist(), [False, True])


class SourceAssignmentTests(unittest.TestCase):
    def test_presence_overlap_uses_deterministic_latest_exact_acquisition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            older = root / "m_a_20210715.tif"
            newer = root / "m_a_20220715.tif"
            older.touch()
            newer.touch()
            tiles = gpd.GeoDataFrame(
                [
                    _tile_row(older, "m_a_old", "NY", "20210715", box(0, 0, 2, 2)),
                    _tile_row(newer, "m_a_new", "NY", "20220715", box(0, 0, 2, 2)),
                ],
                crs="EPSG:4326",
            )
            points = gpd.GeoDataFrame(geometry=[Point(1, 1)], crs="EPSG:4326")
            assigned = attach_naip_filename_or_url(points, tiles).iloc[0]
            self.assertEqual(assigned["source_acq_date"], "20220715")
            self.assertEqual(assigned["source_tile_key"], "m_a_new")
            self.assertEqual(
                assigned["source_selection_method"],
                "spatial_join_latest_exact_acquisition",
            )

    def test_background_assignment_is_not_overwritten_by_spatial_rejoin(self):
        selected = gpd.GeoDataFrame(
            [{
                "geometry": Point(1, 1),
                "source_tile_key": "selected_tile",
                "source_filename": "selected_20210715.tif",
                "source_local_path": "/selected/selected_20210715.tif",
                "source_archive_state": "NY",
                "source_archive_year": 2021,
                "source_acq_date": "20210715",
                "source_acq_year": 2021,
                "source_acq_month": 7,
                "source_domain_id": "NY|2021|20210715",
                "source_selection_method": "sampled_tile_footprint",
            }],
            crs="EPSG:4326",
        )
        replacement_tiles = gpd.GeoDataFrame(
            [
                _tile_row(
                    Path("/newer/newer_20220715.tif"),
                    "newer_tile",
                    "NY",
                    "20220715",
                    box(0, 0, 2, 2),
                )
            ],
            crs="EPSG:4326",
        )
        assigned = attach_naip_filename_or_url(selected, replacement_tiles).iloc[0]
        self.assertEqual(assigned["source_tile_key"], "selected_tile")
        self.assertEqual(assigned["source_acq_date"], "20210715")
        self.assertEqual(assigned["source_selection_method"], "sampled_tile_footprint")


class BackgroundSamplingTests(unittest.TestCase):
    def test_spatial_thinning_uses_exact_radius_not_buffer_approximation(self):
        distance = 799.6
        angle = np.deg2rad(22.5)
        points = gpd.GeoDataFrame(
            {"presence": [1, 1]},
            geometry=[
                Point(0.0, 0.0),
                Point(distance * np.cos(angle), distance * np.sin(angle)),
            ],
            crs="EPSG:5070",
        )
        thinned = spatially_thin_points(
            points,
            min_distance_m=800.0,
            seed=42,
            projected_crs="EPSG:5070",
        )
        self.assertEqual(len(thinned), 1)

    def test_state_restricted_domain_sampling_preserves_selected_tile(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            wc_path = tmp_path / "wc.tif"
            with rasterio.open(
                wc_path,
                "w",
                driver="GTiff",
                height=40,
                width=40,
                count=1,
                dtype="float32",
                crs="EPSG:4326",
                transform=from_origin(-10, 30, 1.0, 1.0),
            ) as dst:
                dst.write(np.ones((1, 40, 40), dtype=np.float32))

            dummy = tmp_path / "dummy.tif"
            dummy.touch()
            tile_rows = [
                _tile_row(dummy, "ca_1", "CA", "20220415", box(0, 0, 2, 2)),
                _tile_row(dummy, "ca_2", "CA", "20220715", box(2, 0, 4, 2)),
                _tile_row(dummy, "or_1", "OR", "20200615", box(0, 2, 2, 4)),
                _tile_row(dummy, "or_2", "OR", "20200715", box(2, 2, 4, 4)),
                _tile_row(dummy, "nv_1", "NV", "20190615", box(4, 0, 6, 2)),
            ]
            tiles = gpd.GeoDataFrame(tile_rows, crs="EPSG:4326")
            presence = gpd.GeoDataFrame(geometry=[Point(20, 20)], crs="EPSG:4326")

            sampled = sample_background_points_us_naip(
                presence_gdf=presence,
                downloaded_tiles_gdf=tiles,
                wc_reference_raster_fp=str(wc_path),
                background_multiplier=3,
                seed=42,
                inner_buffer_km=0,
                target_count=12,
                allowed_states=["CA", "OR"],
                sampling_design="domain_balanced",
                domain_columns=["source_archive_state"],
                domain_min_count=3,
            )

            self.assertEqual(len(sampled), 12)
            self.assertEqual(set(sampled["source_archive_state"]), {"CA", "OR"})
            self.assertGreaterEqual(int(sampled["source_archive_state"].value_counts().min()), 3)
            source_geometries = {row["tile_key"]: row["geometry"] for row in tile_rows}
            for _, row in sampled.iterrows():
                self.assertTrue(source_geometries[row["source_tile_key"]].contains(row.geometry))
                self.assertEqual(row["source_selection_method"], "sampled_tile_footprint")


class CandidateManifestTests(unittest.TestCase):
    @staticmethod
    def _pool() -> gpd.GeoDataFrame:
        rows = []
        index = 0
        for domain, y in [("A", 0.0), ("B", 10.0), ("C", 20.0)]:
            for offset in range(20):
                rows.append(
                    {
                        "candidate_id": f"candidate_{index:03d}",
                        "background_sampling_domain": domain,
                        "geometry": Point(offset * 0.01, y + offset * 0.01),
                    }
                )
                index += 1
        return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")

    def test_capacity_allocation_is_exact(self):
        weights = pd.Series({"A": 1.0, "B": 2.0, "C": 7.0})
        capacities = pd.Series({"A": 4, "B": 20, "C": 20})
        allocation = _allocate_with_capacity(weights, total=30, minimum_per_domain=5, capacities=capacities)
        self.assertEqual(int(allocation.sum()), 30)
        self.assertLessEqual(int(allocation["A"]), 4)
        self.assertTrue((allocation <= capacities).all())

    def test_balanced_order_has_nested_minimum_depth_prefix(self):
        pool = self._pool()
        areas = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0})
        ordered = assign_balanced_order(
            pool,
            domain_areas=areas,
            target=30,
            minimum_per_domain=3,
            seed=42,
        )
        prefix = ordered.head(30)
        self.assertEqual(len(prefix), 30)
        self.assertTrue((prefix["background_sampling_domain"].value_counts() >= 3).all())
        self.assertEqual(ordered["candidate_id"].nunique(), len(pool))

    def test_weighted_random_order_is_complete_and_deterministic(self):
        pool = self._pool()
        areas = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0})
        first = assign_weighted_random_order(pool, areas, seed=42)
        second = assign_weighted_random_order(pool, areas, seed=42)
        self.assertEqual(first["candidate_id"].tolist(), second["candidate_id"].tolist())
        self.assertEqual(first["candidate_id"].nunique(), len(pool))

    def test_spatial_filter_preserves_presence_and_nested_rank(self):
        candidates = gpd.GeoDataFrame(
            {
                "candidate_id": ["near", "one", "too_close", "two"],
            },
            geometry=[Point(0.001, 0), Point(0.02, 0), Point(0.0201, 0), Point(0.04, 0)],
            crs="EPSG:4326",
        )
        presences = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs="EPSG:4326")
        selected = spatially_filter_order(candidates, presences, target=2, min_distance_m=800)
        self.assertEqual(selected["candidate_id"].tolist(), ["one", "two"])
        self.assertEqual(selected["nested_rank"].tolist(), [1, 2])

    def test_materialization_replenishment_preserves_domains_and_nesting(self):
        ranked = pd.DataFrame(
            {
                "sample_id": [f"background_{index}" for index in range(1, 11)],
                "source_domain_id": ["A", "B", "A", "B", "C", "A", "B", "C", "A", "B"],
                "nested_rank": list(range(1, 11)),
            }
        )
        successful = ranked[~ranked["sample_id"].isin({"background_1", "background_4"})]
        selections, diagnostics = select_domain_replenished_backgrounds(
            ranked,
            successful,
            targets=[4, 7],
        )

        self.assertEqual(len(selections[4]), 4)
        self.assertEqual(len(selections[7]), 7)
        self.assertTrue(set(selections[4]["sample_id"]) < set(selections[7]["sample_id"]))
        self.assertEqual(
            selections[4]["source_domain_id"].value_counts().to_dict(),
            {"A": 2, "B": 2},
        )
        self.assertEqual(diagnostics[4]["intended_vs_actual_domain_total_variation"], 0.0)
        self.assertEqual(diagnostics[7]["remaining_domain_deficit_count"], 0)

    def test_materialization_replenishment_reports_irreducible_domain_loss(self):
        ranked = pd.DataFrame(
            {
                "sample_id": [f"background_{index}" for index in range(1, 7)],
                "source_domain_id": ["rare", "rare", "common", "common", "common", "common"],
                "nested_rank": list(range(1, 7)),
            }
        )
        successful = ranked[ranked["sample_id"] != "background_2"]
        selections, diagnostics = select_domain_replenished_backgrounds(
            ranked,
            successful,
            targets=[4],
        )

        self.assertEqual(len(selections[4]), 4)
        self.assertEqual(diagnostics[4]["domain_capacity_shortfall_count"], 1)
        self.assertEqual(diagnostics[4]["remaining_domain_deficit_count"], 1)
        self.assertAlmostEqual(
            diagnostics[4]["intended_vs_actual_domain_total_variation"],
            0.25,
        )


class StrictChipSourceTests(unittest.TestCase):
    def _run_chip_case(self, second_date: str, second_state: str = "CA"):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        first = root / "m_a_20220715.tif"
        second = root / f"m_b_{second_date}.tif"
        _write_four_band_raster(first, 10, left=0, top=8)
        _write_four_band_raster(second, 20, left=4, top=8)
        tiles = gpd.GeoDataFrame(
            [
                _tile_row(first, "m_a", "CA", "20220715", box(0, 0, 4, 8)),
                _tile_row(second, "m_b", second_state, second_date, box(4, 0, 8, 8)),
            ],
            crs="EPSG:4326",
        )
        canonical = add_canonical_naip_source_metadata(tiles)
        source_record = canonical.iloc[0]
        output = root / "chip.tif"
        return temporary, tiles, source_record, output

    def test_strict_source_lock_allows_same_domain_constituents(self):
        temporary, tiles, source_record, output = self._run_chip_case("20220715")
        try:
            provenance = extract_naip_chip_for_point(
                lon=3.5,
                lat=4.0,
                out_fp=str(output),
                chip_size=4,
                tileindex_gdf=tiles,
                naip_file_index={},
                source_record=source_record,
                source_lock_mode="strict",
                min_valid_frac=1.0,
            )
            self.assertIsNotNone(provenance)
            self.assertEqual(provenance["actual_source_count"], 2)
            self.assertEqual(provenance["actual_source_tile_keys"], "m_a;m_b")
            self.assertTrue(provenance["intended_source_used"])
            self.assertTrue(provenance["source_provenance_match"])
            self.assertFalse(provenance["actual_multi_state"])
            self.assertFalse(provenance["actual_multi_acquisition"])
            self.assertAlmostEqual(provenance["naip_valid_frac"], 1.0)
            with rasterio.open(output) as src:
                values = src.read()
                self.assertEqual(set(np.unique(values)), {10, 20})
                self.assertEqual(src.tags()["naip_source_lock_mode"], "strict")
        finally:
            temporary.cleanup()

    def test_strict_source_lock_ignores_irrelevant_fractional_grid_neighbor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intended = root / "m_a_20220715.tif"
            neighbor = root / "m_b_20220715.tif"
            _write_four_band_raster(intended, 10, left=0, top=8, width=8, height=8)
            _write_four_band_raster(neighbor, 20, left=4.3, top=8, width=4, height=8)
            tiles = gpd.GeoDataFrame(
                [
                    _tile_row(intended, "m_a", "CA", "20220715", box(0, 0, 8, 8)),
                    # Deliberately broad index geometry: it intersects the chip
                    # although the intended source already supplies every pixel.
                    _tile_row(neighbor, "m_b", "CA", "20220715", box(3, 0, 8.3, 8)),
                ],
                crs="EPSG:4326",
            )
            canonical = add_canonical_naip_source_metadata(tiles)
            output = root / "chip.tif"
            provenance = extract_naip_chip_for_point(
                lon=2.5,
                lat=4.0,
                out_fp=str(output),
                chip_size=4,
                tileindex_gdf=tiles,
                naip_file_index={},
                source_record=canonical.iloc[0],
                source_lock_mode="strict",
                min_valid_frac=1.0,
            )
            self.assertIsNotNone(provenance)
            self.assertEqual(provenance["actual_source_count"], 1)
            self.assertEqual(provenance["actual_source_tile_keys"], "m_a")
            self.assertFalse(provenance["actual_source_resampled"])
            with rasterio.open(output) as dataset:
                self.assertEqual(set(np.unique(dataset.read())), {10})

    def test_strict_source_lock_resamples_same_domain_neighbor_to_intended_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intended = root / "m_a_20220715.tif"
            neighbor = root / "m_b_20220715.tif"
            _write_four_band_raster(intended, 10, left=0, top=8)
            _write_four_band_raster(neighbor, 20, left=3.6, top=8)
            tiles = gpd.GeoDataFrame(
                [
                    _tile_row(intended, "m_a", "CA", "20220715", box(0, 0, 4, 8)),
                    _tile_row(neighbor, "m_b", "CA", "20220715", box(3.6, 0, 7.6, 8)),
                ],
                crs="EPSG:4326",
            )
            canonical = add_canonical_naip_source_metadata(tiles)
            output = root / "chip.tif"
            provenance = extract_naip_chip_for_point(
                lon=3.5,
                lat=4.0,
                out_fp=str(output),
                chip_size=4,
                tileindex_gdf=tiles,
                naip_file_index={},
                source_record=canonical.iloc[0],
                source_lock_mode="strict",
                min_valid_frac=1.0,
            )
            self.assertIsNotNone(provenance)
            self.assertEqual(provenance["actual_source_count"], 2)
            self.assertTrue(provenance["actual_source_resampled"])
            self.assertEqual(
                provenance["actual_source_resampling_method"],
                "nearest_to_intended_source_grid",
            )

    def test_strict_source_lock_rejects_cross_acquisition_fill(self):
        temporary, tiles, source_record, output = self._run_chip_case("20210715")
        try:
            provenance = extract_naip_chip_for_point(
                lon=3.5,
                lat=4.0,
                out_fp=str(output),
                chip_size=4,
                tileindex_gdf=tiles,
                naip_file_index={},
                source_record=source_record,
                source_lock_mode="strict",
                min_valid_frac=1.0,
            )
            self.assertIsNone(provenance)
            self.assertFalse(output.exists())
        finally:
            temporary.cleanup()

    def test_strict_source_lock_rejects_cross_state_fill(self):
        temporary, tiles, source_record, output = self._run_chip_case(
            "20220715",
            second_state="OR",
        )
        try:
            provenance = extract_naip_chip_for_point(
                lon=3.5,
                lat=4.0,
                out_fp=str(output),
                chip_size=4,
                tileindex_gdf=tiles,
                naip_file_index={},
                source_record=source_record,
                source_lock_mode="strict",
                min_valid_frac=1.0,
            )
            self.assertIsNone(provenance)
            self.assertFalse(output.exists())
        finally:
            temporary.cleanup()

    def test_strict_source_lock_rejects_source_absent_from_tile_index(self):
        temporary, tiles, source_record, output = self._run_chip_case("20220715")
        try:
            unindexed = Path(temporary.name) / "m_unindexed_20220715.tif"
            _write_four_band_raster(unindexed, 30, left=0, top=8)
            source_record = source_record.copy()
            source_record["source_local_path"] = str(unindexed)
            source_record["source_tile_key"] = "m_unindexed"
            provenance = extract_naip_chip_for_point(
                lon=3.5,
                lat=4.0,
                out_fp=str(output),
                chip_size=4,
                tileindex_gdf=tiles,
                naip_file_index={str(unindexed.name).lower(): str(unindexed)},
                source_record=source_record,
                source_lock_mode="strict",
                min_valid_frac=1.0,
            )
            self.assertIsNone(provenance)
            self.assertFalse(output.exists())
        finally:
            temporary.cleanup()

    def test_strict_source_lock_rejects_resolution_mismatch(self):
        temporary, tiles, source_record, output = self._run_chip_case("20220715")
        try:
            second_path = Path(temporary.name) / "m_b_20220715.tif"
            _write_four_band_raster(
                second_path,
                20,
                left=4,
                top=8,
                width=8,
                height=16,
                pixel_size=0.5,
            )
            provenance = extract_naip_chip_for_point(
                lon=3.5,
                lat=4.0,
                out_fp=str(output),
                chip_size=4,
                tileindex_gdf=tiles,
                naip_file_index={},
                source_record=source_record,
                source_lock_mode="strict",
                min_valid_frac=1.0,
            )
            self.assertIsNone(provenance)
            self.assertFalse(output.exists())
        finally:
            temporary.cleanup()


class ManifestQATests(unittest.TestCase):
    @staticmethod
    def _valid_manifest(root: Path) -> pd.DataFrame:
        chip_a = root / "chip_a.tif"
        chip_b = root / "chip_b.tif"
        chip_a.touch()
        chip_b.touch()
        return pd.DataFrame(
            [
                {
                    "sample_id": "a",
                    "presence": 1,
                    "lat": 35.0,
                    "lon": -78.0,
                    "split": "train",
                    "chip_path": chip_a.name,
                    "source_tile_key": "tile_a",
                    "source_archive_state": "NC",
                    "source_archive_year": 2022,
                    "source_acq_date": "20220615",
                    "source_acq_year": 2022,
                    "source_acq_month": 6,
                    "actual_source_lock_mode": "strict",
                    "actual_source_archive_state": "NC",
                    "actual_source_acq_date": "20220615",
                    "actual_source_tile_keys": "tile_a",
                    "source_provenance_match": True,
                    "intended_source_used": True,
                    "actual_multi_state": False,
                    "actual_multi_acquisition": False,
                    "ghm": 0.2,
                    "lat_norm": -1.0,
                    "lon_norm": -1.0,
                },
                {
                    "sample_id": "b",
                    "presence": 0,
                    "lat": 36.0,
                    "lon": -79.0,
                    "split": "test",
                    "chip_path": chip_b.name,
                    "source_tile_key": "tile_b",
                    "source_archive_state": "NC",
                    "source_archive_year": 2022,
                    "source_acq_date": "20220715",
                    "source_acq_year": 2022,
                    "source_acq_month": 7,
                    "actual_source_lock_mode": "strict",
                    "actual_source_archive_state": "NC",
                    "actual_source_acq_date": "20220715",
                    "actual_source_tile_keys": "tile_b",
                    "source_provenance_match": True,
                    "intended_source_used": True,
                    "actual_multi_state": False,
                    "actual_multi_acquisition": False,
                    "ghm": 0.3,
                    "lat_norm": 1.0,
                    "lon_norm": 1.0,
                },
            ]
        )

    def test_valid_manifest_passes_hard_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            qa = audit_dataset_manifest(
                self._valid_manifest(root),
                dataset_root=root,
                require_all_split_classes=False,
            )
            self.assertFalse(manifest_has_hard_failures(qa))
            self.assertEqual(int(qa["class_split_counts"]["row_count"].sum()), 2)
            self.assertEqual(int(qa["domain_counts"]["row_count"].sum()), 2)
            status_path = write_manifest_qa_package(qa, root / "qa")
            self.assertTrue(status_path.is_file())
            self.assertTrue((root / "qa" / "background_domain_depth_summary.csv").is_file())

    def test_manifest_path_cache_reuses_verified_materialized_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._valid_manifest(root)
            cache = {}
            first = audit_dataset_manifest(
                manifest,
                dataset_root=root,
                require_all_split_classes=False,
                path_existence_cache=cache,
            )
            self.assertFalse(manifest_has_hard_failures(first))
            self.assertEqual(
                cache,
                {
                    str(root / "chip_a.tif"): True,
                    str(root / "chip_b.tif"): True,
                },
            )
            second = audit_dataset_manifest(
                manifest,
                dataset_root=root,
                require_all_split_classes=False,
                path_existence_cache=cache,
            )
            self.assertFalse(manifest_has_hard_failures(second))
            self.assertEqual(len(cache), 2)

    def test_manifest_gate_detects_source_tile_leakage_and_provenance_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._valid_manifest(root)
            manifest.loc[1, "source_tile_key"] = "tile_a"
            manifest.loc[1, "actual_source_acq_date"] = "20210615"
            manifest.loc[1, "actual_source_tile_keys"] = "tile_a;tile_b"
            qa = audit_dataset_manifest(
                manifest,
                dataset_root=root,
                require_all_split_classes=False,
            )
            failures = set(
                qa["findings"].loc[qa["findings"]["status"] == "fail", "check"]
            )
            self.assertIn("source_tile_split_leakage", failures)
            self.assertIn("actual_source_tile_split_leakage", failures)
            self.assertIn("source_provenance_failures", failures)
            self.assertTrue(manifest_has_hard_failures(qa))
            self.assertEqual(len(qa["source_tile_split_leakage"]), 1)


class SpatialSplitTests(unittest.TestCase):
    def test_actual_constituent_components_are_isolated_in_split_and_fold(self):
        rows = []
        for index in range(30):
            actual_tiles = f"tile_{index}"
            if index == 1:
                actual_tiles = "tile_0;tile_1"
            rows.append(
                {
                    "sample_id": f"sample_{index}",
                    "presence": index % 2,
                    "lat": 30.0 + index * 0.2,
                    "lon": -120.0 + index * 0.4,
                    "split": ["train", "val", "test"][index % 3],
                    "block_id": f"candidate_{index}",
                    "fold": index % 5 + 1,
                    "actual_source_tile_keys": actual_tiles,
                }
            )
        assignments, audit = reconcile_actual_source_groups(
            pd.DataFrame(rows),
            seed=42,
            train_frac=0.70,
            val_frac=0.15,
            test_frac=0.15,
            block_size_m=200000.0,
            n_folds=5,
        )
        shared = assignments[assignments["sample_id"].isin(["sample_0", "sample_1"])]
        self.assertEqual(shared["split"].nunique(), 1)
        self.assertEqual(shared["block_id"].nunique(), 1)
        self.assertEqual(shared["fold"].nunique(), 1)
        self.assertEqual(len(assignments), len(rows))
        self.assertEqual(len(audit), len(rows))

    def test_uniform_group_split_keeps_source_tiles_isolated(self):
        rows = []
        for tile_index in range(30):
            for label in [0, 1]:
                rows.append(
                    {
                        "source_tile_key": f"tile_{tile_index}",
                        "presence": label,
                        "lat": 30 + tile_index / 100,
                        "lon": -80 - label / 100,
                    }
                )
        split = stratified_train_val_test_split(
            pd.DataFrame(rows),
            train_frac=0.70,
            val_frac=0.15,
            test_frac=0.15,
            seed=42,
            group_column="source_tile_key",
        )
        self.assertEqual(set(split["split"]), {"train", "val", "test"})
        self.assertTrue((split.groupby("source_tile_key")["split"].nunique() == 1).all())
        class_counts = split.groupby(["split", "presence"]).size().unstack(fill_value=0)
        self.assertTrue((class_counts[0] == class_counts[1]).all())

    def test_source_tile_group_is_forced_into_one_spatial_block(self):
        points = gpd.GeoDataFrame(
            {
                "source_tile_key": ["shared", "shared", "other"],
                "presence": [1, 0, 0],
            },
            geometry=[Point(-100, 35), Point(-95, 35), Point(-80, 35)],
            crs="EPSG:4326",
        )
        assigned = assign_spatial_blocks(
            points,
            block_size_m=200_000,
            n_folds=3,
            seed=42,
            group_column="source_tile_key",
        )
        shared = assigned[assigned["source_tile_key"] == "shared"]
        self.assertEqual(shared["block_id"].nunique(), 1)
        self.assertEqual(shared["fold"].nunique(), 1)

    def test_heldout_fold_validation_has_no_block_overlap(self):
        rows = []
        for fold in range(1, 6):
            for label in [0, 1]:
                rows.append(
                    {
                        "lat": 30.0 + fold + label / 10,
                        "lon": -80.0 - fold,
                        "presence": label,
                        "fold": fold,
                        "block_id": f"block_{fold}",
                        "source_tile_key": f"tile_{fold}_{label}",
                    }
                )
        rounds = make_spatial_cv_rounds(
            pd.DataFrame(rows),
            n_folds=5,
            seed=42,
            validation_mode="heldout_fold",
        )
        for _, round_df in rounds.groupby("cv_round"):
            self.assertTrue({"train", "val", "test"}.issubset(set(round_df["split"])))
            self.assertTrue((round_df.groupby("block_id")["split"].nunique() == 1).all())
            self.assertTrue((round_df.groupby("source_tile_key")["split"].nunique() == 1).all())

if __name__ == "__main__":
    unittest.main()
