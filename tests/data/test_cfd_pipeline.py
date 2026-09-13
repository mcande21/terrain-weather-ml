"""Tests for CFD data pipeline (tasks 3.1-3.5).

Tests target the REAL wind-cfd-trial zarr schema from souravsud/wind-cfd-trial:
  - Variables: Ux, Uy, Uz (3D: ni x nj x nk), dem (2D: ni x nj), h_agl (3D)
  - Attrs: reference_velocity_ms, rotation_deg, reference_height_m, case_id
  - Data loaded via xarray.open_zarr, NOT datasets.load_dataset
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Zarr fixture helpers — replicate real wind-cfd-trial schema
# ---------------------------------------------------------------------------


def _create_zarr_store(path: Path, case_id: str, ni: int = 64, nj: int = 64, nk: int = 16,
                       ref_velocity: float = 10.0, rotation: float = 210.0,
                       ref_height: float = 100.0) -> Path:
    """Create a minimal zarr store matching the real wind-cfd-trial schema.

    Uses xarray to write the zarr store so dimension metadata
    (_ARRAY_DIMENSIONS) is properly set for xarray.open_zarr compatibility.

    Real variables: Ux, Uy, Uz (ni, nj, nk), dem (ni, nj), h_agl (ni, nj, nk).
    Real attrs: reference_velocity_ms, rotation_deg, reference_height_m, case_id.
    """
    import xarray as xr

    store_path = path / f"{case_id}.zarr"
    rng = np.random.default_rng(42)

    # Build xarray Dataset matching real schema
    ds = xr.Dataset(
        data_vars={
            "Ux": (["ni", "nj", "nk"],
                   rng.standard_normal((ni, nj, nk)).astype(np.float32)),
            "Uy": (["ni", "nj", "nk"],
                   rng.standard_normal((ni, nj, nk)).astype(np.float32)),
            "Uz": (["ni", "nj", "nk"],
                   rng.standard_normal((ni, nj, nk)).astype(np.float32)),
            "dem": (["ni", "nj"],
                    rng.uniform(0, 500, (ni, nj)).astype(np.float32)),
            "h_agl": (["ni", "nj", "nk"],
                      np.stack([
                          np.full((ni, nj), 2.0 + k * 4.0, dtype=np.float32)
                          for k in range(nk)
                      ], axis=-1)),
            "roughness": (["ni", "nj"],
                          rng.uniform(0.01, 1.0, (ni, nj)).astype(np.float32)),
        },
        attrs={
            "case_id": case_id,
            "reference_velocity_ms": ref_velocity,
            "rotation_deg": rotation,
            "reference_height_m": ref_height,
            "grid_ni": ni,
            "grid_nj": nj,
            "grid_nk": nk,
            "converged": True,
            "solver": "simpleFoam",
        },
    )
    ds.to_zarr(str(store_path), mode="w")

    return store_path


@pytest.fixture
def zarr_cases(tmp_path):
    """Create two zarr cases matching the real wind-cfd-trial trial dataset."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    case1 = _create_zarr_store(data_dir, "grid_bench_askervein_fix_210deg",
                               ref_velocity=10.0, rotation=210.0)
    case2 = _create_zarr_store(data_dir, "grid_bench_baseline_Z0smooth_G2_5_225deg",
                               ref_velocity=10.0, rotation=225.0)
    return tmp_path, [case1, case2]


# ---------------------------------------------------------------------------
# Task 3.1 — wind-cfd-trial download and indexing (zarr-based)
# ---------------------------------------------------------------------------


class TestCFDDownloadAndIndexing:
    """wind-cfd-trial HuggingFace download, indexing, and case lookup.

    The real dataset uses zarr stores with xarray, NOT tabular datasets.
    """

    def test_repo_id_is_correct(self):
        """Repo ID should be souravsud/wind-cfd-trial (not daisy1212)."""
        from terrain_weather_ml.data.cfd_pipeline import WIND_CFD_TRIAL_REPO

        assert WIND_CFD_TRIAL_REPO == "souravsud/wind-cfd-trial"

    def test_from_huggingface_indexes_zarr_cases(self, zarr_cases):
        """from_huggingface should index zarr stores, not tabular rows."""
        from terrain_weather_ml.data.cfd_pipeline import CFDDatasetIndex

        root_dir, case_paths = zarr_cases
        # Mock the HuggingFace download to return local zarr paths
        with patch("terrain_weather_ml.data.cfd_pipeline.download_zarr_cases") as mock_dl:
            mock_dl.return_value = case_paths
            index = CFDDatasetIndex.from_huggingface(cache_dir=root_dir)

        assert len(index) == 2

    def test_index_metadata_from_zarr_attrs(self, zarr_cases):
        """Index metadata should come from zarr .attrs, not tabular columns."""
        from terrain_weather_ml.data.cfd_pipeline import CFDDatasetIndex

        root_dir, case_paths = zarr_cases
        with patch("terrain_weather_ml.data.cfd_pipeline.download_zarr_cases") as mock_dl:
            mock_dl.return_value = case_paths
            index = CFDDatasetIndex.from_huggingface(cache_dir=root_dir)

        meta = index.get_case("grid_bench_askervein_fix_210deg")
        assert meta is not None
        assert meta["wind_speed"] == pytest.approx(10.0)
        assert meta["wind_direction"] == pytest.approx(210.0)
        assert meta["reference_height"] == pytest.approx(100.0)

    def test_index_stores_zarr_path(self, zarr_cases):
        """Index should store the zarr path for later loading."""
        from terrain_weather_ml.data.cfd_pipeline import CFDDatasetIndex

        root_dir, case_paths = zarr_cases
        with patch("terrain_weather_ml.data.cfd_pipeline.download_zarr_cases") as mock_dl:
            mock_dl.return_value = case_paths
            index = CFDDatasetIndex.from_huggingface(cache_dir=root_dir)

        meta = index.get_case("grid_bench_askervein_fix_210deg")
        assert "zarr_path" in meta
        assert Path(meta["zarr_path"]).exists()

    def test_index_lookup_by_case_id(self, zarr_cases):
        """Index should support O(1) lookup by case_id."""
        from terrain_weather_ml.data.cfd_pipeline import CFDDatasetIndex

        root_dir, case_paths = zarr_cases
        with patch("terrain_weather_ml.data.cfd_pipeline.download_zarr_cases") as mock_dl:
            mock_dl.return_value = case_paths
            index = CFDDatasetIndex.from_huggingface(cache_dir=root_dir)

        meta = index.get_case("grid_bench_baseline_Z0smooth_G2_5_225deg")
        assert meta is not None
        assert meta["case_id"] == "grid_bench_baseline_Z0smooth_G2_5_225deg"

    def test_case_ids_returns_all(self, zarr_cases):
        """case_ids() should return all indexed case IDs."""
        from terrain_weather_ml.data.cfd_pipeline import CFDDatasetIndex

        root_dir, case_paths = zarr_cases
        with patch("terrain_weather_ml.data.cfd_pipeline.download_zarr_cases") as mock_dl:
            mock_dl.return_value = case_paths
            index = CFDDatasetIndex.from_huggingface(cache_dir=root_dir)

        ids = index.case_ids()
        assert len(ids) == 2
        assert "grid_bench_askervein_fix_210deg" in ids

    def test_handles_small_dataset_gracefully(self, tmp_path):
        """Pipeline should handle datasets with only 1-2 cases."""
        from terrain_weather_ml.data.cfd_pipeline import CFDDatasetIndex

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        single_case = _create_zarr_store(data_dir, "single_case")

        with patch("terrain_weather_ml.data.cfd_pipeline.download_zarr_cases") as mock_dl:
            mock_dl.return_value = [single_case]
            index = CFDDatasetIndex.from_huggingface(cache_dir=tmp_path)

        assert len(index) == 1


# ---------------------------------------------------------------------------
# Task 3.2 — Input/output tensor formatting (3D zarr -> 2D extraction)
# ---------------------------------------------------------------------------


def _make_raw_zarr_case(ni: int = 64, nj: int = 64, nk: int = 16) -> dict:
    """Create a raw case dict matching the REAL zarr schema.

    3D wind fields (Ux, Uy, Uz), 2D dem, 3D h_agl, attrs-based metadata.
    """
    rng = np.random.default_rng(42)
    h_agl = np.zeros((ni, nj, nk), dtype=np.float32)
    for k_idx in range(nk):
        h_agl[:, :, k_idx] = 2.0 + k_idx * 4.0

    return {
        "case_id": "test_case",
        "Ux": rng.standard_normal((ni, nj, nk)).astype(np.float32),
        "Uy": rng.standard_normal((ni, nj, nk)).astype(np.float32),
        "Uz": rng.standard_normal((ni, nj, nk)).astype(np.float32),
        "dem": rng.uniform(0, 500, (ni, nj)).astype(np.float32),
        "h_agl": h_agl,
        "wind_speed": 10.0,
        "wind_direction": 270.0,
        "reference_height": 100.0,
    }


class TestTensorFormatting:
    """Input/output tensor formatting for CFD cases with 3D->2D extraction."""

    def test_format_extracts_2d_from_3d_wind(self):
        """format_cfd_tensors should extract a 2D slice from 3D wind fields."""
        from terrain_weather_ml.data.cfd_pipeline import format_cfd_tensors

        raw = _make_raw_zarr_case(ni=64, nj=64, nk=16)
        sample = format_cfd_tensors(raw)
        # Output wind field should be 2D: (3, H, W), NOT (3, H, W, K)
        assert sample.wind_field.shape == (3, 64, 64)
        assert sample.wind_field.dtype == torch.float32

    def test_format_accepts_variable_name_mapping(self):
        """format_cfd_tensors should accept Ux/Uy/Uz (real names), not u_wind/v_wind/w_wind."""
        from terrain_weather_ml.data.cfd_pipeline import format_cfd_tensors

        raw = _make_raw_zarr_case()
        # Should NOT require old keys u_wind/v_wind/w_wind
        assert "u_wind" not in raw
        assert "Ux" in raw
        sample = format_cfd_tensors(raw)
        assert sample.wind_field.ndim == 3

    def test_dem_patch_shape(self):
        """DEM patch tensor should have shape (1, H, W)."""
        from terrain_weather_ml.data.cfd_pipeline import format_cfd_tensors

        raw = _make_raw_zarr_case(ni=64, nj=64)
        sample = format_cfd_tensors(raw)
        assert sample.dem.shape == (1, 64, 64)
        assert sample.dem.dtype == torch.float32

    def test_boundary_conditions_shape(self):
        """Boundary conditions tensor should be (3,): speed, direction, height."""
        from terrain_weather_ml.data.cfd_pipeline import format_cfd_tensors

        raw = _make_raw_zarr_case()
        sample = format_cfd_tensors(raw)
        assert sample.boundary_conditions.shape == (3,)
        assert sample.boundary_conditions[0].item() == pytest.approx(10.0)
        assert sample.boundary_conditions[1].item() == pytest.approx(270.0)
        assert sample.boundary_conditions[2].item() == pytest.approx(100.0)

    def test_spatial_dimensions_consistent(self):
        """DEM and wind field spatial dims must match."""
        from terrain_weather_ml.data.cfd_pipeline import format_cfd_tensors

        raw = _make_raw_zarr_case(ni=128, nj=128)
        sample = format_cfd_tensors(raw)
        _, h_dem, w_dem = sample.dem.shape
        _, h_wind, w_wind = sample.wind_field.shape
        assert h_dem == h_wind
        assert w_dem == w_wind

    def test_terrain_features_shape(self):
        """Terrain features tensor should have shape (C_terrain, H, W)."""
        from terrain_weather_ml.data.cfd_pipeline import format_cfd_tensors

        raw = _make_raw_zarr_case(ni=64, nj=64)
        sample = format_cfd_tensors(raw)
        assert sample.terrain_features.ndim == 3
        c, h, w = sample.terrain_features.shape
        assert c >= 1
        assert h == 64
        assert w == 64

    def test_vertical_level_selection_default(self):
        """Default should extract near-surface level (k=0)."""
        from terrain_weather_ml.data.cfd_pipeline import format_cfd_tensors

        raw = _make_raw_zarr_case(ni=32, nj=32, nk=16)
        sample = format_cfd_tensors(raw)
        # Wind at k=0 should match the extracted slice
        expected_u = torch.from_numpy(raw["Ux"][:, :, 0])
        assert torch.allclose(sample.wind_field[0], expected_u)

    def test_vertical_level_selection_custom(self):
        """Should support selecting a specific vertical level."""
        from terrain_weather_ml.data.cfd_pipeline import format_cfd_tensors

        raw = _make_raw_zarr_case(ni=32, nj=32, nk=16)
        sample = format_cfd_tensors(raw, vertical_level=5)
        expected_u = torch.from_numpy(raw["Ux"][:, :, 5])
        assert torch.allclose(sample.wind_field[0], expected_u)

    def test_format_handles_already_2d_wind(self):
        """Should still work with pre-extracted 2D wind fields for backward compat."""
        from terrain_weather_ml.data.cfd_pipeline import format_cfd_tensors

        raw = {
            "case_id": "legacy_case",
            "Ux": np.random.rand(64, 64).astype(np.float32),
            "Uy": np.random.rand(64, 64).astype(np.float32),
            "Uz": np.random.rand(64, 64).astype(np.float32),
            "dem": np.random.rand(64, 64).astype(np.float32),
            "wind_speed": 10.0,
            "wind_direction": 270.0,
            "reference_height": 10.0,
        }
        sample = format_cfd_tensors(raw)
        assert sample.wind_field.shape == (3, 64, 64)


# ---------------------------------------------------------------------------
# Task 3.3 — WindNinja synthetic generation
# ---------------------------------------------------------------------------


class TestWindNinjaSynthetic:
    """WindNinja integration with graceful absence handling."""

    @patch("shutil.which", return_value=None)
    def test_windninja_not_found_logs_warning(self, mock_which, caplog):
        """When WindNinja binary is absent, log a warning and return None."""
        from terrain_weather_ml.data.cfd_pipeline import WindNinjaGenerator

        gen = WindNinjaGenerator()
        dem = torch.randn(1, 64, 64)
        bc = torch.tensor([10.0, 270.0, 10.0])

        with caplog.at_level(logging.WARNING):
            result = gen.generate(dem, bc)

        assert result is None
        assert any("WindNinja" in msg for msg in caplog.messages)

    @patch("shutil.which", return_value=None)
    def test_pipeline_continues_without_windninja(self, mock_which, caplog):
        """Pipeline should operate on wind-cfd-trial data only when no WindNinja."""
        from terrain_weather_ml.data.cfd_pipeline import WindNinjaGenerator

        gen = WindNinjaGenerator()
        assert not gen.is_available()

    @patch("shutil.which", return_value="/usr/local/bin/WindNinja_cli")
    def test_windninja_available_detected(self, mock_which):
        """When binary exists, is_available should return True."""
        from terrain_weather_ml.data.cfd_pipeline import WindNinjaGenerator

        gen = WindNinjaGenerator()
        assert gen.is_available()

    @patch("shutil.which", return_value="/usr/local/bin/WindNinja_cli")
    @patch("subprocess.run")
    def test_windninja_produces_valid_tensors(self, mock_run, mock_which, tmp_path):
        """When WindNinja is available and succeeds, output should be a CFDSample."""
        from terrain_weather_ml.data.cfd_pipeline import WindNinjaGenerator

        h, w = 64, 64
        mock_run.return_value = MagicMock(returncode=0)

        gen = WindNinjaGenerator(work_dir=tmp_path)

        # Create fake output files that WindNinja would produce
        u_out = np.random.rand(h, w).astype(np.float32)
        v_out = np.random.rand(h, w).astype(np.float32)
        w_out = np.zeros((h, w), dtype=np.float32)
        np.save(tmp_path / "u_wind.npy", u_out)
        np.save(tmp_path / "v_wind.npy", v_out)
        np.save(tmp_path / "w_wind.npy", w_out)

        dem = torch.randn(1, h, w)
        bc = torch.tensor([10.0, 270.0, 10.0])

        result = gen.generate(dem, bc)
        assert result is not None
        assert result.wind_field.shape == (3, h, w)
        assert result.dem.shape == (1, h, w)


# ---------------------------------------------------------------------------
# Task 3.4 — Domain randomization
# ---------------------------------------------------------------------------


def _make_sample(h: int = 128, w: int = 128):
    """Create a deterministic CFDSample for augmentation tests."""
    from terrain_weather_ml.data.cfd_pipeline import CFDSample

    return CFDSample(
        dem=torch.ones(1, h, w, dtype=torch.float32) * 1000.0,
        terrain_features=torch.ones(1, h, w, dtype=torch.float32) * 1000.0,
        boundary_conditions=torch.tensor([10.0, 90.0, 10.0]),
        wind_field=torch.stack([
            torch.ones(h, w) * 5.0,   # u
            torch.ones(h, w) * 3.0,   # v
            torch.zeros(h, w),         # w
        ]),
        case_id="test_case",
    )


class TestDomainRandomization:
    """Domain randomization augmentations for CFD training samples."""

    def test_wind_direction_rotation(self):
        """Wind direction rotation should rotate both boundary and wind field."""
        from terrain_weather_ml.data.cfd_pipeline import augment_wind_direction

        sample = _make_sample()
        augmented = augment_wind_direction(sample, rotation_deg=90.0)

        new_dir = augmented.boundary_conditions[1].item()
        assert new_dir == pytest.approx(180.0, abs=1e-4)

        assert augmented.wind_field[0].mean().item() == pytest.approx(-3.0, abs=1e-4)
        assert augmented.wind_field[1].mean().item() == pytest.approx(5.0, abs=1e-4)

    def test_wind_speed_scaling(self):
        """Wind speed scaling should scale both boundary and wind field."""
        from terrain_weather_ml.data.cfd_pipeline import augment_wind_speed

        sample = _make_sample()
        augmented = augment_wind_speed(sample, scale_factor=2.0)

        assert augmented.boundary_conditions[0].item() == pytest.approx(20.0)
        assert augmented.wind_field[0].mean().item() == pytest.approx(10.0, abs=1e-4)
        assert augmented.wind_field[1].mean().item() == pytest.approx(6.0, abs=1e-4)

    def test_wind_speed_clamped(self):
        """Scaled speed must stay within physical bounds [0.5, 30]."""
        from terrain_weather_ml.data.cfd_pipeline import augment_wind_speed

        sample = _make_sample()
        augmented = augment_wind_speed(sample, scale_factor=100.0)
        assert augmented.boundary_conditions[0].item() <= 30.0

    def test_random_crop_minimum_size(self):
        """Random crop must produce at least 64x64."""
        from terrain_weather_ml.data.cfd_pipeline import augment_random_crop

        sample = _make_sample(h=128, w=128)
        augmented = augment_random_crop(sample, min_size=64)
        _, h, w = augmented.dem.shape
        assert h >= 64
        assert w >= 64

    def test_random_crop_consistent(self):
        """Crop must be applied consistently to DEM and wind field."""
        from terrain_weather_ml.data.cfd_pipeline import augment_random_crop

        sample = _make_sample(h=128, w=128)
        augmented = augment_random_crop(sample, min_size=64)
        _, h_dem, w_dem = augmented.dem.shape
        _, h_wind, w_wind = augmented.wind_field.shape
        assert h_dem == h_wind
        assert w_dem == w_wind

    def test_elevation_noise(self):
        """Gaussian elevation noise should change the DEM but not the wind field."""
        from terrain_weather_ml.data.cfd_pipeline import augment_elevation_noise

        sample = _make_sample()
        augmented = augment_elevation_noise(sample, sigma=5.0)

        assert not torch.allclose(augmented.dem, sample.dem)
        assert torch.allclose(augmented.wind_field, sample.wind_field)

    def test_full_augmentation_pipeline(self):
        """Full augmentation pipeline should produce valid shapes."""
        from terrain_weather_ml.data.cfd_pipeline import augment_sample

        sample = _make_sample(h=128, w=128)
        torch.manual_seed(42)
        aug1 = augment_sample(sample)
        torch.manual_seed(99)
        aug2 = augment_sample(sample)

        assert aug1.dem.ndim == 3
        assert aug1.wind_field.ndim == 3
        assert aug2.dem.ndim == 3

        assert not torch.allclose(
            aug1.boundary_conditions, aug2.boundary_conditions
        )


# ---------------------------------------------------------------------------
# Task 3.5 — Mass conservation labeling
# ---------------------------------------------------------------------------


class TestMassConservationLabeling:
    """Divergence computation and threshold-based exclusion."""

    def test_divergence_free_field_returns_near_zero(self):
        """A known divergence-free field (uniform) should have ~0 divergence."""
        from terrain_weather_ml.data.cfd_pipeline import compute_divergence

        h, w = 64, 64
        u = torch.ones(h, w) * 5.0
        v = torch.ones(h, w) * 3.0
        wind_field = torch.stack([u, v, torch.zeros(h, w)])

        div_field = compute_divergence(wind_field)
        assert div_field.shape == (h, w)
        assert div_field.abs().max().item() < 1e-6

    def test_non_conservative_field_has_nonzero_divergence(self):
        """A linearly increasing u-field should have positive divergence."""
        from terrain_weather_ml.data.cfd_pipeline import compute_divergence

        h, w = 64, 64
        x = torch.linspace(0, 10, w).unsqueeze(0).expand(h, w)
        v = torch.zeros(h, w)
        wind_field = torch.stack([x, v, torch.zeros(h, w)])

        div_field = compute_divergence(wind_field)
        assert div_field[1:-1, 1:-1].mean().item() > 0

    def test_label_passes_conservative_field(self):
        """Uniform wind should pass mass conservation check."""
        from terrain_weather_ml.data.cfd_pipeline import label_mass_conservation

        sample = _make_sample()
        labeled = label_mass_conservation(sample)

        assert hasattr(labeled, "divergence_field")
        assert hasattr(labeled, "passes_conservation")
        assert labeled.passes_conservation is True

    def test_label_flags_non_conservative_field(self):
        """Field with high divergence should be flagged for exclusion."""
        from terrain_weather_ml.data.cfd_pipeline import (
            CFDSample,
            label_mass_conservation,
        )

        h, w = 64, 64
        x = torch.linspace(0, 100, w).unsqueeze(0).expand(h, w)
        wind_field = torch.stack([x, torch.zeros(h, w), torch.zeros(h, w)])

        sample = CFDSample(
            dem=torch.ones(1, h, w),
            terrain_features=torch.ones(1, h, w),
            boundary_conditions=torch.tensor([10.0, 0.0, 10.0]),
            wind_field=wind_field,
            case_id="bad_case",
        )

        labeled = label_mass_conservation(sample, threshold=0.01)
        assert labeled.passes_conservation is False

    def test_configurable_threshold(self):
        """Conservation threshold should be configurable for real RANS data.

        Real near-surface 2D slices of 3D terrain-following RANS have max |div|
        of 7-10 s^-1 due to missing dw/dz term. The default must accommodate this.
        """
        from terrain_weather_ml.data.cfd_pipeline import DEFAULT_DIVERGENCE_THRESHOLD

        # Default should be reasonable for near-surface RANS slices (not 0.01)
        assert DEFAULT_DIVERGENCE_THRESHOLD > 1.0, (
            f"Default threshold {DEFAULT_DIVERGENCE_THRESHOLD} is too strict for "
            "near-surface 2D slices of 3D RANS data (real max |div| ~ 7-10 s^-1)"
        )

    def test_threshold_based_filtering(self):
        """Filter function should exclude samples exceeding threshold."""
        from terrain_weather_ml.data.cfd_pipeline import (
            CFDSample,
            filter_by_conservation,
            label_mass_conservation,
        )

        good = _make_sample()
        good_labeled = label_mass_conservation(good)

        h, w = 128, 128
        x = torch.linspace(0, 100, w).unsqueeze(0).expand(h, w)
        bad_field = torch.stack([x, torch.zeros(h, w), torch.zeros(h, w)])
        bad = CFDSample(
            dem=torch.ones(1, h, w),
            terrain_features=torch.ones(1, h, w),
            boundary_conditions=torch.tensor([10.0, 0.0, 10.0]),
            wind_field=bad_field,
            case_id="bad_case",
        )
        bad_labeled = label_mass_conservation(bad, threshold=0.01)

        samples = [good_labeled, bad_labeled]
        filtered = filter_by_conservation(samples)

        assert len(filtered) == 1
        assert filtered[0].case_id == good.case_id


# ---------------------------------------------------------------------------
# Load from zarr integration test
# ---------------------------------------------------------------------------


class TestLoadFromZarr:
    """Integration: load zarr store -> format -> CFDSample."""

    def test_load_zarr_case_produces_cfd_sample(self, zarr_cases):
        """Loading a zarr case should produce a valid CFDSample."""
        from terrain_weather_ml.data.cfd_pipeline import format_cfd_tensors, load_zarr_case

        _, case_paths = zarr_cases
        raw = load_zarr_case(case_paths[0])

        assert "Ux" in raw
        assert "dem" in raw
        assert "case_id" in raw
        assert "wind_speed" in raw

        sample = format_cfd_tensors(raw)
        assert sample.wind_field.shape == (3, 64, 64)
        assert sample.dem.shape == (1, 64, 64)

    def test_load_zarr_extracts_attrs_as_metadata(self, zarr_cases):
        """Zarr attrs should be mapped to pipeline metadata keys."""
        from terrain_weather_ml.data.cfd_pipeline import load_zarr_case

        _, case_paths = zarr_cases
        raw = load_zarr_case(case_paths[0])

        # Attrs are mapped: reference_velocity_ms -> wind_speed, etc.
        assert raw["wind_speed"] == pytest.approx(10.0)
        assert raw["wind_direction"] == pytest.approx(210.0)
        assert raw["reference_height"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Real data validation (skipped if data not present)
# ---------------------------------------------------------------------------

REAL_DATA_DIR = Path(
    "/Users/cooperanderson/work/personal/code/terrain-weather-ml/data/cfd/data"
)
REAL_CASES = [
    "grid_bench_askervein_fix_210deg",
    "grid_bench_baseline_Z0smooth_G2_5_225deg",
]


@pytest.mark.skipif(
    not REAL_DATA_DIR.exists(),
    reason="Real CFD data not available at data/cfd/data/",
)
class TestRealDataValidation:
    """Validate the pipeline against real wind-cfd-trial zarr stores."""

    @pytest.mark.parametrize("case_name", REAL_CASES)
    def test_load_real_zarr_produces_cfd_sample(self, case_name):
        """Real zarr data should load and produce valid CFDSample objects."""
        from terrain_weather_ml.data.cfd_pipeline import format_cfd_tensors, load_zarr_case

        zarr_path = REAL_DATA_DIR / f"{case_name}.zarr"
        if not zarr_path.exists():
            pytest.skip(f"Real zarr store not found: {zarr_path}")

        raw = load_zarr_case(zarr_path)
        sample = format_cfd_tensors(raw)

        # Real data is 298x298 with 64 vertical levels
        assert sample.wind_field.shape == (3, 298, 298)
        assert sample.dem.shape == (1, 298, 298)
        assert sample.boundary_conditions.shape == (3,)
        assert sample.case_id == case_name

    @pytest.mark.parametrize("case_name", REAL_CASES)
    def test_real_data_metadata_mapping(self, case_name):
        """Real zarr attrs should map correctly to pipeline metadata."""
        from terrain_weather_ml.data.cfd_pipeline import load_zarr_case

        zarr_path = REAL_DATA_DIR / f"{case_name}.zarr"
        if not zarr_path.exists():
            pytest.skip(f"Real zarr store not found: {zarr_path}")

        raw = load_zarr_case(zarr_path)

        assert raw["wind_speed"] == pytest.approx(10.0)  # reference_velocity_ms
        assert raw["reference_height"] == pytest.approx(100.0)  # reference_height_m
        assert raw["wind_direction"] in (210.0, 225.0)  # rotation_deg

    def test_real_data_vertical_level_selection(self):
        """Should extract different 2D slices from 3D real data."""
        from terrain_weather_ml.data.cfd_pipeline import format_cfd_tensors, load_zarr_case

        zarr_path = REAL_DATA_DIR / f"{REAL_CASES[0]}.zarr"
        if not zarr_path.exists():
            pytest.skip("Real zarr store not found")

        raw = load_zarr_case(zarr_path)

        sample_k0 = format_cfd_tensors(raw, vertical_level=0)
        sample_k5 = format_cfd_tensors(raw, vertical_level=5)

        # Both should be 2D
        assert sample_k0.wind_field.shape == (3, 298, 298)
        assert sample_k5.wind_field.shape == (3, 298, 298)

        # But different values at different levels
        assert not torch.allclose(sample_k0.wind_field, sample_k5.wind_field)
