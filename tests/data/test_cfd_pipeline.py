"""Tests for CFD data pipeline (tasks 3.1-3.5)."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Task 3.1 — wind-cfd-trial download and indexing
# ---------------------------------------------------------------------------


class TestCFDDownloadAndIndexing:
    """wind-cfd-trial HuggingFace download, indexing, and case lookup."""

    def _make_mock_case(self, case_id: int, h: int = 64, w: int = 64):
        """Create a synthetic CFD case dict matching HuggingFace schema."""
        return {
            "case_id": f"case_{case_id:05d}",
            "dem": np.random.rand(h, w).astype(np.float32),
            "wind_speed": float(np.random.uniform(1, 20)),
            "wind_direction": float(np.random.uniform(0, 360)),
            "reference_height": 10.0,
            "u_wind": np.random.rand(h, w).astype(np.float32),
            "v_wind": np.random.rand(h, w).astype(np.float32),
            "w_wind": np.random.rand(h, w).astype(np.float32),
        }

    @patch("terrain_weather_ml.data.cfd_pipeline.load_dataset")
    def test_download_returns_index_with_expected_count(self, mock_load):
        """Downloading should return an index with one entry per case."""
        from terrain_weather_ml.data.cfd_pipeline import CFDDatasetIndex

        cases = [self._make_mock_case(i) for i in range(25)]
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = MagicMock(return_value=iter(cases))
        mock_dataset.__len__ = MagicMock(return_value=len(cases))
        mock_load.return_value = {"train": mock_dataset}

        index = CFDDatasetIndex.from_huggingface(cache_dir="/tmp/cfd_test")
        assert len(index) == 25

    @patch("terrain_weather_ml.data.cfd_pipeline.load_dataset")
    def test_index_lookup_by_case_id(self, mock_load):
        """Index should support O(1) lookup by case_id."""
        from terrain_weather_ml.data.cfd_pipeline import CFDDatasetIndex

        cases = [self._make_mock_case(i) for i in range(10)]
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = MagicMock(return_value=iter(cases))
        mock_dataset.__len__ = MagicMock(return_value=len(cases))
        mock_load.return_value = {"train": mock_dataset}

        index = CFDDatasetIndex.from_huggingface(cache_dir="/tmp/cfd_test")

        meta = index.get_case("case_00003")
        assert meta is not None
        assert meta["case_id"] == "case_00003"
        assert "wind_speed" in meta
        assert "wind_direction" in meta

    @patch("terrain_weather_ml.data.cfd_pipeline.load_dataset")
    def test_index_metadata_fields(self, mock_load):
        """Each indexed case should contain boundary condition metadata."""
        from terrain_weather_ml.data.cfd_pipeline import CFDDatasetIndex

        cases = [self._make_mock_case(0)]
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = MagicMock(return_value=iter(cases))
        mock_dataset.__len__ = MagicMock(return_value=len(cases))
        mock_load.return_value = {"train": mock_dataset}

        index = CFDDatasetIndex.from_huggingface(cache_dir="/tmp/cfd_test")
        meta = index.get_case("case_00000")

        required_keys = {"case_id", "wind_speed", "wind_direction", "reference_height"}
        assert required_keys.issubset(set(meta.keys()))

    @patch("terrain_weather_ml.data.cfd_pipeline.load_dataset")
    def test_index_iteration(self, mock_load):
        """Index should be iterable over all case IDs."""
        from terrain_weather_ml.data.cfd_pipeline import CFDDatasetIndex

        cases = [self._make_mock_case(i) for i in range(5)]
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = MagicMock(return_value=iter(cases))
        mock_dataset.__len__ = MagicMock(return_value=len(cases))
        mock_load.return_value = {"train": mock_dataset}

        index = CFDDatasetIndex.from_huggingface(cache_dir="/tmp/cfd_test")
        case_ids = list(index.case_ids())
        assert len(case_ids) == 5
        assert "case_00000" in case_ids


# ---------------------------------------------------------------------------
# Task 3.2 — Input/output tensor formatting
# ---------------------------------------------------------------------------


def _make_raw_case(h: int = 64, w: int = 64) -> dict:
    """Create a raw HuggingFace-style case dict."""
    return {
        "case_id": "case_00000",
        "dem": np.random.rand(h, w).astype(np.float32),
        "wind_speed": 10.0,
        "wind_direction": 270.0,
        "reference_height": 10.0,
        "u_wind": np.random.rand(h, w).astype(np.float32),
        "v_wind": np.random.rand(h, w).astype(np.float32),
        "w_wind": np.random.rand(h, w).astype(np.float32),
    }


class TestTensorFormatting:
    """Input/output tensor formatting for CFD cases."""

    def test_dem_patch_shape(self):
        """DEM patch tensor should have shape (1, H, W)."""
        from terrain_weather_ml.data.cfd_pipeline import format_cfd_tensors

        raw = _make_raw_case(h=64, w=64)
        sample = format_cfd_tensors(raw)
        assert sample.dem.shape == (1, 64, 64)
        assert sample.dem.dtype == torch.float32

    def test_boundary_conditions_shape(self):
        """Boundary conditions tensor should be (3,): speed, direction, height."""
        from terrain_weather_ml.data.cfd_pipeline import format_cfd_tensors

        raw = _make_raw_case()
        sample = format_cfd_tensors(raw)
        assert sample.boundary_conditions.shape == (3,)
        assert sample.boundary_conditions[0].item() == pytest.approx(10.0)  # speed
        assert sample.boundary_conditions[1].item() == pytest.approx(270.0)  # direction
        assert sample.boundary_conditions[2].item() == pytest.approx(10.0)  # height

    def test_wind_field_shape(self):
        """Target wind field should be (3, H, W) for u, v, w."""
        from terrain_weather_ml.data.cfd_pipeline import format_cfd_tensors

        raw = _make_raw_case(h=80, w=96)
        sample = format_cfd_tensors(raw)
        assert sample.wind_field.shape == (3, 80, 96)
        assert sample.wind_field.dtype == torch.float32

    def test_spatial_dimensions_consistent(self):
        """DEM and wind field spatial dims must match."""
        from terrain_weather_ml.data.cfd_pipeline import format_cfd_tensors

        raw = _make_raw_case(h=128, w=128)
        sample = format_cfd_tensors(raw)
        _, h_dem, w_dem = sample.dem.shape
        _, h_wind, w_wind = sample.wind_field.shape
        assert h_dem == h_wind
        assert w_dem == w_wind

    def test_terrain_features_shape(self):
        """Terrain features tensor should have shape (C_terrain, H, W).

        For now, C_terrain is at least 1 (derived from DEM), but the terrain
        encoder will expand this later. The pipeline should produce a placeholder.
        """
        from terrain_weather_ml.data.cfd_pipeline import format_cfd_tensors

        raw = _make_raw_case(h=64, w=64)
        sample = format_cfd_tensors(raw)
        assert sample.terrain_features.ndim == 3
        c, h, w = sample.terrain_features.shape
        assert c >= 1
        assert h == 64
        assert w == 64


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
        # Simulate WindNinja writing output files
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

        # Direction should be rotated
        new_dir = augmented.boundary_conditions[1].item()
        assert new_dir == pytest.approx(180.0, abs=1e-4)

        # u, v should be rotated: (5, 3) rotated 90 deg CCW => (-3, 5)
        assert augmented.wind_field[0].mean().item() == pytest.approx(-3.0, abs=1e-4)
        assert augmented.wind_field[1].mean().item() == pytest.approx(5.0, abs=1e-4)

    def test_wind_speed_scaling(self):
        """Wind speed scaling should scale both boundary and wind field."""
        from terrain_weather_ml.data.cfd_pipeline import augment_wind_speed

        sample = _make_sample()
        augmented = augment_wind_speed(sample, scale_factor=2.0)

        assert augmented.boundary_conditions[0].item() == pytest.approx(20.0)
        # Wind components should also be scaled
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

        # DEM should be different
        assert not torch.allclose(augmented.dem, sample.dem)
        # Wind field should be unchanged
        assert torch.allclose(augmented.wind_field, sample.wind_field)

    def test_full_augmentation_pipeline(self):
        """Full augmentation pipeline should produce valid shapes."""
        from terrain_weather_ml.data.cfd_pipeline import augment_sample

        sample = _make_sample(h=128, w=128)
        torch.manual_seed(42)
        aug1 = augment_sample(sample)
        torch.manual_seed(99)
        aug2 = augment_sample(sample)

        # Both should have valid shapes
        assert aug1.dem.ndim == 3
        assert aug1.wind_field.ndim == 3
        assert aug2.dem.ndim == 3

        # With different seeds, augmentations should differ in at least one
        # property. Boundary conditions (not affected by crop size) are a
        # reliable comparison target.
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

        # Uniform (u=5, v=3) over the grid is divergence-free: du/dx=0, dv/dy=0
        h, w = 64, 64
        u = torch.ones(h, w) * 5.0
        v = torch.ones(h, w) * 3.0
        wind_field = torch.stack([u, v, torch.zeros(h, w)])  # (3, H, W)

        div_field = compute_divergence(wind_field)
        assert div_field.shape == (h, w)
        assert div_field.abs().max().item() < 1e-6

    def test_non_conservative_field_has_nonzero_divergence(self):
        """A linearly increasing u-field should have positive divergence."""
        from terrain_weather_ml.data.cfd_pipeline import compute_divergence

        h, w = 64, 64
        # u increases linearly in x: du/dx = 1/dx (nonzero)
        x = torch.linspace(0, 10, w).unsqueeze(0).expand(h, w)
        v = torch.zeros(h, w)
        wind_field = torch.stack([x, v, torch.zeros(h, w)])

        div_field = compute_divergence(wind_field)
        # Interior should have positive divergence (du/dx > 0)
        assert div_field[1:-1, 1:-1].mean().item() > 0

    def test_label_passes_conservative_field(self):
        """Uniform wind should pass mass conservation check."""
        from terrain_weather_ml.data.cfd_pipeline import label_mass_conservation

        sample = _make_sample()
        # Uniform wind => divergence-free
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
        # Create a strongly non-conservative field
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

    def test_threshold_based_filtering(self):
        """Filter function should exclude samples exceeding threshold."""
        from terrain_weather_ml.data.cfd_pipeline import (
            CFDSample,
            filter_by_conservation,
            label_mass_conservation,
        )

        # Create a mix of good and bad samples
        good = _make_sample()  # uniform wind => divergence-free
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
