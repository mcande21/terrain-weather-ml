"""Tests for ERA5 conditioner (tasks 2.1-2.4).

Tests cover:
- ERA5Conditioner class with CDS API download for configurable domains
- Variable extraction and mapping (ERA5 short names -> project schema)
- Accumulated precipitation -> instantaneous rate conversion
- Temporal alignment with PeakWeather/IMIS observations
- Resumable download with checkpoint tracking
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import numpy as np
import pytest
import xarray as xr

from terrain_weather_ml.backbone.era5_conditioner import (
    CDS_VARIABLE_MAP,
    PREDEFINED_DOMAINS,
    ERA5Conditioner,
    ERA5Config,
    ERA5Data,
    ERA5TemporalAligner,
)
from terrain_weather_ml.data.reynolds_creek import DownloadCheckpoint

# -------------------------------------------------------------------------
# 2.1 — ERA5Conditioner class and domain configuration
# -------------------------------------------------------------------------


class TestERA5Config:
    """Test ERA5Config with predefined and custom domains."""

    def test_alpine_domain_defaults(self):
        config = ERA5Config(domain="alpine")
        assert config.lat_min == 45.0
        assert config.lat_max == 48.0
        assert config.lon_min == 5.0
        assert config.lon_max == 11.0

    def test_colorado_domain(self):
        config = ERA5Config(domain="colorado")
        assert config.lat_min == 37.0
        assert config.lat_max == 41.0
        assert config.lon_min == -109.0
        assert config.lon_max == -105.0

    def test_custom_domain(self):
        config = ERA5Config(
            domain="custom",
            lat_min=50.0,
            lat_max=55.0,
            lon_min=10.0,
            lon_max=15.0,
        )
        assert config.lat_min == 50.0
        assert config.lat_max == 55.0

    def test_custom_domain_requires_bounds(self):
        with pytest.raises(ValueError, match="custom.*bounds"):
            ERA5Config(domain="custom")

    def test_invalid_domain(self):
        with pytest.raises(ValueError, match="Unknown domain"):
            ERA5Config(domain="mars")

    def test_predefined_domains_exist(self):
        assert "alpine" in PREDEFINED_DOMAINS
        assert "colorado" in PREDEFINED_DOMAINS


class TestERA5Data:
    """Test ERA5Data container."""

    def test_era5_data_creation(self):
        fields = {"T2M": np.zeros((10, 12))}
        data = ERA5Data(
            fields=fields,
            grid_shape=(10, 12),
            valid_time="2020-01-01T00:00:00Z",
            variables=["T2M"],
        )
        assert data.grid_shape == (10, 12)
        assert "T2M" in data.fields


class TestERA5ConditionerInit:
    """Test ERA5Conditioner initialization and configuration."""

    def test_conditioner_creation(self):
        config = ERA5Config(domain="alpine")
        conditioner = ERA5Conditioner(config)
        assert conditioner.config is config

    def test_conditioner_default_variables(self):
        config = ERA5Config(domain="alpine")
        conditioner = ERA5Conditioner(config)
        expected = {"T2M", "U10", "V10", "PRATE", "SP", "BLH"}
        assert set(conditioner.config.variables) == expected


class TestCDSAPIMocking:
    """Test CDS API interaction with mocked responses."""

    def test_download_creates_request(self, tmp_path):
        config = ERA5Config(domain="alpine")
        conditioner = ERA5Conditioner(config, output_dir=tmp_path)

        mock_client = MagicMock()
        # Simulate CDS API: the retrieve method writes a file
        dummy_ds = xr.Dataset(
            {
                "t2m": (["time", "latitude", "longitude"],
                        np.random.rand(1, 13, 25).astype(np.float32)),
                "u10": (["time", "latitude", "longitude"],
                        np.random.rand(1, 13, 25).astype(np.float32)),
                "v10": (["time", "latitude", "longitude"],
                        np.random.rand(1, 13, 25).astype(np.float32)),
                "tp": (["time", "latitude", "longitude"],
                       np.random.rand(1, 13, 25).astype(np.float32)),
                "sp": (["time", "latitude", "longitude"],
                       np.random.rand(1, 13, 25).astype(np.float32)),
                "blh": (["time", "latitude", "longitude"],
                        np.random.rand(1, 13, 25).astype(np.float32)),
            },
            coords={
                "time": [np.datetime64("2020-01-01T00:00")],
                "latitude": np.linspace(48, 45, 13),
                "longitude": np.linspace(5, 11, 25),
            },
        )
        out_file = tmp_path / "era5_test.nc"
        dummy_ds.to_netcdf(out_file)

        def mock_retrieve(dataset, request, target):
            import shutil
            shutil.copy(out_file, target)

        mock_client.retrieve = mock_retrieve

        result = conditioner.download_timestep(
            datetime(2020, 1, 1, tzinfo=UTC),
            cds_client=mock_client,
        )
        assert result is not None

    def test_download_builds_correct_cds_request(self, tmp_path):
        """Verify the CDS request parameters match the Alpine domain."""
        config = ERA5Config(domain="alpine")
        conditioner = ERA5Conditioner(config, output_dir=tmp_path)

        mock_client = MagicMock()
        captured_requests = []

        dummy_ds = xr.Dataset(
            {
                "t2m": (["time", "latitude", "longitude"],
                        np.random.rand(1, 13, 25).astype(np.float32)),
                "u10": (["time", "latitude", "longitude"],
                        np.random.rand(1, 13, 25).astype(np.float32)),
                "v10": (["time", "latitude", "longitude"],
                        np.random.rand(1, 13, 25).astype(np.float32)),
                "tp": (["time", "latitude", "longitude"],
                       np.random.rand(1, 13, 25).astype(np.float32)),
                "sp": (["time", "latitude", "longitude"],
                       np.random.rand(1, 13, 25).astype(np.float32)),
                "blh": (["time", "latitude", "longitude"],
                        np.random.rand(1, 13, 25).astype(np.float32)),
            },
            coords={
                "time": [np.datetime64("2020-01-01T00:00")],
                "latitude": np.linspace(48, 45, 13),
                "longitude": np.linspace(5, 11, 25),
            },
        )
        out_file = tmp_path / "era5_test.nc"
        dummy_ds.to_netcdf(out_file)

        def mock_retrieve(dataset, request, target):
            import shutil
            captured_requests.append((dataset, request))
            shutil.copy(out_file, target)

        mock_client.retrieve = mock_retrieve

        conditioner.download_timestep(
            datetime(2020, 1, 1, tzinfo=UTC),
            cds_client=mock_client,
        )

        assert len(captured_requests) == 1
        dataset, request = captured_requests[0]
        assert dataset == "reanalysis-era5-single-levels"
        # Area should be [north, west, south, east]
        assert request["area"] == [48.0, 5.0, 45.0, 11.0]
        assert "2m_temperature" in request["variable"]
        assert "10m_u_component_of_wind" in request["variable"]

    def test_beta_api_support(self, tmp_path):
        """Verify beta CDS API uses correct dataset name."""
        config = ERA5Config(domain="alpine", cds_api_version="beta")
        conditioner = ERA5Conditioner(config, output_dir=tmp_path)

        mock_client = MagicMock()
        captured = []

        dummy_ds = xr.Dataset(
            {
                "t2m": (["time", "latitude", "longitude"],
                        np.random.rand(1, 13, 25).astype(np.float32)),
                "u10": (["time", "latitude", "longitude"],
                        np.random.rand(1, 13, 25).astype(np.float32)),
                "v10": (["time", "latitude", "longitude"],
                        np.random.rand(1, 13, 25).astype(np.float32)),
                "tp": (["time", "latitude", "longitude"],
                       np.random.rand(1, 13, 25).astype(np.float32)),
                "sp": (["time", "latitude", "longitude"],
                       np.random.rand(1, 13, 25).astype(np.float32)),
                "blh": (["time", "latitude", "longitude"],
                        np.random.rand(1, 13, 25).astype(np.float32)),
            },
            coords={
                "time": [np.datetime64("2020-01-01T00:00")],
                "latitude": np.linspace(48, 45, 13),
                "longitude": np.linspace(5, 11, 25),
            },
        )
        out_file = tmp_path / "era5_test.nc"
        dummy_ds.to_netcdf(out_file)

        def mock_retrieve(dataset, request, target):
            import shutil
            captured.append(dataset)
            shutil.copy(out_file, target)

        mock_client.retrieve = mock_retrieve

        conditioner.download_timestep(
            datetime(2020, 1, 1, tzinfo=UTC),
            cds_client=mock_client,
        )
        assert captured[0] == "reanalysis-era5-single-levels"


# -------------------------------------------------------------------------
# 2.2 — Variable extraction and mapping
# -------------------------------------------------------------------------


class TestVariableMapping:
    """Test ERA5 short name -> project schema mapping."""

    def test_variable_map_keys(self):
        """All 6 ERA5 short names are mapped."""
        assert "2t" in CDS_VARIABLE_MAP
        assert "10u" in CDS_VARIABLE_MAP
        assert "10v" in CDS_VARIABLE_MAP
        assert "tp" in CDS_VARIABLE_MAP
        assert "sp" in CDS_VARIABLE_MAP
        assert "blh" in CDS_VARIABLE_MAP

    def test_variable_map_values(self):
        """Mapped to correct project schema names."""
        assert CDS_VARIABLE_MAP["2t"] == "T2M"
        assert CDS_VARIABLE_MAP["10u"] == "U10"
        assert CDS_VARIABLE_MAP["10v"] == "V10"
        assert CDS_VARIABLE_MAP["tp"] == "PRATE"
        assert CDS_VARIABLE_MAP["sp"] == "SP"
        assert CDS_VARIABLE_MAP["blh"] == "BLH"

    def test_extract_variables_from_dataset(self, tmp_path):
        """Extract and rename ERA5 variables to project schema."""
        config = ERA5Config(domain="alpine")
        conditioner = ERA5Conditioner(config, output_dir=tmp_path)

        # Create a mock ERA5 xarray dataset with ERA5 short names
        ds = xr.Dataset(
            {
                "t2m": (["time", "latitude", "longitude"],
                        np.full((2, 5, 5), 280.0, dtype=np.float32)),
                "u10": (["time", "latitude", "longitude"],
                        np.full((2, 5, 5), 3.0, dtype=np.float32)),
                "v10": (["time", "latitude", "longitude"],
                        np.full((2, 5, 5), -1.0, dtype=np.float32)),
                "tp": (["time", "latitude", "longitude"],
                       np.array([[[0.001] * 5] * 5,
                                 [[0.003] * 5] * 5], dtype=np.float32)),
                "sp": (["time", "latitude", "longitude"],
                       np.full((2, 5, 5), 85000.0, dtype=np.float32)),
                "blh": (["time", "latitude", "longitude"],
                        np.full((2, 5, 5), 1500.0, dtype=np.float32)),
            },
            coords={
                "time": [
                    np.datetime64("2020-01-01T00:00"),
                    np.datetime64("2020-01-01T01:00"),
                ],
                "latitude": np.linspace(48, 45, 5),
                "longitude": np.linspace(5, 11, 5),
            },
        )

        result = conditioner.extract_variables(ds, time_idx=1)
        assert "T2M" in result.fields
        assert "U10" in result.fields
        assert "V10" in result.fields
        assert "PRATE" in result.fields
        assert "SP" in result.fields
        assert "BLH" in result.fields
        assert result.grid_shape == (5, 5)

    def test_precipitation_accumulated_to_rate(self, tmp_path):
        """ERA5 accumulated precip is differenced to get instantaneous rate."""
        config = ERA5Config(domain="alpine")
        conditioner = ERA5Conditioner(config, output_dir=tmp_path)

        # ERA5 tp is accumulated from the start of the forecast.
        # Consecutive timesteps: t0=0.001m, t1=0.003m -> rate = 0.002 m/hr
        # Converted to mm/hr: 0.002 * 1000 = 2.0 mm/hr
        ds = xr.Dataset(
            {
                "t2m": (["time", "latitude", "longitude"],
                        np.full((2, 3, 3), 280.0, dtype=np.float32)),
                "u10": (["time", "latitude", "longitude"],
                        np.full((2, 3, 3), 3.0, dtype=np.float32)),
                "v10": (["time", "latitude", "longitude"],
                        np.full((2, 3, 3), -1.0, dtype=np.float32)),
                "tp": (["time", "latitude", "longitude"],
                       np.array([[[0.001] * 3] * 3,
                                 [[0.003] * 3] * 3], dtype=np.float32)),
                "sp": (["time", "latitude", "longitude"],
                       np.full((2, 3, 3), 85000.0, dtype=np.float32)),
                "blh": (["time", "latitude", "longitude"],
                        np.full((2, 3, 3), 1500.0, dtype=np.float32)),
            },
            coords={
                "time": [
                    np.datetime64("2020-01-01T00:00"),
                    np.datetime64("2020-01-01T01:00"),
                ],
                "latitude": np.linspace(48, 46, 3),
                "longitude": np.linspace(5, 7, 3),
            },
        )

        result = conditioner.extract_variables(ds, time_idx=1)
        # (0.003 - 0.001) m * 1000 = 2.0 mm/hr
        np.testing.assert_allclose(result.fields["PRATE"], 2.0, atol=1e-5)

    def test_precipitation_first_timestep_uses_absolute(self, tmp_path):
        """First timestep has no predecessor; use absolute accumulated value."""
        config = ERA5Config(domain="alpine")
        conditioner = ERA5Conditioner(config, output_dir=tmp_path)

        ds = xr.Dataset(
            {
                "t2m": (["time", "latitude", "longitude"],
                        np.full((1, 3, 3), 280.0, dtype=np.float32)),
                "u10": (["time", "latitude", "longitude"],
                        np.full((1, 3, 3), 3.0, dtype=np.float32)),
                "v10": (["time", "latitude", "longitude"],
                        np.full((1, 3, 3), -1.0, dtype=np.float32)),
                "tp": (["time", "latitude", "longitude"],
                       np.full((1, 3, 3), 0.005, dtype=np.float32)),
                "sp": (["time", "latitude", "longitude"],
                       np.full((1, 3, 3), 85000.0, dtype=np.float32)),
                "blh": (["time", "latitude", "longitude"],
                        np.full((1, 3, 3), 1500.0, dtype=np.float32)),
            },
            coords={
                "time": [np.datetime64("2020-01-01T00:00")],
                "latitude": np.linspace(48, 46, 3),
                "longitude": np.linspace(5, 7, 3),
            },
        )

        result = conditioner.extract_variables(ds, time_idx=0)
        # First timestep: 0.005 m * 1000 = 5.0 mm/hr
        np.testing.assert_allclose(result.fields["PRATE"], 5.0, atol=1e-5)


# -------------------------------------------------------------------------
# 2.3 — Temporal alignment
# -------------------------------------------------------------------------


class TestTemporalAlignment:
    """Test ERA5 hourly <-> station observation temporal matching."""

    def test_exact_hour_match(self):
        """Station observation exactly on the hour matches that ERA5 step."""
        aligner = ERA5TemporalAligner(tolerance_minutes=30)

        era5_times = [
            datetime(2020, 1, 1, 0, tzinfo=UTC),
            datetime(2020, 1, 1, 1, tzinfo=UTC),
            datetime(2020, 1, 1, 2, tzinfo=UTC),
        ]
        obs_time = datetime(2020, 1, 1, 1, tzinfo=UTC)

        matched = aligner.find_nearest(obs_time, era5_times)
        assert matched == datetime(2020, 1, 1, 1, tzinfo=UTC)

    def test_nearest_within_tolerance(self):
        """Observation 15 min past the hour matches the nearest hour."""
        aligner = ERA5TemporalAligner(tolerance_minutes=30)

        era5_times = [
            datetime(2020, 1, 1, 0, tzinfo=UTC),
            datetime(2020, 1, 1, 1, tzinfo=UTC),
            datetime(2020, 1, 1, 2, tzinfo=UTC),
        ]
        obs_time = datetime(2020, 1, 1, 1, 15, tzinfo=UTC)

        matched = aligner.find_nearest(obs_time, era5_times)
        assert matched == datetime(2020, 1, 1, 1, tzinfo=UTC)

    def test_observation_before_hour_matches(self):
        """Observation 20 min before the hour matches the nearest hour."""
        aligner = ERA5TemporalAligner(tolerance_minutes=30)

        era5_times = [
            datetime(2020, 1, 1, 0, tzinfo=UTC),
            datetime(2020, 1, 1, 1, tzinfo=UTC),
        ]
        obs_time = datetime(2020, 1, 1, 0, 40, tzinfo=UTC)

        matched = aligner.find_nearest(obs_time, era5_times)
        assert matched == datetime(2020, 1, 1, 1, tzinfo=UTC)

    def test_out_of_tolerance_raises(self):
        """Observation beyond tolerance raises ValueError."""
        aligner = ERA5TemporalAligner(tolerance_minutes=30)

        era5_times = [
            datetime(2020, 1, 1, 0, tzinfo=UTC),
            datetime(2020, 1, 1, 2, tzinfo=UTC),  # 2-hour gap
        ]
        obs_time = datetime(2020, 1, 1, 1, tzinfo=UTC)

        with pytest.raises(ValueError, match="tolerance"):
            aligner.find_nearest(obs_time, era5_times)

    def test_custom_tolerance(self):
        """Configurable tolerance of 15 minutes."""
        aligner = ERA5TemporalAligner(tolerance_minutes=15)

        era5_times = [
            datetime(2020, 1, 1, 0, tzinfo=UTC),
            datetime(2020, 1, 1, 1, tzinfo=UTC),
        ]

        # 10 min from nearest: within 15-min tolerance
        obs_ok = datetime(2020, 1, 1, 0, 10, tzinfo=UTC)
        assert aligner.find_nearest(obs_ok, era5_times) is not None

        # 20 min from nearest: outside 15-min tolerance
        obs_bad = datetime(2020, 1, 1, 0, 20, tzinfo=UTC)
        with pytest.raises(ValueError, match="tolerance"):
            aligner.find_nearest(obs_bad, era5_times)

    def test_peakweather_10min_alignment(self):
        """PeakWeather 10-min obs: each maps to nearest ERA5 hour."""
        aligner = ERA5TemporalAligner(tolerance_minutes=30)

        era5_times = [
            datetime(2020, 1, 1, h, tzinfo=UTC)
            for h in range(24)
        ]

        # 10-min observations around 14:00
        obs_times = [
            datetime(2020, 1, 1, 13, 50, tzinfo=UTC),
            datetime(2020, 1, 1, 14, 0, tzinfo=UTC),
            datetime(2020, 1, 1, 14, 10, tzinfo=UTC),
            datetime(2020, 1, 1, 14, 20, tzinfo=UTC),
        ]

        matched = [aligner.find_nearest(t, era5_times) for t in obs_times]
        assert matched[0] == datetime(2020, 1, 1, 14, tzinfo=UTC)
        assert matched[1] == datetime(2020, 1, 1, 14, tzinfo=UTC)
        assert matched[2] == datetime(2020, 1, 1, 14, tzinfo=UTC)
        assert matched[3] == datetime(2020, 1, 1, 14, tzinfo=UTC)

    def test_imis_30min_alignment(self):
        """IMIS 30-min obs: :00 matches exact, :30 equidistant picks earlier."""
        aligner = ERA5TemporalAligner(tolerance_minutes=30)

        era5_times = [
            datetime(2020, 1, 1, h, tzinfo=UTC)
            for h in range(24)
        ]

        obs_on_hour = datetime(2020, 1, 1, 12, 0, tzinfo=UTC)
        obs_half_hour = datetime(2020, 1, 1, 12, 30, tzinfo=UTC)

        assert aligner.find_nearest(obs_on_hour, era5_times) == \
            datetime(2020, 1, 1, 12, tzinfo=UTC)
        # At exactly 30 min from both 12:00 and 13:00, the earlier match
        # is returned (first encountered). Both are within tolerance.
        matched = aligner.find_nearest(obs_half_hour, era5_times)
        assert matched in (
            datetime(2020, 1, 1, 12, tzinfo=UTC),
            datetime(2020, 1, 1, 13, tzinfo=UTC),
        )

    def test_align_batch(self):
        """Batch alignment returns dict of obs_time -> era5_time pairs."""
        aligner = ERA5TemporalAligner(tolerance_minutes=30)

        era5_times = [
            datetime(2020, 1, 1, h, tzinfo=UTC)
            for h in range(24)
        ]
        obs_times = [
            datetime(2020, 1, 1, 0, 5, tzinfo=UTC),
            datetime(2020, 1, 1, 1, 10, tzinfo=UTC),
            datetime(2020, 1, 1, 2, 20, tzinfo=UTC),
        ]

        pairs = aligner.align_batch(obs_times, era5_times)
        assert len(pairs) == 3
        for era5_t in pairs.values():
            assert isinstance(era5_t, datetime)

    def test_empty_era5_times_raises(self):
        """Empty ERA5 times list raises ValueError."""
        aligner = ERA5TemporalAligner(tolerance_minutes=30)
        with pytest.raises(ValueError, match="empty"):
            aligner.find_nearest(
                datetime(2020, 1, 1, tzinfo=UTC), []
            )


# -------------------------------------------------------------------------
# 2.4 — Resumable download with checkpoint tracking
# -------------------------------------------------------------------------


class TestResumableDownload:
    """Test checkpoint-based resumable downloads (reusing DownloadCheckpoint)."""

    def test_checkpoint_skips_completed_timestep(self, tmp_path):
        """Already-downloaded timesteps are skipped on re-invocation."""
        config = ERA5Config(domain="alpine")
        checkpoint_path = tmp_path / ".era5_checkpoint.json"
        conditioner = ERA5Conditioner(
            config,
            output_dir=tmp_path,
            checkpoint_path=checkpoint_path,
        )

        # Pre-populate checkpoint with one completed timestep
        checkpoint = DownloadCheckpoint(checkpoint_path)
        checkpoint.mark_complete("era5_2020-01-01T00", "abc123")
        checkpoint.save()

        # Reload conditioner to pick up checkpoint
        conditioner = ERA5Conditioner(
            config,
            output_dir=tmp_path,
            checkpoint_path=checkpoint_path,
        )

        assert conditioner.is_timestep_complete(
            datetime(2020, 1, 1, 0, tzinfo=UTC)
        )
        assert not conditioner.is_timestep_complete(
            datetime(2020, 1, 1, 1, tzinfo=UTC)
        )

    def test_download_marks_checkpoint_on_success(self, tmp_path):
        """Successful download marks the timestep complete in checkpoint."""
        config = ERA5Config(domain="alpine")
        checkpoint_path = tmp_path / ".era5_checkpoint.json"
        conditioner = ERA5Conditioner(
            config,
            output_dir=tmp_path,
            checkpoint_path=checkpoint_path,
        )

        # Create a mock CDS client that writes a valid file
        mock_client = MagicMock()
        dummy_ds = xr.Dataset(
            {
                "t2m": (["time", "latitude", "longitude"],
                        np.random.rand(1, 5, 5).astype(np.float32)),
                "u10": (["time", "latitude", "longitude"],
                        np.random.rand(1, 5, 5).astype(np.float32)),
                "v10": (["time", "latitude", "longitude"],
                        np.random.rand(1, 5, 5).astype(np.float32)),
                "tp": (["time", "latitude", "longitude"],
                       np.random.rand(1, 5, 5).astype(np.float32)),
                "sp": (["time", "latitude", "longitude"],
                       np.random.rand(1, 5, 5).astype(np.float32)),
                "blh": (["time", "latitude", "longitude"],
                        np.random.rand(1, 5, 5).astype(np.float32)),
            },
            coords={
                "time": [np.datetime64("2020-01-01T00:00")],
                "latitude": np.linspace(48, 45, 5),
                "longitude": np.linspace(5, 11, 5),
            },
        )
        out_file = tmp_path / "era5_mock.nc"
        dummy_ds.to_netcdf(out_file)

        def mock_retrieve(dataset, request, target):
            import shutil
            shutil.copy(out_file, target)

        mock_client.retrieve = mock_retrieve

        ts = datetime(2020, 1, 1, 0, tzinfo=UTC)
        conditioner.download_timestep(ts, cds_client=mock_client)

        assert conditioner.is_timestep_complete(ts)

    def test_checkpoint_persists_to_disk(self, tmp_path):
        """Checkpoint state survives across conditioner instances."""
        config = ERA5Config(domain="alpine")
        checkpoint_path = tmp_path / ".era5_checkpoint.json"

        # First instance marks a timestep
        c1 = ERA5Conditioner(
            config,
            output_dir=tmp_path,
            checkpoint_path=checkpoint_path,
        )

        mock_client = MagicMock()
        dummy_ds = xr.Dataset(
            {
                "t2m": (["time", "latitude", "longitude"],
                        np.random.rand(1, 5, 5).astype(np.float32)),
                "u10": (["time", "latitude", "longitude"],
                        np.random.rand(1, 5, 5).astype(np.float32)),
                "v10": (["time", "latitude", "longitude"],
                        np.random.rand(1, 5, 5).astype(np.float32)),
                "tp": (["time", "latitude", "longitude"],
                       np.random.rand(1, 5, 5).astype(np.float32)),
                "sp": (["time", "latitude", "longitude"],
                       np.random.rand(1, 5, 5).astype(np.float32)),
                "blh": (["time", "latitude", "longitude"],
                        np.random.rand(1, 5, 5).astype(np.float32)),
            },
            coords={
                "time": [np.datetime64("2020-01-01T00:00")],
                "latitude": np.linspace(48, 45, 5),
                "longitude": np.linspace(5, 11, 5),
            },
        )
        out_file = tmp_path / "era5_mock.nc"
        dummy_ds.to_netcdf(out_file)

        def mock_retrieve(dataset, request, target):
            import shutil
            shutil.copy(out_file, target)

        mock_client.retrieve = mock_retrieve

        ts = datetime(2020, 1, 1, 0, tzinfo=UTC)
        c1.download_timestep(ts, cds_client=mock_client)

        # Second instance reads from disk
        c2 = ERA5Conditioner(
            config,
            output_dir=tmp_path,
            checkpoint_path=checkpoint_path,
        )
        assert c2.is_timestep_complete(ts)

    def test_timestep_key_format(self, tmp_path):
        """Verify the timestep key format matches expected pattern."""
        config = ERA5Config(domain="alpine")
        conditioner = ERA5Conditioner(config, output_dir=tmp_path)

        ts = datetime(2020, 6, 15, 12, tzinfo=UTC)
        key = conditioner._make_timestep_key(ts)
        assert key == "era5_2020-06-15T12"

    def test_download_skips_if_complete(self, tmp_path):
        """download_timestep returns None for already-completed timestep."""
        config = ERA5Config(domain="alpine")
        checkpoint_path = tmp_path / ".era5_checkpoint.json"

        checkpoint = DownloadCheckpoint(checkpoint_path)
        checkpoint.mark_complete("era5_2020-01-01T00", "abc123")
        checkpoint.save()

        conditioner = ERA5Conditioner(
            config,
            output_dir=tmp_path,
            checkpoint_path=checkpoint_path,
        )

        mock_client = MagicMock()
        result = conditioner.download_timestep(
            datetime(2020, 1, 1, 0, tzinfo=UTC),
            cds_client=mock_client,
        )
        assert result is None
        # CDS client should not have been called
        mock_client.retrieve.assert_not_called()
