"""Tests for Reynolds Creek data pipeline (tasks 8.1-8.3).

Reynolds Creek Experimental Watershed: 21TB hourly gridded product at 10m
resolution. These tests validate variable/temporal subsetting, staged resumable
downloads, and NetCDF format conversion.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import numpy as np
import xarray as xr

# ---------------------------------------------------------------------------
# Task 8.1 — Variable and temporal subsetting
# ---------------------------------------------------------------------------


class TestVariableSubsetting:
    """Variable filter rejects files containing only excluded variables."""

    def test_core_variables_defined(self):
        """Pipeline should define exactly 6 core variables."""
        from terrain_weather_ml.data.reynolds_creek import CORE_VARIABLES

        assert len(CORE_VARIABLES) == 6
        expected = {
            "air_temperature",
            "wind_speed",
            "wind_direction",
            "precipitation",
            "relative_humidity",
            "shortwave_radiation",
        }
        assert set(CORE_VARIABLES.keys()) == expected

    def test_variable_filter_accepts_core_variable_files(self):
        """Files containing a core variable should pass the filter."""
        from terrain_weather_ml.data.reynolds_creek import ReynoldsCreekConfig

        config = ReynoldsCreekConfig()
        # A filename pattern like "rme_tair_2018.nc" contains air temperature
        assert config.is_variable_included("air_temperature")
        assert config.is_variable_included("wind_speed")

    def test_variable_filter_rejects_excluded_variables(self):
        """Files containing only excluded variables should be rejected."""
        from terrain_weather_ml.data.reynolds_creek import ReynoldsCreekConfig

        config = ReynoldsCreekConfig()
        assert not config.is_variable_included("snow_depth")
        assert not config.is_variable_included("soil_temperature")
        assert not config.is_variable_included("snow_water_equivalent")


class TestTemporalSubsetting:
    """Temporal bounds produce correct date ranges."""

    def test_default_five_water_years(self):
        """Default config should produce 5 most recent complete water years."""
        from terrain_weather_ml.data.reynolds_creek import ReynoldsCreekConfig

        config = ReynoldsCreekConfig()
        start, end = config.get_temporal_bounds()

        # Should span exactly 5 years
        # Water year N runs Oct 1 of year N-1 to Sep 30 of year N
        assert start.month == 10
        assert start.day == 1
        assert end.month == 9
        assert end.day == 30
        # 5 water years: WY N starts Oct of year N-1, ends Sep of year N.
        # WY 2016-2020: Oct 2015 to Sep 2020 = 5-year span.
        assert end.year - start.year == 5

    def test_custom_temporal_window(self):
        """Custom start/end dates should override defaults."""
        from terrain_weather_ml.data.reynolds_creek import ReynoldsCreekConfig

        config = ReynoldsCreekConfig(
            start_date=date(2010, 10, 1),
            end_date=date(2015, 9, 30),
        )
        start, end = config.get_temporal_bounds()
        assert start == date(2010, 10, 1)
        assert end == date(2015, 9, 30)

    def test_date_to_file_list(self):
        """Temporal bounds should generate a list of daily dates."""
        from terrain_weather_ml.data.reynolds_creek import ReynoldsCreekConfig

        config = ReynoldsCreekConfig(
            start_date=date(2018, 10, 1),
            end_date=date(2018, 10, 3),
        )
        dates = config.get_date_range()
        assert len(dates) == 3
        assert dates[0] == date(2018, 10, 1)
        assert dates[2] == date(2018, 10, 3)


class TestSpatialExtent:
    """Full spatial extent covers entire watershed boundary."""

    def test_full_watershed_domain(self):
        """Config should specify full Reynolds Creek spatial domain."""
        from terrain_weather_ml.data.reynolds_creek import ReynoldsCreekConfig

        config = ReynoldsCreekConfig()
        bounds = config.spatial_bounds
        # Reynolds Creek is in southwestern Idaho
        # Approximate bounds: lat 43.0-43.2, lon -116.9--116.6
        assert bounds["lat_min"] < 43.1
        assert bounds["lat_max"] > 43.1
        assert bounds["lon_min"] < -116.7
        assert bounds["lon_max"] > -116.7

    def test_no_spatial_subsetting(self):
        """Config should not subset spatially (full domain)."""
        from terrain_weather_ml.data.reynolds_creek import ReynoldsCreekConfig

        config = ReynoldsCreekConfig()
        # spatial_subset should be False (full domain)
        assert config.spatial_subset is False


# ---------------------------------------------------------------------------
# Task 8.2 — Staged download with checkpointing
# ---------------------------------------------------------------------------


class TestStagedDownload:
    """Resumable download with checkpoint files and hash verification."""

    def test_checkpoint_tracks_completed_files(self, tmp_path):
        """Checkpoint should track which files have been downloaded."""
        from terrain_weather_ml.data.reynolds_creek import DownloadCheckpoint

        ckpt = DownloadCheckpoint(tmp_path / "checkpoint.json")
        ckpt.mark_complete("air_temperature/2018/air_temperature_20181001.nc", "abc123")
        ckpt.save()

        # Reload and verify
        ckpt2 = DownloadCheckpoint(tmp_path / "checkpoint.json")
        ckpt2.load()
        assert ckpt2.is_complete("air_temperature/2018/air_temperature_20181001.nc")

    def test_checkpoint_detects_incomplete_files(self, tmp_path):
        """Files not in checkpoint should be detected as incomplete."""
        from terrain_weather_ml.data.reynolds_creek import DownloadCheckpoint

        ckpt = DownloadCheckpoint(tmp_path / "checkpoint.json")
        ckpt.load()
        assert not ckpt.is_complete("air_temperature/2018/air_temperature_20181001.nc")

    def test_resume_skips_completed_files(self, tmp_path):
        """Restart should skip already-downloaded files."""
        from terrain_weather_ml.data.reynolds_creek import (
            DownloadCheckpoint,
            ReynoldsCreekDownloader,
        )

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Pre-populate checkpoint with one file
        ckpt = DownloadCheckpoint(tmp_path / "checkpoint.json")
        file_key = "air_temperature/2018/air_temperature_20181001.nc"
        ckpt.mark_complete(file_key, "hash1")
        ckpt.save()

        downloader = ReynoldsCreekDownloader(
            output_dir=output_dir,
            checkpoint_path=tmp_path / "checkpoint.json",
        )
        # The file marked complete should be skipped
        assert downloader.checkpoint.is_complete(file_key)

    def test_hash_mismatch_triggers_redownload(self, tmp_path):
        """File with hash mismatch should be flagged for re-download."""
        from terrain_weather_ml.data.reynolds_creek import DownloadCheckpoint

        ckpt = DownloadCheckpoint(tmp_path / "checkpoint.json")
        file_key = "air_temperature/2018/air_temperature_20181001.nc"
        ckpt.mark_complete(file_key, "expected_hash")
        ckpt.save()

        ckpt2 = DownloadCheckpoint(tmp_path / "checkpoint.json")
        ckpt2.load()
        # Verify with wrong hash should fail
        assert not ckpt2.verify_hash(file_key, "different_hash")
        # Verify with correct hash should pass
        assert ckpt2.verify_hash(file_key, "expected_hash")

    def test_download_progress_reporting(self, tmp_path):
        """Downloader should report progress (completed/total files)."""
        from terrain_weather_ml.data.reynolds_creek import DownloadCheckpoint

        ckpt = DownloadCheckpoint(tmp_path / "checkpoint.json")
        ckpt.mark_complete("file1.nc", "hash1")
        ckpt.mark_complete("file2.nc", "hash2")

        assert ckpt.completed_count == 2

    def test_simulated_interruption_and_resume(self, tmp_path):
        """Simulated interruption: mark some files done, verify resume state."""
        from terrain_weather_ml.data.reynolds_creek import (
            DownloadCheckpoint,
            ReynoldsCreekConfig,
            ReynoldsCreekDownloader,
        )

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        config = ReynoldsCreekConfig(
            start_date=date(2018, 10, 1),
            end_date=date(2018, 10, 3),
        )

        # Create downloader with some files already checkpointed
        ckpt_path = tmp_path / "checkpoint.json"
        ckpt = DownloadCheckpoint(ckpt_path)
        ckpt.mark_complete("air_temperature/2018/air_temperature_20181001.nc", "h1")
        ckpt.mark_complete("air_temperature/2018/air_temperature_20181002.nc", "h2")
        ckpt.save()

        downloader = ReynoldsCreekDownloader(
            output_dir=output_dir,
            checkpoint_path=ckpt_path,
            config=config,
        )

        # Plan should exclude already-downloaded files
        plan = downloader.plan_downloads()
        completed_keys = {f.file_key for f in plan if f.is_complete}
        pending_keys = {f.file_key for f in plan if not f.is_complete}

        assert "air_temperature/2018/air_temperature_20181001.nc" in completed_keys
        assert "air_temperature/2018/air_temperature_20181002.nc" in completed_keys
        # Day 3 for air_temperature should be pending
        assert "air_temperature/2018/air_temperature_20181003.nc" in pending_keys


# ---------------------------------------------------------------------------
# Task 8.3 — NetCDF format conversion
# ---------------------------------------------------------------------------


def _make_synthetic_raw_nc(path: Path, variable: str, dt: date):
    """Create a synthetic raw Reynolds Creek NetCDF for testing conversion.

    Mimics the raw hourly data with spatial dimensions matching a 10m grid
    subset (small for testing: 20x30).
    """
    n_hours = 24
    n_lat = 20
    n_lon = 30
    times = [datetime(dt.year, dt.month, dt.day, h) for h in range(n_hours)]
    lats = np.linspace(43.05, 43.15, n_lat)
    lons = np.linspace(-116.85, -116.70, n_lon)

    data = np.random.rand(n_hours, n_lat, n_lon).astype(np.float32)

    ds = xr.Dataset(
        {variable: (["time", "lat", "lon"], data)},
        coords={
            "time": times,
            "lat": lats,
            "lon": lons,
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)
    return ds


class TestNetCDFConversion:
    """NetCDF format conversion to per-variable daily files."""

    def test_output_directory_structure(self, tmp_path):
        """Converted files should follow {variable}/{year}/ structure."""
        from terrain_weather_ml.data.reynolds_creek import NetCDFConverter

        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "converted"

        # Create synthetic raw file
        raw_path = raw_dir / "raw_tair_20181001.nc"
        _make_synthetic_raw_nc(raw_path, "air_temperature", date(2018, 10, 1))

        converter = NetCDFConverter(output_dir=out_dir)
        result = converter.convert_file(
            raw_path,
            variable="air_temperature",
            target_date=date(2018, 10, 1),
        )

        expected = out_dir / "air_temperature" / "2018" / "air_temperature_20181001.nc"
        assert result == expected
        assert result.exists()

    def test_converted_file_has_correct_dimensions(self, tmp_path):
        """Converted NetCDF should have time=24, plus lat and lon dimensions."""
        from terrain_weather_ml.data.reynolds_creek import NetCDFConverter

        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "converted"

        raw_path = raw_dir / "raw_tair_20181001.nc"
        _make_synthetic_raw_nc(raw_path, "air_temperature", date(2018, 10, 1))

        converter = NetCDFConverter(output_dir=out_dir)
        result = converter.convert_file(
            raw_path,
            variable="air_temperature",
            target_date=date(2018, 10, 1),
        )

        ds = xr.open_dataset(result)
        assert "time" in ds.dims
        assert ds.sizes["time"] == 24
        assert "lat" in ds.dims
        assert "lon" in ds.dims
        ds.close()

    def test_converted_file_has_variable_metadata(self, tmp_path):
        """Converted NetCDF should include variable with standard attributes."""
        from terrain_weather_ml.data.reynolds_creek import NetCDFConverter

        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "converted"

        raw_path = raw_dir / "raw_tair_20181001.nc"
        _make_synthetic_raw_nc(raw_path, "air_temperature", date(2018, 10, 1))

        converter = NetCDFConverter(output_dir=out_dir)
        result = converter.convert_file(
            raw_path,
            variable="air_temperature",
            target_date=date(2018, 10, 1),
        )

        ds = xr.open_dataset(result)
        assert "air_temperature" in ds.data_vars
        # Should have units attribute
        assert "units" in ds["air_temperature"].attrs
        ds.close()

    def test_converted_spatial_matches_10m_grid(self, tmp_path):
        """Converted file spatial dimensions should match input 10m grid."""
        from terrain_weather_ml.data.reynolds_creek import NetCDFConverter

        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "converted"

        raw_path = raw_dir / "raw_wind_20181001.nc"
        raw_ds = _make_synthetic_raw_nc(
            raw_path, "wind_speed", date(2018, 10, 1)
        )

        converter = NetCDFConverter(output_dir=out_dir)
        result = converter.convert_file(
            raw_path,
            variable="wind_speed",
            target_date=date(2018, 10, 1),
        )

        ds = xr.open_dataset(result)
        assert ds.sizes["lat"] == raw_ds.sizes["lat"]
        assert ds.sizes["lon"] == raw_ds.sizes["lon"]
        ds.close()

    def test_batch_conversion(self, tmp_path):
        """Batch conversion should process multiple files."""
        from terrain_weather_ml.data.reynolds_creek import NetCDFConverter

        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "converted"

        # Create 3 days of data for one variable
        raw_files = []
        for day in [1, 2, 3]:
            dt = date(2018, 10, day)
            path = raw_dir / f"raw_tair_{dt.strftime('%Y%m%d')}.nc"
            _make_synthetic_raw_nc(path, "air_temperature", dt)
            raw_files.append((path, "air_temperature", dt))

        converter = NetCDFConverter(output_dir=out_dir)
        results = converter.convert_batch(raw_files)

        assert len(results) == 3
        for result in results:
            assert result.exists()
            assert "air_temperature" in str(result)
            assert "2018" in str(result)
