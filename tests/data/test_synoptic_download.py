"""Tests for IEM wind data download script."""

import sys
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
import download_synoptic_wind as dws


@pytest.fixture
def mock_stations_df():
    return pl.DataFrame({
        "station_id": ["LXV", "TEX", "0CO", "LOW"],
        "name": ["Leadville", "Telluride", "Berthoud Pass", "Denver"],
        "latitude": [39.228, 37.954, 39.794, 39.856],
        "longitude": [-106.316, -107.909, -105.764, -104.674],
        "elevation_m": [3026.0, 2769.0, 3792.0, 1655.0],
        "network": ["CO_ASOS"] * 4,
        "online": [True, True, True, True],
        "archive_begin": [None] * 4,
        "archive_end": [None] * 4,
    })


@pytest.fixture
def mock_raw_obs():
    return pl.DataFrame({
        "station": ["LXV", "LXV", "LXV", "TEX", "TEX"],
        "valid": [
            "2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 02:00",
            "2024-01-01 00:00", "2024-01-01 01:00",
        ],
        "lon": [-106.316, -106.316, -106.316, -107.909, -107.909],
        "lat": [39.228, 39.228, 39.228, 37.954, 37.954],
        "elevation": [3026.0, 3026.0, 3026.0, 2769.0, 2769.0],
        "tmpf": ["32.0", "28.0", "M", "25.0", "M"],
        "sknt": ["10.0", "15.0", "8.0", "M", "12.0"],
        "drct": ["270.0", "250.0", "280.0", "M", "180.0"],
    })


class TestStationFiltering:
    def test_filters_by_elevation(self, mock_stations_df):
        mountain = mock_stations_df.filter(
            pl.col("elevation_m") >= dws.MIN_ELEVATION_M
        )
        assert "LOW" not in mountain["station_id"].to_list()

    def test_keeps_mountain_stations(self, mock_stations_df):
        mountain = mock_stations_df.filter(
            pl.col("elevation_m") >= dws.MIN_ELEVATION_M
        )
        assert set(mountain["station_id"].to_list()) == {"LXV", "TEX", "0CO"}


class TestConvertObservations:
    def test_converts_knots_to_ms(self, mock_raw_obs):
        result = dws.convert_observations(mock_raw_obs)
        ws = result.filter(pl.col("station_id") == "LXV").sort("date_time")["wind_speed"]
        assert abs(ws[0] - 10.0 * 0.514444) < 0.01

    def test_converts_fahrenheit_to_celsius(self, mock_raw_obs):
        result = dws.convert_observations(mock_raw_obs)
        temp = result.filter(pl.col("station_id") == "LXV").sort("date_time")["air_temp"]
        assert abs(temp[0] - 0.0) < 0.01  # 32F = 0C

    def test_missing_values_become_null(self, mock_raw_obs):
        result = dws.convert_observations(mock_raw_obs)
        lxv = result.filter(pl.col("station_id") == "LXV").sort("date_time")
        assert lxv["air_temp"][2] is None

        tex = result.filter(pl.col("station_id") == "TEX").sort("date_time")
        assert tex["wind_speed"][0] is None

    def test_output_columns(self, mock_raw_obs):
        result = dws.convert_observations(mock_raw_obs)
        expected = {
            "station_id", "date_time", "latitude", "longitude",
            "elevation_m", "wind_speed", "wind_direction", "air_temp",
        }
        assert set(result.columns) == expected

    def test_empty_input(self):
        result = dws.convert_observations(pl.DataFrame())
        assert result.is_empty()


class TestSaveMetadata:
    def test_saves_csv(self, mock_stations_df, tmp_path):
        with patch.object(dws, "DATA_DIR", tmp_path):
            mountain = mock_stations_df.filter(pl.col("elevation_m") >= 2500)
            dws.save_station_metadata(mountain)
            csv = tmp_path / "stations.csv"
            assert csv.exists()
            df = pl.read_csv(csv)
            assert df.shape[0] == 3
            assert "elevation_m" in df.columns


class TestSaveObservations:
    def test_saves_parquet_and_csv(self, mock_raw_obs, tmp_path):
        converted = dws.convert_observations(mock_raw_obs)
        with patch.object(dws, "DATA_DIR", tmp_path):
            dws.save_observations(converted, "_test")
            assert (tmp_path / "observations_test.parquet").exists()
            assert (tmp_path / "observations_test.csv").exists()
            result = pl.read_parquet(tmp_path / "observations_test.parquet")
            assert "wind_speed" in result.columns
            assert result.shape[0] == 5


class TestConstants:
    def test_elevation_threshold(self):
        assert dws.MIN_ELEVATION_M == 2500

    def test_network(self):
        assert dws.NETWORK == "CO_ASOS"
