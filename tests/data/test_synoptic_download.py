"""Tests for Synoptic wind data download script."""

import sys
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
import download_synoptic_wind as dws


@pytest.fixture
def mock_metadata_df():
    return pl.DataFrame({
        "stid": ["STA1", "STA2", "STA3", "AP001", "LOW1"],
        "name": ["High Peak", "Ridge Top", "Summit", "CWOP Bad", "Valley Low"],
        "elevation": [10000.0, 12000.0, 9000.0, 20000.0, 5000.0],
        "latitude": [39.0, 40.0, 38.5, 40.0, 39.5],
        "longitude": [-105.0, -106.0, -105.5, -105.0, -105.0],
        "shortname": ["RAWS", "CDOT", "SNOTEL", "APRSWXNET/CWOP", "RAWS"],
        "mnet_id": [2, 17, 5, 37, 2],
        "state": ["CO"] * 5,
        "elev_dem": [3048.0, 3657.0, 2743.0, 1600.0, 1524.0],
        "period_of_record_start": [None] * 5,
        "period_of_record_end": [None] * 5,
        "is_active": [True, True, True, True, True],
        "sensorvars": [None] * 5,
    })


class TestStationFiltering:
    def test_excludes_low_elevation(self, mock_metadata_df):
        """Stations below 8200ft should be filtered out."""
        filtered = mock_metadata_df.filter(
            (pl.col("elevation") > dws.MIN_ELEVATION_FT)
            & (pl.col("elevation") < dws.MAX_ELEVATION_FT)
            & (~pl.col("shortname").is_in(dws.EXCLUDED_NETWORKS))
        )
        stids = filtered["stid"].to_list()
        assert "LOW1" not in stids

    def test_excludes_cwop_bad_elevation(self, mock_metadata_df):
        """APRSWXNET/CWOP stations should be excluded (known bad elevation data)."""
        filtered = mock_metadata_df.filter(
            (pl.col("elevation") > dws.MIN_ELEVATION_FT)
            & (pl.col("elevation") < dws.MAX_ELEVATION_FT)
            & (~pl.col("shortname").is_in(dws.EXCLUDED_NETWORKS))
        )
        stids = filtered["stid"].to_list()
        assert "AP001" not in stids

    def test_keeps_valid_mountain_stations(self, mock_metadata_df):
        """Valid mountain stations should be kept."""
        filtered = mock_metadata_df.filter(
            (pl.col("elevation") > dws.MIN_ELEVATION_FT)
            & (pl.col("elevation") < dws.MAX_ELEVATION_FT)
            & (~pl.col("shortname").is_in(dws.EXCLUDED_NETWORKS))
        )
        stids = filtered["stid"].to_list()
        assert set(stids) == {"STA1", "STA2", "STA3"}


class TestSaveMetadata:
    def test_saves_csv_with_elevation_m(self, mock_metadata_df, tmp_path):
        with patch.object(dws, "DATA_DIR", tmp_path):
            stations = mock_metadata_df.filter(pl.col("stid").is_in(["STA1", "STA2"]))
            dws.save_station_metadata(stations)
            csv = tmp_path / "stations.csv"
            assert csv.exists()
            df = pl.read_csv(csv)
            assert "elevation_m" in df.columns
            assert "network" in df.columns
            assert df.shape[0] == 2


class TestSaveObservations:
    def test_pivots_and_saves(self, tmp_path):
        obs = pl.DataFrame({
            "stid": ["S1", "S1", "S1", "S2", "S2", "S2"],
            "date_time": [
                "2024-01-01T00:00:00+00:00",
                "2024-01-01T00:00:00+00:00",
                "2024-01-01T00:00:00+00:00",
                "2024-01-01T01:00:00+00:00",
                "2024-01-01T01:00:00+00:00",
                "2024-01-01T01:00:00+00:00",
            ],
            "variable": ["wind_speed", "wind_direction", "air_temp",
                          "wind_speed", "wind_direction", "air_temp"],
            "value": [5.0, 270.0, -3.0, 8.0, 180.0, -1.0],
            "units": ["m/s", "deg", "C", "m/s", "deg", "C"],
            "latitude": [39.0] * 6,
            "longitude": [-105.0] * 6,
            "elevation": [10000.0] * 6,
            "qc_flagged": [False] * 6,
        }).with_columns(
            pl.col("date_time").str.to_datetime(time_zone="UTC"),
        )

        with patch.object(dws, "DATA_DIR", tmp_path):
            dws.save_observations(obs, "_test")
            pq = tmp_path / "observations_test.parquet"
            assert pq.exists()
            result = pl.read_parquet(pq)
            assert "wind_speed" in result.columns
            assert "wind_direction" in result.columns
            assert result.shape[0] == 2


class TestConstants:
    def test_elevation_bounds(self):
        assert dws.MIN_ELEVATION_FT == 8200
        assert dws.MAX_ELEVATION_FT == 14500

    def test_excluded_networks(self):
        assert "APRSWXNET/CWOP" in dws.EXCLUDED_NETWORKS
