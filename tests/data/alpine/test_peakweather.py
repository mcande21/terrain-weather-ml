"""Tests for Task 2.1: PeakWeather HuggingFace download.

Verifies:
- Mocked HuggingFace responses return expected station count (302) and variables
- Incremental mode skips already-downloaded ranges
- Date range parsing and variable extraction
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from terrain_weather_ml.data.alpine.peakweather import PeakWeatherClient
from terrain_weather_ml.data.alpine.schema import Network, QualityFlag


def _make_synthetic_peakweather_df(
    n_stations: int = 302,
    n_timesteps: int = 6,
    start: datetime | None = None,
) -> pd.DataFrame:
    """Create synthetic PeakWeather data mirroring real schema."""
    start = start or datetime(2024, 1, 1, 0, 0)
    rows = []
    for i in range(n_stations):
        sid = f"STN{i:04d}"
        lat = 46.0 + (i % 50) * 0.01
        lon = 7.0 + (i // 50) * 0.01
        elev = 400.0 + i * 5
        for t in range(n_timesteps):
            ts = datetime(
                start.year, start.month, start.day,
                start.hour + (t * 10) // 60,
                (start.minute + t * 10) % 60,
            )
            rows.append({
                "stn": sid,
                "time": ts,
                "latitude": lat,
                "longitude": lon,
                "elevation": elev,
                "ta": 5.0 + i * 0.01,       # Celsius
                "fu": 3.0 + t * 0.1,         # m/s
                "dkl": 180.0 + t,            # degrees
                "rre": 0.1 * t,              # mm/10min
                "prestas": 950.0 - i * 0.1,  # hPa
                "ure": 65.0 + t,             # percent
                "gre": 200.0 + t * 10,       # W/m2
            })
    return pd.DataFrame(rows)


class MockHFDownloader:
    """Mock HuggingFace downloader for testing."""

    def __init__(self, data: pd.DataFrame | None = None):
        self.data = data if data is not None else _make_synthetic_peakweather_df()
        self.download_calls: list[tuple] = []

    def download(
        self,
        repo_id: str,
        start_date: datetime,
        end_date: datetime,
        local_dir: Path,
    ) -> list[Path]:
        self.download_calls.append((repo_id, start_date, end_date))
        # Write synthetic data to CSV (no pyarrow dependency)
        out_path = local_dir / "data.csv"
        self.data.to_csv(out_path, index=False)
        return [out_path]

    def list_files(self, repo_id: str) -> list[str]:
        return ["data.csv"]


class TestPeakWeatherClient:
    """Tests for PeakWeather download and parsing."""

    def test_parse_stations_returns_expected_count(self, tmp_path: Path):
        """Mocked data returns 302 stations."""
        df = _make_synthetic_peakweather_df(n_stations=302)
        client = PeakWeatherClient(
            cache_dir=tmp_path, downloader=MockHFDownloader(df)
        )
        stations = client.parse_stations(df)
        assert len(stations) == 302

    def test_parse_stations_all_fields_present(self, tmp_path: Path):
        """Every station has all metadata fields."""
        df = _make_synthetic_peakweather_df(n_stations=5)
        client = PeakWeatherClient(
            cache_dir=tmp_path, downloader=MockHFDownloader(df)
        )
        stations = client.parse_stations(df)
        for sid, meta in stations.items():
            assert meta.station_id == sid
            assert meta.network == Network.PEAKWEATHER
            assert meta.latitude is not None
            assert meta.longitude is not None
            assert meta.elevation is not None
            assert meta.terrain_ref_lat is not None
            assert meta.terrain_ref_lon is not None

    def test_parse_observations_extracts_all_variables(self, tmp_path: Path):
        """Parsed observations contain all expected weather variables."""
        df = _make_synthetic_peakweather_df(n_stations=1, n_timesteps=3)
        client = PeakWeatherClient(
            cache_dir=tmp_path, downloader=MockHFDownloader(df)
        )
        records = client.parse_observations(df)
        assert len(records) == 3

        rec = records[0]
        # All PeakWeather variables should be set (not NaN)
        assert rec.temperature == pytest.approx(5.0, abs=0.1)
        assert rec.wind_speed == pytest.approx(3.0, abs=0.1)
        assert rec.wind_direction == pytest.approx(180.0, abs=1)
        assert rec.pressure == pytest.approx(950.0, abs=1)
        assert rec.humidity == pytest.approx(65.0, abs=1)
        assert rec.global_radiation == pytest.approx(200.0, abs=10)

        # QC flags should be GOOD for present values
        assert rec.temperature_qc == QualityFlag.GOOD
        assert rec.wind_speed_qc == QualityFlag.GOOD

    def test_download_incremental_skips_existing(self, tmp_path: Path):
        """Incremental mode skips already-downloaded ranges."""
        mock = MockHFDownloader()
        client = PeakWeatherClient(cache_dir=tmp_path, downloader=mock)

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)

        # First download
        client.download(start, end)
        assert len(mock.download_calls) == 1

        # Second download of same range -- should skip
        mock.download_calls.clear()
        client.download(start, end)
        assert len(mock.download_calls) == 0

    def test_download_incremental_fetches_new_range(self, tmp_path: Path):
        """Incremental mode downloads only missing portions."""
        mock = MockHFDownloader()
        client = PeakWeatherClient(cache_dir=tmp_path, downloader=mock)

        # Download January
        client.download(datetime(2024, 1, 1), datetime(2024, 1, 31))
        assert len(mock.download_calls) == 1

        # Now request Jan-Feb -- should only download Feb
        mock.download_calls.clear()
        client.download(datetime(2024, 1, 1), datetime(2024, 2, 28))
        assert len(mock.download_calls) == 1
        _, call_start, _call_end = mock.download_calls[0]
        assert call_start == datetime(2024, 1, 31)

    def test_parse_stations_with_alternate_column_names(self, tmp_path: Path):
        """Handles alternate column names (stn, lat, lon, alt)."""
        df = pd.DataFrame([{
            "stn": "ALT1",
            "time": datetime(2024, 1, 1),
            "lat": 46.5,
            "lon": 7.5,
            "alt": 1200.0,
            "ta": 3.0,
        }])
        client = PeakWeatherClient(
            cache_dir=tmp_path, downloader=MockHFDownloader(df)
        )
        stations = client.parse_stations(df)
        assert "ALT1" in stations
        assert stations["ALT1"].latitude == 46.5
        assert stations["ALT1"].elevation == 1200.0
