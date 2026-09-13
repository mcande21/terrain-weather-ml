"""Tests for PeakWeather HuggingFace download and parsing.

Verifies the client handles the REAL HuggingFace dataset schema:
- Station metadata: nat_abbr index, station_height, 13 terrain features
- Observations: MultiIndex columns (nat_abbr, param_short), datetime index
- Real MeteoSwiss parameter codes (tre200s0, fkl010z0, etc.)
- Unit conversions: Celsius->Kelvin, humidity %->fraction
- Vectorized observation parsing (no iterrows)
- File range filtering: observations vs metadata
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from terrain_weather_ml.data.alpine.peakweather import (
    PEAKWEATHER_VAR_MAP,
    DefaultHFDownloader,
    PeakWeatherClient,
)
from terrain_weather_ml.data.alpine.schema import Network, QualityFlag

# -- Terrain feature columns present in real stations.parquet --
TERRAIN_FEATURE_COLS = [
    "ASPECT_2000M_SIGRATIO1",
    "WE_DERIVATIVE_2000M_SIGRATIO1",
    "TPI_2000M",
    "SN_DERIVATIVE_10000M_SIGRATIO1",
    "dem",
    "SN_DERIVATIVE_2000M_SIGRATIO1",
    "SLOPE_10000M_SIGRATIO1",
    "ASPECT_10000M_SIGRATIO1",
    "SLOPE_2000M_SIGRATIO1",
    "STD_2000M",
    "STD_10000M",
    "TPI_10000M",
    "WE_DERIVATIVE_10000M_SIGRATIO1",
]

# Real MeteoSwiss parameter codes
REAL_PARAMS = [
    "tre200s0",   # temperature (Celsius)
    "fkl010z0",   # wind speed (m/s)
    "dkl010z0",   # wind direction (degrees)
    "rre150z0",   # precipitation (mm/10min)
    "prestas0",   # station pressure (hPa)
    "ure200s0",   # relative humidity (%)
    "sre000z0",   # sunshine duration (min)
]


def _make_real_station_df(n_stations: int = 302) -> pd.DataFrame:
    """Create synthetic station metadata matching the real PeakWeather schema.

    Real schema: DataFrame with nat_abbr as index, station_height column,
    13 pre-computed terrain features.
    """
    rows = {}
    for i in range(n_stations):
        sid = f"STN{i:04d}"
        rows[sid] = {
            "station_name": f"Station {i}",
            "latitude": 46.0 + (i % 50) * 0.01,
            "longitude": 7.0 + (i // 50) * 0.01,
            "station_height": 400.0 + i * 5,
            "swiss_easting": 2700000.0 + i * 100,
            "swiss_northing": 1200000.0 + i * 100,
            "ASPECT_2000M_SIGRATIO1": 30.0 + i * 0.1,
            "WE_DERIVATIVE_2000M_SIGRATIO1": -0.01 + i * 0.001,
            "TPI_2000M": -20.0 + i * 0.5,
            "SN_DERIVATIVE_10000M_SIGRATIO1": -0.02 + i * 0.001,
            "dem": 395.0 + i * 5,
            "SN_DERIVATIVE_2000M_SIGRATIO1": -0.03 + i * 0.001,
            "SLOPE_10000M_SIGRATIO1": 1.0 + i * 0.01,
            "ASPECT_10000M_SIGRATIO1": 20.0 + i * 0.1,
            "SLOPE_2000M_SIGRATIO1": 1.5 + i * 0.01,
            "STD_2000M": 5.0 + i * 0.1,
            "STD_10000M": 80.0 + i * 0.5,
            "TPI_10000M": -50.0 + i * 1.0,
            "WE_DERIVATIVE_10000M_SIGRATIO1": -0.01 + i * 0.001,
            "station_type": "meteo_station",
        }
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "nat_abbr"
    return df


def _make_real_observation_df(
    station_ids: list[str] | None = None,
    n_timesteps: int = 6,
    start: datetime | None = None,
    include_gust: bool = False,
) -> pd.DataFrame:
    """Create synthetic observations matching real MultiIndex column schema.

    Real schema: datetime index, MultiIndex columns (nat_abbr, param_short).
    """
    start = start or datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    station_ids = station_ids or ["STN0000", "STN0001", "STN0002"]

    params = list(REAL_PARAMS)
    if include_gust:
        params.append("fkl010z1")

    timestamps = pd.date_range(start=start, periods=n_timesteps, freq="10min")
    data: dict[tuple[str, str], list[float]] = {}

    for sid_idx, sid in enumerate(station_ids):
        for param in params:
            values = []
            for t in range(n_timesteps):
                if param == "tre200s0":
                    values.append(5.0 + sid_idx * 0.01)  # Celsius
                elif param == "fkl010z0":
                    values.append(3.0 + t * 0.1)
                elif param == "dkl010z0":
                    values.append(180.0 + t)
                elif param == "rre150z0":
                    values.append(0.1 * t)
                elif param == "prestas0":
                    values.append(950.0 - sid_idx * 0.1)
                elif param == "ure200s0":
                    values.append(65.0 + t)  # percent
                elif param == "sre000z0":
                    values.append(5.0 + t * 0.5)
                elif param == "fkl010z1":
                    values.append(8.0 + t * 0.2)
                else:
                    values.append(float("nan"))
            data[(sid, param)] = values

    col_tuples = [(sid, param) for sid in station_ids for param in params]
    columns = pd.MultiIndex.from_tuples(col_tuples, names=["nat_abbr", "param_short"])
    df = pd.DataFrame(data, index=timestamps, columns=columns)
    df.index.name = "datetime"
    return df


class MockHFDownloader:
    """Mock HuggingFace downloader for testing."""

    def __init__(self, data: pd.DataFrame | None = None):
        self.data = data if data is not None else _make_real_observation_df()
        self.download_calls: list[tuple] = []

    def download(
        self,
        repo_id: str,
        start_date: datetime,
        end_date: datetime,
        local_dir: Path,
    ) -> list[Path]:
        self.download_calls.append((repo_id, start_date, end_date))
        out_path = local_dir / "data.csv"
        self.data.to_csv(out_path)
        return [out_path]

    def list_files(self, repo_id: str) -> list[str]:
        return ["data/observations/2024.parquet"]


class TestParseStations:
    """Station metadata parsing from real schema."""

    def test_handles_index_as_station_id(self, tmp_path: Path):
        """Station ID comes from nat_abbr index, not a column."""
        df = _make_real_station_df(n_stations=5)
        assert df.index.name == "nat_abbr"
        assert "station_id" not in df.columns
        assert "stn" not in df.columns

        client = PeakWeatherClient(
            cache_dir=tmp_path, downloader=MockHFDownloader()
        )
        stations = client.parse_stations(df)
        assert len(stations) == 5
        assert "STN0000" in stations
        assert stations["STN0000"].station_id == "STN0000"

    def test_station_height_column(self, tmp_path: Path):
        """Elevation comes from station_height column, not elevation/alt/elev."""
        df = _make_real_station_df(n_stations=3)
        assert "station_height" in df.columns
        assert "elevation" not in df.columns

        client = PeakWeatherClient(
            cache_dir=tmp_path, downloader=MockHFDownloader()
        )
        stations = client.parse_stations(df)
        assert stations["STN0000"].elevation == pytest.approx(400.0)
        assert stations["STN0002"].elevation == pytest.approx(410.0)

    def test_returns_expected_count(self, tmp_path: Path):
        """Full dataset returns 302 stations."""
        df = _make_real_station_df(n_stations=302)
        client = PeakWeatherClient(
            cache_dir=tmp_path, downloader=MockHFDownloader()
        )
        stations = client.parse_stations(df)
        assert len(stations) == 302

    def test_all_core_fields_present(self, tmp_path: Path):
        """Every station has all required metadata fields."""
        df = _make_real_station_df(n_stations=5)
        client = PeakWeatherClient(
            cache_dir=tmp_path, downloader=MockHFDownloader()
        )
        stations = client.parse_stations(df)
        for sid, meta in stations.items():
            assert meta.station_id == sid
            assert meta.network == Network.PEAKWEATHER
            assert meta.latitude is not None
            assert meta.longitude is not None
            assert meta.elevation is not None
            assert meta.terrain_ref_lat == meta.latitude
            assert meta.terrain_ref_lon == meta.longitude

    def test_preserves_terrain_features(self, tmp_path: Path):
        """13 pre-computed terrain features are preserved as optional metadata."""
        df = _make_real_station_df(n_stations=3)
        client = PeakWeatherClient(
            cache_dir=tmp_path, downloader=MockHFDownloader()
        )
        stations = client.parse_stations(df)
        meta = stations["STN0000"]
        tf_dict = dict(meta.terrain_features)
        assert len(tf_dict) == 13
        for col in TERRAIN_FEATURE_COLS:
            assert col in tf_dict, f"Missing terrain feature: {col}"
        assert tf_dict["dem"] == pytest.approx(395.0)
        assert tf_dict["TPI_2000M"] == pytest.approx(-20.0)

    def test_preserves_station_name(self, tmp_path: Path):
        """Station name from real data is preserved."""
        df = _make_real_station_df(n_stations=3)
        client = PeakWeatherClient(
            cache_dir=tmp_path, downloader=MockHFDownloader()
        )
        stations = client.parse_stations(df)
        assert stations["STN0000"].station_name == "Station 0"


class TestParseObservations:
    """Observation parsing from real MultiIndex schema."""

    def test_multiindex_columns_parsed(self, tmp_path: Path):
        """Handles real MultiIndex (nat_abbr, param_short) columns."""
        obs = _make_real_observation_df(
            station_ids=["STN0000"], n_timesteps=3
        )
        assert isinstance(obs.columns, pd.MultiIndex)

        client = PeakWeatherClient(
            cache_dir=tmp_path, downloader=MockHFDownloader()
        )
        records = client.parse_observations(obs)
        assert len(records) == 3
        assert all(r.station_id == "STN0000" for r in records)

    def test_real_parameter_codes_mapped(self, tmp_path: Path):
        """Real MeteoSwiss codes (tre200s0 etc.) are mapped to schema fields."""
        obs = _make_real_observation_df(
            station_ids=["STN0000"], n_timesteps=1
        )
        client = PeakWeatherClient(
            cache_dir=tmp_path, downloader=MockHFDownloader()
        )
        records = client.parse_observations(obs)
        rec = records[0]

        # Temperature: 5.0 C -> 278.15 K
        assert rec.temperature == pytest.approx(278.15, abs=0.01)
        assert rec.wind_speed == pytest.approx(3.0, abs=0.1)
        assert rec.wind_direction == pytest.approx(180.0, abs=1)
        assert rec.precipitation == pytest.approx(0.0, abs=0.01)
        assert rec.pressure == pytest.approx(950.0, abs=1)
        # Humidity: 65% -> 0.65
        assert rec.humidity == pytest.approx(0.65, abs=0.01)
        # Sunshine, NOT global_radiation
        assert rec.sunshine == pytest.approx(5.0, abs=0.5)

    def test_unit_conversions_celsius_to_kelvin(self, tmp_path: Path):
        """Temperature is converted from Celsius to Kelvin."""
        obs = _make_real_observation_df(
            station_ids=["STN0000"], n_timesteps=2
        )
        client = PeakWeatherClient(
            cache_dir=tmp_path, downloader=MockHFDownloader()
        )
        records = client.parse_observations(obs)
        # Raw value is 5.0 C -> should be 278.15 K
        assert records[0].temperature == pytest.approx(278.15, abs=0.01)

    def test_unit_conversions_humidity_to_fraction(self, tmp_path: Path):
        """Humidity is converted from percentage to fraction."""
        obs = _make_real_observation_df(
            station_ids=["STN0000"], n_timesteps=2
        )
        client = PeakWeatherClient(
            cache_dir=tmp_path, downloader=MockHFDownloader()
        )
        records = client.parse_observations(obs)
        # Raw value is 65% -> should be 0.65
        assert records[0].humidity == pytest.approx(0.65, abs=0.01)
        # Second timestep: 66% -> 0.66
        assert records[1].humidity == pytest.approx(0.66, abs=0.01)

    def test_qc_flags_set_for_present_values(self, tmp_path: Path):
        """QC flags are GOOD for all present values."""
        obs = _make_real_observation_df(
            station_ids=["STN0000"], n_timesteps=1
        )
        client = PeakWeatherClient(
            cache_dir=tmp_path, downloader=MockHFDownloader()
        )
        records = client.parse_observations(obs)
        rec = records[0]
        assert rec.temperature_qc == QualityFlag.GOOD
        assert rec.wind_speed_qc == QualityFlag.GOOD
        assert rec.humidity_qc == QualityFlag.GOOD
        assert rec.sunshine_qc == QualityFlag.GOOD

    def test_wind_gust_optional(self, tmp_path: Path):
        """Wind gust (fkl010z1) is parsed when present."""
        obs = _make_real_observation_df(
            station_ids=["STN0000"], n_timesteps=1, include_gust=True
        )
        client = PeakWeatherClient(
            cache_dir=tmp_path, downloader=MockHFDownloader()
        )
        records = client.parse_observations(obs)
        assert records[0].wind_gust == pytest.approx(8.0, abs=0.1)
        assert records[0].wind_gust_qc == QualityFlag.GOOD

    def test_multiple_stations_produces_correct_records(self, tmp_path: Path):
        """MultiIndex with multiple stations produces records for each."""
        obs = _make_real_observation_df(
            station_ids=["STN0000", "STN0001", "STN0002"],
            n_timesteps=4,
        )
        client = PeakWeatherClient(
            cache_dir=tmp_path, downloader=MockHFDownloader()
        )
        records = client.parse_observations(obs)
        # 3 stations x 4 timesteps = 12 records
        assert len(records) == 12
        stn_ids = {r.station_id for r in records}
        assert stn_ids == {"STN0000", "STN0001", "STN0002"}

    def test_no_iterrows_used(self, tmp_path: Path):
        """parse_observations uses vectorized ops, not iterrows."""
        import inspect
        client = PeakWeatherClient(
            cache_dir=tmp_path, downloader=MockHFDownloader()
        )
        source = inspect.getsource(client.parse_observations)
        assert "iterrows" not in source, (
            "parse_observations should use vectorized operations, not iterrows()"
        )


class TestVarMap:
    """Variable mapping uses real MeteoSwiss parameter codes."""

    def test_real_meteoswiss_codes(self):
        """PEAKWEATHER_VAR_MAP uses real MeteoSwiss parameter codes."""
        assert "tre200s0" in PEAKWEATHER_VAR_MAP
        assert "fkl010z0" in PEAKWEATHER_VAR_MAP
        assert "dkl010z0" in PEAKWEATHER_VAR_MAP
        assert "rre150z0" in PEAKWEATHER_VAR_MAP
        assert "prestas0" in PEAKWEATHER_VAR_MAP
        assert "ure200s0" in PEAKWEATHER_VAR_MAP
        assert "sre000z0" in PEAKWEATHER_VAR_MAP

    def test_old_codes_removed(self):
        """Old wrong codes are no longer in the var map."""
        assert "ta" not in PEAKWEATHER_VAR_MAP
        assert "fu" not in PEAKWEATHER_VAR_MAP
        assert "dkl" not in PEAKWEATHER_VAR_MAP
        assert "rre" not in PEAKWEATHER_VAR_MAP
        assert "prestas" not in PEAKWEATHER_VAR_MAP
        assert "ure" not in PEAKWEATHER_VAR_MAP
        assert "gre" not in PEAKWEATHER_VAR_MAP

    def test_sunshine_not_global_radiation(self):
        """sre000z0 maps to sunshine, not global_radiation."""
        assert PEAKWEATHER_VAR_MAP["sre000z0"] == "sunshine"

    def test_wind_gust_in_map(self):
        """fkl010z1 (wind gust) is in the var map."""
        assert "fkl010z1" in PEAKWEATHER_VAR_MAP
        assert PEAKWEATHER_VAR_MAP["fkl010z1"] == "wind_gust"


class TestFileInRange:
    """File range filtering for real PeakWeather file organization."""

    def test_metadata_files_always_included(self):
        """stations.parquet, parameters.parquet, installation.parquet always pass."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)
        assert DefaultHFDownloader._file_in_range(
            "data/stations.parquet", start, end
        )
        assert DefaultHFDownloader._file_in_range(
            "data/parameters.parquet", start, end
        )
        assert DefaultHFDownloader._file_in_range(
            "data/installation.parquet", start, end
        )

    def test_observation_year_in_range(self):
        """Observation files within date range are included."""
        start = datetime(2023, 6, 1)
        end = datetime(2024, 6, 1)
        assert DefaultHFDownloader._file_in_range(
            "data/observations/2024.parquet", start, end
        )
        assert DefaultHFDownloader._file_in_range(
            "data/observations/2023.parquet", start, end
        )

    def test_observation_year_out_of_range(self):
        """Observation files outside date range are excluded."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)
        assert not DefaultHFDownloader._file_in_range(
            "data/observations/2020.parquet", start, end
        )
        assert not DefaultHFDownloader._file_in_range(
            "data/observations/2025.parquet", start, end
        )

    def test_old_csv_pattern_rejected(self):
        """Old generic CSV matching no longer applies."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)
        assert not DefaultHFDownloader._file_in_range(
            "random_data.csv", start, end
        )


class TestDownloadIncremental:
    """Incremental download behavior."""

    def test_skips_existing_range(self, tmp_path: Path):
        """Incremental mode skips already-downloaded ranges."""
        mock = MockHFDownloader()
        client = PeakWeatherClient(cache_dir=tmp_path, downloader=mock)

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)

        client.download(start, end)
        assert len(mock.download_calls) == 1

        mock.download_calls.clear()
        client.download(start, end)
        assert len(mock.download_calls) == 0

    def test_fetches_new_range(self, tmp_path: Path):
        """Incremental mode downloads only missing portions."""
        mock = MockHFDownloader()
        client = PeakWeatherClient(cache_dir=tmp_path, downloader=mock)

        client.download(datetime(2024, 1, 1), datetime(2024, 1, 31))
        assert len(mock.download_calls) == 1

        mock.download_calls.clear()
        client.download(datetime(2024, 1, 1), datetime(2024, 2, 28))
        assert len(mock.download_calls) == 1
        _, call_start, _ = mock.download_calls[0]
        assert call_start == datetime(2024, 1, 31)


class TestRealDataIntegration:
    """Integration tests using real PeakWeather sample data."""

    REAL_DATA_DIR = Path(
        "/Users/cooperanderson/work/personal/code/terrain-weather-ml"
        "/data/raw/peakweather/data"
    )

    @pytest.fixture()
    def real_stations(self) -> pd.DataFrame:
        path = self.REAL_DATA_DIR / "stations.parquet"
        if not path.exists():
            pytest.skip("Real PeakWeather data not available")
        return pd.read_parquet(path)

    @pytest.fixture()
    def real_observations(self) -> pd.DataFrame:
        path = self.REAL_DATA_DIR / "observations" / "2024.parquet"
        if not path.exists():
            pytest.skip("Real PeakWeather data not available")
        # Load only first 100 rows to keep test fast
        return pd.read_parquet(path).head(100)

    def test_real_stations_parse(self, tmp_path: Path, real_stations: pd.DataFrame):
        """Parse real stations.parquet through the fixed client."""
        client = PeakWeatherClient(
            cache_dir=tmp_path, downloader=MockHFDownloader()
        )
        stations = client.parse_stations(real_stations)
        assert len(stations) == 302

        # Spot-check a known station
        for sid, meta in stations.items():
            assert meta.station_id == sid
            assert meta.network == Network.PEAKWEATHER
            assert -90 <= meta.latitude <= 90
            assert -180 <= meta.longitude <= 180
            assert meta.elevation > 0
            assert len(dict(meta.terrain_features)) == 13
            break

    def test_real_observations_parse(
        self, tmp_path: Path, real_observations: pd.DataFrame
    ):
        """Parse real 2024.parquet observations through the fixed client."""
        client = PeakWeatherClient(
            cache_dir=tmp_path, downloader=MockHFDownloader()
        )
        records = client.parse_observations(real_observations)
        assert len(records) > 0

        rec = records[0]
        assert rec.network == Network.PEAKWEATHER
        assert isinstance(rec.timestamp, datetime)
        # Temperature should be in Kelvin (> 200)
        if not np.isnan(rec.temperature):
            assert rec.temperature > 200, (
                f"Temperature {rec.temperature} looks like Celsius, not Kelvin"
            )
        # Humidity should be fraction (0-1)
        if not np.isnan(rec.humidity):
            assert rec.humidity <= 1.0, (
                f"Humidity {rec.humidity} looks like percentage, not fraction"
            )
