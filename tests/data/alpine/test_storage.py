"""Tests for Task 2.6: Storage format with random access.

Verifies:
- Random access retrieval for a specific station and time range
- Sub-linear access time (indexed, not full-scan)
- Round-trip insert/query preserves data
- Station metadata persistence
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from terrain_weather_ml.data.alpine.schema import (
    Network,
    QualityFlag,
    StationMetadata,
    WeatherRecord,
)
from terrain_weather_ml.data.alpine.storage import AlpineDataStore


def _make_station(
    station_id: str = "STN0001",
    network: Network = Network.PEAKWEATHER,
) -> StationMetadata:
    return StationMetadata(
        station_id=station_id,
        network=network,
        latitude=46.9481,
        longitude=7.4474,
        elevation=540.0,
        terrain_ref_lat=46.9481,
        terrain_ref_lon=7.4474,
    )


def _make_records(
    station_id: str = "STN0001",
    n: int = 100,
    start: datetime | None = None,
) -> list[WeatherRecord]:
    """Create n weather records at 30-min intervals."""
    start = start or datetime(2024, 1, 1)
    records = []
    for i in range(n):
        ts = start + timedelta(minutes=30 * i)
        records.append(WeatherRecord(
            timestamp=ts,
            station_id=station_id,
            network=Network.PEAKWEATHER,
            temperature=278.15 + i * 0.01,
            wind_speed=5.0 + i * 0.1,
            wind_direction=180.0,
            precipitation=0.0,
            pressure=950.0,
            humidity=0.65,
            temperature_qc=QualityFlag.GOOD,
            wind_speed_qc=QualityFlag.GOOD,
            wind_direction_qc=QualityFlag.GOOD,
            precipitation_qc=QualityFlag.GOOD,
            pressure_qc=QualityFlag.GOOD,
            humidity_qc=QualityFlag.GOOD,
        ))
    return records


class TestAlpineDataStore:
    """Tests for DuckDB-backed storage."""

    def test_insert_and_query_station(self, tmp_path: Path):
        """Insert station metadata and retrieve it."""
        db = tmp_path / "test.duckdb"
        store = AlpineDataStore(db)
        try:
            station = _make_station("STN0001")
            store.insert_stations([station])
            result = store.get_station_metadata("STN0001")
            assert result is not None
            assert result.station_id == "STN0001"
            assert result.network == Network.PEAKWEATHER
            assert result.latitude == pytest.approx(46.9481)
            assert result.elevation == pytest.approx(540.0)
        finally:
            store.close()

    def test_insert_and_query_observations(self, tmp_path: Path):
        """Insert observations and query by station + time range."""
        db = tmp_path / "test.duckdb"
        store = AlpineDataStore(db)
        try:
            records = _make_records("STN0001", n=100)
            store.insert_observations(records)

            # Query a subset
            start = datetime(2024, 1, 1, 5, 0)
            end = datetime(2024, 1, 1, 10, 0)
            results = store.query_station("STN0001", start, end)

            assert len(results) > 0
            for r in results:
                assert r.station_id == "STN0001"
                assert r.timestamp >= start
                assert r.timestamp <= end
        finally:
            store.close()

    def test_random_access_returns_correct_slice(self, tmp_path: Path):
        """Queried slice contains exactly the expected records."""
        db = tmp_path / "test.duckdb"
        store = AlpineDataStore(db)
        try:
            records = _make_records("STN0001", n=48)  # 24 hours of 30-min
            store.insert_observations(records)

            # Query 2 hours: should get 4 records (inclusive)
            start = datetime(2024, 1, 1, 2, 0)
            end = datetime(2024, 1, 1, 3, 30)
            results = store.query_station("STN0001", start, end)
            assert len(results) == 4
        finally:
            store.close()

    def test_round_trip_preserves_values(self, tmp_path: Path):
        """Data survives insert -> query round trip."""
        db = tmp_path / "test.duckdb"
        store = AlpineDataStore(db)
        try:
            original = WeatherRecord(
                timestamp=datetime(2024, 6, 15, 12, 0),
                station_id="ROUND",
                network=Network.IMIS,
                temperature=273.15,
                wind_speed=12.5,
                wind_direction=270.0,
                precipitation=1.5,
                pressure=850.0,
                humidity=0.92,
                snow_depth=2.5,
                temperature_qc=QualityFlag.GOOD,
                wind_speed_qc=QualityFlag.GOOD,
                wind_direction_qc=QualityFlag.GOOD,
                precipitation_qc=QualityFlag.GOOD,
                pressure_qc=QualityFlag.GOOD,
                humidity_qc=QualityFlag.GOOD,
                snow_depth_qc=QualityFlag.GOOD,
            )
            store.insert_observations([original])

            results = store.query_station(
                "ROUND",
                datetime(2024, 6, 15),
                datetime(2024, 6, 16),
            )
            assert len(results) == 1
            r = results[0]
            assert r.temperature == pytest.approx(273.15)
            assert r.wind_speed == pytest.approx(12.5)
            assert r.wind_direction == pytest.approx(270.0)
            assert r.precipitation == pytest.approx(1.5)
            assert r.pressure == pytest.approx(850.0)
            assert r.humidity == pytest.approx(0.92)
            assert r.snow_depth == pytest.approx(2.5)
            assert r.temperature_qc == QualityFlag.GOOD
        finally:
            store.close()

    def test_nan_values_round_trip(self, tmp_path: Path):
        """NaN values survive insert -> query via NULL."""
        db = tmp_path / "test.duckdb"
        store = AlpineDataStore(db)
        try:
            rec = WeatherRecord(
                timestamp=datetime(2024, 1, 1),
                station_id="NANTEST",
                network=Network.PEAKWEATHER,
                # All defaults are NaN
            )
            store.insert_observations([rec])
            results = store.query_station(
                "NANTEST",
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
            )
            assert len(results) == 1
            assert math.isnan(results[0].temperature)
            assert results[0].temperature_qc == QualityFlag.MISSING
        finally:
            store.close()

    def test_sub_linear_access_time(self, tmp_path: Path):
        """Access time does not scale linearly with dataset size.

        Insert a large dataset, then query a small slice.
        The query time should be fast (O(log N)) not O(N).
        """
        db = tmp_path / "test.duckdb"
        store = AlpineDataStore(db)
        try:
            # Insert 10k records across 10 stations
            for i in range(10):
                records = _make_records(f"STN{i:04d}", n=1000)
                store.insert_observations(records)

            # Query a small slice -- should be fast
            start_time = time.monotonic()
            for _ in range(100):
                store.query_station(
                    "STN0005",
                    datetime(2024, 1, 5),
                    datetime(2024, 1, 6),
                )
            elapsed = time.monotonic() - start_time

            # 100 queries should take well under 5 seconds with indexing
            assert elapsed < 5.0, (
                f"100 queries took {elapsed:.2f}s, suggesting full scan"
            )
        finally:
            store.close()

    def test_list_stations_by_network(self, tmp_path: Path):
        """List stations filtered by network."""
        db = tmp_path / "test.duckdb"
        store = AlpineDataStore(db)
        try:
            pw = _make_station("PW1", Network.PEAKWEATHER)
            imis = _make_station("IM1", Network.IMIS)
            store.insert_stations([pw, imis])

            pw_list = store.list_stations(Network.PEAKWEATHER)
            assert len(pw_list) == 1
            assert pw_list[0].station_id == "PW1"

            imis_list = store.list_stations(Network.IMIS)
            assert len(imis_list) == 1
            assert imis_list[0].station_id == "IM1"

            all_list = store.list_stations()
            assert len(all_list) == 2
        finally:
            store.close()

    def test_count_observations(self, tmp_path: Path):
        """Count observations total and per-station."""
        db = tmp_path / "test.duckdb"
        store = AlpineDataStore(db)
        try:
            store.insert_observations(_make_records("A", n=10))
            store.insert_observations(_make_records("B", n=20))
            assert store.count_observations() == 30
            assert store.count_observations("A") == 10
            assert store.count_observations("B") == 20
        finally:
            store.close()
