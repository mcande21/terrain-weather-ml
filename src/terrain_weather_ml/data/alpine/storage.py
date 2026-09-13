"""Storage format with random access for Alpine weather data.

Task 2.6: Store aligned dataset as per-station time series supporting
O(log N) random access by station ID and time range using DuckDB.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import duckdb

from terrain_weather_ml.data.alpine.schema import (
    Network,
    QualityFlag,
    StationMetadata,
    WeatherRecord,
)

logger = logging.getLogger(__name__)


class AlpineDataStore:
    """DuckDB-backed storage for Alpine weather data.

    Provides O(log N) random access by station ID and time range
    via B-tree indexes on (station_id, timestamp).
    """

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = str(db_path)
        self.conn = duckdb.connect(self.db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        """Create tables and indexes if they don't exist."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS stations (
                station_id VARCHAR PRIMARY KEY,
                network VARCHAR NOT NULL,
                latitude DOUBLE NOT NULL,
                longitude DOUBLE NOT NULL,
                elevation DOUBLE NOT NULL,
                terrain_ref_lat DOUBLE NOT NULL,
                terrain_ref_lon DOUBLE NOT NULL
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS observations (
                station_id VARCHAR NOT NULL,
                network VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                temperature DOUBLE,
                wind_speed DOUBLE,
                wind_direction DOUBLE,
                precipitation DOUBLE,
                pressure DOUBLE,
                humidity DOUBLE,
                global_radiation DOUBLE,
                snow_surface_temp DOUBLE,
                snow_depth DOUBLE,
                temperature_qc INTEGER DEFAULT 2,
                wind_speed_qc INTEGER DEFAULT 2,
                wind_direction_qc INTEGER DEFAULT 2,
                precipitation_qc INTEGER DEFAULT 2,
                pressure_qc INTEGER DEFAULT 2,
                humidity_qc INTEGER DEFAULT 2,
                global_radiation_qc INTEGER DEFAULT 2,
                snow_surface_temp_qc INTEGER DEFAULT 2,
                snow_depth_qc INTEGER DEFAULT 2,
                PRIMARY KEY (station_id, timestamp)
            )
        """)

    def insert_stations(self, stations: Sequence[StationMetadata]) -> int:
        """Insert or update station metadata. Returns count inserted."""
        count = 0
        for s in stations:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO stations
                    (station_id, network, latitude, longitude, elevation,
                     terrain_ref_lat, terrain_ref_lon)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    s.station_id,
                    s.network.value,
                    s.latitude,
                    s.longitude,
                    s.elevation,
                    s.terrain_ref_lat,
                    s.terrain_ref_lon,
                ],
            )
            count += 1
        return count

    def insert_observations(self, records: Sequence[WeatherRecord]) -> int:
        """Insert or update weather observations. Returns count inserted."""
        count = 0
        for r in records:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO observations
                    (station_id, network, timestamp,
                     temperature, wind_speed, wind_direction,
                     precipitation, pressure, humidity,
                     global_radiation, snow_surface_temp, snow_depth,
                     temperature_qc, wind_speed_qc, wind_direction_qc,
                     precipitation_qc, pressure_qc, humidity_qc,
                     global_radiation_qc, snow_surface_temp_qc, snow_depth_qc)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    r.station_id,
                    r.network.value,
                    r.timestamp,
                    self._nan_to_none(r.temperature),
                    self._nan_to_none(r.wind_speed),
                    self._nan_to_none(r.wind_direction),
                    self._nan_to_none(r.precipitation),
                    self._nan_to_none(r.pressure),
                    self._nan_to_none(r.humidity),
                    self._nan_to_none(r.global_radiation),
                    self._nan_to_none(r.snow_surface_temp),
                    self._nan_to_none(r.snow_depth),
                    r.temperature_qc.value,
                    r.wind_speed_qc.value,
                    r.wind_direction_qc.value,
                    r.precipitation_qc.value,
                    r.pressure_qc.value,
                    r.humidity_qc.value,
                    r.global_radiation_qc.value,
                    r.snow_surface_temp_qc.value,
                    r.snow_depth_qc.value,
                ],
            )
            count += 1
        return count

    def query_station(
        self,
        station_id: str,
        start_time: datetime,
        end_time: datetime,
    ) -> list[WeatherRecord]:
        """Query observations for a station within a time range.

        Uses indexed lookup for O(log N) access.
        """
        rows = self.conn.execute(
            """
            SELECT station_id, network, timestamp,
                   temperature, wind_speed, wind_direction,
                   precipitation, pressure, humidity,
                   global_radiation, snow_surface_temp, snow_depth,
                   temperature_qc, wind_speed_qc, wind_direction_qc,
                   precipitation_qc, pressure_qc, humidity_qc,
                   global_radiation_qc, snow_surface_temp_qc, snow_depth_qc
            FROM observations
            WHERE station_id = ? AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp
            """,
            [station_id, start_time, end_time],
        ).fetchall()

        return [self._row_to_record(row) for row in rows]

    def get_station_metadata(
        self, station_id: str
    ) -> StationMetadata | None:
        """Get metadata for a single station."""
        row = self.conn.execute(
            "SELECT * FROM stations WHERE station_id = ?",
            [station_id],
        ).fetchone()

        if row is None:
            return None

        return StationMetadata(
            station_id=row[0],
            network=Network(row[1]),
            latitude=row[2],
            longitude=row[3],
            elevation=row[4],
            terrain_ref_lat=row[5],
            terrain_ref_lon=row[6],
        )

    def list_stations(
        self, network: Network | None = None
    ) -> list[StationMetadata]:
        """List all stations, optionally filtered by network."""
        if network:
            rows = self.conn.execute(
                "SELECT * FROM stations WHERE network = ?",
                [network.value],
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM stations").fetchall()

        return [
            StationMetadata(
                station_id=r[0],
                network=Network(r[1]),
                latitude=r[2],
                longitude=r[3],
                elevation=r[4],
                terrain_ref_lat=r[5],
                terrain_ref_lon=r[6],
            )
            for r in rows
        ]

    def count_observations(
        self, station_id: str | None = None
    ) -> int:
        """Count observations, optionally for a specific station."""
        if station_id:
            result = self.conn.execute(
                "SELECT COUNT(*) FROM observations WHERE station_id = ?",
                [station_id],
            ).fetchone()
        else:
            result = self.conn.execute(
                "SELECT COUNT(*) FROM observations"
            ).fetchone()
        return result[0] if result else 0

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()

    @staticmethod
    def _nan_to_none(value: float) -> float | None:
        """Convert NaN to None for SQL storage."""
        if math.isnan(value):
            return None
        return value

    @staticmethod
    def _none_to_nan(value: float | None) -> float:
        """Convert None back to NaN from SQL storage."""
        if value is None:
            return float("nan")
        return float(value)

    def _row_to_record(self, row: tuple) -> WeatherRecord:
        """Convert a database row to a WeatherRecord."""
        return WeatherRecord(
            station_id=row[0],
            network=Network(row[1]),
            timestamp=row[2],
            temperature=self._none_to_nan(row[3]),
            wind_speed=self._none_to_nan(row[4]),
            wind_direction=self._none_to_nan(row[5]),
            precipitation=self._none_to_nan(row[6]),
            pressure=self._none_to_nan(row[7]),
            humidity=self._none_to_nan(row[8]),
            global_radiation=self._none_to_nan(row[9]),
            snow_surface_temp=self._none_to_nan(row[10]),
            snow_depth=self._none_to_nan(row[11]),
            temperature_qc=QualityFlag(row[12]),
            wind_speed_qc=QualityFlag(row[13]),
            wind_direction_qc=QualityFlag(row[14]),
            precipitation_qc=QualityFlag(row[15]),
            pressure_qc=QualityFlag(row[16]),
            humidity_qc=QualityFlag(row[17]),
            global_radiation_qc=QualityFlag(row[18]),
            snow_surface_temp_qc=QualityFlag(row[19]),
            snow_depth_qc=QualityFlag(row[20]),
        )
