"""Temporal alignment of Alpine weather data.

Task 2.4: Aggregate PeakWeather 10-min to configurable target resolution
(default 30-min) with correct aggregation rules per variable.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime

from terrain_weather_ml.data.alpine.schema import (
    QualityFlag,
    WeatherRecord,
)


class TemporalAligner:
    """Aligns weather records to a common temporal resolution.

    PeakWeather (10-min) is aggregated to the target resolution.
    IMIS (30-min) is passed through if target matches its native resolution.

    Aggregation rules:
    - Temperature, pressure, humidity: arithmetic mean
    - Wind: vector mean (decompose to u/v, average, recompose)
    - Precipitation: sum
    - Global radiation: mean
    - Snow depth, snow surface temp: mean

    Incomplete intervals (fewer than min_records of expected) are marked NaN.
    """

    def __init__(
        self,
        target_minutes: int = 30,
        min_fraction: float = 2 / 3,
    ) -> None:
        """Initialize the temporal aligner.

        Args:
            target_minutes: Target temporal resolution in minutes.
            min_fraction: Minimum fraction of expected records required
                to compute an aggregate. Below this, the interval is NaN.
                Default 2/3 (e.g., 2 of 3 for 10-min -> 30-min).
        """
        if target_minutes <= 0:
            raise ValueError("target_minutes must be positive")
        self.target_minutes = target_minutes
        self.min_fraction = min_fraction

    def align(
        self,
        records: Sequence[WeatherRecord],
        source_minutes: int = 10,
    ) -> list[WeatherRecord]:
        """Align records to the target resolution.

        Args:
            records: Input weather records (assumed same station, sorted by time).
            source_minutes: Source temporal resolution in minutes.

        Returns:
            Aggregated records at target resolution.
        """
        if not records:
            return []

        if source_minutes == self.target_minutes:
            return list(records)

        if source_minutes > self.target_minutes:
            raise ValueError(
                f"Cannot aggregate from {source_minutes}-min to "
                f"{self.target_minutes}-min (upsampling not supported)"
            )

        expected_per_window = self.target_minutes // source_minutes
        min_required = max(1, math.ceil(expected_per_window * self.min_fraction))

        # Group records by target time window
        windows: dict[datetime, list[WeatherRecord]] = {}
        for rec in records:
            window_start = self._snap_to_window(rec.timestamp)
            windows.setdefault(window_start, []).append(rec)

        # Aggregate each window
        aggregated: list[WeatherRecord] = []
        for window_start, window_records in sorted(windows.items()):
            agg = self._aggregate_window(
                window_start,
                window_records,
                min_required,
            )
            aggregated.append(agg)

        return aggregated

    def _snap_to_window(self, ts: datetime) -> datetime:
        """Snap a timestamp to the start of its target window."""
        minutes = ts.hour * 60 + ts.minute
        window_start_minutes = (minutes // self.target_minutes) * self.target_minutes
        return ts.replace(
            hour=window_start_minutes // 60,
            minute=window_start_minutes % 60,
            second=0,
            microsecond=0,
        )

    def _aggregate_window(
        self,
        window_start: datetime,
        records: list[WeatherRecord],
        min_required: int,
    ) -> WeatherRecord:
        """Aggregate records within a single time window."""
        n = len(records)
        station_id = records[0].station_id
        network = records[0].network

        result = WeatherRecord(
            timestamp=window_start,
            station_id=station_id,
            network=network,
        )

        if n < min_required:
            # Insufficient data -- all values stay NaN, flags stay MISSING
            return result

        # Mean-aggregated variables
        for var in ("temperature", "pressure", "humidity", "global_radiation",
                    "snow_surface_temp", "snow_depth"):
            values = [
                getattr(r, var) for r in records
                if not math.isnan(getattr(r, var))
            ]
            if len(values) >= min_required:
                setattr(result, var, sum(values) / len(values))
                setattr(result, f"{var}_qc", QualityFlag.GOOD)

        # Sum-aggregated: precipitation
        precip_values = [
            r.precipitation for r in records
            if not math.isnan(r.precipitation)
        ]
        if len(precip_values) >= min_required:
            result.precipitation = sum(precip_values)
            result.precipitation_qc = QualityFlag.GOOD

        # Vector-mean aggregated: wind
        wind_speeds = []
        wind_dirs = []
        for r in records:
            if not math.isnan(r.wind_speed) and not math.isnan(r.wind_direction):
                wind_speeds.append(r.wind_speed)
                wind_dirs.append(r.wind_direction)

        if len(wind_speeds) >= min_required:
            result.wind_speed, result.wind_direction = self._vector_mean_wind(
                wind_speeds, wind_dirs
            )
            result.wind_speed_qc = QualityFlag.GOOD
            result.wind_direction_qc = QualityFlag.GOOD

        return result

    @staticmethod
    def _vector_mean_wind(
        speeds: list[float], directions: list[float]
    ) -> tuple[float, float]:
        """Compute vector-mean wind speed and direction.

        Decomposes wind into u/v components, averages, and recomposes.
        Direction uses meteorological convention (0=N, 90=E).
        """
        u_sum = 0.0
        v_sum = 0.0
        for speed, direction in zip(speeds, directions):
            rad = math.radians(direction)
            # Meteorological convention: direction wind comes FROM
            u_sum += -speed * math.sin(rad)
            v_sum += -speed * math.cos(rad)

        n = len(speeds)
        u_mean = u_sum / n
        v_mean = v_sum / n

        mean_speed = math.sqrt(u_mean**2 + v_mean**2)
        mean_direction = math.degrees(math.atan2(-u_mean, -v_mean)) % 360

        return mean_speed, mean_direction
