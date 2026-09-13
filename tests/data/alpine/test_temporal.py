"""Tests for Task 2.4: Temporal alignment.

Verifies:
- Aggregation rules on synthetic 10-min data
- Incomplete-interval NaN marking (fewer than 2-of-3)
- Aligned timestamps match IMIS 30-min boundaries exactly
- Vector-mean wind aggregation correctness
"""

from __future__ import annotations

import math
from datetime import datetime

import pytest

from terrain_weather_ml.data.alpine.schema import (
    Network,
    QualityFlag,
    WeatherRecord,
)
from terrain_weather_ml.data.alpine.temporal import TemporalAligner


def _make_10min_records(
    n_windows: int = 2,
    records_per_window: int = 3,
    station_id: str = "STN0001",
) -> list[WeatherRecord]:
    """Create synthetic 10-min records for testing aggregation."""
    records = []
    for w in range(n_windows):
        for r in range(records_per_window):
            t = w * 3 + r  # 0,1,2,3,4,5...
            ts = datetime(
                2024, 1, 1,
                (t * 10) // 60,
                (t * 10) % 60,
            )
            records.append(WeatherRecord(
                timestamp=ts,
                station_id=station_id,
                network=Network.PEAKWEATHER,
                temperature=273.15 + 5.0 + r * 0.1,  # ~5.0-5.2 C in K
                wind_speed=3.0 + r * 0.5,
                wind_direction=180.0,  # constant south wind
                precipitation=0.1 * (r + 1),  # 0.1, 0.2, 0.3
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


class TestTemporalAligner:
    """Tests for temporal alignment."""

    def test_aggregate_10min_to_30min_temperature_mean(self):
        """Temperature aggregation uses arithmetic mean."""
        records = _make_10min_records(n_windows=1, records_per_window=3)
        aligner = TemporalAligner(target_minutes=30)
        result = aligner.align(records, source_minutes=10)

        assert len(result) == 1
        # Mean of 278.15, 278.25, 278.35 = 278.25
        assert result[0].temperature == pytest.approx(278.25, abs=0.01)

    def test_aggregate_10min_to_30min_precipitation_sum(self):
        """Precipitation aggregation uses sum."""
        records = _make_10min_records(n_windows=1, records_per_window=3)
        aligner = TemporalAligner(target_minutes=30)
        result = aligner.align(records, source_minutes=10)

        assert len(result) == 1
        # Sum of 0.1, 0.2, 0.3 = 0.6
        assert result[0].precipitation == pytest.approx(0.6, abs=0.01)

    def test_aggregate_10min_to_30min_wind_vector_mean(self):
        """Wind aggregation uses vector mean."""
        records = []
        for i, direction in enumerate([0.0, 90.0, 180.0]):
            records.append(WeatherRecord(
                timestamp=datetime(2024, 1, 1, 0, i * 10),
                station_id="STN0001",
                network=Network.PEAKWEATHER,
                wind_speed=10.0,
                wind_direction=direction,
                wind_speed_qc=QualityFlag.GOOD,
                wind_direction_qc=QualityFlag.GOOD,
            ))

        aligner = TemporalAligner(target_minutes=30)
        result = aligner.align(records, source_minutes=10)

        assert len(result) == 1
        # Vector mean of N(10), E(10), S(10) should give ~NE with reduced speed
        assert result[0].wind_speed < 10.0  # Vector mean < scalar mean
        assert result[0].wind_speed_qc == QualityFlag.GOOD

    def test_incomplete_interval_marked_nan(self):
        """Intervals with < 2 of 3 expected records are NaN."""
        # Only 1 record in a 30-min window (need at least 2)
        records = [WeatherRecord(
            timestamp=datetime(2024, 1, 1, 0, 0),
            station_id="STN0001",
            network=Network.PEAKWEATHER,
            temperature=278.15,
            wind_speed=5.0,
            wind_direction=180.0,
            precipitation=0.1,
            pressure=950.0,
            humidity=0.65,
            temperature_qc=QualityFlag.GOOD,
            wind_speed_qc=QualityFlag.GOOD,
            wind_direction_qc=QualityFlag.GOOD,
            precipitation_qc=QualityFlag.GOOD,
            pressure_qc=QualityFlag.GOOD,
            humidity_qc=QualityFlag.GOOD,
        )]

        aligner = TemporalAligner(target_minutes=30)
        result = aligner.align(records, source_minutes=10)

        assert len(result) == 1
        # All values should be NaN due to insufficient records
        assert math.isnan(result[0].temperature)
        assert math.isnan(result[0].wind_speed)
        assert math.isnan(result[0].precipitation)

    def test_two_records_sufficient_for_aggregation(self):
        """2 of 3 expected records is enough (min_fraction=2/3)."""
        records = [
            WeatherRecord(
                timestamp=datetime(2024, 1, 1, 0, 0),
                station_id="STN0001",
                network=Network.PEAKWEATHER,
                temperature=278.15,
                wind_speed=5.0,
                wind_direction=180.0,
                precipitation=0.1,
                pressure=950.0,
                humidity=0.65,
                temperature_qc=QualityFlag.GOOD,
                wind_speed_qc=QualityFlag.GOOD,
                wind_direction_qc=QualityFlag.GOOD,
                precipitation_qc=QualityFlag.GOOD,
                pressure_qc=QualityFlag.GOOD,
                humidity_qc=QualityFlag.GOOD,
            ),
            WeatherRecord(
                timestamp=datetime(2024, 1, 1, 0, 10),
                station_id="STN0001",
                network=Network.PEAKWEATHER,
                temperature=278.25,
                wind_speed=6.0,
                wind_direction=180.0,
                precipitation=0.2,
                pressure=951.0,
                humidity=0.66,
                temperature_qc=QualityFlag.GOOD,
                wind_speed_qc=QualityFlag.GOOD,
                wind_direction_qc=QualityFlag.GOOD,
                precipitation_qc=QualityFlag.GOOD,
                pressure_qc=QualityFlag.GOOD,
                humidity_qc=QualityFlag.GOOD,
            ),
        ]

        aligner = TemporalAligner(target_minutes=30)
        result = aligner.align(records, source_minutes=10)

        assert len(result) == 1
        assert not math.isnan(result[0].temperature)
        assert result[0].temperature == pytest.approx(278.20, abs=0.01)

    def test_aligned_timestamps_match_30min_boundaries(self):
        """Output timestamps are at exact 30-min boundaries."""
        records = _make_10min_records(n_windows=4, records_per_window=3)
        aligner = TemporalAligner(target_minutes=30)
        result = aligner.align(records, source_minutes=10)

        for rec in result:
            assert rec.timestamp.minute % 30 == 0
            assert rec.timestamp.second == 0

    def test_passthrough_when_source_equals_target(self):
        """Records pass through unchanged when source == target resolution."""
        records = _make_10min_records(n_windows=1, records_per_window=3)
        aligner = TemporalAligner(target_minutes=10)
        result = aligner.align(records, source_minutes=10)
        assert len(result) == 3  # No aggregation

    def test_empty_input_returns_empty(self):
        aligner = TemporalAligner(target_minutes=30)
        assert aligner.align([], source_minutes=10) == []

    def test_upsampling_raises_error(self):
        aligner = TemporalAligner(target_minutes=5)
        with pytest.raises(ValueError, match="upsampling"):
            aligner.align(
                _make_10min_records(n_windows=1), source_minutes=10
            )
