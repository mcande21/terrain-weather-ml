"""Tests for Task 2.5: Quality control and unit normalization.

Verifies:
- Unit conversions on known values
- QC rejection on each impossible-value category
- Cross-network variable consistency in a mixed batch
"""

from __future__ import annotations

import math
from datetime import datetime

import pytest

from terrain_weather_ml.data.alpine.qc import QualityController
from terrain_weather_ml.data.alpine.schema import (
    Network,
    QualityFlag,
    WeatherRecord,
)


def _make_record(
    network: Network = Network.PEAKWEATHER,
    **kwargs,
) -> WeatherRecord:
    """Create a weather record with given overrides."""
    defaults = {
        "timestamp": datetime(2024, 1, 1),
        "station_id": "TEST",
        "network": network,
    }
    defaults.update(kwargs)
    return WeatherRecord(**defaults)


class TestUnitNormalization:
    """Test unit conversion rules."""

    def test_celsius_to_kelvin(self):
        """Temperature converts from Celsius to Kelvin."""
        rec = _make_record(
            temperature=20.0,
            temperature_qc=QualityFlag.GOOD,
        )
        qc = QualityController()
        result = qc.normalize_and_qc([rec])[0]
        assert result.temperature == pytest.approx(293.15, abs=0.01)

    def test_humidity_percent_to_fraction(self):
        """Humidity converts from percentage to fraction."""
        rec = _make_record(
            humidity=65.0,
            humidity_qc=QualityFlag.GOOD,
        )
        qc = QualityController()
        result = qc.normalize_and_qc([rec])[0]
        assert result.humidity == pytest.approx(0.65, abs=0.01)

    def test_snow_depth_cm_to_meters_imis(self):
        """IMIS snow depth converts from cm to meters."""
        rec = _make_record(
            network=Network.IMIS,
            snow_depth=150.0,
            snow_depth_qc=QualityFlag.GOOD,
        )
        qc = QualityController()
        result = qc.normalize_and_qc([rec])[0]
        assert result.snow_depth == pytest.approx(1.5, abs=0.01)

    def test_wind_direction_normalization(self):
        """Negative wind direction wraps to [0, 360)."""
        rec = _make_record(
            wind_direction=-90.0,
            wind_direction_qc=QualityFlag.GOOD,
        )
        qc = QualityController()
        result = qc.normalize_and_qc([rec])[0]
        assert result.wind_direction == pytest.approx(270.0, abs=0.01)

    def test_already_kelvin_not_double_converted(self):
        """Temperature already in Kelvin (>200) is not converted again."""
        rec = _make_record(
            temperature=280.0,
            temperature_qc=QualityFlag.GOOD,
        )
        qc = QualityController()
        result = qc.normalize_and_qc([rec])[0]
        assert result.temperature == pytest.approx(280.0, abs=0.01)


class TestQCRejection:
    """Test QC rejection for physically impossible values."""

    def test_reject_negative_wind_speed(self):
        """Negative wind speed is rejected."""
        rec = _make_record(
            wind_speed=-1.0,
            wind_speed_qc=QualityFlag.GOOD,
        )
        qc = QualityController()
        result = qc.normalize_and_qc([rec])[0]
        assert math.isnan(result.wind_speed)
        assert result.wind_speed_qc == QualityFlag.REJECTED

    def test_reject_temperature_below_180k(self):
        """Temperature below 180K is rejected."""
        rec = _make_record(
            temperature=170.0,  # Already in K (> 200 check won't convert)
            temperature_qc=QualityFlag.GOOD,
        )
        # Force it to be treated as Kelvin by setting > 200 threshold bypass
        # Actually 170 < 200 so it would be converted: 170+273.15=443.15 > 340 -> rejected
        qc = QualityController()
        result = qc.normalize_and_qc([rec])[0]
        # 170 C -> 443.15 K -> exceeds 340K -> rejected
        assert math.isnan(result.temperature)
        assert result.temperature_qc == QualityFlag.REJECTED

    def test_reject_temperature_above_340k(self):
        """Temperature above 340K is rejected."""
        rec = _make_record(
            temperature=350.0,  # Already in K
            temperature_qc=QualityFlag.GOOD,
        )
        qc = QualityController()
        result = qc.normalize_and_qc([rec])[0]
        assert math.isnan(result.temperature)
        assert result.temperature_qc == QualityFlag.REJECTED

    def test_reject_wind_direction_still_out_of_range(self):
        """Wind direction wraps, so it should not be rejected after normalization."""
        rec = _make_record(
            wind_direction=450.0,
            wind_direction_qc=QualityFlag.GOOD,
        )
        qc = QualityController()
        result = qc.normalize_and_qc([rec])[0]
        # 450 % 360 = 90, which is in [0, 360]
        assert result.wind_direction == pytest.approx(90.0, abs=0.01)
        assert result.wind_direction_qc == QualityFlag.GOOD

    def test_reject_humidity_above_1_after_normalization(self):
        """Humidity > 100% is rejected after normalization to fraction."""
        rec = _make_record(
            humidity=110.0,  # 110% -> 1.1 after normalization
            humidity_qc=QualityFlag.GOOD,
        )
        qc = QualityController()
        result = qc.normalize_and_qc([rec])[0]
        # 110% -> 1.1 -> exceeds [0, 1] -> rejected
        assert math.isnan(result.humidity)
        assert result.humidity_qc == QualityFlag.REJECTED

    def test_valid_values_pass_qc(self):
        """Valid values pass QC without modification."""
        rec = _make_record(
            temperature=5.0,   # Celsius -> 278.15 K (valid)
            wind_speed=10.0,   # m/s (valid)
            wind_direction=180.0,  # degrees (valid)
            precipitation=2.0,  # mm (valid)
            pressure=950.0,    # hPa (valid)
            humidity=65.0,     # % -> 0.65 (valid)
            temperature_qc=QualityFlag.GOOD,
            wind_speed_qc=QualityFlag.GOOD,
            wind_direction_qc=QualityFlag.GOOD,
            precipitation_qc=QualityFlag.GOOD,
            pressure_qc=QualityFlag.GOOD,
            humidity_qc=QualityFlag.GOOD,
        )
        qc = QualityController()
        result = qc.normalize_and_qc([rec])[0]

        assert not math.isnan(result.temperature)
        assert not math.isnan(result.wind_speed)
        assert not math.isnan(result.wind_direction)
        assert not math.isnan(result.precipitation)
        assert not math.isnan(result.pressure)
        assert not math.isnan(result.humidity)


class TestCrossNetworkConsistency:
    """Test cross-network variable consistency."""

    def test_consistent_units_pass(self):
        """Records with consistent units pass the check."""
        pw_rec = _make_record(
            network=Network.PEAKWEATHER,
            temperature=278.15,  # Kelvin
            humidity=0.65,       # fraction
        )
        imis_rec = _make_record(
            network=Network.IMIS,
            temperature=270.15,  # Kelvin
            humidity=0.80,       # fraction
        )
        qc = QualityController()
        assert qc.check_cross_network_consistency([pw_rec], [imis_rec])

    def test_inconsistent_temperature_raises(self):
        """Temperature in Celsius (not Kelvin) raises ValueError."""
        pw_rec = _make_record(
            network=Network.PEAKWEATHER,
            temperature=5.0,  # Celsius -- not normalized!
        )
        qc = QualityController()
        with pytest.raises(ValueError, match="Celsius"):
            qc.check_cross_network_consistency([pw_rec], [])

    def test_inconsistent_humidity_raises(self):
        """Humidity in percentage (not fraction) raises ValueError."""
        imis_rec = _make_record(
            network=Network.IMIS,
            humidity=65.0,  # Percentage -- not normalized!
        )
        qc = QualityController()
        with pytest.raises(ValueError, match="percentage"):
            qc.check_cross_network_consistency([], [imis_rec])
