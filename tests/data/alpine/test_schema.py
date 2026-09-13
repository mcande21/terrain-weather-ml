"""Tests for Task 2.3: Station metadata schema.

Verifies:
- Both PeakWeather and IMIS produce records with all required metadata fields
- No field is None
- Validation rejects invalid inputs
"""

from __future__ import annotations

import math

import pytest

from terrain_weather_ml.data.alpine.schema import (
    Network,
    QualityFlag,
    StationMetadata,
    WeatherRecord,
)


class TestStationMetadata:
    """Test unified station metadata across both networks."""

    def test_peakweather_metadata_all_fields_present(self):
        """PeakWeather station has all required fields, none None."""
        meta = StationMetadata(
            station_id="BER",
            network=Network.PEAKWEATHER,
            latitude=46.9481,
            longitude=7.4474,
            elevation=540.0,
            terrain_ref_lat=46.9481,
            terrain_ref_lon=7.4474,
        )
        assert meta.station_id == "BER"
        assert meta.network == Network.PEAKWEATHER
        assert meta.latitude == 46.9481
        assert meta.longitude == 7.4474
        assert meta.elevation == 540.0
        assert meta.terrain_ref_lat == 46.9481
        assert meta.terrain_ref_lon == 7.4474

    def test_imis_metadata_all_fields_present(self):
        """IMIS station has all required fields, none None."""
        meta = StationMetadata(
            station_id="WFJ2",
            network=Network.IMIS,
            latitude=46.8296,
            longitude=9.8092,
            elevation=2540.0,
            terrain_ref_lat=46.8296,
            terrain_ref_lon=9.8092,
        )
        assert meta.station_id == "WFJ2"
        assert meta.network == Network.IMIS
        assert meta.latitude == 46.8296
        assert meta.longitude == 9.8092
        assert meta.elevation == 2540.0
        assert meta.terrain_ref_lat is not None
        assert meta.terrain_ref_lon is not None

    def test_empty_station_id_rejected(self):
        with pytest.raises(ValueError, match="station_id"):
            StationMetadata(
                station_id="",
                network=Network.PEAKWEATHER,
                latitude=46.0,
                longitude=7.0,
                elevation=500.0,
                terrain_ref_lat=46.0,
                terrain_ref_lon=7.0,
            )

    def test_invalid_latitude_rejected(self):
        with pytest.raises(ValueError, match="latitude"):
            StationMetadata(
                station_id="BAD",
                network=Network.PEAKWEATHER,
                latitude=91.0,
                longitude=7.0,
                elevation=500.0,
                terrain_ref_lat=91.0,
                terrain_ref_lon=7.0,
            )

    def test_invalid_longitude_rejected(self):
        with pytest.raises(ValueError, match="longitude"):
            StationMetadata(
                station_id="BAD",
                network=Network.PEAKWEATHER,
                latitude=46.0,
                longitude=181.0,
                elevation=500.0,
                terrain_ref_lat=46.0,
                terrain_ref_lon=181.0,
            )

    def test_frozen_immutability(self):
        meta = StationMetadata(
            station_id="TEST",
            network=Network.IMIS,
            latitude=46.0,
            longitude=7.0,
            elevation=1000.0,
            terrain_ref_lat=46.0,
            terrain_ref_lon=7.0,
        )
        with pytest.raises(AttributeError):
            meta.station_id = "CHANGED"  # type: ignore[misc]


class TestWeatherRecord:
    """Test weather record dataclass."""

    def test_default_values_are_nan(self):
        from datetime import datetime

        rec = WeatherRecord(
            timestamp=datetime(2024, 1, 1),
            station_id="TEST",
            network=Network.PEAKWEATHER,
        )
        assert math.isnan(rec.temperature)
        assert math.isnan(rec.wind_speed)
        assert math.isnan(rec.wind_direction)
        assert math.isnan(rec.precipitation)
        assert math.isnan(rec.pressure)
        assert math.isnan(rec.humidity)

    def test_default_qc_flags_are_missing(self):
        from datetime import datetime

        rec = WeatherRecord(
            timestamp=datetime(2024, 1, 1),
            station_id="TEST",
            network=Network.IMIS,
        )
        assert rec.temperature_qc == QualityFlag.MISSING
        assert rec.wind_speed_qc == QualityFlag.MISSING

    def test_weather_vars_list(self):
        from datetime import datetime

        rec = WeatherRecord(
            timestamp=datetime(2024, 1, 1),
            station_id="TEST",
            network=Network.PEAKWEATHER,
        )
        assert "temperature" in rec.WEATHER_VARS
        assert "wind_speed" in rec.WEATHER_VARS
        assert "wind_direction" in rec.WEATHER_VARS
        assert "precipitation" in rec.WEATHER_VARS
        assert "pressure" in rec.WEATHER_VARS
        assert "humidity" in rec.WEATHER_VARS


class TestNetwork:
    def test_network_values(self):
        assert Network.PEAKWEATHER.value == "peakweather"
        assert Network.IMIS.value == "imis"


class TestQualityFlag:
    def test_flag_values(self):
        assert QualityFlag.GOOD.value == 0
        assert QualityFlag.SUSPECT.value == 1
        assert QualityFlag.MISSING.value == 2
        assert QualityFlag.REJECTED.value == 3
