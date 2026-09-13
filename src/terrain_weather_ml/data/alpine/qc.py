"""Quality control and unit normalization for Alpine weather data.

Task 2.5: Normalize to common units, reject physically impossible values.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

from terrain_weather_ml.data.alpine.schema import (
    Network,
    QualityFlag,
    WeatherRecord,
)

logger = logging.getLogger(__name__)


# Physical bounds for quality control
QC_BOUNDS: dict[str, tuple[float, float]] = {
    "temperature": (180.0, 340.0),  # Kelvin
    "wind_speed": (0.0, 120.0),  # m/s (allow up to category 5 hurricane+)
    "wind_direction": (0.0, 360.0),  # degrees
    "precipitation": (0.0, 500.0),  # mm/interval
    "pressure": (300.0, 1100.0),  # hPa
    "humidity": (0.0, 1.0),  # fraction
    "snow_surface_temp": (180.0, 340.0),  # Kelvin
    "snow_depth": (0.0, 30.0),  # meters
    "global_radiation": (0.0, 1500.0),  # W/m2
}


class QualityController:
    """Quality control and unit normalization for weather records.

    Unit conversions applied:
    - Temperature: Celsius -> Kelvin (add 273.15)
    - Humidity: percentage -> fraction (divide by 100)
    - Snow depth: cm -> meters (divide by 100)
    - Wind direction: ensure 0-360 range

    QC rejection rules:
    - Negative wind speed -> REJECTED
    - Temperature < 180K or > 340K -> REJECTED
    - Wind direction outside [0, 360] -> REJECTED
    - Humidity outside [0, 1] (after normalization) -> REJECTED
    """

    def __init__(
        self,
        bounds: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self.bounds = bounds or QC_BOUNDS

    def normalize_and_qc(
        self, records: Sequence[WeatherRecord]
    ) -> list[WeatherRecord]:
        """Apply unit normalization and quality control to records.

        Returns new records with normalized units and updated QC flags.
        Records are modified in place and returned.
        """
        result: list[WeatherRecord] = []
        for record in records:
            normalized = self._normalize_units(record)
            qc_checked = self._apply_qc(normalized)
            result.append(qc_checked)
        return result

    def _normalize_units(self, record: WeatherRecord) -> WeatherRecord:
        """Convert raw units to normalized schema units."""
        # Temperature: Celsius -> Kelvin (value > 200 suggests already Kelvin)
        if not math.isnan(record.temperature) and record.temperature < 200:
            record.temperature += 273.15

        # Snow surface temperature: Celsius -> Kelvin
        if not math.isnan(record.snow_surface_temp) and record.snow_surface_temp < 200:
            record.snow_surface_temp += 273.15

        # Humidity: percentage -> fraction
        if not math.isnan(record.humidity) and record.humidity > 1.0:
            record.humidity /= 100.0

        # Snow depth: cm -> meters (IMIS reports in cm, convert if > 30)
        if (
            not math.isnan(record.snow_depth)
            and record.network == Network.IMIS
            and record.snow_depth > 30
        ):
            record.snow_depth /= 100.0

        # Wind direction: normalize to [0, 360)
        if not math.isnan(record.wind_direction):
            record.wind_direction = record.wind_direction % 360

        return record

    def _apply_qc(self, record: WeatherRecord) -> WeatherRecord:
        """Apply quality control bounds checks."""
        for var_name, (low, high) in self.bounds.items():
            value = getattr(record, var_name, None)
            if value is None or math.isnan(value):
                continue

            qc_field = f"{var_name}_qc"
            if value < low or value > high:
                setattr(record, var_name, float("nan"))
                setattr(record, qc_field, QualityFlag.REJECTED)
                logger.debug(
                    "QC rejected %s=%s at station %s (bounds [%s, %s])",
                    var_name,
                    value,
                    record.station_id,
                    low,
                    high,
                )

        return record

    def check_cross_network_consistency(
        self,
        peakweather_records: Sequence[WeatherRecord],
        imis_records: Sequence[WeatherRecord],
    ) -> bool:
        """Verify that PeakWeather and IMIS records use identical units.

        Returns True if consistent, raises ValueError if not.
        """
        # Check a sample of records for unit consistency
        pw_sample = peakweather_records[:10] if peakweather_records else []
        imis_sample = imis_records[:10] if imis_records else []

        for sample, name in [(pw_sample, "PeakWeather"), (imis_sample, "IMIS")]:
            for rec in sample:
                # Temperature should be in Kelvin (> 200)
                if not math.isnan(rec.temperature) and rec.temperature < 200:
                    raise ValueError(
                        f"{name} temperature {rec.temperature} appears to be "
                        f"in Celsius, not Kelvin"
                    )
                # Humidity should be fraction (0-1)
                if not math.isnan(rec.humidity) and rec.humidity > 1.0:
                    raise ValueError(
                        f"{name} humidity {rec.humidity} appears to be "
                        f"in percentage, not fraction"
                    )

        return True
