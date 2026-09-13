"""Unified station metadata and weather record schema for Alpine networks.

Task 2.3: Station metadata schema -- unified metadata dataclass across both
PeakWeather and IMIS networks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class Network(str, Enum):
    """Alpine station network identifier."""

    PEAKWEATHER = "peakweather"
    IMIS = "imis"


class QualityFlag(int, Enum):
    """Per-variable quality control flag."""

    GOOD = 0
    SUSPECT = 1
    MISSING = 2
    REJECTED = 3


@dataclass(frozen=True)
class StationMetadata:
    """Unified station metadata across PeakWeather and IMIS networks.

    All fields are required -- no field may be None.
    """

    station_id: str
    network: Network
    latitude: float
    longitude: float
    elevation: float  # meters above sea level
    terrain_ref_lat: float  # latitude for terrain encoder linkage
    terrain_ref_lon: float  # longitude for terrain encoder linkage

    def __post_init__(self) -> None:
        if not self.station_id:
            raise ValueError("station_id must be non-empty")
        if not isinstance(self.network, Network):
            raise TypeError(f"network must be a Network enum, got {type(self.network)}")
        if not (-90 <= self.latitude <= 90):
            raise ValueError(f"latitude must be in [-90, 90], got {self.latitude}")
        if not (-180 <= self.longitude <= 180):
            raise ValueError(f"longitude must be in [-180, 180], got {self.longitude}")


@dataclass
class WeatherRecord:
    """Single weather observation with quality flags.

    Variables use normalized units:
    - temperature: Kelvin
    - wind_speed: m/s
    - wind_direction: degrees (meteorological, 0=N)
    - precipitation: mm/interval
    - pressure: hPa
    - humidity: fraction (0-1)
    - global_radiation: W/m^2 (PeakWeather only, NaN for IMIS)
    - snow_surface_temp: Kelvin (IMIS only, NaN for PeakWeather)
    - snow_depth: meters (IMIS only, NaN for PeakWeather)
    """

    timestamp: datetime
    station_id: str
    network: Network

    # Weather variables (normalized units)
    temperature: float = float("nan")
    wind_speed: float = float("nan")
    wind_direction: float = float("nan")
    precipitation: float = float("nan")
    pressure: float = float("nan")
    humidity: float = float("nan")
    global_radiation: float = float("nan")
    snow_surface_temp: float = float("nan")
    snow_depth: float = float("nan")

    # Per-variable quality flags
    temperature_qc: QualityFlag = QualityFlag.MISSING
    wind_speed_qc: QualityFlag = QualityFlag.MISSING
    wind_direction_qc: QualityFlag = QualityFlag.MISSING
    precipitation_qc: QualityFlag = QualityFlag.MISSING
    pressure_qc: QualityFlag = QualityFlag.MISSING
    humidity_qc: QualityFlag = QualityFlag.MISSING
    global_radiation_qc: QualityFlag = QualityFlag.MISSING
    snow_surface_temp_qc: QualityFlag = QualityFlag.MISSING
    snow_depth_qc: QualityFlag = QualityFlag.MISSING

    # Common weather variable names (normalized across both networks)
    WEATHER_VARS: tuple[str, ...] = field(
        default=(
            "temperature",
            "wind_speed",
            "wind_direction",
            "precipitation",
            "pressure",
            "humidity",
        ),
        init=False,
        repr=False,
    )
