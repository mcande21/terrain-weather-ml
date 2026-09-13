"""Request/response schemas for the micro-weather inference API.

Pydantic models defining the API contract with colorado-avalanche-ml
and other consumers.
"""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Domain bounds (configurable at runtime; these are sensible defaults
# for the Colorado SNOTEL domain)
# ---------------------------------------------------------------------------

DEFAULT_DOMAIN_BOUNDS = {
    "lat_min": 36.5,
    "lat_max": 41.5,
    "lon_min": -109.5,
    "lon_max": -103.5,
}

MAX_FORECAST_HOURS = 18
MAX_BATCH_SIZE = 1000


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------

class PredictionRequest(BaseModel):
    """Single-point prediction request."""

    latitude: float = Field(..., ge=-90, le=90, description="WGS84 latitude")
    longitude: float = Field(..., ge=-180, le=180, description="WGS84 longitude")
    datetime: dt.datetime = Field(..., description="ISO 8601 UTC datetime")


class PredictionResponse(BaseModel):
    """Single-point prediction response."""

    wind_speed: float = Field(..., description="Wind speed (m/s)")
    wind_direction: float = Field(..., description="Wind direction (degrees, met. convention)")
    temperature: float = Field(..., description="Temperature (K)")
    precipitation_rate: float = Field(..., description="Precipitation rate (mm/hr)")
    elevation: float = Field(..., description="DEM elevation at query point (m)")
    aspect: float = Field(..., description="DEM aspect at query point (degrees)")
    model_version: str = Field(..., description="Checkpoint identifier")
    source_hrrr_cycle: str = Field(..., description="HRRR analysis cycle used")


# ---------------------------------------------------------------------------
# /predict/batch
# ---------------------------------------------------------------------------

class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""

    points: list[PredictionRequest] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        description="Up to 1000 prediction points",
    )


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""

    predictions: list[PredictionResponse]


# ---------------------------------------------------------------------------
# /terrain-weather (colorado-avalanche-ml integration)
# ---------------------------------------------------------------------------

class TerrainPoint(BaseModel):
    """A terrain point for the avalanche integration endpoint."""

    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    elevation: float = Field(..., description="Elevation (m)")
    aspect: float = Field(..., ge=0, lt=360, description="Aspect (degrees)")


class TerrainWeatherRequest(BaseModel):
    """colorado-avalanche-ml integration request."""

    points: list[TerrainPoint] = Field(
        ..., min_length=1, max_length=MAX_BATCH_SIZE,
    )
    datetime: dt.datetime = Field(..., description="Forecast start (ISO 8601 UTC)")
    horizon_hours: int = Field(default=24, ge=1, le=48)


class HourlyForecast(BaseModel):
    """One hourly forecast entry."""

    datetime: str = Field(..., description="ISO 8601 UTC")
    temperature: float = Field(..., description="Temperature (K)")
    wind_speed: float = Field(..., description="Wind speed (m/s)")
    wind_direction: float = Field(..., description="Wind direction (degrees)")
    precipitation_rate: float = Field(..., description="Precipitation rate (mm/hr)")
    surface_pressure: float = Field(..., description="Surface pressure (hPa)")


class PointForecast(BaseModel):
    """Forecast for a single terrain point."""

    latitude: float
    longitude: float
    elevation: float
    aspect: float
    hourly: list[HourlyForecast]


class TerrainWeatherResponse(BaseModel):
    """colorado-avalanche-ml integration response."""

    forecasts: list[PointForecast]
    model_version: str
    source_hrrr_cycle: str


# ---------------------------------------------------------------------------
# /health, /metadata
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    backbone: str = Field(..., description="Active backbone adapter (nwp or stormcast)")
    data_source: str = Field(..., description="Data source(s) for the active backbone")


class MetadataResponse(BaseModel):
    """Model metadata response."""

    model_version: str
    checkpoint_hash: str
    domain_bounds: dict[str, float]
    max_forecast_hours: int
    backbone: str
    data_source: str
    output_variables: list[dict[str, str]]
