"""Core inference service.

Encapsulates model loading, terrain lookups, HRRR refresh, and
prediction logic. The FastAPI routes are thin wrappers around this.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import torch

from terrain_weather_ml.inference.schemas import (
    DEFAULT_DOMAIN_BOUNDS,
    MAX_FORECAST_HOURS,
    HourlyForecast,
    PointForecast,
    PredictionResponse,
)
from terrain_weather_ml.terrain.downscaling.output import uv_to_speed_direction

logger = logging.getLogger(__name__)


def _get_device() -> str:
    """Auto-detect best available device."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class InferenceConfig:
    """Configuration for the inference service.

    Args:
        model_version: Human-readable version string.
        checkpoint_hash: SHA256 of the model checkpoint.
        domain_bounds: Spatial domain bounds {lat_min, lat_max, lon_min, lon_max}.
        hrrr_refresh_seconds: Interval for HRRR refresh (default 3600 = hourly).
        device: PyTorch device string. Auto-detected if None.
    """

    model_version: str = "terrain-weather-ml-v0.1.0"
    checkpoint_hash: str = "none"
    domain_bounds: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_DOMAIN_BOUNDS)
    )
    hrrr_refresh_seconds: int = 3600
    device: str | None = None


class InferenceService:
    """Stateful inference service managing model, terrain, and HRRR state.

    Designed for dependency injection: the model, terrain lookup, and
    HRRR fetcher are callables so tests can supply mocks without touching
    the real StormCast/DEM infrastructure.
    """

    def __init__(
        self,
        config: InferenceConfig,
        model_fn: Callable[..., torch.Tensor] | None = None,
        terrain_lookup_fn: Callable[..., dict[str, float]] | None = None,
        hrrr_fetch_fn: Callable[..., Any] | None = None,
    ):
        self.config = config
        self.device = config.device or _get_device()
        self._model_fn = model_fn
        self._terrain_lookup_fn = terrain_lookup_fn
        self._hrrr_fetch_fn = hrrr_fetch_fn

        self._model_loaded = False
        self._current_hrrr_cycle: str = "unknown"
        self._hrrr_refresh_task: asyncio.Task | None = None

    # ----- lifecycle -----

    @property
    def is_ready(self) -> bool:
        return self._model_loaded

    def load_model(self) -> None:
        """Load the model (lazy, called on first request)."""
        if self._model_loaded:
            return
        logger.info("Loading inference model (version=%s)", self.config.model_version)
        # In production this loads StormCast + LoRA + terrain head from
        # safetensors. For now the model_fn callable is the abstraction.
        self._model_loaded = True
        logger.info("Model loaded successfully")

    # ----- domain validation -----

    def validate_coordinates(self, lat: float, lon: float) -> str | None:
        """Return an error message if coordinates are outside domain, else None."""
        bounds = self.config.domain_bounds
        if not (bounds["lat_min"] <= lat <= bounds["lat_max"]):
            return (
                f"Latitude {lat} outside model domain "
                f"[{bounds['lat_min']}, {bounds['lat_max']}]"
            )
        if not (bounds["lon_min"] <= lon <= bounds["lon_max"]):
            return (
                f"Longitude {lon} outside model domain "
                f"[{bounds['lon_min']}, {bounds['lon_max']}]"
            )
        return None

    def validate_datetime(self, dt: datetime) -> str | None:
        """Return an error message if datetime is beyond forecast horizon."""
        now = datetime.now(UTC)
        if dt.tzinfo is None:
            # Treat naive as UTC
            dt = dt.replace(tzinfo=UTC)
        horizon = now + timedelta(hours=MAX_FORECAST_HOURS)
        if dt > horizon:
            return (
                f"Requested datetime {dt.isoformat()} exceeds maximum "
                f"forecast horizon of {MAX_FORECAST_HOURS} hours from now"
            )
        return None

    # ----- terrain lookup -----

    def lookup_terrain(self, lat: float, lon: float) -> dict[str, float]:
        """Look up elevation and aspect from DEM at (lat, lon)."""
        if self._terrain_lookup_fn is not None:
            return self._terrain_lookup_fn(lat, lon)
        # Fallback: return defaults (production would query the terrain encoder)
        return {"elevation": 0.0, "aspect": 0.0}

    # ----- prediction -----

    def predict_single(
        self,
        lat: float,
        lon: float,
        dt: datetime,
    ) -> PredictionResponse:
        """Run inference for a single point."""
        if not self._model_loaded:
            self.load_model()

        terrain = self.lookup_terrain(lat, lon)

        if self._model_fn is not None:
            raw = self._model_fn(lat, lon, dt)
        else:
            # Mock output: (u, v, temperature, precipitation)
            raw = torch.tensor([1.0, 1.0, 268.0, 0.001])

        u, v = raw[0], raw[1]
        speed_t, direction_t = uv_to_speed_direction(
            u.unsqueeze(0), v.unsqueeze(0),
        )

        return PredictionResponse(
            wind_speed=float(speed_t.item()),
            wind_direction=float(direction_t.item()),
            temperature=float(raw[2].item()),
            precipitation_rate=float(raw[3].item()),
            elevation=terrain["elevation"],
            aspect=terrain["aspect"],
            model_version=self.config.model_version,
            source_hrrr_cycle=self._current_hrrr_cycle,
        )

    def predict_batch(
        self,
        points: list[tuple[float, float, datetime]],
    ) -> list[PredictionResponse]:
        """Run inference for a batch of points.

        Groups points by datetime to share StormCast forward passes.
        """
        if not self._model_loaded:
            self.load_model()

        # Group by datetime for efficient batching
        dt_groups: dict[datetime, list[tuple[int, float, float]]] = {}
        for i, (lat, lon, dt) in enumerate(points):
            dt_groups.setdefault(dt, []).append((i, lat, lon))

        results: list[PredictionResponse | None] = [None] * len(points)

        for dt, group in dt_groups.items():
            # Batch forward pass per datetime group
            for idx, lat, lon in group:
                results[idx] = self.predict_single(lat, lon, dt)

        return [r for r in results if r is not None]

    def predict_terrain_weather(
        self,
        points: list[dict[str, float]],
        start_dt: datetime,
        horizon_hours: int,
    ) -> list[PointForecast]:
        """Run multi-hour forecasts for colorado-avalanche-ml.

        Args:
            points: List of dicts with lat, lon, elevation, aspect.
            start_dt: Forecast start datetime.
            horizon_hours: Number of hourly forecasts.

        Returns:
            List of PointForecast, one per input point.
        """
        if not self._model_loaded:
            self.load_model()

        forecasts = []
        for pt in points:
            hourly = []
            for h in range(horizon_hours):
                dt = start_dt + timedelta(hours=h)
                pred = self.predict_single(pt["latitude"], pt["longitude"], dt)
                hourly.append(HourlyForecast(
                    datetime=dt.isoformat(),
                    temperature=pred.temperature,
                    wind_speed=pred.wind_speed,
                    wind_direction=pred.wind_direction,
                    precipitation_rate=pred.precipitation_rate,
                    surface_pressure=1013.25,  # placeholder; production reads from backbone
                ))
            forecasts.append(PointForecast(
                latitude=pt["latitude"],
                longitude=pt["longitude"],
                elevation=pt["elevation"],
                aspect=pt["aspect"],
                hourly=hourly,
            ))
        return forecasts

    # ----- HRRR refresh -----

    @property
    def current_hrrr_cycle(self) -> str:
        return self._current_hrrr_cycle

    def set_hrrr_cycle(self, cycle: str) -> None:
        self._current_hrrr_cycle = cycle

    async def refresh_hrrr(self) -> bool:
        """Attempt to fetch the latest HRRR cycle.

        Returns True on success, False on failure (stale cycle kept).
        """
        if self._hrrr_fetch_fn is None:
            return False
        try:
            new_cycle = self._hrrr_fetch_fn()
            if new_cycle is not None:
                self._current_hrrr_cycle = new_cycle
                logger.info("HRRR cycle updated to %s", new_cycle)
                return True
        except Exception:
            logger.warning(
                "HRRR fetch failed, continuing with cycle %s",
                self._current_hrrr_cycle,
                exc_info=True,
            )
        return False

    async def start_hrrr_refresh_loop(self) -> None:
        """Start the periodic HRRR refresh background task."""
        async def _loop() -> None:
            while True:
                await asyncio.sleep(self.config.hrrr_refresh_seconds)
                await self.refresh_hrrr()

        self._hrrr_refresh_task = asyncio.create_task(_loop())

    async def stop_hrrr_refresh_loop(self) -> None:
        """Stop the periodic HRRR refresh background task."""
        if self._hrrr_refresh_task is not None:
            self._hrrr_refresh_task.cancel()
            try:
                await self._hrrr_refresh_task
            except asyncio.CancelledError:
                pass
            self._hrrr_refresh_task = None

    # ----- metadata -----

    def get_metadata(self) -> dict:
        """Return model metadata for the /metadata endpoint."""
        return {
            "model_version": self.config.model_version,
            "checkpoint_hash": self.config.checkpoint_hash,
            "domain_bounds": self.config.domain_bounds,
            "max_forecast_hours": MAX_FORECAST_HOURS,
            "output_variables": [
                {"name": "wind_speed", "unit": "m/s"},
                {"name": "wind_direction", "unit": "degrees"},
                {"name": "temperature", "unit": "K"},
                {"name": "precipitation_rate", "unit": "mm/hr"},
                {"name": "elevation", "unit": "m"},
                {"name": "aspect", "unit": "degrees"},
            ],
        }
