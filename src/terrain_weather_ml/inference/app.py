"""FastAPI application for micro-weather inference.

Thin routing layer over the InferenceService. Handles HTTP
concerns (validation errors, status codes) while delegating
prediction logic to the service.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException

from terrain_weather_ml.inference.schemas import (
    MAX_BATCH_SIZE,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    MetadataResponse,
    PredictionRequest,
    PredictionResponse,
    TerrainWeatherRequest,
    TerrainWeatherResponse,
)
from terrain_weather_ml.inference.service import InferenceConfig, InferenceService

# ---------------------------------------------------------------------------
# Module-level service instance (set by create_app or tests)
# ---------------------------------------------------------------------------

_service: InferenceService | None = None


def get_service() -> InferenceService:
    """Return the active InferenceService, raising if not configured."""
    if _service is None:
        raise RuntimeError("InferenceService not initialized")
    return _service


def set_service(service: InferenceService) -> None:
    """Set the module-level service (used by create_app and tests)."""
    global _service
    _service = service


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app(
    config: InferenceConfig | None = None,
    **service_kwargs: Any,
) -> FastAPI:
    """Create a configured FastAPI application.

    Args:
        config: Inference configuration. Uses defaults if None.
        **service_kwargs: Forwarded to InferenceService (model_fn, etc.).

    Returns:
        Configured FastAPI app.
    """
    config = config or InferenceConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service = InferenceService(config, **service_kwargs)
        set_service(service)
        await service.start_hrrr_refresh_loop()
        yield
        await service.stop_hrrr_refresh_loop()

    app = FastAPI(
        title="Terrain Weather ML - Micro-Weather API",
        version=config.model_version,
        lifespan=lifespan,
    )

    # Register routes
    app.include_router(_build_router())

    return app


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def _build_router():
    from fastapi import APIRouter

    router = APIRouter()

    @router.get("/health", response_model=HealthResponse)
    async def health():
        svc = get_service()
        if not svc.is_ready:
            raise HTTPException(
                status_code=503,
                detail="Model is loading or failed to load",
            )
        return HealthResponse(status="healthy")

    @router.get("/metadata", response_model=MetadataResponse)
    async def metadata():
        svc = get_service()
        meta = svc.get_metadata()
        return MetadataResponse(**meta)

    @router.post("/predict", response_model=PredictionResponse)
    async def predict(request: PredictionRequest):
        svc = get_service()

        # Validate coordinates
        coord_err = svc.validate_coordinates(request.latitude, request.longitude)
        if coord_err is not None:
            raise HTTPException(status_code=422, detail=coord_err)

        # Validate datetime
        dt_err = svc.validate_datetime(request.datetime)
        if dt_err is not None:
            raise HTTPException(status_code=422, detail=dt_err)

        return svc.predict_single(
            request.latitude, request.longitude, request.datetime,
        )

    @router.post("/predict/batch", response_model=BatchPredictionResponse)
    async def predict_batch(request: BatchPredictionRequest):
        svc = get_service()

        if len(request.points) > MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=422,
                detail=f"Batch size {len(request.points)} exceeds maximum of {MAX_BATCH_SIZE}",
            )

        # Validate all points
        for pt in request.points:
            coord_err = svc.validate_coordinates(pt.latitude, pt.longitude)
            if coord_err is not None:
                raise HTTPException(status_code=422, detail=coord_err)
            dt_err = svc.validate_datetime(pt.datetime)
            if dt_err is not None:
                raise HTTPException(status_code=422, detail=dt_err)

        points = [
            (pt.latitude, pt.longitude, pt.datetime)
            for pt in request.points
        ]
        predictions = svc.predict_batch(points)

        return BatchPredictionResponse(predictions=predictions)

    @router.post("/terrain-weather", response_model=TerrainWeatherResponse)
    async def terrain_weather(request: TerrainWeatherRequest):
        svc = get_service()

        points_dicts = [
            {
                "latitude": pt.latitude,
                "longitude": pt.longitude,
                "elevation": pt.elevation,
                "aspect": pt.aspect,
            }
            for pt in request.points
        ]

        forecasts = svc.predict_terrain_weather(
            points_dicts,
            request.datetime,
            request.horizon_hours,
        )

        return TerrainWeatherResponse(
            forecasts=forecasts,
            model_version=svc.config.model_version,
            source_hrrr_cycle=svc.current_hrrr_cycle,
        )

    return router
