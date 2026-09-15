"""Evaluation framework for terrain-weather models."""

from terrain_weather_ml.evaluation.loso import (
    ElevationBandConfig,
    LOSOConfig,
    LOSOEvaluator,
    LOSOResult,
    StationData,
    TemporalSplit,
    bootstrap_confidence_intervals,
    classify_elevation_band,
    classify_season,
    compute_seasonal_metrics,
)

__all__ = [
    "ElevationBandConfig",
    "LOSOConfig",
    "LOSOEvaluator",
    "LOSOResult",
    "StationData",
    "TemporalSplit",
    "bootstrap_confidence_intervals",
    "classify_elevation_band",
    "classify_season",
    "compute_seasonal_metrics",
]
