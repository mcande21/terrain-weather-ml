"""Micro-weather inference API.

FastAPI endpoints serving sub-km weather predictions at arbitrary
terrain coordinates. Provides the input layer for colorado-avalanche-ml.
"""

from terrain_weather_ml.inference.app import create_app
from terrain_weather_ml.inference.service import InferenceConfig, InferenceService

__all__ = [
    "InferenceConfig",
    "InferenceService",
    "create_app",
]
