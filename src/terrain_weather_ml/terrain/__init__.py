"""Terrain feature encoder module.

DEM-derived terrain feature computation and encoding for the
downscaling head's static terrain input branch.
"""

from terrain_weather_ml.terrain.dem import DEMData, load_dem
from terrain_weather_ml.terrain.encoder import (
    C_TERRAIN,
    CHANNEL_NAMES,
    TerrainFeatureEncoder,
)
from terrain_weather_ml.terrain.normalization import TerrainNormalizer

__all__ = [
    "CHANNEL_NAMES",
    "C_TERRAIN",
    "DEMData",
    "TerrainFeatureEncoder",
    "TerrainNormalizer",
    "load_dem",
]
