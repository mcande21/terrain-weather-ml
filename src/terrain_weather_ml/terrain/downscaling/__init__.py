"""Terrain downscaling head module.

DEVINE-inspired U-Net for downscaling coarse weather grids to sub-km
predictions using static terrain features, with mass-conservation
enforcement on wind output.
"""

from terrain_weather_ml.terrain.downscaling.devine import (
    create_mock_devine_checkpoint,
    load_devine_weights,
)
from terrain_weather_ml.terrain.downscaling.divergence import (
    DivergenceFreeProjection,
    compute_divergence,
    divergence_free_projection,
)
from terrain_weather_ml.terrain.downscaling.head import (
    C_OUT_DEFAULT,
    C_TERRAIN_DEFAULT,
    C_WEATHER_DEFAULT,
    IDX_PRECIPITATION,
    IDX_TEMPERATURE,
    IDX_U_WIND,
    IDX_V_WIND,
    TerrainDownscalingHead,
)
from terrain_weather_ml.terrain.downscaling.loss import (
    DownscalingLoss,
    orographic_precipitation_penalty,
)
from terrain_weather_ml.terrain.downscaling.output import (
    OUTPUT_CHANNELS,
    uv_to_speed_direction,
)
from terrain_weather_ml.terrain.downscaling.unet import UNet

__all__ = [
    "C_OUT_DEFAULT",
    "C_TERRAIN_DEFAULT",
    "C_WEATHER_DEFAULT",
    "IDX_PRECIPITATION",
    "IDX_TEMPERATURE",
    "IDX_U_WIND",
    "IDX_V_WIND",
    "OUTPUT_CHANNELS",
    "DivergenceFreeProjection",
    "DownscalingLoss",
    "TerrainDownscalingHead",
    "UNet",
    "compute_divergence",
    "create_mock_devine_checkpoint",
    "divergence_free_projection",
    "load_devine_weights",
    "orographic_precipitation_penalty",
    "uv_to_speed_direction",
]
