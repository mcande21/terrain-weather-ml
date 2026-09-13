"""Terrain downscaling head (tasks 5.2-5.5).

DEVINE-inspired U-Net that takes coarse weather + static terrain
and produces sub-km multi-variable weather predictions with
mass-conservation constraint on wind output.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import nn

from terrain_weather_ml.terrain.downscaling.divergence import DivergenceFreeProjection
from terrain_weather_ml.terrain.downscaling.unet import UNet

logger = logging.getLogger(__name__)

# Default channel counts
C_TERRAIN_DEFAULT = 17
C_WEATHER_DEFAULT = 6
C_OUT_DEFAULT = 4  # u-wind, v-wind, temperature, precipitation

# Output channel indices
IDX_U_WIND = 0
IDX_V_WIND = 1
IDX_TEMPERATURE = 2
IDX_PRECIPITATION = 3


class TerrainDownscalingHead(nn.Module):
    """DEVINE-inspired terrain downscaling head.

    Dual-branch input: static terrain tensor + dynamic weather tensor.
    The weather tensor is upsampled to match terrain resolution before
    concatenation and U-Net processing.

    Args:
        c_terrain: Number of terrain feature channels (default: 17).
        c_weather: Number of weather input channels (default: 6).
        c_out: Number of output channels (default: 4).
        base_features: Base feature count for U-Net encoder.
        apply_divergence_free: Apply divergence-free projection on wind output.
    """

    def __init__(
        self,
        c_terrain: int = C_TERRAIN_DEFAULT,
        c_weather: int = C_WEATHER_DEFAULT,
        c_out: int = C_OUT_DEFAULT,
        base_features: int = 64,
        apply_divergence_free: bool = True,
    ):
        super().__init__()
        self.c_terrain = c_terrain
        self.c_weather = c_weather
        self.c_out = c_out
        self.apply_divergence_free = apply_divergence_free

        c_in = c_terrain + c_weather
        self.unet = UNet(
            in_channels=c_in,
            out_channels=c_out,
            base_features=base_features,
        )

        # Divergence-free projection on wind channels (hard constraint)
        self.div_free_proj = DivergenceFreeProjection(
            u_idx=IDX_U_WIND, v_idx=IDX_V_WIND
        )

    def forward(
        self,
        terrain: torch.Tensor,
        weather: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            terrain: Static terrain tensor (B, C_terrain, H_fine, W_fine).
            weather: Dynamic weather tensor (B, C_weather, H_coarse, W_coarse).

        Returns:
            Prediction tensor (B, C_out, H_fine, W_fine).
        """
        h_fine, w_fine = terrain.shape[2], terrain.shape[3]

        # Upsample weather to match terrain resolution
        if weather.shape[2:] != terrain.shape[2:]:
            weather_up = F.interpolate(
                weather,
                size=(h_fine, w_fine),
                mode="bilinear",
                align_corners=False,
            )
        else:
            weather_up = weather

        # Concatenate along channel dimension
        x = torch.cat([terrain, weather_up], dim=1)

        # U-Net forward pass
        out = self.unet(x)

        # Apply divergence-free projection to wind channels
        if self.apply_divergence_free:
            out = self.div_free_proj(out)

        return out
