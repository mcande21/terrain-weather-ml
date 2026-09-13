"""Tests for multi-variable output head (task 5.3)."""

import math

import torch

from terrain_weather_ml.terrain.downscaling.head import TerrainDownscalingHead
from terrain_weather_ml.terrain.downscaling.output import (
    OUTPUT_CHANNELS,
    uv_to_speed_direction,
)


class TestMultiVariableOutput:
    """Multi-variable output: u-wind, v-wind, temperature, precipitation."""

    def test_output_shape(self):
        """Output has 4 channels at fine resolution."""
        head = TerrainDownscalingHead()
        terrain = torch.randn(1, 17, 16, 16)
        weather = torch.randn(1, 6, 4, 4)
        out = head(terrain, weather)
        assert out.shape == (1, 4, 16, 16)

    def test_output_channel_names(self):
        """Output channel constants are defined."""
        assert OUTPUT_CHANNELS == ["u_wind", "v_wind", "temperature", "precipitation"]

    def test_uv_to_speed_direction_north(self):
        """Pure northward wind: u=0, v>0 -> speed=v, direction=180 degrees."""
        u = torch.tensor([[[[0.0]]]])
        v = torch.tensor([[[[5.0]]]])
        speed, direction = uv_to_speed_direction(u, v)
        assert torch.allclose(speed, torch.tensor([[[[5.0]]]]))
        # Meteorological convention: wind FROM direction
        # v>0 means wind blowing north, which comes FROM south (180)
        assert torch.allclose(direction, torch.tensor([[[[180.0]]]]), atol=0.1)

    def test_uv_to_speed_direction_east(self):
        """Pure eastward wind: u>0, v=0 -> speed=u, direction=270 degrees."""
        u = torch.tensor([[[[5.0]]]])
        v = torch.tensor([[[[0.0]]]])
        speed, direction = uv_to_speed_direction(u, v)
        assert torch.allclose(speed, torch.tensor([[[[5.0]]]]))
        # u>0 means wind blowing east, which comes FROM west (270)
        assert torch.allclose(direction, torch.tensor([[[[270.0]]]]), atol=0.1)

    def test_uv_to_speed_direction_southwest(self):
        """Wind from NE: u=-1, v=-1 -> direction=45 degrees."""
        u = torch.tensor([[[[-1.0]]]])
        v = torch.tensor([[[[-1.0]]]])
        speed, direction = uv_to_speed_direction(u, v)
        expected_speed = torch.tensor([[[[math.sqrt(2)]]]])
        assert torch.allclose(speed, expected_speed, atol=1e-5)
        # u<0,v<0 means wind blowing SW, comes FROM NE (45)
        assert torch.allclose(direction, torch.tensor([[[[45.0]]]]), atol=0.1)

    def test_uv_to_speed_direction_calm(self):
        """Calm wind: u=0, v=0 -> speed=0, direction=0."""
        u = torch.tensor([[[[0.0]]]])
        v = torch.tensor([[[[0.0]]]])
        speed, direction = uv_to_speed_direction(u, v)
        assert torch.allclose(speed, torch.tensor([[[[0.0]]]]))
        assert torch.allclose(direction, torch.tensor([[[[0.0]]]]))

    def test_uv_to_speed_direction_batch(self):
        """Conversion works on batched tensors."""
        u = torch.randn(4, 1, 8, 8)
        v = torch.randn(4, 1, 8, 8)
        speed, direction = uv_to_speed_direction(u, v)
        assert speed.shape == (4, 1, 8, 8)
        assert direction.shape == (4, 1, 8, 8)
        assert (speed >= 0).all()
        assert (direction >= 0).all()
        assert (direction <= 360).all()
