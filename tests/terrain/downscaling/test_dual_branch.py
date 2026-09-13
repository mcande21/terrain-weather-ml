"""Tests for dual-branch input processing (task 5.2)."""

import torch

from terrain_weather_ml.terrain.downscaling.head import TerrainDownscalingHead


class TestDualBranchInput:
    """Dual-branch: terrain (static) + weather (dynamic) input processing."""

    def test_upsample_and_concat(self):
        """Coarse weather upsampled to match fine terrain, then concatenated."""
        head = TerrainDownscalingHead()
        terrain = torch.randn(1, 17, 32, 32)
        weather = torch.randn(1, 6, 4, 4)  # 8x coarser
        out = head(terrain, weather)
        # Output should be at fine resolution
        assert out.shape[2:] == (32, 32)

    def test_23_channel_concat(self):
        """17 terrain + 6 weather = 23 channels going into U-Net."""
        head = TerrainDownscalingHead()
        torch.randn(1, 17, 16, 16)
        torch.randn(1, 6, 4, 4)
        # The internal U-Net should have 23 input channels
        assert head.unet.init_conv.block[0].in_channels == 23

    def test_output_channels(self):
        """Output has 4 channels: u-wind, v-wind, temperature, precipitation."""
        head = TerrainDownscalingHead()
        terrain = torch.randn(1, 17, 16, 16)
        weather = torch.randn(1, 6, 4, 4)
        out = head(terrain, weather)
        assert out.shape[1] == 4

    def test_batch_processing(self):
        """Batched input produces correct output shape."""
        head = TerrainDownscalingHead()
        terrain = torch.randn(4, 17, 32, 32)
        weather = torch.randn(4, 6, 8, 8)
        out = head(terrain, weather)
        assert out.shape == (4, 4, 32, 32)

    def test_same_resolution_input(self):
        """Weather at same resolution as terrain works (no upsampling needed)."""
        head = TerrainDownscalingHead()
        terrain = torch.randn(1, 17, 16, 16)
        weather = torch.randn(1, 6, 16, 16)
        out = head(terrain, weather)
        assert out.shape == (1, 4, 16, 16)

    def test_custom_channel_counts(self):
        """Custom terrain/weather channel counts."""
        head = TerrainDownscalingHead(
            c_terrain=10, c_weather=4, c_out=6
        )
        terrain = torch.randn(1, 10, 16, 16)
        weather = torch.randn(1, 4, 8, 8)
        out = head(terrain, weather)
        assert out.shape == (1, 6, 16, 16)
