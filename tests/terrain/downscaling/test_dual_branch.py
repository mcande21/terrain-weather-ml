"""Tests for dual-branch input processing (task 5.2)."""

import torch

from terrain_weather_ml.terrain.downscaling.head import TerrainDownscalingHead


class TestDualBranchInput:
    """Dual-branch: terrain (static) + weather (dynamic) input processing."""

    def test_upsample_weather_to_terrain_resolution(self):
        """Coarse weather upsampled to match fine terrain resolution."""
        head = TerrainDownscalingHead()
        terrain = torch.randn(1, 17, 32, 32)
        weather = torch.randn(1, 6, 4, 4)  # 8x coarser
        out = head(terrain, weather)
        # Output should be at fine resolution
        assert out.shape[2:] == (32, 32)

    def test_weather_only_unet_input(self):
        """FiLM head sends only weather (6ch) into U-Net, not concatenated."""
        head = TerrainDownscalingHead()
        # The internal U-Net should have c_weather input channels (6)
        assert head.unet.init_conv.block[0].in_channels == 6

    def test_film_generator_exists(self):
        """FiLM head has a FiLM generator for terrain conditioning."""
        head = TerrainDownscalingHead()
        assert hasattr(head, "film_gen")

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
        """Custom terrain/weather channel counts with FiLM."""
        head = TerrainDownscalingHead(
            c_terrain=10, c_weather=4, c_out=6
        )
        terrain = torch.randn(1, 10, 16, 16)
        weather = torch.randn(1, 4, 8, 8)
        out = head(terrain, weather)
        assert out.shape == (1, 6, 16, 16)

    def test_use_film_false_concat_fallback(self):
        """use_film=False falls back to concatenation (backward compat)."""
        head = TerrainDownscalingHead(use_film=False)
        # Should use concatenated 23-channel input
        assert head.unet.init_conv.block[0].in_channels == 23
        terrain = torch.randn(1, 17, 16, 16)
        weather = torch.randn(1, 6, 4, 4)
        out = head(terrain, weather)
        assert out.shape == (1, 4, 16, 16)

    def test_gradient_flows_through_film(self):
        """Gradients flow through FiLM generator to terrain input."""
        head = TerrainDownscalingHead(apply_divergence_free=False)
        terrain = torch.randn(1, 17, 16, 16, requires_grad=True)
        weather = torch.randn(1, 6, 4, 4)
        out = head(terrain, weather)
        out.sum().backward()
        assert terrain.grad is not None
        assert terrain.grad.abs().sum() > 0
