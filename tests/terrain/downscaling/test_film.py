"""Tests for FiLM conditioning module."""

import torch

from terrain_weather_ml.terrain.downscaling.film import FiLMBlock, FiLMGenerator


class TestFiLMGenerator:
    """FiLM generator: terrain tensor -> per-decoder-block (gamma, beta)."""

    def test_output_count_matches_decoder_blocks(self):
        """Generator produces one (gamma, beta) pair per decoder block (4)."""
        gen = FiLMGenerator(c_terrain=17, base_features=64)
        terrain = torch.randn(2, 17, 32, 32)
        params = gen(terrain)
        assert len(params) == 4

    def test_output_shapes_base64(self):
        """Each (gamma, beta) matches the decoder block's channel count."""
        gen = FiLMGenerator(c_terrain=17, base_features=64)
        terrain = torch.randn(2, 17, 32, 32)
        params = gen(terrain)
        # Decoder blocks: 512, 256, 128, 64
        expected_channels = [512, 256, 128, 64]
        for i, (gamma, beta) in enumerate(params):
            assert gamma.shape == (2, expected_channels[i], 1, 1), (
                f"Block {i}: gamma shape {gamma.shape} != "
                f"(2, {expected_channels[i]}, 1, 1)"
            )
            assert beta.shape == (2, expected_channels[i], 1, 1), (
                f"Block {i}: beta shape {beta.shape} != "
                f"(2, {expected_channels[i]}, 1, 1)"
            )

    def test_output_shapes_base32(self):
        """Works with base_features=32."""
        gen = FiLMGenerator(c_terrain=17, base_features=32)
        terrain = torch.randn(1, 17, 16, 16)
        params = gen(terrain)
        expected_channels = [256, 128, 64, 32]
        for i, (gamma, beta) in enumerate(params):
            assert gamma.shape == (1, expected_channels[i], 1, 1)
            assert beta.shape == (1, expected_channels[i], 1, 1)

    def test_identity_initialization(self):
        """At initialization, gamma=1 and beta=0 (identity transform)."""
        gen = FiLMGenerator(c_terrain=17, base_features=64)
        terrain = torch.randn(1, 17, 16, 16)
        params = gen(terrain)
        for i, (gamma, beta) in enumerate(params):
            assert torch.allclose(gamma, torch.ones_like(gamma), atol=0.15), (
                f"Block {i}: gamma not near 1.0 at init"
            )
            assert torch.allclose(beta, torch.zeros_like(beta), atol=0.15), (
                f"Block {i}: beta not near 0.0 at init"
            )

    def test_parameter_count_under_10_percent(self):
        """FiLM generator has <10% of U-Net parameter count."""
        from terrain_weather_ml.terrain.downscaling.unet import UNet

        gen = FiLMGenerator(c_terrain=17, base_features=64)
        unet = UNet(in_channels=6, out_channels=4, base_features=64)

        film_params = sum(p.numel() for p in gen.parameters())
        unet_params = sum(p.numel() for p in unet.parameters())

        ratio = film_params / unet_params
        assert ratio < 0.10, (
            f"FiLM generator has {ratio:.1%} of U-Net params, "
            f"expected <10%"
        )

    def test_custom_terrain_channels(self):
        """Works with non-default terrain channel count."""
        gen = FiLMGenerator(c_terrain=10, base_features=64)
        terrain = torch.randn(1, 10, 16, 16)
        params = gen(terrain)
        assert len(params) == 4

    def test_gradient_flows(self):
        """Gradients flow back through the generator."""
        gen = FiLMGenerator(c_terrain=17, base_features=64)
        terrain = torch.randn(1, 17, 16, 16, requires_grad=True)
        params = gen(terrain)
        loss = sum(g.sum() + b.sum() for g, b in params)
        loss.backward()
        assert terrain.grad is not None
        assert terrain.grad.abs().sum() > 0


class TestFiLMBlock:
    """FiLM block: affine modulation after GroupNorm."""

    def test_identity_passthrough(self):
        """gamma=1, beta=0 produces same output as plain GroupNorm."""
        channels = 64
        num_groups = min(32, channels)
        norm = torch.nn.GroupNorm(num_groups, channels)
        film_block = FiLMBlock(channels)

        torch.manual_seed(42)
        x = torch.randn(2, channels, 8, 8)

        expected = norm(x)

        # Copy norm weights to FiLMBlock's norm
        film_block.norm.load_state_dict(norm.state_dict())

        gamma = torch.ones(2, channels, 1, 1)
        beta = torch.zeros(2, channels, 1, 1)
        result = film_block(x, gamma, beta)

        assert torch.allclose(result, expected, atol=1e-6), (
            "Identity FiLM should match plain GroupNorm"
        )

    def test_non_identity_changes_output(self):
        """Non-identity gamma/beta produces different output than GroupNorm."""
        channels = 64
        film_block = FiLMBlock(channels)

        torch.manual_seed(42)
        x = torch.randn(2, channels, 8, 8)

        gamma_identity = torch.ones(2, channels, 1, 1)
        beta_identity = torch.zeros(2, channels, 1, 1)
        out_identity = film_block(x, gamma_identity, beta_identity)

        gamma = torch.full((2, channels, 1, 1), 2.0)
        beta = torch.full((2, channels, 1, 1), 0.5)
        out_modulated = film_block(x, gamma, beta)

        assert not torch.allclose(out_identity, out_modulated, atol=1e-6), (
            "Non-identity FiLM params should change output"
        )

    def test_modulation_formula(self):
        """Verifies gamma * GroupNorm(x) + beta."""
        channels = 32
        film_block = FiLMBlock(channels)

        torch.manual_seed(42)
        x = torch.randn(1, channels, 4, 4)

        gamma = torch.full((1, channels, 1, 1), 3.0)
        beta = torch.full((1, channels, 1, 1), -1.0)

        normed = film_block.norm(x)
        expected = gamma * normed + beta
        result = film_block(x, gamma, beta)

        assert torch.allclose(result, expected, atol=1e-6)

    def test_output_shape_preserved(self):
        """Output shape matches input shape."""
        channels = 128
        film_block = FiLMBlock(channels)
        x = torch.randn(4, channels, 16, 16)
        gamma = torch.ones(4, channels, 1, 1)
        beta = torch.zeros(4, channels, 1, 1)
        result = film_block(x, gamma, beta)
        assert result.shape == x.shape

    def test_gradient_flows(self):
        """Gradients flow through FiLMBlock."""
        channels = 64
        film_block = FiLMBlock(channels)
        x = torch.randn(1, channels, 8, 8, requires_grad=True)
        gamma = torch.ones(1, channels, 1, 1, requires_grad=True)
        beta = torch.zeros(1, channels, 1, 1, requires_grad=True)
        result = film_block(x, gamma, beta)
        loss = result.sum()
        loss.backward()
        assert x.grad is not None
        assert gamma.grad is not None
        assert beta.grad is not None
