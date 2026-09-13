"""Tests for U-Net encoder-decoder architecture (task 5.1)."""

import torch

from terrain_weather_ml.terrain.downscaling.unet import UNet


class TestUNetArchitecture:
    """U-Net encoder-decoder with skip connections."""

    def test_forward_preserves_spatial_dims(self):
        """Output spatial dims match input spatial dims."""
        model = UNet(in_channels=23, out_channels=4)
        x = torch.randn(1, 23, 16, 16)
        y = model(x)
        assert y.shape == (1, 4, 16, 16)

    def test_forward_batch_size(self):
        """Batched input produces matching batch output."""
        model = UNet(in_channels=23, out_channels=4)
        x = torch.randn(4, 23, 32, 32)
        y = model(x)
        assert y.shape == (4, 4, 32, 32)

    def test_four_downsampling_stages(self):
        """Encoder has at least 4 downsampling stages."""
        model = UNet(in_channels=23, out_channels=4)
        assert len(model.encoder_blocks) >= 4

    def test_skip_connections_affect_output(self):
        """Removing a skip connection changes the output, proving they are active."""
        model = UNet(in_channels=23, out_channels=4)
        model.eval()
        x = torch.randn(1, 23, 16, 16)

        with torch.no_grad():
            y_normal = model(x).clone()

        # Zero out a skip connection by hooking into the forward pass
        # We verify skip connections exist and affect output by testing
        # that the model has decoder blocks that receive skip connections
        assert len(model.decoder_blocks) >= 4

        # Ablation: run encoder, zero one skip, then run decoder
        with torch.no_grad():
            y_ablated = model.forward_ablated(x, zero_skip_idx=0)

        assert not torch.allclose(y_normal, y_ablated, atol=1e-6), (
            "Zeroing a skip connection should change the output"
        )

    def test_different_spatial_sizes(self):
        """U-Net handles various spatial dimensions (must be divisible by 2^4=16)."""
        model = UNet(in_channels=23, out_channels=4)
        for size in [16, 32, 64]:
            x = torch.randn(1, 23, size, size)
            y = model(x)
            assert y.shape == (1, 4, size, size), f"Failed for spatial size {size}"

    def test_custom_channel_counts(self):
        """U-Net respects custom in/out channel counts."""
        model = UNet(in_channels=10, out_channels=8)
        x = torch.randn(1, 10, 16, 16)
        y = model(x)
        assert y.shape == (1, 8, 16, 16)

    def test_gradient_flows(self):
        """Gradients flow through the full U-Net."""
        model = UNet(in_channels=23, out_channels=4)
        x = torch.randn(1, 23, 16, 16, requires_grad=True)
        y = model(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0
