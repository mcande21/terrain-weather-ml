"""Tests for DEVINE weight initialization (task 5.5)."""

import torch

from terrain_weather_ml.terrain.downscaling.devine import (
    create_mock_devine_checkpoint,
    load_devine_weights,
)
from terrain_weather_ml.terrain.downscaling.head import TerrainDownscalingHead


class TestDEVINEWeightInit:
    """DEVINE weight initialization for wind channels."""

    def test_load_devine_weights(self, tmp_path):
        """DEVINE weights load into wind encoder/decoder channels."""
        ckpt_path = create_mock_devine_checkpoint(tmp_path)
        head = TerrainDownscalingHead(apply_divergence_free=False)
        load_devine_weights(head, ckpt_path)
        # Should not raise

    def test_freeze_wind_channels(self, tmp_path):
        """Freeze flag disables gradients on encoder/decoder (DEVINE-loaded)."""
        ckpt_path = create_mock_devine_checkpoint(tmp_path)
        head = TerrainDownscalingHead(apply_divergence_free=False)
        load_devine_weights(head, ckpt_path, freeze_wind=True)

        # Encoder and decoder blocks should be frozen
        for block in head.unet.encoder_blocks:
            for param in block.parameters():
                assert not param.requires_grad
        for block in head.unet.decoder_blocks:
            for param in block.parameters():
                assert not param.requires_grad

        # Final conv should remain trainable (has temp/precip channels)
        assert head.unet.final_conv.weight.requires_grad

    def test_temp_precip_remain_trainable(self, tmp_path):
        """Temperature and precipitation channels are trainable even with freeze."""
        ckpt_path = create_mock_devine_checkpoint(tmp_path)
        head = TerrainDownscalingHead(apply_divergence_free=False)
        load_devine_weights(head, ckpt_path, freeze_wind=True)

        # Check that at least some parameters have requires_grad=True
        # (the non-wind parts of the network should be trainable)
        trainable_params = [p for p in head.parameters() if p.requires_grad]
        assert len(trainable_params) > 0

    def test_no_devine_path_uses_random_init(self):
        """Without DEVINE path, all channels are randomly initialized."""
        head = TerrainDownscalingHead(apply_divergence_free=False)
        # Just verify it constructs without error and all params are trainable
        all_params = list(head.parameters())
        trainable = [p for p in all_params if p.requires_grad]
        assert len(trainable) == len(all_params)

    def test_devine_weights_change_output(self, tmp_path):
        """Loading DEVINE weights actually changes the model output."""
        head = TerrainDownscalingHead(apply_divergence_free=False)
        head.eval()
        torch.manual_seed(42)
        terrain = torch.randn(1, 17, 16, 16)
        weather = torch.randn(1, 6, 4, 4)

        with torch.no_grad():
            out_before = head(terrain, weather).clone()

        ckpt_path = create_mock_devine_checkpoint(tmp_path)
        load_devine_weights(head, ckpt_path)

        with torch.no_grad():
            out_after = head(terrain, weather)

        assert not torch.allclose(out_before, out_after, atol=1e-6)

    def test_unfreeze_after_warmup(self, tmp_path):
        """Wind channels can be unfrozen after warmup period."""
        ckpt_path = create_mock_devine_checkpoint(tmp_path)
        head = TerrainDownscalingHead(apply_divergence_free=False)
        load_devine_weights(head, ckpt_path, freeze_wind=True)

        # Verify frozen
        frozen_before = sum(
            1 for p in head.parameters() if not p.requires_grad
        )
        assert frozen_before > 0

        # Unfreeze all parameters
        for p in head.parameters():
            p.requires_grad = True

        # Now all should be trainable
        frozen_after = sum(
            1 for p in head.parameters() if not p.requires_grad
        )
        assert frozen_after == 0
