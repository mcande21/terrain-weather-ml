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

    def test_init_conv_skipped_on_shape_mismatch(self, tmp_path):
        """init_conv weights skipped when DEVINE has 23ch but head has 6ch."""
        # DEVINE checkpoint with 23-channel input (old concat approach)
        ckpt_path = create_mock_devine_checkpoint(
            tmp_path, in_channels=23
        )

        # FiLM head with 6-channel input
        head = TerrainDownscalingHead(apply_divergence_free=False)
        assert head.unet.init_conv.block[0].in_channels == 6

        # Save init_conv weight before loading
        init_weight_before = head.unet.init_conv.block[0].weight.clone()

        load_devine_weights(head, ckpt_path)

        # init_conv should NOT have been overwritten (shape mismatch)
        init_weight_after = head.unet.init_conv.block[0].weight
        assert torch.allclose(init_weight_before, init_weight_after), (
            "init_conv should be skipped when shapes don't match"
        )

    def test_decoder_weights_loaded_despite_init_conv_mismatch(self, tmp_path):
        """Decoder weights load even when init_conv shapes don't match."""
        ckpt_path = create_mock_devine_checkpoint(
            tmp_path, in_channels=23
        )
        head = TerrainDownscalingHead(apply_divergence_free=False)

        # Save a decoder weight before loading
        dec_weight_before = (
            head.unet.decoder_blocks[0].conv.block[0].weight.clone()
        )

        load_devine_weights(head, ckpt_path)

        # Decoder weight SHOULD have been overwritten
        dec_weight_after = head.unet.decoder_blocks[0].conv.block[0].weight
        assert not torch.allclose(dec_weight_before, dec_weight_after), (
            "Decoder weights should be loaded from DEVINE"
        )

    def test_film_generator_excluded_from_devine(self, tmp_path):
        """FiLM generator parameters are not affected by DEVINE loading."""
        head = TerrainDownscalingHead(apply_divergence_free=False)

        # Save FiLM generator weights before
        film_weights_before = {
            name: p.clone()
            for name, p in head.film_gen.named_parameters()
        }

        ckpt_path = create_mock_devine_checkpoint(tmp_path)
        load_devine_weights(head, ckpt_path)

        # FiLM generator weights should be unchanged
        for name, p in head.film_gen.named_parameters():
            assert torch.allclose(film_weights_before[name], p), (
                f"FiLM param {name} should not change during DEVINE loading"
            )

    def test_freeze_wind_keeps_film_trainable(self, tmp_path):
        """Freezing wind params leaves FiLM generator trainable."""
        ckpt_path = create_mock_devine_checkpoint(tmp_path)
        head = TerrainDownscalingHead(apply_divergence_free=False)
        load_devine_weights(head, ckpt_path, freeze_wind=True)

        # FiLM generator should remain trainable
        for param in head.film_gen.parameters():
            assert param.requires_grad, (
                "FiLM generator should remain trainable after freeze_wind"
            )

    def test_use_film_false_loads_normally(self, tmp_path):
        """use_film=False head loads DEVINE weights including init_conv."""
        ckpt_path = create_mock_devine_checkpoint(
            tmp_path, in_channels=23
        )
        head = TerrainDownscalingHead(
            apply_divergence_free=False, use_film=False
        )
        assert head.unet.init_conv.block[0].in_channels == 23

        init_weight_before = head.unet.init_conv.block[0].weight.clone()
        load_devine_weights(head, ckpt_path)

        # init_conv SHOULD be loaded (shapes match)
        init_weight_after = head.unet.init_conv.block[0].weight
        assert not torch.allclose(init_weight_before, init_weight_after), (
            "init_conv should be loaded when shapes match (use_film=False)"
        )
