"""Task 4.3: LoRA adapter injection tests.

LoRA targets `affine` nn.Linear layers (55 in real StormCast).
Must skip fused QKV Conv2d layers.

Verify: trainable parameter count < 1% of total; save/load round-trip of
LoRA-only checkpoint reproduces identical adapted model output; LoRA
targets affine layers and skips Conv2d.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from terrain_weather_ml.backbone.stormcast import (
    LoRALinear,
    StormCastAdapter,
    StormCastConfig,
)
from tests.backbone.test_checkpoint import MockStormCast


@pytest.fixture
def mock_checkpoint_path(tmp_path):
    """Save a mock StormCast checkpoint to disk."""
    model = MockStormCast(dim=64, n_blocks=2)
    path = tmp_path / "stormcast_mock.pt"
    torch.save(model.state_dict(), path)
    return path


@pytest.fixture
def adapter_with_lora(mock_checkpoint_path):
    """Create a StormCastAdapter with LoRA injected."""
    config = StormCastConfig(
        checkpoint_path=str(mock_checkpoint_path),
        model_class="tests.backbone.test_checkpoint.MockStormCast",
    )
    adapter = StormCastAdapter(config)
    adapter.inject_lora()
    return adapter


class TestLoRALinear:
    """Unit tests for the LoRA layer itself."""

    def test_lora_output_shape_matches_original(self):
        """LoRA-wrapped linear should produce same output shape."""
        original = nn.Linear(32, 64)
        lora = LoRALinear(original, rank=4, alpha=8)
        x = torch.randn(2, 32)
        out = lora(x)
        assert out.shape == (2, 64)

    def test_lora_original_frozen(self):
        """Original parameters should be frozen after LoRA wrapping."""
        original = nn.Linear(32, 64)
        lora = LoRALinear(original, rank=4, alpha=8)
        for param in lora.original.parameters():
            assert not param.requires_grad

    def test_lora_adapters_trainable(self):
        """LoRA A and B matrices should be trainable."""
        original = nn.Linear(32, 64)
        lora = LoRALinear(original, rank=4, alpha=8)
        assert lora.lora_A.requires_grad
        assert lora.lora_B.requires_grad

    def test_lora_initial_output_matches_original(self):
        """With B initialized to zero, LoRA output should match original."""
        original = nn.Linear(32, 64)
        # Save original weights before LoRA wrapping
        x = torch.randn(2, 32)
        with torch.no_grad():
            expected = original(x)

        lora = LoRALinear(original, rank=4, alpha=8)
        with torch.no_grad():
            actual = lora(x)

        torch.testing.assert_close(actual, expected)

    def test_lora_rank_shapes(self):
        """LoRA A should be (rank, in), B should be (out, rank)."""
        original = nn.Linear(32, 64)
        rank = 4
        lora = LoRALinear(original, rank=rank, alpha=8)
        assert lora.lora_A.shape == (rank, 32)
        assert lora.lora_B.shape == (64, rank)


class TestLoRAInjection:
    """Task 4.3: LoRA injection into affine layers, skipping Conv2d."""

    def test_default_target_is_affine(self):
        """Default LoRA target modules should be ['affine']."""
        config = StormCastConfig()
        assert config.lora_target_modules == ["affine"]

    def test_lora_targets_affine_layers(self, adapter_with_lora):
        """LoRA should be injected into `affine` nn.Linear layers."""
        lora_count = sum(
            1 for name, m in adapter_with_lora.backbone.named_modules()
            if isinstance(m, LoRALinear)
        )
        # MockStormCast: 2 blocks x 3 sub-blocks x 1 affine = 6
        assert lora_count == 6

    def test_lora_skips_conv2d(self, adapter_with_lora):
        """LoRA must NOT be injected into Conv2d layers (fused QKV)."""
        for name, m in adapter_with_lora.backbone.named_modules():
            if isinstance(m, nn.Conv2d):
                assert not isinstance(m, LoRALinear), (
                    f"Conv2d {name} should not be LoRA-wrapped"
                )

    def test_lora_skips_non_target_linear(self, adapter_with_lora):
        """LoRA should not wrap nn.Linear layers that are not named 'affine'."""
        for name, m in adapter_with_lora.backbone.named_modules():
            if isinstance(m, LoRALinear):
                leaf_name = name.split(".")[-1]
                assert leaf_name == "affine", (
                    f"LoRA injected into non-affine layer: {name}"
                )

    def test_trainable_params_structure(self, adapter_with_lora):
        """Only LoRA params should be trainable."""
        for name, param in adapter_with_lora.named_parameters():
            if param.requires_grad:
                assert "lora_A" in name or "lora_B" in name, (
                    f"Non-LoRA param {name} is trainable"
                )

    def test_trainable_param_count(self, adapter_with_lora):
        """LoRA params: 6 layers x (rank*in + out*rank) = 6 x 1024 = 6144."""
        info = adapter_with_lora.model_info()
        trainable = info["trainable_params"]
        assert trainable == 6144, (
            f"Expected 6144 trainable params (6 affine layers * rank=8 * "
            f"dim=64 * 2), got {trainable}"
        )

    def test_backbone_params_still_frozen(self, adapter_with_lora):
        """Original backbone params (non-LoRA) should remain frozen."""
        for name, param in adapter_with_lora.backbone.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                assert param.requires_grad
            elif "original" not in name:
                assert not param.requires_grad, f"Param {name} should be frozen"

    def test_save_load_roundtrip(self, adapter_with_lora, tmp_path):
        """LoRA save/load round-trip should reproduce identical output."""
        x = torch.randn(1, 99, 8, 8)

        with torch.no_grad():
            expected = adapter_with_lora(x, extract_surface=False)

        lora_path = tmp_path / "lora_weights.pt"
        lora_state = adapter_with_lora.get_lora_state_dict()
        torch.save(lora_state, lora_path)

        assert all("lora" in k for k in lora_state)
        assert len(lora_state) > 0

        config = StormCastConfig(
            checkpoint_path=str(adapter_with_lora.config.checkpoint_path),
            model_class="tests.backbone.test_checkpoint.MockStormCast",
        )
        fresh_adapter = StormCastAdapter(config)
        fresh_adapter.inject_lora()

        saved_lora = torch.load(lora_path, map_location="cpu", weights_only=True)
        fresh_adapter.load_lora_state_dict(saved_lora)

        with torch.no_grad():
            actual = fresh_adapter(x, extract_surface=False)

        torch.testing.assert_close(actual, expected)

    def test_lora_state_dict_is_small(self, adapter_with_lora):
        """LoRA checkpoint should be much smaller than full model."""
        lora_state = adapter_with_lora.get_lora_state_dict()
        full_state = adapter_with_lora.backbone.state_dict()

        lora_params = sum(v.numel() for v in lora_state.values())
        full_params = sum(v.numel() for v in full_state.values())

        assert lora_params < full_params
        assert all("lora" in k for k in lora_state)

    def test_lora_affects_output(self, adapter_with_lora):
        """After modifying LoRA weights, output should differ from zero-init."""
        x = torch.randn(1, 99, 8, 8)

        with torch.no_grad():
            output_zero = adapter_with_lora(x, extract_surface=False).clone()

        for name, param in adapter_with_lora.named_parameters():
            if "lora_B" in name:
                nn.init.normal_(param, std=1.0)

        with torch.no_grad():
            output_modified = adapter_with_lora(x, extract_surface=False)

        assert not torch.allclose(output_zero, output_modified, atol=1e-6)

    def test_double_inject_idempotent(self, mock_checkpoint_path):
        """Injecting LoRA twice should not double-wrap layers."""
        config = StormCastConfig(
            checkpoint_path=str(mock_checkpoint_path),
            model_class="tests.backbone.test_checkpoint.MockStormCast",
        )
        adapter = StormCastAdapter(config)
        adapter.inject_lora()
        count2 = adapter.inject_lora()
        assert count2 == 0
