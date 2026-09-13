"""Task 4.1: StormCast checkpoint loading tests.

Verify: successful load reports non-zero parameter count, all parameters have
requires_grad=False; missing checkpoint raises descriptive error.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from terrain_weather_ml.backbone.stormcast import (
    StormCastAdapter,
    StormCastConfig,
)

# ---------------------------------------------------------------------------
# Mock StormCast model for testing without real weights
# ---------------------------------------------------------------------------

class MockStormCastBlock(nn.Module):
    """Mimics a StormCast transformer block with attention + MLP."""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.attn_q = nn.Linear(dim, dim)
        self.attn_k = nn.Linear(dim, dim)
        self.attn_v = nn.Linear(dim, dim)
        self.attn_out = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        flat = x.reshape(b, c, h * w).permute(0, 2, 1)  # (B, HW, C)
        normed = self.norm1(flat)
        q = self.attn_q(normed)
        k = self.attn_k(normed)
        v = self.attn_v(normed)
        attn = torch.softmax(q @ k.transpose(-2, -1) / (c ** 0.5), dim=-1)
        out = self.attn_out(attn @ v)
        flat = flat + out
        flat = flat + self.mlp(self.norm2(flat))
        return flat.permute(0, 2, 1).reshape(b, c, h, w)


class MockStormCast(nn.Module):
    """Lightweight mock of StormCast architecture for unit tests.

    Produces 99-variable output from 99-variable input, mimicking the
    real StormCast's interface without the full model size.
    """

    NUM_VARS = 99

    def __init__(self, dim: int = 64, n_blocks: int = 2):
        super().__init__()
        self.input_proj = nn.Conv2d(self.NUM_VARS, dim, 1)
        self.blocks = nn.ModuleList([MockStormCastBlock(dim) for _ in range(n_blocks)])
        self.output_proj = nn.Conv2d(dim, self.NUM_VARS, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.output_proj(h)


@pytest.fixture
def mock_checkpoint_path(tmp_path):
    """Save a mock StormCast checkpoint to disk."""
    model = MockStormCast(dim=64, n_blocks=2)
    path = tmp_path / "stormcast_mock.pt"
    torch.save(model.state_dict(), path)
    return path


@pytest.fixture
def mock_config(mock_checkpoint_path):
    """Config pointing at the mock checkpoint."""
    return StormCastConfig(
        checkpoint_path=str(mock_checkpoint_path),
        model_class="tests.backbone.test_checkpoint.MockStormCast",
    )


class TestCheckpointLoading:
    """Task 4.1: StormCast checkpoint loading."""

    def test_load_reports_nonzero_param_count(self, mock_config):
        """Successful load reports non-zero parameter count."""
        adapter = StormCastAdapter(mock_config)
        info = adapter.model_info()
        assert info["total_params"] > 0

    def test_all_params_frozen(self, mock_config):
        """All backbone parameters must have requires_grad=False."""
        adapter = StormCastAdapter(mock_config)
        for name, param in adapter.backbone.named_parameters():
            assert not param.requires_grad, f"Parameter {name} is not frozen"

    def test_memory_footprint_reported(self, mock_config):
        """Model info should include memory footprint."""
        adapter = StormCastAdapter(mock_config)
        info = adapter.model_info()
        assert info["memory_mb"] > 0

    def test_missing_checkpoint_raises_error(self, tmp_path):
        """Missing checkpoint should raise a descriptive FileNotFoundError."""
        config = StormCastConfig(
            checkpoint_path=str(tmp_path / "nonexistent.pt"),
            model_class="tests.backbone.test_checkpoint.MockStormCast",
        )
        with pytest.raises(FileNotFoundError, match="checkpoint"):
            StormCastAdapter(config)

    def test_model_in_eval_mode(self, mock_config):
        """Loaded model should be in eval mode."""
        adapter = StormCastAdapter(mock_config)
        assert not adapter.backbone.training
