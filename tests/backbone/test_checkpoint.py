"""Task 4.1: StormCast checkpoint loading tests.

Verify: successful load reports non-zero parameter count, all parameters have
requires_grad=False; missing checkpoint raises descriptive error; .mdlus format
loading repacks zip to strip model/ prefix.
"""

from __future__ import annotations

import io
import zipfile

import pytest
import torch
from torch import nn

from terrain_weather_ml.backbone.stormcast import (
    StormCastAdapter,
    StormCastConfig,
)

# ---------------------------------------------------------------------------
# Mock StormCast model for testing without real weights
#
# Mirrors real StormCast structure:
# - Each block has `affine` nn.Linear layers (LoRA targets)
# - A fused QKV Conv2d per block (LoRA should skip these)
# - 99-channel input/output
# ---------------------------------------------------------------------------

class MockStormCastSubBlock(nn.Module):
    """Sub-block with an `affine` linear layer -- a valid LoRA target."""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.affine = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.affine(x)) + x


class MockStormCastBlock(nn.Module):
    """Mimics a StormCast UNet block.

    Has 3 sub-blocks each with an `affine` nn.Linear (valid LoRA targets)
    and a fused QKV Conv2d (should be skipped by LoRA injection).
    """

    def __init__(self, dim: int = 64):
        super().__init__()
        # Fused QKV attention (Conv2d -- should NOT be targeted by LoRA)
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.attn_out = nn.Linear(dim, dim)
        # 3 sub-blocks each with an `affine` layer
        self.sub1 = MockStormCastSubBlock(dim)
        self.sub2 = MockStormCastSubBlock(dim)
        self.sub3 = MockStormCastSubBlock(dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        flat = x.reshape(b, c, h * w).permute(0, 2, 1)  # (B, HW, C)
        flat = self.sub1(flat)
        flat = self.sub2(flat)
        flat = self.sub3(flat)
        flat = self.norm(flat)
        return flat.permute(0, 2, 1).reshape(b, c, h, w)


class MockStormCast(nn.Module):
    """Lightweight mock of StormCast architecture for unit tests.

    Produces 99-variable output from 99-variable input, mimicking the
    real StormCast's interface without the full model size.

    Structure mirrors real StormCast:
    - `affine` nn.Linear layers in sub-blocks (valid LoRA targets)
    - Fused QKV Conv2d per block (LoRA must skip)
    - 2 blocks x 3 sub-blocks = 6 `affine` layers total
    """

    NUM_VARS = 99

    def __init__(self, dim: int = 64, n_blocks: int = 2):
        super().__init__()
        self.input_proj = nn.Conv2d(self.NUM_VARS, dim, 1)
        self.blocks = nn.ModuleList(
            [MockStormCastBlock(dim) for _ in range(n_blocks)]
        )
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
def mock_mdlus_path(tmp_path):
    """Save a mock StormCast checkpoint in .mdlus (Modulus zip) format.

    The real .mdlus format stores weights under a model/ prefix inside
    a zip archive. The loader must strip this prefix before loading.
    """
    model = MockStormCast(dim=64, n_blocks=2)
    state_dict = model.state_dict()

    # Serialize state_dict to bytes
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    weights_bytes = buf.getvalue()

    # Pack into a zip with model/ prefix (real Modulus format)
    mdlus_path = tmp_path / "StormCastUNet.0.0.mdlus"
    with zipfile.ZipFile(mdlus_path, "w") as zf:
        zf.writestr("model/weights.pt", weights_bytes)

    return mdlus_path


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

    def test_marked_experimental(self, mock_config):
        """StormCastAdapter must be marked as experimental."""
        adapter = StormCastAdapter(mock_config)
        assert adapter.experimental is True

    def test_mdlus_format_loading(self, mock_mdlus_path):
        """NVIDIA Modulus .mdlus format should be loaded by repacking zip."""
        config = StormCastConfig(
            checkpoint_path=str(mock_mdlus_path),
            model_class="tests.backbone.test_checkpoint.MockStormCast",
        )
        adapter = StormCastAdapter(config)
        info = adapter.model_info()
        assert info["total_params"] > 0
        assert info["frozen"]
