"""Task 4.2: Surface variable extraction tests.

Verify: extraction from a mock 99-channel tensor produces shape (6, H, W);
variable index validation catches metadata mismatch.
"""

from __future__ import annotations

import pytest
import torch

from terrain_weather_ml.backbone.stormcast import (
    C_WEATHER,
    SURFACE_VARIABLE_NAMES,
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
def adapter(mock_checkpoint_path):
    """Create a StormCastAdapter with mock model."""
    config = StormCastConfig(
        checkpoint_path=str(mock_checkpoint_path),
        model_class="tests.backbone.test_checkpoint.MockStormCast",
    )
    return StormCastAdapter(config)


class TestSurfaceExtraction:
    """Task 4.2: Surface variable extraction."""

    def test_c_weather_is_6(self):
        """C_WEATHER constant should be 6."""
        assert C_WEATHER == 6

    def test_surface_variable_names_count(self):
        """Should have exactly 6 surface variable names."""
        assert len(SURFACE_VARIABLE_NAMES) == C_WEATHER

    def test_surface_variable_names_content(self):
        """Variable names should match spec."""
        expected = ["T2M", "U10", "V10", "PRATE", "SP", "BLH"]
        assert SURFACE_VARIABLE_NAMES == expected

    def test_extraction_shape_from_99_channels(self, adapter):
        """Extract 6 variables from 99-channel output -> (B, 6, H, W)."""
        batch, h, w = 2, 16, 16
        full_output = torch.randn(batch, 99, h, w)
        surface = adapter.extract_surface_variables(full_output)
        assert surface.shape == (batch, C_WEATHER, h, w)

    def test_extraction_preserves_values(self, adapter):
        """Extracted values should match the original at the correct indices."""
        batch, h, w = 1, 8, 8
        full_output = torch.randn(batch, 99, h, w)
        surface = adapter.extract_surface_variables(full_output)

        for i, idx in enumerate(adapter.surface_indices):
            torch.testing.assert_close(surface[:, i], full_output[:, idx])

    def test_extraction_single_sample(self, adapter):
        """Should work with batch size 1."""
        full_output = torch.randn(1, 99, 32, 32)
        surface = adapter.extract_surface_variables(full_output)
        assert surface.shape == (1, C_WEATHER, 32, 32)

    def test_extraction_insufficient_channels_raises(self, adapter):
        """Should raise ValueError if output has fewer channels than needed."""
        full_output = torch.randn(1, 3, 8, 8)
        with pytest.raises(ValueError, match="channels"):
            adapter.extract_surface_variables(full_output)

    def test_forward_extracts_surface_by_default(self, adapter):
        """forward() with extract_surface=True returns 6-channel output."""
        x = torch.randn(1, 99, 8, 8)
        output = adapter(x, extract_surface=True)
        assert output.shape[1] == C_WEATHER

    def test_forward_full_output_when_not_extracting(self, adapter):
        """forward() with extract_surface=False returns full 99-channel output."""
        x = torch.randn(1, 99, 8, 8)
        output = adapter(x, extract_surface=False)
        assert output.shape[1] == 99

    def test_custom_surface_indices(self, mock_checkpoint_path):
        """Custom surface indices should select different channels."""
        custom_indices = [10, 20, 30, 40, 50, 60]
        config = StormCastConfig(
            checkpoint_path=str(mock_checkpoint_path),
            model_class="tests.backbone.test_checkpoint.MockStormCast",
            surface_indices=custom_indices,
        )
        adapter = StormCastAdapter(config)

        full_output = torch.randn(1, 99, 8, 8)
        surface = adapter.extract_surface_variables(full_output)
        assert surface.shape == (1, 6, 8, 8)

        for i, idx in enumerate(custom_indices):
            torch.testing.assert_close(surface[:, i], full_output[:, idx])
