"""Task 4.2: Surface variable extraction tests.

StormCast provides 4 surface variables (T2M, U10, V10, MSLP) at indices
[2, 0, 1, 3]. The remaining 2 (PRATE, BLH) come from HRRR passthrough
to reach C_WEATHER=6 total.

Verify: StormCast extraction produces (B, 4, H, W) from correct indices;
extract() method combines 4+2 to produce (B, 6, H, W).
"""

from __future__ import annotations

import pytest
import torch

from terrain_weather_ml.backbone.stormcast import (
    C_WEATHER,
    HRRR_PASSTHROUGH_NAMES,
    STORMCAST_SURFACE_INDICES,
    STORMCAST_SURFACE_NAMES,
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


class TestSurfaceConstants:
    """Surface variable constants reflect real StormCast output."""

    def test_c_weather_is_6(self):
        """C_WEATHER constant should be 6 (combined output)."""
        assert C_WEATHER == 6

    def test_stormcast_provides_4_surface_vars(self):
        """StormCast provides exactly 4 surface variables."""
        assert len(STORMCAST_SURFACE_NAMES) == 4
        assert STORMCAST_SURFACE_NAMES == ["T2M", "U10", "V10", "MSLP"]

    def test_stormcast_surface_indices_correct(self):
        """StormCast surface indices map to real output positions."""
        # T2M=2, U10=0, V10=1, MSLP=3 in StormCast's 99-channel output
        assert STORMCAST_SURFACE_INDICES == [2, 0, 1, 3]

    def test_hrrr_passthrough_provides_2_vars(self):
        """HRRR passthrough supplies 2 variables StormCast lacks."""
        assert len(HRRR_PASSTHROUGH_NAMES) == 2
        assert HRRR_PASSTHROUGH_NAMES == ["PRATE", "BLH"]

    def test_combined_names_total_6(self):
        """Combined surface variable names should be 6."""
        assert len(SURFACE_VARIABLE_NAMES) == C_WEATHER
        expected = ["T2M", "U10", "V10", "MSLP", "PRATE", "BLH"]
        assert SURFACE_VARIABLE_NAMES == expected


class TestStormCastExtraction:
    """StormCast-only extraction (4 channels from 99-channel output)."""

    def test_extract_4_from_99(self, adapter):
        """Extract 4 surface variables from 99-channel output."""
        batch, h, w = 2, 16, 16
        full_output = torch.randn(batch, 99, h, w)
        surface = adapter.extract_surface_variables(full_output)
        assert surface.shape == (batch, 4, h, w)

    def test_extraction_uses_correct_indices(self, adapter):
        """Extracted values match output at indices [2, 0, 1, 3]."""
        batch, h, w = 1, 8, 8
        full_output = torch.randn(batch, 99, h, w)
        surface = adapter.extract_surface_variables(full_output)

        for i, idx in enumerate(STORMCAST_SURFACE_INDICES):
            torch.testing.assert_close(surface[:, i], full_output[:, idx])

    def test_extraction_single_sample(self, adapter):
        """Should work with batch size 1."""
        full_output = torch.randn(1, 99, 32, 32)
        surface = adapter.extract_surface_variables(full_output)
        assert surface.shape == (1, 4, 32, 32)

    def test_extraction_insufficient_channels_raises(self, adapter):
        """Should raise ValueError if output has fewer channels than needed."""
        full_output = torch.randn(1, 3, 8, 8)
        with pytest.raises(ValueError, match="channels"):
            adapter.extract_surface_variables(full_output)


class TestExtractMethod:
    """Protocol-conforming extract() combines 4 StormCast + 2 HRRR = 6."""

    def test_extract_produces_6_channels(self, adapter):
        """extract() returns (B, 6, H, W) tensor."""
        h, w = 8, 8
        nwp_data = {
            "stormcast_input": torch.randn(1, 99, h, w),
            "PRATE": torch.randn(1, 1, h, w),
            "BLH": torch.randn(1, 1, h, w),
        }
        result = adapter.extract(nwp_data)
        assert result.shape == (1, C_WEATHER, h, w)

    def test_extract_channels_in_correct_order(self, adapter):
        """First 4 channels from StormCast, last 2 from HRRR passthrough."""
        h, w = 8, 8
        stormcast_input = torch.randn(1, 99, h, w)
        prate = torch.randn(1, 1, h, w)
        blh = torch.randn(1, 1, h, w)
        nwp_data = {
            "stormcast_input": stormcast_input,
            "PRATE": prate,
            "BLH": blh,
        }
        result = adapter.extract(nwp_data)

        # Channels 4 and 5 should be PRATE and BLH
        torch.testing.assert_close(result[:, 4:5], prate)
        torch.testing.assert_close(result[:, 5:6], blh)

    def test_extract_missing_passthrough_raises(self, adapter):
        """extract() should raise if HRRR passthrough vars are missing."""
        nwp_data = {
            "stormcast_input": torch.randn(1, 99, 8, 8),
            # Missing PRATE and BLH
        }
        with pytest.raises(KeyError):
            adapter.extract(nwp_data)

    def test_forward_extracts_4_channels(self, adapter):
        """forward() with extract_surface=True returns 4-channel output."""
        x = torch.randn(1, 99, 8, 8)
        output = adapter(x, extract_surface=True)
        assert output.shape[1] == 4

    def test_forward_full_output_when_not_extracting(self, adapter):
        """forward() with extract_surface=False returns full 99-channel."""
        x = torch.randn(1, 99, 8, 8)
        output = adapter(x, extract_surface=False)
        assert output.shape[1] == 99
