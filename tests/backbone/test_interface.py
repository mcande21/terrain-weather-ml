"""Task 4.5: Interface contract validation tests.

Verify: matching shapes pass validation; mismatched channel count
raises descriptive error.
"""

from __future__ import annotations

import pytest
import torch

from terrain_weather_ml.backbone.interface import (
    InterfaceContractError,
    validate_interface_contract,
)
from terrain_weather_ml.backbone.stormcast import (
    C_WEATHER,
    StormCastAdapter,
    StormCastConfig,
)
from terrain_weather_ml.terrain import C_TERRAIN
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
    """Create a StormCastAdapter."""
    config = StormCastConfig(
        checkpoint_path=str(mock_checkpoint_path),
        model_class="tests.backbone.test_checkpoint.MockStormCast",
    )
    return StormCastAdapter(config)


class TestInterfaceContract:
    """Task 4.5: Interface contract validation."""

    def test_matching_channels_pass(self, adapter):
        """Adapter output (6 channels) + terrain (17 channels) = 23 total."""
        # Validate the adapter's output channels match the expected dynamic
        # input for the downscaling head
        validate_interface_contract(
            weather_channels=C_WEATHER,
            terrain_channels=C_TERRAIN,
            expected_weather=6,
            expected_terrain=17,
        )

    def test_correct_total_input_channels(self, adapter):
        """Total concatenated channels should be C_WEATHER + C_TERRAIN = 23."""
        total = C_WEATHER + C_TERRAIN
        assert total == 23

    def test_mismatched_weather_channels_raises(self):
        """Wrong weather channel count should raise InterfaceContractError."""
        with pytest.raises(InterfaceContractError, match="weather"):
            validate_interface_contract(
                weather_channels=4,  # wrong, should be 6
                terrain_channels=C_TERRAIN,
                expected_weather=6,
                expected_terrain=17,
            )

    def test_mismatched_terrain_channels_raises(self):
        """Wrong terrain channel count should raise InterfaceContractError."""
        with pytest.raises(InterfaceContractError, match="terrain"):
            validate_interface_contract(
                weather_channels=C_WEATHER,
                terrain_channels=10,  # wrong, should be 17
                expected_weather=6,
                expected_terrain=17,
            )

    def test_adapter_output_shape_matches_contract(self, adapter):
        """Adapter forward pass should produce the contracted output shape."""
        x = torch.randn(1, 99, 16, 16)
        output = adapter(x, extract_surface=True)
        assert output.shape[1] == C_WEATHER

    def test_validate_with_tensors(self, adapter):
        """validate_interface_contract should work with actual tensor shapes."""
        weather = torch.randn(1, C_WEATHER, 8, 8)
        terrain = torch.randn(1, C_TERRAIN, 64, 64)

        validate_interface_contract(
            weather_channels=weather.shape[1],
            terrain_channels=terrain.shape[1],
            expected_weather=C_WEATHER,
            expected_terrain=C_TERRAIN,
        )

    def test_error_message_is_descriptive(self):
        """Error message should describe the expected vs actual channels."""
        with pytest.raises(InterfaceContractError) as exc_info:
            validate_interface_contract(
                weather_channels=4,
                terrain_channels=C_TERRAIN,
                expected_weather=6,
                expected_terrain=17,
            )
        msg = str(exc_info.value)
        assert "4" in msg
        assert "6" in msg
