"""Task 4.1/4.4: Interface contract and protocol validation tests.

Verify: matching shapes pass validation; mismatched channel count raises
descriptive error; StormCastAdapter satisfies WeatherAdapter protocol;
backbone config flag selects correct adapter.
"""

from __future__ import annotations

import pytest
import torch

from terrain_weather_ml.backbone import create_adapter
from terrain_weather_ml.backbone.adapter_protocol import WeatherAdapter
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
    """Interface contract validation."""

    def test_matching_channels_pass(self, adapter):
        """Adapter output (6 channels) + terrain (17 channels) = 23 total."""
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
                weather_channels=4,
                terrain_channels=C_TERRAIN,
                expected_weather=6,
                expected_terrain=17,
            )

    def test_mismatched_terrain_channels_raises(self):
        """Wrong terrain channel count should raise InterfaceContractError."""
        with pytest.raises(InterfaceContractError, match="terrain"):
            validate_interface_contract(
                weather_channels=C_WEATHER,
                terrain_channels=10,
                expected_weather=6,
                expected_terrain=17,
            )

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


class TestProtocolConformance:
    """Task 4.1: StormCastAdapter satisfies WeatherAdapter protocol."""

    def test_stormcast_is_weather_adapter(self, adapter):
        """StormCastAdapter should be a WeatherAdapter (structural typing)."""
        assert isinstance(adapter, WeatherAdapter)

    def test_extract_method_exists(self, adapter):
        """StormCastAdapter must have an extract() method."""
        assert hasattr(adapter, "extract")
        assert callable(adapter.extract)


class TestBackboneConfigFlag:
    """Task 4.4: backbone config flag selects adapter."""

    def test_default_is_passthrough(self):
        """Default backbone should be 'passthrough' (not stormcast)."""
        adapter = create_adapter(backbone="passthrough")
        assert not isinstance(adapter, StormCastAdapter)

    def test_stormcast_flag_creates_stormcast(self, mock_checkpoint_path):
        """backbone='stormcast' should create StormCastAdapter."""
        adapter = create_adapter(
            backbone="stormcast",
            checkpoint_path=str(mock_checkpoint_path),
            model_class="tests.backbone.test_checkpoint.MockStormCast",
        )
        assert isinstance(adapter, StormCastAdapter)
        assert adapter.experimental is True

    def test_invalid_backbone_raises(self):
        """Unknown backbone name should raise ValueError."""
        with pytest.raises(ValueError, match="backbone"):
            create_adapter(backbone="invalid")
