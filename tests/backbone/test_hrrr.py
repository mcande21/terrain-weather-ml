"""Task 4.4: HRRR input conditioning tests.

Verify: GRIB2 processing on a fixture produces a valid input tensor
matching StormCast's expected shape and value ranges.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from terrain_weather_ml.backbone.hrrr import (
    HRRRConditioner,
    HRRRConfig,
    HRRRData,
)


@pytest.fixture
def hrrr_config():
    """Default HRRR conditioner configuration."""
    return HRRRConfig(
        num_input_vars=99,
        grid_height=32,
        grid_width=32,
    )


@pytest.fixture
def conditioner(hrrr_config):
    """Create an HRRRConditioner."""
    return HRRRConditioner(hrrr_config)


@pytest.fixture
def mock_grib_data():
    """Create mock HRRR data mimicking GRIB2-extracted fields.

    Returns raw numpy arrays simulating what cfgrib/xarray would produce.
    """
    h, w = 32, 32
    # Realistic value ranges for key HRRR variables
    fields = {}

    # Temperature fields (Kelvin) ~250-310K
    for i in range(20):
        fields[f"t_{i}"] = np.random.uniform(250, 310, (h, w)).astype(np.float32)

    # Wind fields (m/s) ~-40 to 40
    for i in range(20):
        fields[f"u_{i}"] = np.random.uniform(-40, 40, (h, w)).astype(np.float32)
        fields[f"v_{i}"] = np.random.uniform(-40, 40, (h, w)).astype(np.float32)

    # Pressure fields (Pa) ~50000-105000
    for i in range(10):
        fields[f"p_{i}"] = np.random.uniform(50000, 105000, (h, w)).astype(np.float32)

    # Remaining fields
    for i in range(99 - 70):
        fields[f"misc_{i}"] = np.random.uniform(-100, 100, (h, w)).astype(np.float32)

    return HRRRData(
        fields=fields,
        grid_shape=(h, w),
        valid_time="2024-01-15T12:00:00",
        cycle="2024-01-15T06:00:00",
    )


class TestHRRRConditioner:
    """Task 4.4: HRRR input conditioning."""

    def test_format_produces_correct_shape(self, conditioner, mock_grib_data):
        """Formatted HRRR data should produce (1, 99, H, W) tensor."""
        tensor = conditioner.format_input(mock_grib_data)
        assert tensor.shape == (1, 99, 32, 32)

    def test_output_is_float32(self, conditioner, mock_grib_data):
        """Output tensor should be float32."""
        tensor = conditioner.format_input(mock_grib_data)
        assert tensor.dtype == torch.float32

    def test_output_has_no_nan(self, conditioner, mock_grib_data):
        """Output tensor should not contain NaN values."""
        tensor = conditioner.format_input(mock_grib_data)
        assert not torch.isnan(tensor).any()

    def test_output_has_no_inf(self, conditioner, mock_grib_data):
        """Output tensor should not contain Inf values."""
        tensor = conditioner.format_input(mock_grib_data)
        assert not torch.isinf(tensor).any()

    def test_normalization_applied(self, conditioner, mock_grib_data):
        """After normalization, values should be in a reasonable range."""
        tensor = conditioner.format_input(mock_grib_data)
        # After z-score normalization, most values should be within [-5, 5]
        assert tensor.abs().max() < 100  # loose bound for safety

    def test_batch_formatting(self, conditioner, mock_grib_data):
        """Should handle a list of HRRR data as a batch."""
        tensors = conditioner.format_batch([mock_grib_data, mock_grib_data])
        assert tensors.shape == (2, 99, 32, 32)

    def test_missing_fields_raises(self, conditioner):
        """Should raise ValueError when required fields are missing."""
        incomplete = HRRRData(
            fields={"t_0": np.zeros((32, 32), dtype=np.float32)},
            grid_shape=(32, 32),
            valid_time="2024-01-15T12:00:00",
            cycle="2024-01-15T06:00:00",
        )
        with pytest.raises(ValueError, match="Expected 99"):
            conditioner.format_input(incomplete)

    def test_grid_shape_mismatch_raises(self, conditioner):
        """Fields with inconsistent spatial dimensions should raise."""
        fields = {f"var_{i}": np.zeros((32, 32), dtype=np.float32) for i in range(98)}
        fields["var_98"] = np.zeros((16, 16), dtype=np.float32)  # wrong shape
        bad_data = HRRRData(
            fields=fields,
            grid_shape=(32, 32),
            valid_time="2024-01-15T12:00:00",
            cycle="2024-01-15T06:00:00",
        )
        with pytest.raises(ValueError, match="shape"):
            conditioner.format_input(bad_data)


class TestHRRRData:
    """HRRRData container tests."""

    def test_hrrr_data_fields(self, mock_grib_data):
        """HRRRData should store fields and metadata."""
        assert isinstance(mock_grib_data.fields, dict)
        assert mock_grib_data.grid_shape == (32, 32)
        assert mock_grib_data.valid_time is not None
        assert mock_grib_data.cycle is not None
