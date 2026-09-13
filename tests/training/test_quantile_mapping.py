"""Task 6.4: Quantile mapping module tests.

Verify: mapping transforms a sample from Alpine distribution to match Colorado
distribution moments; per-season stratification produces different mappings
for winter vs. summer; round-trip map/inverse-map recovers original values
within tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from terrain_weather_ml.training.quantile_mapping import (
    QuantileMapper,
    Season,
    compute_empirical_cdf,
)


@pytest.fixture
def alpine_data():
    """Synthetic Alpine station data: higher precip, lower temp."""
    rng = np.random.default_rng(42)
    n = 2000
    return {
        "temperature": torch.tensor(
            rng.normal(loc=275.0, scale=10.0, size=n), dtype=torch.float32
        ),
        "wind_speed": torch.tensor(
            rng.exponential(scale=5.0, size=n), dtype=torch.float32
        ),
        "precipitation": torch.tensor(
            rng.exponential(scale=3.0, size=n), dtype=torch.float32
        ),
    }


@pytest.fixture
def colorado_data():
    """Synthetic Colorado station data: lower precip, higher temp."""
    rng = np.random.default_rng(123)
    n = 1500
    return {
        "temperature": torch.tensor(
            rng.normal(loc=280.0, scale=12.0, size=n), dtype=torch.float32
        ),
        "wind_speed": torch.tensor(
            rng.exponential(scale=7.0, size=n), dtype=torch.float32
        ),
        "precipitation": torch.tensor(
            rng.exponential(scale=2.0, size=n), dtype=torch.float32
        ),
    }


@pytest.fixture
def alpine_seasonal():
    """Synthetic Alpine data with winter and summer seasons."""
    rng = np.random.default_rng(42)
    n = 1000
    return {
        Season.WINTER: {
            "temperature": torch.tensor(
                rng.normal(loc=265.0, scale=8.0, size=n), dtype=torch.float32
            ),
        },
        Season.SUMMER: {
            "temperature": torch.tensor(
                rng.normal(loc=290.0, scale=6.0, size=n), dtype=torch.float32
            ),
        },
    }


@pytest.fixture
def colorado_seasonal():
    """Synthetic Colorado data with winter and summer seasons."""
    rng = np.random.default_rng(123)
    n = 800
    return {
        Season.WINTER: {
            "temperature": torch.tensor(
                rng.normal(loc=260.0, scale=10.0, size=n), dtype=torch.float32
            ),
        },
        Season.SUMMER: {
            "temperature": torch.tensor(
                rng.normal(loc=295.0, scale=7.0, size=n), dtype=torch.float32
            ),
        },
    }


class TestEmpiricalCDF:
    """Empirical CDF computation."""

    def test_cdf_sorted_quantiles(self, alpine_data):
        """CDF quantile values should be monotonically non-decreasing."""
        values, quantiles = compute_empirical_cdf(alpine_data["temperature"])
        assert torch.all(values[1:] >= values[:-1])
        assert quantiles[0] == pytest.approx(0.0, abs=0.01)
        assert quantiles[-1] == pytest.approx(1.0, abs=0.01)

    def test_cdf_covers_data_range(self, alpine_data):
        """CDF should cover the full range of the input data."""
        data = alpine_data["temperature"]
        values, _ = compute_empirical_cdf(data)
        assert values[0] <= data.min()
        assert values[-1] >= data.max()


class TestQuantileMapper:
    """Quantile mapping between Alpine and Colorado distributions."""

    def test_mapped_distribution_moments(self, alpine_data, colorado_data):
        """Mapped Alpine data should have moments closer to Colorado."""
        mapper = QuantileMapper()
        mapper.fit(
            source_data=alpine_data,
            target_data=colorado_data,
        )

        mapped_temp = mapper.transform(
            alpine_data["temperature"], variable="temperature"
        )

        # Mapped mean should be closer to Colorado mean than Alpine mean
        alpine_mean = alpine_data["temperature"].mean().item()
        colorado_mean = colorado_data["temperature"].mean().item()
        mapped_mean = mapped_temp.mean().item()

        assert abs(mapped_mean - colorado_mean) < abs(alpine_mean - colorado_mean)

    def test_mapped_preserves_ordering(self, alpine_data, colorado_data):
        """Quantile mapping should preserve the rank ordering of values."""
        mapper = QuantileMapper()
        mapper.fit(source_data=alpine_data, target_data=colorado_data)

        original = alpine_data["temperature"]
        mapped = mapper.transform(original, variable="temperature")

        # Sort indices should be identical
        orig_order = torch.argsort(original)
        mapped_order = torch.argsort(mapped)
        assert torch.equal(orig_order, mapped_order)

    def test_per_season_different_mappings(
        self, alpine_seasonal, colorado_seasonal
    ):
        """Winter and summer mappings should differ."""
        mapper_winter = QuantileMapper()
        mapper_winter.fit(
            source_data=alpine_seasonal[Season.WINTER],
            target_data=colorado_seasonal[Season.WINTER],
        )

        mapper_summer = QuantileMapper()
        mapper_summer.fit(
            source_data=alpine_seasonal[Season.SUMMER],
            target_data=colorado_seasonal[Season.SUMMER],
        )

        test_val = torch.tensor([275.0])
        mapped_winter = mapper_winter.transform(test_val, variable="temperature")
        mapped_summer = mapper_summer.transform(test_val, variable="temperature")

        assert not torch.allclose(mapped_winter, mapped_summer, atol=0.1)

    def test_roundtrip_recovery(self, alpine_data, colorado_data):
        """Map then inverse-map should recover original values."""
        mapper = QuantileMapper()
        mapper.fit(source_data=alpine_data, target_data=colorado_data)

        original = alpine_data["temperature"]
        mapped = mapper.transform(original, variable="temperature")
        recovered = mapper.inverse_transform(mapped, variable="temperature")

        torch.testing.assert_close(recovered, original, atol=0.5, rtol=0.01)

    def test_serialization(self, alpine_data, colorado_data, tmp_path):
        """Quantile mapper should be serializable for checkpoint storage."""
        mapper = QuantileMapper()
        mapper.fit(source_data=alpine_data, target_data=colorado_data)

        params = mapper.get_params()
        assert isinstance(params, dict)
        assert "temperature" in params

        # Load into a new mapper
        mapper2 = QuantileMapper()
        mapper2.load_params(params)

        test_val = alpine_data["temperature"][:10]
        mapped1 = mapper.transform(test_val, variable="temperature")
        mapped2 = mapper2.transform(test_val, variable="temperature")
        torch.testing.assert_close(mapped1, mapped2)

    def test_multiple_variables(self, alpine_data, colorado_data):
        """Mapper should handle multiple variables independently."""
        mapper = QuantileMapper()
        mapper.fit(source_data=alpine_data, target_data=colorado_data)

        for var in ["temperature", "wind_speed", "precipitation"]:
            mapped = mapper.transform(
                alpine_data[var], variable=var
            )
            assert mapped.shape == alpine_data[var].shape

    def test_extrapolation_clamps(self, alpine_data, colorado_data):
        """Values outside the training range should be clamped."""
        mapper = QuantileMapper()
        mapper.fit(source_data=alpine_data, target_data=colorado_data)

        extreme = torch.tensor([150.0, 400.0])
        mapped = mapper.transform(extreme, variable="temperature")
        # Should not produce NaN or Inf
        assert torch.isfinite(mapped).all()
