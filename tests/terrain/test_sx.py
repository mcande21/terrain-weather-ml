"""Task 1.4: Sx wind exposure computation tests.

Verify: ridgeline pixel has negative Sx (exposed), valley-bottom pixel has
positive Sx (sheltered), values are in degrees.
"""

import numpy as np
import pytest

from terrain_weather_ml.terrain.sx import compute_sx, SX_DIRECTIONS


class TestSx:
    def test_sx_directions_count(self):
        """Should compute Sx for 8 cardinal directions."""
        assert len(SX_DIRECTIONS) == 8

    def test_ridgeline_negative_sx(self):
        """A ridgeline pixel should have negative Sx (exposed) in cross-ridge directions."""
        size = 101
        center = size // 2
        dem = np.zeros((size, size), dtype=np.float32)
        # Create a north-south ridge along the center column
        for j in range(size):
            # Ridge profile: Gaussian peak at center column
            dem[:, j] = 500.0 * np.exp(-((j - center) ** 2) / (2 * 5**2))

        sx = compute_sx(dem, resolution=30.0, max_radius=900.0)
        # At the ridge top, looking east or west should be exposed (negative Sx)
        # E direction is index 2, W direction is index 6
        assert sx[2, center, center] < 0  # East: looking downhill = exposed
        assert sx[6, center, center] < 0  # West: looking downhill = exposed

    def test_valley_positive_sx(self):
        """A valley-bottom pixel should have positive Sx (sheltered)."""
        size = 101
        center = size // 2
        dem = np.zeros((size, size), dtype=np.float32)
        # Create a valley: higher elevation on sides
        for j in range(size):
            dem[:, j] = 500.0 * (1 - np.exp(-((j - center) ** 2) / (2 * 5**2)))

        sx = compute_sx(dem, resolution=30.0, max_radius=900.0)
        # At valley bottom, looking east or west should be sheltered (positive Sx)
        assert sx[2, center, center] > 0  # East: looking uphill = sheltered
        assert sx[6, center, center] > 0  # West: looking uphill = sheltered

    def test_flat_surface_zero_sx(self):
        """A flat surface should have Sx approximately 0."""
        dem = np.ones((80, 80), dtype=np.float32) * 500.0
        sx = compute_sx(dem, resolution=30.0, max_radius=900.0)
        interior = sx[:, 10:-10, 10:-10]
        np.testing.assert_allclose(interior, 0.0, atol=0.5)

    def test_sx_output_shape(self):
        """Output shape should be (8, H, W)."""
        dem = np.random.rand(40, 50).astype(np.float32) * 1000
        sx = compute_sx(dem, resolution=30.0, max_radius=600.0)
        assert sx.shape == (8, 40, 50)

    def test_sx_values_in_degrees(self):
        """Sx values should be in reasonable degree range [-90, 90]."""
        np.random.seed(42)
        dem = np.random.rand(60, 60).astype(np.float32) * 1000
        sx = compute_sx(dem, resolution=30.0, max_radius=600.0)
        assert sx.min() >= -90.0
        assert sx.max() <= 90.0
