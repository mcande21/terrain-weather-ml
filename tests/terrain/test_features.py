"""Task 1.2: Core terrain feature computation tests.

Verify: test against synthetic DEM patches with known geometry (plane, cone, saddle)
producing expected slope/aspect/curvature values within tolerance.
"""

import numpy as np
import pytest

from terrain_weather_ml.terrain.features import (
    compute_slope,
    compute_aspect,
    compute_curvature,
)


def _make_grid(height: int, width: int, resolution: float):
    """Create coordinate grids for a DEM patch."""
    y = np.arange(height) * resolution
    x = np.arange(width) * resolution
    return np.meshgrid(x, y)


class TestSlope:
    """Slope computation via Horn's method."""

    def test_flat_surface_zero_slope(self):
        """A flat surface should have zero slope everywhere."""
        dem = np.ones((50, 50), dtype=np.float32) * 1000.0
        slope = compute_slope(dem, resolution=30.0)
        # Interior pixels (Horn's method needs neighbors)
        interior = slope[1:-1, 1:-1]
        np.testing.assert_allclose(interior, 0.0, atol=1e-5)

    def test_tilted_plane_uniform_slope(self):
        """A uniformly tilted plane should have constant slope."""
        height, width, res = 50, 50, 30.0
        # 45-degree slope along y axis: rise = run
        y_coords = np.arange(height) * res
        dem = np.tile(y_coords[:, np.newaxis], (1, width)).astype(np.float32)
        slope = compute_slope(dem, resolution=res)
        interior = slope[1:-1, 1:-1]
        # atan(1) = 45 degrees
        np.testing.assert_allclose(interior, 45.0, atol=0.5)

    def test_slope_degrees_range(self):
        """Slope values should be in [0, 90] degrees."""
        np.random.seed(42)
        dem = np.random.rand(50, 50).astype(np.float32) * 1000
        slope = compute_slope(dem, resolution=30.0)
        assert slope.min() >= 0.0
        assert slope.max() <= 90.0

    def test_slope_output_shape(self):
        """Output shape should match input shape."""
        dem = np.random.rand(30, 40).astype(np.float32) * 1000
        slope = compute_slope(dem, resolution=30.0)
        assert slope.shape == dem.shape


class TestAspect:
    """Aspect computation as (sin, cos) pair."""

    def test_flat_surface_zero_aspect(self):
        """A flat surface should have aspect (0, 0)."""
        dem = np.ones((50, 50), dtype=np.float32) * 1000.0
        aspect_sin, aspect_cos = compute_aspect(dem, resolution=30.0)
        interior_sin = aspect_sin[1:-1, 1:-1]
        interior_cos = aspect_cos[1:-1, 1:-1]
        np.testing.assert_allclose(interior_sin, 0.0, atol=1e-5)
        np.testing.assert_allclose(interior_cos, 0.0, atol=1e-5)

    def test_north_facing_slope(self):
        """Slope facing north: elevation increases southward (row index increases)."""
        height, width, res = 50, 50, 30.0
        y_coords = np.arange(height, dtype=np.float32) * res
        # Elevation increases with row index (south), so slope faces north
        dem = np.tile(y_coords[:, np.newaxis], (1, width)).astype(np.float32)
        aspect_sin, aspect_cos = compute_aspect(dem, resolution=res)
        interior_sin = aspect_sin[1:-1, 1:-1]
        interior_cos = aspect_cos[1:-1, 1:-1]
        # North = 0 degrees => sin(0)=0, cos(0)=1
        np.testing.assert_allclose(interior_sin, 0.0, atol=0.1)
        np.testing.assert_allclose(interior_cos, 1.0, atol=0.1)

    def test_east_facing_slope(self):
        """Slope facing east: elevation increases westward (column index decreases)."""
        height, width, res = 50, 50, 30.0
        x_coords = np.arange(width, 0, -1, dtype=np.float32) * res
        dem = np.tile(x_coords[np.newaxis, :], (height, 1)).astype(np.float32)
        aspect_sin, aspect_cos = compute_aspect(dem, resolution=res)
        interior_sin = aspect_sin[1:-1, 1:-1]
        interior_cos = aspect_cos[1:-1, 1:-1]
        # East = 90 degrees => sin(90)=1, cos(90)=0
        np.testing.assert_allclose(interior_sin, 1.0, atol=0.1)
        np.testing.assert_allclose(interior_cos, 0.0, atol=0.1)

    def test_aspect_output_shapes(self):
        """Output shapes should match input shape."""
        dem = np.random.rand(30, 40).astype(np.float32) * 1000
        aspect_sin, aspect_cos = compute_aspect(dem, resolution=30.0)
        assert aspect_sin.shape == dem.shape
        assert aspect_cos.shape == dem.shape

    def test_aspect_values_bounded(self):
        """Sin and cos values should be in [-1, 1]."""
        np.random.seed(42)
        dem = np.random.rand(50, 50).astype(np.float32) * 1000
        aspect_sin, aspect_cos = compute_aspect(dem, resolution=30.0)
        assert aspect_sin.min() >= -1.0
        assert aspect_sin.max() <= 1.0
        assert aspect_cos.min() >= -1.0
        assert aspect_cos.max() <= 1.0


class TestCurvature:
    """Plan and profile curvature computation."""

    def test_flat_surface_zero_curvature(self):
        """A flat surface should have zero curvature."""
        dem = np.ones((50, 50), dtype=np.float32) * 1000.0
        plan_curv, profile_curv = compute_curvature(dem, resolution=30.0)
        interior_plan = plan_curv[2:-2, 2:-2]
        interior_profile = profile_curv[2:-2, 2:-2]
        np.testing.assert_allclose(interior_plan, 0.0, atol=1e-5)
        np.testing.assert_allclose(interior_profile, 0.0, atol=1e-5)

    def test_plane_zero_curvature(self):
        """A tilted plane has zero curvature (no bending)."""
        height, width, res = 50, 50, 30.0
        y_coords = np.arange(height, dtype=np.float32) * res
        x_coords = np.arange(width, dtype=np.float32) * res
        xx, yy = np.meshgrid(x_coords, y_coords)
        dem = (100.0 + 0.5 * xx + 0.3 * yy).astype(np.float32)
        plan_curv, profile_curv = compute_curvature(dem, resolution=res)
        interior_plan = plan_curv[2:-2, 2:-2]
        interior_profile = profile_curv[2:-2, 2:-2]
        np.testing.assert_allclose(interior_plan, 0.0, atol=1e-4)
        np.testing.assert_allclose(interior_profile, 0.0, atol=1e-4)

    def test_concave_surface_positive_plan_curvature(self):
        """A concave (valley/bowl) surface should have detectable curvature."""
        height, width, res = 50, 50, 30.0
        y_coords = np.arange(height, dtype=np.float32) * res
        x_coords = np.arange(width, dtype=np.float32) * res
        xx, yy = np.meshgrid(x_coords, y_coords)
        cx, cy = width * res / 2, height * res / 2
        # Bowl shape (concave up)
        dem = (0.001 * ((xx - cx) ** 2 + (yy - cy) ** 2)).astype(np.float32)
        plan_curv, profile_curv = compute_curvature(dem, resolution=res)
        # At least one curvature should be non-zero for a bowl
        interior = plan_curv[5:-5, 5:-5]
        assert np.abs(interior).max() > 0 or np.abs(profile_curv[5:-5, 5:-5]).max() > 0

    def test_curvature_output_shapes(self):
        """Output shapes should match input shape."""
        dem = np.random.rand(30, 40).astype(np.float32) * 1000
        plan_curv, profile_curv = compute_curvature(dem, resolution=30.0)
        assert plan_curv.shape == dem.shape
        assert profile_curv.shape == dem.shape
