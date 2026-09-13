"""Task 1.1: DEM ingestion module tests.

Verify: test loads a synthetic DEM, reprojects CRS, outputs elevation grid
at requested resolution with CRS metadata preserved.
"""

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from terrain_weather_ml.terrain.dem import load_dem, DEMData


@pytest.fixture
def synthetic_dem_path(tmp_path):
    """Create a small synthetic GeoTIFF DEM in EPSG:4326."""
    filepath = tmp_path / "test_dem.tif"
    height, width = 100, 120
    # Synthetic elevation: a tilted plane + gaussian peak
    y = np.linspace(0, 1, height)
    x = np.linspace(0, 1, width)
    xx, yy = np.meshgrid(x, y)
    elevation = 1000.0 + 500.0 * yy + 200.0 * np.exp(
        -((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / 0.02
    )
    elevation = elevation.astype(np.float32)

    # WGS84 bounding box in Swiss Alps
    west, south, east, north = 7.0, 46.0, 7.1, 46.1
    transform = from_bounds(west, south, east, north, width, height)

    with rasterio.open(
        filepath,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=CRS.from_epsg(4326),
        transform=transform,
    ) as dst:
        dst.write(elevation, 1)

    return filepath


def test_load_dem_returns_dem_data(synthetic_dem_path):
    """Loading a DEM returns a DEMData with elevation array and metadata."""
    result = load_dem(synthetic_dem_path)
    assert isinstance(result, DEMData)
    assert result.elevation.ndim == 2
    assert result.elevation.shape[0] > 0
    assert result.elevation.shape[1] > 0
    assert result.crs is not None
    assert result.transform is not None
    assert result.resolution is not None


def test_load_dem_preserves_values(synthetic_dem_path):
    """Loaded elevation values should match the source data."""
    result = load_dem(synthetic_dem_path)
    # Values should be in the range we created (1000-1700 roughly)
    assert result.elevation.min() >= 900.0
    assert result.elevation.max() <= 1800.0
    assert result.elevation.dtype == np.float32


def test_load_dem_with_target_crs(synthetic_dem_path):
    """Reprojecting from EPSG:4326 to UTM should produce a UTM CRS."""
    target_crs = CRS.from_epsg(32632)  # UTM zone 32N
    result = load_dem(synthetic_dem_path, target_crs=target_crs)
    assert result.crs == target_crs
    assert result.elevation.ndim == 2
    assert result.elevation.shape[0] > 0


def test_load_dem_with_target_resolution(synthetic_dem_path):
    """Resampling to a target resolution should change grid dimensions."""
    # Reproject to UTM first (meters), then resample to 100m
    target_crs = CRS.from_epsg(32632)
    result = load_dem(
        synthetic_dem_path, target_crs=target_crs, target_resolution=100.0
    )
    assert result.crs == target_crs
    # Resolution should be approximately 100m
    assert abs(result.resolution - 100.0) < 1.0


def test_load_dem_mismatched_crs(synthetic_dem_path):
    """When input CRS differs from target, it should be reprojected."""
    # Input is EPSG:4326, target is UTM
    target_crs = CRS.from_epsg(32632)
    result = load_dem(synthetic_dem_path, target_crs=target_crs)
    # CRS should be the target, not the input
    assert result.crs == target_crs
    # Elevation values should be preserved (check valid pixels only,
    # reprojection introduces NaN at borders)
    valid = result.elevation[~np.isnan(result.elevation)]
    assert valid.min() >= 900.0
    assert valid.max() <= 1800.0
