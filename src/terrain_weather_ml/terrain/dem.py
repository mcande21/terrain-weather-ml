"""DEM ingestion module.

Accepts GeoTIFF DEM rasters (SRTM 30m, Copernicus 30m, USGS 1/3 arc-second),
reprojects to target CRS, resamples to target resolution via rasterio.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import calculate_default_transform, reproject


@dataclass
class DEMData:
    """Container for DEM elevation data with geospatial metadata."""

    elevation: np.ndarray  # shape (H, W), float32
    crs: CRS
    transform: Affine
    resolution: float  # cell size in CRS units (meters for projected CRS)

    @property
    def height(self) -> int:
        return self.elevation.shape[0]

    @property
    def width(self) -> int:
        return self.elevation.shape[1]


def load_dem(
    path: str | Path,
    target_crs: CRS | None = None,
    target_resolution: float | None = None,
) -> DEMData:
    """Load a GeoTIFF DEM, optionally reprojecting and resampling.

    Args:
        path: Path to GeoTIFF DEM file.
        target_crs: Target coordinate reference system. If None, keeps source CRS.
        target_resolution: Target cell size in CRS units. If None, keeps source resolution.

    Returns:
        DEMData with elevation array and geospatial metadata.
    """
    path = Path(path)

    with rasterio.open(path) as src:
        src_crs = src.crs
        src_transform = src.transform
        src_height = src.height
        src_width = src.width
        elevation = src.read(1).astype(np.float32)

    need_reproject = target_crs is not None and target_crs != src_crs
    need_resample = target_resolution is not None

    if not need_reproject and not need_resample:
        # Compute resolution from transform (use absolute value of pixel size)
        res = abs(src_transform.a)
        return DEMData(
            elevation=elevation,
            crs=src_crs,
            transform=src_transform,
            resolution=res,
        )

    # Determine the effective CRS and compute the transform
    dst_crs = target_crs if target_crs is not None else src_crs

    if target_resolution is not None:
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs,
            dst_crs,
            src_width,
            src_height,
            *rasterio.open(path).bounds,
            resolution=target_resolution,
        )
    else:
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs,
            dst_crs,
            src_width,
            src_height,
            *rasterio.open(path).bounds,
        )

    dst_elevation = np.full(
        (dst_height, dst_width), fill_value=np.nan, dtype=np.float32
    )

    reproject(
        source=elevation,
        destination=dst_elevation,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
        dst_nodata=np.nan,
    )

    res = abs(dst_transform.a)

    return DEMData(
        elevation=dst_elevation,
        crs=dst_crs,
        transform=dst_transform,
        resolution=res,
    )
