"""Topographic Position Index (TPI) computation.

TPI = elevation - mean elevation within a specified radius.
Positive = peak/ridge, negative = valley/depression.
"""

from __future__ import annotations

import numpy as np


def compute_tpi(
    dem: np.ndarray,
    resolution: float,
    radius: float = 500.0,
) -> np.ndarray:
    """Compute Topographic Position Index at a given radius.

    TPI is the difference between the elevation of a cell and the mean
    elevation of its neighborhood within the specified radius.

    Args:
        dem: Elevation array (H, W) in meters.
        resolution: Cell size in meters.
        radius: Neighborhood radius in meters.

    Returns:
        TPI array in meters, shape (H, W).
    """
    kernel_radius = max(int(radius / resolution), 1)

    h, w = dem.shape

    # Create circular kernel
    y, x = np.ogrid[-kernel_radius:kernel_radius + 1, -kernel_radius:kernel_radius + 1]
    mask = (x**2 + y**2) <= kernel_radius**2
    # Exclude center pixel
    mask[kernel_radius, kernel_radius] = False

    # Compute mean elevation within radius using a padded array
    pad = kernel_radius
    padded = np.pad(dem, pad, mode="reflect")

    # Sum using the circular kernel
    mean_elev = np.zeros_like(dem)
    count = np.zeros_like(dem)

    for di in range(-kernel_radius, kernel_radius + 1):
        for dj in range(-kernel_radius, kernel_radius + 1):
            if di**2 + dj**2 <= kernel_radius**2 and not (di == 0 and dj == 0):
                shifted = padded[
                    pad + di : pad + di + h,
                    pad + dj : pad + dj + w,
                ]
                mean_elev += shifted
                count += 1

    mean_elev /= np.maximum(count, 1)
    tpi = (dem - mean_elev).astype(np.float32)

    return tpi
