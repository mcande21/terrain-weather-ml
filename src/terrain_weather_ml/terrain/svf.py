"""Sky-view factor (SVF) computation.

36-azimuth-sector ray-tracing with configurable maximum search radius.
SVF is the fraction of the hemisphere visible from each point.
"""

from __future__ import annotations

import numpy as np


def compute_svf(
    dem: np.ndarray,
    resolution: float,
    max_radius: float = 5000.0,
    n_azimuths: int = 36,
) -> np.ndarray:
    """Compute sky-view factor via horizon angle ray-tracing.

    For each pixel, traces rays in n_azimuths directions and finds the maximum
    elevation angle (horizon angle) along each ray. SVF is computed as:
        SVF = 1 - mean(sin(horizon_angle)^2) over all azimuths

    Args:
        dem: Elevation array (H, W) in meters.
        resolution: Cell size in meters.
        max_radius: Maximum search radius in meters.
        n_azimuths: Number of azimuth directions (default 36 = 10-degree spacing).

    Returns:
        SVF array, shape (H, W), values in [0, 1].
    """
    h, w = dem.shape
    max_steps = int(max_radius / resolution) + 1
    azimuths = np.linspace(0, 2 * np.pi, n_azimuths, endpoint=False)

    # Pre-compute direction vectors in pixel coordinates
    dy = -np.cos(azimuths)  # row direction (north = -row)
    dx = np.sin(azimuths)   # col direction (east = +col)

    # Initialize horizon angles array: (n_azimuths, H, W)
    horizon_angles = np.zeros((n_azimuths, h, w), dtype=np.float32)

    row_coords = np.arange(h, dtype=np.float32)
    col_coords = np.arange(w, dtype=np.float32)
    rr, cc = np.meshgrid(row_coords, col_coords, indexing="ij")

    for az_idx in range(n_azimuths):
        max_angle = np.full((h, w), 0.0, dtype=np.float32)

        for step in range(1, max_steps + 1):
            # Target pixel coordinates along this ray
            target_r = rr + step * dy[az_idx]
            target_c = cc + step * dx[az_idx]

            # Integer pixel indices (nearest neighbor)
            tr = np.round(target_r).astype(np.int32)
            tc = np.round(target_c).astype(np.int32)

            # Mask for valid pixels
            valid = (tr >= 0) & (tr < h) & (tc >= 0) & (tc < w)

            # Get elevation at target (use 0 for out-of-bounds, will be masked)
            target_elev = np.where(valid, dem[np.clip(tr, 0, h - 1), np.clip(tc, 0, w - 1)], 0)

            # Horizontal distance
            dist = step * resolution

            # Elevation angle = atan2(dz, dist)
            dz = target_elev - dem
            angle = np.arctan2(dz, dist)

            # Update max angle only for valid pixels
            max_angle = np.where(valid & (angle > max_angle), angle, max_angle)

        horizon_angles[az_idx] = max_angle

    # SVF = 1 - mean(sin^2(horizon_angle)) over all azimuths
    # Clamp negative angles to 0 (below horizon doesn't reduce sky view)
    horizon_angles = np.maximum(horizon_angles, 0)
    svf = 1.0 - np.mean(np.sin(horizon_angles) ** 2, axis=0)

    return np.clip(svf, 0.0, 1.0).astype(np.float32)
