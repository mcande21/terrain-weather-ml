"""Sx upwind exposure computation.

8 cardinal directions (N, NE, E, SE, S, SW, W, NW).
Positive = sheltered, negative = exposed. Values in degrees.
Configurable search radius (default 3km).
"""

from __future__ import annotations

import numpy as np

# Direction order: N, NE, E, SE, S, SW, W, NW
# Angles in radians clockwise from north
SX_DIRECTIONS = [
    ("N", 0.0),
    ("NE", np.pi / 4),
    ("E", np.pi / 2),
    ("SE", 3 * np.pi / 4),
    ("S", np.pi),
    ("SW", 5 * np.pi / 4),
    ("W", 3 * np.pi / 2),
    ("NW", 7 * np.pi / 4),
]


def compute_sx(
    dem: np.ndarray,
    resolution: float,
    max_radius: float = 3000.0,
) -> np.ndarray:
    """Compute Sx wind exposure for 8 cardinal directions.

    Sx is the maximum upward angle to the horizon in each direction.
    Positive values indicate sheltering (terrain rises in that direction),
    negative values indicate exposure (terrain drops in that direction).

    Args:
        dem: Elevation array (H, W) in meters.
        resolution: Cell size in meters.
        max_radius: Maximum search radius in meters.

    Returns:
        Sx array, shape (8, H, W), values in degrees.
    """
    h, w = dem.shape
    max_steps = int(max_radius / resolution) + 1
    sx = np.zeros((8, h, w), dtype=np.float32)

    row_coords = np.arange(h, dtype=np.float32)
    col_coords = np.arange(w, dtype=np.float32)
    rr, cc = np.meshgrid(row_coords, col_coords, indexing="ij")

    for dir_idx, (_, azimuth) in enumerate(SX_DIRECTIONS):
        # Direction vector in pixel coordinates
        dy = -np.cos(azimuth)  # row direction (north = -row)
        dx = np.sin(azimuth)   # col direction (east = +col)

        max_angle = np.full((h, w), -90.0, dtype=np.float32)

        for step in range(1, max_steps + 1):
            target_r = rr + step * dy
            target_c = cc + step * dx

            tr = np.round(target_r).astype(np.int32)
            tc = np.round(target_c).astype(np.int32)

            valid = (tr >= 0) & (tr < h) & (tc >= 0) & (tc < w)

            target_elev = np.where(
                valid,
                dem[np.clip(tr, 0, h - 1), np.clip(tc, 0, w - 1)],
                0,
            )

            dist = step * resolution
            dz = target_elev - dem
            angle = np.degrees(np.arctan2(dz, dist))

            max_angle = np.where(valid & (angle > max_angle), angle, max_angle)

        # Where no valid pixels were found, set to 0
        no_valid = max_angle <= -89.9
        max_angle[no_valid] = 0.0

        sx[dir_idx] = max_angle

    return sx
