"""Core terrain feature computation.

Implements elevation passthrough, slope via Horn's method (degrees),
aspect as (sin, cos) pair, plan curvature and profile curvature (1/m).
"""

from __future__ import annotations

import numpy as np


def _horn_gradients(dem: np.ndarray, resolution: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute dz/dx and dz/dy using Horn's method (3x3 weighted finite differences).

    Returns:
        (dzdx, dzdy) gradient arrays with same shape as input.
        Border pixels use forward/backward differences.
    """
    dzdx = np.zeros_like(dem)
    dzdy = np.zeros_like(dem)

    z = dem

    # Horn's method:
    # z0=z[r-1,c-1], z1=z[r-1,c], z2=z[r-1,c+1]
    # z3=z[r,c-1],   z4=z[r,c],   z5=z[r,c+1]
    # z6=z[r+1,c-1], z7=z[r+1,c], z8=z[r+1,c+1]
    z0 = z[:-2, :-2]
    z1 = z[:-2, 1:-1]
    z2 = z[:-2, 2:]
    z3 = z[1:-1, :-2]
    z5 = z[1:-1, 2:]
    z6 = z[2:, :-2]
    z7 = z[2:, 1:-1]
    z8 = z[2:, 2:]

    dzdx[1:-1, 1:-1] = ((z2 + 2 * z5 + z8) - (z0 + 2 * z3 + z6)) / (8 * resolution)
    dzdy[1:-1, 1:-1] = ((z6 + 2 * z7 + z8) - (z0 + 2 * z1 + z2)) / (8 * resolution)

    # Border: simple forward/backward differences
    # Top row
    dzdx[0, 1:-1] = (z[0, 2:] - z[0, :-2]) / (2 * resolution)
    dzdy[0, :] = (z[1, :] - z[0, :]) / resolution
    # Bottom row
    dzdx[-1, 1:-1] = (z[-1, 2:] - z[-1, :-2]) / (2 * resolution)
    dzdy[-1, :] = (z[-1, :] - z[-2, :]) / resolution
    # Left column
    dzdx[:, 0] = (z[:, 1] - z[:, 0]) / resolution
    dzdy[1:-1, 0] = (z[2:, 0] - z[:-2, 0]) / (2 * resolution)
    # Right column
    dzdx[:, -1] = (z[:, -1] - z[:, -2]) / resolution
    dzdy[1:-1, -1] = (z[2:, -1] - z[:-2, -1]) / (2 * resolution)

    return dzdx, dzdy


def compute_slope(dem: np.ndarray, resolution: float) -> np.ndarray:
    """Compute slope in degrees using Horn's method.

    Args:
        dem: Elevation array (H, W) in meters.
        resolution: Cell size in meters.

    Returns:
        Slope in degrees, shape (H, W).
    """
    dzdx, dzdy = _horn_gradients(dem, resolution)
    slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    return np.degrees(slope_rad).astype(np.float32)


def compute_aspect(
    dem: np.ndarray, resolution: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compute aspect as (sin, cos) pair to avoid 0/360 discontinuity.

    Convention: aspect is measured clockwise from north.
    Flat pixels (zero gradient) get aspect (0, 0).

    Args:
        dem: Elevation array (H, W) in meters.
        resolution: Cell size in meters.

    Returns:
        (aspect_sin, aspect_cos) arrays, shape (H, W) each.
    """
    dzdx, dzdy = _horn_gradients(dem, resolution)

    # Aspect angle clockwise from north
    # North is the negative y direction (row index decreases)
    # atan2(-dzdx, -dzdy) gives angle from north, but we need to handle
    # the convention that dzdy positive means elevation increases downward (south)
    # For a north-facing slope, elevation increases to the south (dzdy > 0)
    # aspect = atan2(-dzdx, dzdy) would give north
    # Actually: aspect = atan2(dzdx, -dzdy) clockwise from north
    # Let's be precise: north-facing means steepest descent is northward
    # Gradient points uphill. For north-facing slope, gradient points south (dzdy>0)
    # Aspect = direction of steepest descent = opposite of gradient
    # descent_x = -dzdx, descent_y = -dzdy
    # angle from north CW = atan2(descent_x, -descent_y)
    # = atan2(-dzdx, dzdy)

    # Wait, the y-axis convention: in raster data, row 0 is north, row increases south.
    # So dzdy positive means elevation increases southward (gradient points south).
    # Steepest descent = -gradient, so descent points north when gradient points south.
    # Aspect (direction slope faces = direction of steepest descent)
    # For north-facing: descent is toward north (decreasing row), aspect = 0 degrees
    # aspect_angle = atan2(-dzdx, dzdy) -- this gives 0 when dzdx=0, dzdy>0 (north-facing)

    gradient_magnitude = np.sqrt(dzdx**2 + dzdy**2)
    is_flat = gradient_magnitude < 1e-8

    # Aspect angle in radians, clockwise from north
    aspect_rad = np.arctan2(-dzdx, dzdy)

    aspect_sin = np.sin(aspect_rad).astype(np.float32)
    aspect_cos = np.cos(aspect_rad).astype(np.float32)

    # Flat pixels get (0, 0)
    aspect_sin[is_flat] = 0.0
    aspect_cos[is_flat] = 0.0

    return aspect_sin, aspect_cos


def compute_curvature(
    dem: np.ndarray, resolution: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compute plan and profile curvature.

    Plan curvature: perpendicular to slope direction (convergence/divergence).
    Profile curvature: parallel to slope direction (acceleration/deceleration).

    Uses second-order finite differences on a 3x3 neighborhood.

    Args:
        dem: Elevation array (H, W) in meters.
        resolution: Cell size in meters.

    Returns:
        (plan_curvature, profile_curvature) arrays in 1/m, shape (H, W) each.
    """
    plan_curv = np.zeros_like(dem)
    profile_curv = np.zeros_like(dem)

    z = dem
    r = resolution

    # Evans-Young method using 3x3 window
    # z1=NW, z2=N, z3=NE, z4=W, z5=center, z6=E, z7=SW, z8=S, z9=SE
    z1 = z[:-2, :-2]
    z2 = z[:-2, 1:-1]
    z3 = z[:-2, 2:]
    z4 = z[1:-1, :-2]
    z5 = z[1:-1, 1:-1]
    z6 = z[1:-1, 2:]
    z7 = z[2:, :-2]
    z8 = z[2:, 1:-1]
    z9 = z[2:, 2:]

    # Partial derivatives
    p = (z6 - z4) / (2 * r)  # dz/dx
    q = (z2 - z8) / (2 * r)  # dz/dy (note: row decreases = north = positive y)
    r_xx = (z4 - 2 * z5 + z6) / (r**2)  # d2z/dx2
    s_xy = (z3 - z1 - z9 + z7) / (4 * r**2)  # d2z/dxdy
    t_yy = (z2 - 2 * z5 + z8) / (r**2)  # d2z/dy2

    # Denominators
    pq2 = p**2 + q**2
    denom = pq2 + 1e-10  # avoid division by zero

    # Plan curvature (horizontal curvature)
    plan_curv[1:-1, 1:-1] = -(
        (r_xx * q**2 - 2 * s_xy * p * q + t_yy * p**2) / (denom * np.sqrt(denom))
    )

    # Profile curvature (vertical curvature)
    profile_curv[1:-1, 1:-1] = -(
        (r_xx * p**2 + 2 * s_xy * p * q + t_yy * q**2) / (denom * np.sqrt(denom))
    )

    # Where surface is flat, curvature is zero (already initialized to 0)
    flat_mask = np.zeros(dem.shape, dtype=bool)
    flat_mask[1:-1, 1:-1] = pq2 < 1e-10
    plan_curv[flat_mask] = 0.0
    profile_curv[flat_mask] = 0.0

    return plan_curv.astype(np.float32), profile_curv.astype(np.float32)
