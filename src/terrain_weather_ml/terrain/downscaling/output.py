"""Multi-variable output utilities (task 5.3).

Output channel definitions and u/v to speed/direction conversion.
"""

from __future__ import annotations

import torch

# Output channel ordering
OUTPUT_CHANNELS = ["u_wind", "v_wind", "temperature", "precipitation"]

IDX_U_WIND = 0
IDX_V_WIND = 1
IDX_TEMPERATURE = 2
IDX_PRECIPITATION = 3


def uv_to_speed_direction(
    u: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert u/v wind components to speed and meteorological direction.

    Meteorological convention: direction is where the wind comes FROM,
    measured clockwise from north (0=N, 90=E, 180=S, 270=W).

    Args:
        u: Eastward wind component (positive = eastward).
        v: Northward wind component (positive = northward).

    Returns:
        Tuple of (speed, direction) tensors with same shape as input.
        Speed in m/s, direction in degrees [0, 360).
    """
    speed = torch.sqrt(u**2 + v**2)

    # atan2 gives angle of the wind vector (where wind is going TO)
    # Meteorological direction is FROM, so add 180 degrees.
    # Also, atan2(u, v) gives angle from north (y-axis), clockwise.
    # We want: FROM direction = TO direction + 180
    direction_rad = torch.atan2(u, v)
    direction_deg = torch.rad2deg(direction_rad) + 180.0

    # Wrap to [0, 360)
    direction_deg = direction_deg % 360.0

    # Calm wind: set direction to 0
    calm_mask = speed < 1e-10
    direction_deg = torch.where(calm_mask, torch.zeros_like(direction_deg), direction_deg)

    return speed, direction_deg
