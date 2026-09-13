"""Physics consistency checks for terrain-weather models (task 7.3).

Three checks:
(a) Wind mass conservation: 2D divergence of predicted wind field
(b) Cold air pool reproduction: valley-ridgeline temperature bias
(c) Orographic precipitation: elevation-precipitation gradient vs PRISM
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PhysicsCheckResult:
    """Result of a single physics consistency check.

    Attributes:
        name: Check identifier.
        passed: Whether the check passed its threshold.
        mean_value: Mean of the diagnostic quantity.
        max_value: Maximum of the diagnostic quantity.
        threshold: Pass/fail threshold used.
        not_applicable: True if the check cannot be evaluated
            (e.g., no inversions for cold air pool).
        details: Additional diagnostic information.
    """

    name: str
    passed: bool
    mean_value: float
    max_value: float
    threshold: float
    not_applicable: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "mean_value": self.mean_value,
            "max_value": self.max_value,
            "threshold": self.threshold,
            "not_applicable": self.not_applicable,
            "details": self.details,
        }


def _compute_divergence_2d(
    u: np.ndarray,
    v: np.ndarray,
    dx: float = 1.0,
) -> np.ndarray:
    """Compute 2D divergence via central finite differences.

    div(u, v) = du/dx + dv/dy

    Args:
        u: Eastward wind component (H, W) or (T, H, W).
        v: Northward wind component, same shape as u.
        dx: Grid spacing (uniform in both directions).

    Returns:
        Divergence field, same shape as input.
    """
    dudx = np.zeros_like(u)
    # Central differences for interior
    dudx[..., 1:-1] = (u[..., 2:] - u[..., :-2]) / (2 * dx)
    # Forward/backward at boundaries
    dudx[..., 0] = (u[..., 1] - u[..., 0]) / dx
    dudx[..., -1] = (u[..., -1] - u[..., -2]) / dx

    dvdy = np.zeros_like(v)
    dvdy[..., 1:-1, :] = (v[..., 2:, :] - v[..., :-2, :]) / (2 * dx)
    dvdy[..., 0, :] = (v[..., 1, :] - v[..., 0, :]) / dx
    dvdy[..., -1, :] = (v[..., -1, :] - v[..., -2, :]) / dx

    return dudx + dvdy


def check_wind_mass_conservation(
    u_fields: np.ndarray,
    v_fields: np.ndarray,
    dx: float = 1.0,
    threshold: float = 0.01,
) -> PhysicsCheckResult:
    """Check wind mass conservation via divergence.

    Computes 2D divergence at each timestep and reports mean and max
    absolute divergence across the evaluation domain.

    Args:
        u_fields: Eastward wind component (T, H, W).
        v_fields: Northward wind component (T, H, W).
        dx: Grid spacing.
        threshold: Max absolute divergence threshold (s^-1).

    Returns:
        PhysicsCheckResult with pass/fail against threshold.
    """
    div = _compute_divergence_2d(u_fields, v_fields, dx=dx)
    abs_div = np.abs(div)

    mean_div = float(np.mean(abs_div))
    max_div = float(np.max(abs_div))

    return PhysicsCheckResult(
        name="wind_mass_conservation",
        passed=max_div < threshold,
        mean_value=mean_div,
        max_value=max_div,
        threshold=threshold,
        details={
            "n_timesteps": u_fields.shape[0],
            "grid_shape": list(u_fields.shape[1:]),
        },
    )


def check_cold_air_pool(
    observed_valley_temps: np.ndarray,
    observed_ridge_temps: np.ndarray,
    predicted_valley_temps: np.ndarray,
    threshold: float = 2.0,
) -> PhysicsCheckResult:
    """Check cold air pool reproduction.

    During nocturnal inversions (valley colder than ridge), the model
    should reproduce the valley-ridgeline temperature difference within
    the threshold.

    Args:
        observed_valley_temps: Observed temperatures at valley stations (K).
        observed_ridge_temps: Observed temperatures at ridgeline stations (K).
        predicted_valley_temps: Model-predicted valley temperatures (K).
        threshold: Maximum acceptable valley temperature bias (K).

    Returns:
        PhysicsCheckResult with pass/fail. If no inversions are
        identified (valley warmer than ridge), returns not_applicable.
    """
    # Identify inversions: valley colder than ridge
    inversion_mask = observed_valley_temps < observed_ridge_temps

    if not np.any(inversion_mask):
        return PhysicsCheckResult(
            name="cold_air_pool",
            passed=False,
            mean_value=0.0,
            max_value=0.0,
            threshold=threshold,
            not_applicable=True,
            details={"reason": "No nocturnal inversions identified"},
        )

    # During inversions, compare predicted vs observed valley temps
    obs_inv = observed_valley_temps[inversion_mask]
    pred_inv = predicted_valley_temps[inversion_mask]

    # Valley temperature bias: positive = model too warm (missing cold pool)
    bias = pred_inv - obs_inv
    mean_bias = float(np.mean(bias))
    max_bias = float(np.max(np.abs(bias)))

    return PhysicsCheckResult(
        name="cold_air_pool",
        passed=abs(mean_bias) < threshold,
        mean_value=mean_bias,
        max_value=max_bias,
        threshold=threshold,
        details={
            "n_inversions": int(inversion_mask.sum()),
            "mean_obs_valley_temp": float(np.mean(obs_inv)),
            "mean_pred_valley_temp": float(np.mean(pred_inv)),
        },
    )


def _linear_regression_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Simple OLS slope: dy/dx."""
    n = len(x)
    if n < 2:
        return float("nan")
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    ss_xy = np.sum((x - x_mean) * (y - y_mean))
    ss_xx = np.sum((x - x_mean) ** 2)
    if ss_xx == 0:
        return float("nan")
    return float(ss_xy / ss_xx)


def check_orographic_precipitation(
    elevations: np.ndarray,
    prism_precip: np.ndarray,
    predicted_precip: np.ndarray,
    agreement_threshold: float = 0.20,
) -> PhysicsCheckResult:
    """Check orographic precipitation consistency vs PRISM.

    Compares the model's elevation-precipitation gradient against
    PRISM 30-year normals. The predicted slope should agree with
    PRISM within the agreement threshold (fraction).

    Args:
        elevations: Station elevations (m).
        prism_precip: PRISM annual precipitation (mm).
        predicted_precip: Model-predicted annual precipitation (mm).
        agreement_threshold: Fractional agreement (default 0.20 = 20%).

    Returns:
        PhysicsCheckResult with slopes and percent agreement.
    """
    prism_slope = _linear_regression_slope(elevations, prism_precip)
    predicted_slope = _linear_regression_slope(elevations, predicted_precip)

    # Percent agreement
    if abs(prism_slope) > 1e-10:
        pct_diff = abs(predicted_slope - prism_slope) / abs(prism_slope)
    else:
        pct_diff = 0.0 if abs(predicted_slope) < 1e-10 else 1.0

    return PhysicsCheckResult(
        name="orographic_precipitation",
        passed=pct_diff <= agreement_threshold,
        mean_value=pct_diff,
        max_value=pct_diff,
        threshold=agreement_threshold,
        details={
            "prism_slope": prism_slope,
            "predicted_slope": predicted_slope,
            "percent_difference": pct_diff,
        },
    )
