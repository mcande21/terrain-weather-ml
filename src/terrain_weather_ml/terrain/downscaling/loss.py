"""Loss function for terrain downscaling head (task 5.6).

L = MSE(pred, target) + lambda_div * div_penalty + lambda_oro * oro_penalty

- MSE: Per-variable weighted mean squared error.
- div_penalty: Mean squared divergence of the pre-projection wind field,
  encouraging the model to learn near-divergence-free predictions.
- oro_penalty: Orographic precipitation consistency -- predicted
  precipitation should increase on windward slopes and decrease on
  leeward slopes relative to terrain gradient and wind direction.
"""

from __future__ import annotations

import torch
from torch import nn

from terrain_weather_ml.terrain.downscaling.divergence import compute_divergence
from terrain_weather_ml.terrain.downscaling.head import (
    IDX_PRECIPITATION,
    IDX_U_WIND,
    IDX_V_WIND,
)


def orographic_precipitation_penalty(
    pred: torch.Tensor,
    terrain: torch.Tensor,
    dx: float = 1.0,
) -> torch.Tensor:
    """Orographic precipitation consistency penalty.

    Penalizes predicted precipitation that does not correlate with
    windward vs leeward slope exposure. On windward slopes (where wind
    blows uphill), precipitation should increase; on leeward slopes
    (wind blows downhill), precipitation should decrease.

    Formulated as a correlation penalty: we compute the dot product of
    the predicted wind with the terrain gradient (slope exposure), and
    check that precipitation correlates negatively (more precip where
    wind goes uphill, i.e., where wind dot grad(elevation) > 0).

    Args:
        pred: Prediction tensor (B, C_out, H, W).
        terrain: Terrain feature tensor (B, C_terrain, H, W).
            Channel 0 is elevation.
        dx: Grid spacing.

    Returns:
        Scalar penalty (non-negative).
    """
    # Extract predicted wind components
    u = pred[:, IDX_U_WIND]  # (B, H, W)
    v = pred[:, IDX_V_WIND]  # (B, H, W)
    precip = pred[:, IDX_PRECIPITATION]  # (B, H, W)

    # Extract elevation from terrain (channel 0)
    elevation = terrain[:, 0]  # (B, H, W)

    # Compute terrain gradient (uphill direction)
    # dz/dx along W dimension
    dzdx = torch.zeros_like(elevation)
    dzdx[:, :, 1:-1] = (elevation[:, :, 2:] - elevation[:, :, :-2]) / (2 * dx)
    dzdx[:, :, 0] = (elevation[:, :, 1] - elevation[:, :, 0]) / dx
    dzdx[:, :, -1] = (elevation[:, :, -1] - elevation[:, :, -2]) / dx

    # dz/dy along H dimension
    dzdy = torch.zeros_like(elevation)
    dzdy[:, 1:-1, :] = (elevation[:, 2:, :] - elevation[:, :-2, :]) / (2 * dx)
    dzdy[:, 0, :] = (elevation[:, 1, :] - elevation[:, 0, :]) / dx
    dzdy[:, -1, :] = (elevation[:, -1, :] - elevation[:, -2, :]) / dx

    # Wind dot terrain gradient: positive = wind going uphill (windward)
    windward = u * dzdx + v * dzdy  # (B, H, W)

    # Penalty: windward exposure should positively correlate with precipitation.
    # We penalize negative correlation: when windward > 0, precip should be high,
    # and when windward < 0 (leeward), precip should be low.
    # Use mean squared anti-correlation: penalty = mean(max(0, -windward * precip))^2
    # Simpler: penalty = mean(relu(-windward * precip))
    # This penalizes cases where windward and precip have opposite signs.
    anti_correlation = torch.relu(-windward * precip)
    penalty = anti_correlation.mean()

    return penalty


class DownscalingLoss(nn.Module):
    """Combined loss for terrain downscaling head training.

    L = MSE(pred, target) + lambda_div * ||div(u, v)||^2
        + lambda_oro * orographic_penalty

    The divergence penalty is computed on the pre-projection wind field
    (the raw U-Net output), encouraging the model to learn to produce
    near-divergence-free fields naturally.

    Args:
        lambda_div: Weight for divergence penalty (default: 0.1).
        lambda_oro: Weight for orographic penalty (default: 0.0, disabled).
        variable_weights: Per-variable MSE weights [u, v, T, P].
            Default: equal weighting (all 1.0).
        dx: Grid spacing for divergence computation.
    """

    def __init__(
        self,
        lambda_div: float = 0.1,
        lambda_oro: float = 0.0,
        variable_weights: list[float] | None = None,
        dx: float = 1.0,
    ):
        super().__init__()
        self.lambda_div = lambda_div
        self.lambda_oro = lambda_oro
        self.dx = dx

        if variable_weights is None:
            variable_weights = [1.0, 1.0, 1.0, 1.0]
        self.register_buffer(
            "variable_weights",
            torch.tensor(variable_weights, dtype=torch.float32),
        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        terrain: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute combined loss.

        Args:
            pred: Prediction tensor (B, C_out, H, W) -- pre-projection.
            target: Target tensor (B, C_out, H, W).
            terrain: Terrain feature tensor (B, C_terrain, H, W).

        Returns:
            Dict with keys: total, mse, divergence, orographic (if enabled).
        """
        result = {}

        # Per-variable weighted MSE
        # pred and target shape: (B, C, H, W)
        mse_per_var = ((pred - target) ** 2).mean(dim=(0, 2, 3))  # (C,)
        weights = self.variable_weights[: mse_per_var.shape[0]]
        mse = (weights * mse_per_var).sum() / weights.sum()
        result["mse"] = mse

        # Divergence penalty on pre-projection wind field
        u = pred[:, IDX_U_WIND : IDX_U_WIND + 1]
        v = pred[:, IDX_V_WIND : IDX_V_WIND + 1]
        div = compute_divergence(u, v, dx=self.dx)
        div_penalty = (div ** 2).mean()
        result["divergence"] = div_penalty

        # Total loss
        total = mse + self.lambda_div * div_penalty

        # Orographic penalty (optional)
        if self.lambda_oro > 0:
            oro_penalty = orographic_precipitation_penalty(
                pred, terrain, dx=self.dx
            )
            result["orographic"] = oro_penalty
            total = total + self.lambda_oro * oro_penalty

        result["total"] = total

        return result
