"""Quantile mapping for cross-domain alignment (task 6.4).

Implements the RainShift protocol: per-variable, per-season empirical CDF
matching between Alpine and Colorado weather distributions. Applied to
coarse weather input during Phase 3 LoRA adaptation to bridge the domain
gap between the Alps and Rockies.

The mapping is:
    x_target = F_target^{-1}(F_source(x_source))

where F_source and F_target are empirical CDFs. Values outside the
training range are clamped to the target distribution bounds.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

import torch

logger = logging.getLogger(__name__)

# Number of quantile points for CDF approximation
DEFAULT_N_QUANTILES = 1000


class Season(str, Enum):
    """Meteorological season for stratified mapping."""

    WINTER = "winter"  # DJF
    SPRING = "spring"  # MAM
    SUMMER = "summer"  # JJA
    FALL = "fall"      # SON


def compute_empirical_cdf(
    data: torch.Tensor,
    n_quantiles: int = DEFAULT_N_QUANTILES,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute empirical CDF from a 1-D data tensor.

    Args:
        data: 1-D tensor of observed values.
        n_quantiles: Number of quantile points (default 1000).

    Returns:
        Tuple of (values, quantiles) where values are sorted data
        points at evenly spaced quantile levels [0, 1].
    """
    quantile_levels = torch.linspace(0.0, 1.0, n_quantiles, dtype=data.dtype)
    sorted_data = torch.sort(data)[0]

    # Compute quantile values at each level
    values = torch.quantile(sorted_data.float(), quantile_levels.float())

    return values.to(data.dtype), quantile_levels


def _apply_quantile_mapping(
    x: torch.Tensor,
    source_values: torch.Tensor,
    source_quantiles: torch.Tensor,
    target_values: torch.Tensor,
    target_quantiles: torch.Tensor,
) -> torch.Tensor:
    """Map values from source distribution to target distribution.

    Steps:
    1. Find the quantile of each x in the source CDF (via interpolation)
    2. Map that quantile to the target CDF (via interpolation)

    Values outside the source range are clamped.
    """
    # Step 1: x -> quantile (interpolate in source CDF)
    # searchsorted gives us the insertion point
    flat = x.reshape(-1)

    # Clamp to source range for lookup
    clamped = flat.clamp(source_values[0], source_values[-1])

    # Find quantile for each value in source distribution
    indices = torch.searchsorted(source_values, clamped)
    indices = indices.clamp(1, len(source_values) - 1)

    # Linear interpolation for quantile
    lower = indices - 1
    upper = indices

    v_lo = source_values[lower]
    v_hi = source_values[upper]
    q_lo = source_quantiles[lower]
    q_hi = source_quantiles[upper]

    # Avoid division by zero when two quantile values are equal
    denom = v_hi - v_lo
    safe_denom = torch.where(denom.abs() < 1e-10, torch.ones_like(denom), denom)
    frac = (clamped - v_lo) / safe_denom
    frac = frac.clamp(0.0, 1.0)

    quantiles = q_lo + frac * (q_hi - q_lo)

    # Step 2: quantile -> target value (interpolate in target CDF)
    t_indices = torch.searchsorted(target_quantiles, quantiles)
    t_indices = t_indices.clamp(1, len(target_quantiles) - 1)

    t_lower = t_indices - 1
    t_upper = t_indices

    tq_lo = target_quantiles[t_lower]
    tq_hi = target_quantiles[t_upper]
    tv_lo = target_values[t_lower]
    tv_hi = target_values[t_upper]

    t_denom = tq_hi - tq_lo
    t_safe = torch.where(t_denom.abs() < 1e-10, torch.ones_like(t_denom), t_denom)
    t_frac = (quantiles - tq_lo) / t_safe
    t_frac = t_frac.clamp(0.0, 1.0)

    mapped = tv_lo + t_frac * (tv_hi - tv_lo)

    return mapped.reshape(x.shape)


class QuantileMapper:
    """Per-variable quantile mapping between source and target distributions.

    Fits empirical CDFs for each variable, then maps source values to the
    target distribution via CDF matching (RainShift protocol).

    Usage:
        mapper = QuantileMapper()
        mapper.fit(source_data=alpine_vars, target_data=colorado_vars)
        mapped = mapper.transform(alpine_temp, variable="temperature")
        recovered = mapper.inverse_transform(mapped, variable="temperature")
    """

    def __init__(self, n_quantiles: int = DEFAULT_N_QUANTILES):
        self.n_quantiles = n_quantiles
        # Per-variable CDF storage: {var: (values, quantiles)}
        self._source_cdfs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._target_cdfs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def fit(
        self,
        source_data: dict[str, torch.Tensor],
        target_data: dict[str, torch.Tensor],
    ) -> None:
        """Compute empirical CDFs for each variable.

        Args:
            source_data: Dict of variable name -> 1-D tensor of source values.
            target_data: Dict of variable name -> 1-D tensor of target values.
        """
        for var, src_tensor in source_data.items():
            if var not in target_data:
                logger.warning(
                    "Variable %s in source but not target; skipping", var
                )
                continue

            src_values, src_q = compute_empirical_cdf(
                src_tensor, self.n_quantiles
            )
            tgt_values, tgt_q = compute_empirical_cdf(
                target_data[var], self.n_quantiles
            )

            self._source_cdfs[var] = (src_values, src_q)
            self._target_cdfs[var] = (tgt_values, tgt_q)

        logger.info(
            "Fitted quantile mappings for %d variables: %s",
            len(self._source_cdfs),
            list(self._source_cdfs.keys()),
        )

    def transform(
        self,
        data: torch.Tensor,
        variable: str,
    ) -> torch.Tensor:
        """Map data from source distribution to target distribution.

        Args:
            data: Input tensor of values from the source distribution.
            variable: Variable name (must have been fitted).

        Returns:
            Mapped tensor in the target distribution.
        """
        if variable not in self._source_cdfs:
            raise KeyError(f"Variable '{variable}' not fitted. Available: "
                           f"{list(self._source_cdfs.keys())}")

        src_values, src_q = self._source_cdfs[variable]
        tgt_values, tgt_q = self._target_cdfs[variable]

        return _apply_quantile_mapping(data, src_values, src_q, tgt_values, tgt_q)

    def inverse_transform(
        self,
        data: torch.Tensor,
        variable: str,
    ) -> torch.Tensor:
        """Map data from target distribution back to source distribution.

        Args:
            data: Input tensor of values in the target distribution.
            variable: Variable name (must have been fitted).

        Returns:
            Tensor mapped back to the source distribution.
        """
        if variable not in self._target_cdfs:
            raise KeyError(f"Variable '{variable}' not fitted. Available: "
                           f"{list(self._target_cdfs.keys())}")

        tgt_values, tgt_q = self._target_cdfs[variable]
        src_values, src_q = self._source_cdfs[variable]

        return _apply_quantile_mapping(data, tgt_values, tgt_q, src_values, src_q)

    def get_params(self) -> dict[str, Any]:
        """Serialize mapping parameters for checkpoint storage.

        Returns:
            Dict with per-variable source and target CDF data (as lists
            for JSON/pickle compatibility).
        """
        params = {}
        for var in self._source_cdfs:
            src_values, src_q = self._source_cdfs[var]
            tgt_values, tgt_q = self._target_cdfs[var]
            params[var] = {
                "source_values": src_values.tolist(),
                "source_quantiles": src_q.tolist(),
                "target_values": tgt_values.tolist(),
                "target_quantiles": tgt_q.tolist(),
            }
        return params

    def load_params(self, params: dict[str, Any]) -> None:
        """Load mapping parameters from checkpoint data.

        Args:
            params: Dict produced by ``get_params()``.
        """
        self._source_cdfs.clear()
        self._target_cdfs.clear()

        for var, var_params in params.items():
            self._source_cdfs[var] = (
                torch.tensor(var_params["source_values"], dtype=torch.float32),
                torch.tensor(var_params["source_quantiles"], dtype=torch.float32),
            )
            self._target_cdfs[var] = (
                torch.tensor(var_params["target_values"], dtype=torch.float32),
                torch.tensor(var_params["target_quantiles"], dtype=torch.float32),
            )

        logger.info("Loaded quantile mapping params for %d variables", len(params))
