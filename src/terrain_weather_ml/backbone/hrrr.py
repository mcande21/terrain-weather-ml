"""HRRR input conditioning for StormCast.

Parses HRRR analysis/forecast data and formats it for StormCast inference.
Handles variable ordering, normalization, and grid alignment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class HRRRData:
    """Container for HRRR grid data.

    Args:
        fields: Dict mapping variable names to 2D numpy arrays (H, W).
        grid_shape: Expected spatial dimensions (H, W).
        valid_time: ISO 8601 valid time string.
        cycle: ISO 8601 analysis cycle time string.
    """

    fields: dict[str, np.ndarray]
    grid_shape: tuple[int, int]
    valid_time: str
    cycle: str


@dataclass
class HRRRConfig:
    """Configuration for HRRR conditioning.

    Args:
        num_input_vars: Number of input variables expected by StormCast.
        grid_height: Expected grid height.
        grid_width: Expected grid width.
        normalize: Whether to apply z-score normalization.
        norm_means: Per-variable normalization means.
        norm_stds: Per-variable normalization stds.
    """

    num_input_vars: int = 99
    grid_height: int = 1059
    grid_width: int = 1799
    normalize: bool = True
    norm_means: list[float] | None = None
    norm_stds: list[float] | None = None


class HRRRConditioner:
    """Formats HRRR data for StormCast inference.

    Accepts raw HRRR fields (from GRIB2 parsing), validates them,
    assembles into the correct variable ordering, normalizes,
    and produces the input tensor for StormCast.
    """

    def __init__(self, config: HRRRConfig):
        self.config = config
        self._norm_means = None
        self._norm_stds = None

        if config.norm_means is not None and config.norm_stds is not None:
            self._norm_means = torch.tensor(
                config.norm_means, dtype=torch.float32
            ).reshape(-1, 1, 1)
            self._norm_stds = torch.tensor(
                config.norm_stds, dtype=torch.float32
            ).reshape(-1, 1, 1)

    def _validate(self, data: HRRRData) -> None:
        """Validate HRRR data before formatting."""
        n_fields = len(data.fields)
        if n_fields != self.config.num_input_vars:
            raise ValueError(
                f"Expected {self.config.num_input_vars} input variables, "
                f"got {n_fields}"
            )

        h, w = data.grid_shape
        for name, arr in data.fields.items():
            if arr.shape != (h, w):
                raise ValueError(
                    f"Field '{name}' has shape {arr.shape}, "
                    f"expected {(h, w)}"
                )

    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply z-score normalization if configured."""
        if self._norm_means is not None and self._norm_stds is not None:
            return (tensor - self._norm_means) / (self._norm_stds + 1e-8)

        # Default: per-variable z-score normalization
        # (mean/std computed from the input itself)
        mean = tensor.mean(dim=(-2, -1), keepdim=True)
        std = tensor.std(dim=(-2, -1), keepdim=True)
        return (tensor - mean) / (std + 1e-8)

    def format_input(self, data: HRRRData) -> torch.Tensor:
        """Format HRRR data into a StormCast input tensor.

        Args:
            data: HRRRData containing the raw fields.

        Returns:
            Tensor of shape (1, num_input_vars, H, W).
        """
        self._validate(data)

        # Stack fields in sorted key order for deterministic ordering
        arrays = []
        for key in sorted(data.fields.keys()):
            arr = data.fields[key].astype(np.float32)
            arrays.append(arr)

        stacked = np.stack(arrays, axis=0)  # (C, H, W)
        tensor = torch.from_numpy(stacked).unsqueeze(0)  # (1, C, H, W)

        # Replace NaN/Inf
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)

        if self.config.normalize:
            tensor = self._normalize(tensor)

        return tensor

    def format_batch(self, data_list: list[HRRRData]) -> torch.Tensor:
        """Format multiple HRRR snapshots into a batch tensor.

        Args:
            data_list: List of HRRRData instances.

        Returns:
            Tensor of shape (B, num_input_vars, H, W).
        """
        tensors = [self.format_input(d) for d in data_list]
        return torch.cat(tensors, dim=0)
