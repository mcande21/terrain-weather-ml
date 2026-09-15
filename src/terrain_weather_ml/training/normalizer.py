"""Z-score and anomaly-based normalization for training targets."""

from __future__ import annotations

from collections import defaultdict

import torch


class TargetNormalizer:
    """Per-channel z-score normalization for (B, C, H, W) target tensors.

    Computes mean and std per output channel across all spatial positions
    and batch samples. Targets are normalized before loss computation;
    predictions are denormalized back to physical units for inference.
    """

    def __init__(self) -> None:
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None

    def fit(self, targets: torch.Tensor) -> None:
        """Compute per-channel mean and std from training targets.

        Args:
            targets: (B, C, H, W) tensor of training targets.
        """
        self.mean = targets.mean(dim=(0, 2, 3))
        self.std = targets.std(dim=(0, 2, 3)).clamp(min=1e-6)

    def _check_fitted(self) -> None:
        if self.mean is None or self.std is None:
            raise RuntimeError(
                "TargetNormalizer has not been fitted. Call fit() first."
            )

    def normalize(self, targets: torch.Tensor) -> torch.Tensor:
        """Z-score normalize targets: (x - mean) / std.

        Args:
            targets: (B, C, H, W) tensor.

        Returns:
            Normalized tensor with same shape.
        """
        self._check_fitted()
        mean = self.mean.to(targets.device).view(1, -1, 1, 1)
        std = self.std.to(targets.device).view(1, -1, 1, 1)
        return (targets - mean) / std

    def denormalize(self, predictions: torch.Tensor) -> torch.Tensor:
        """Reverse z-score normalization: x * std + mean.

        Args:
            predictions: (B, C, H, W) normalized tensor.

        Returns:
            Tensor in original physical units.
        """
        self._check_fitted()
        mean = self.mean.to(predictions.device).view(1, -1, 1, 1)
        std = self.std.to(predictions.device).view(1, -1, 1, 1)
        return predictions * std + mean

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return normalizer parameters for checkpoint serialization."""
        self._check_fitted()
        return {"mean": self.mean.clone(), "std": self.std.clone()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Load normalizer parameters from a checkpoint."""
        self.mean = state["mean"].clone()
        self.std = state["std"].clone()


class AnomalyNormalizer:
    """Anomaly-based target normalization per WeatherBench standard.

    Subtracts per-(station, month) climatology before z-scoring residuals.
    This removes the seasonal cycle that causes warm/cold bias when a global
    z-score conflates winter and summer observations.

    Normalization:  anomaly = (obs - climatology[station, month]) / residual_std
    Denormalization: physical = anomaly * residual_std + climatology[station, month]
    """

    def __init__(self) -> None:
        self.climatology: dict[tuple[str, int], torch.Tensor] | None = None
        self.residual_std: torch.Tensor | None = None

    def fit(
        self,
        targets: torch.Tensor,
        station_ids: list[str],
        months: list[int],
    ) -> None:
        """Compute monthly station climatology and global residual std.

        Args:
            targets: (N, C) or (N, C, 1, 1) tensor of training targets.
            station_ids: length-N list of station identifiers.
            months: length-N list of calendar months (1-12).
        """
        n = targets.shape[0]
        if len(station_ids) != n:
            raise ValueError(
                f"station_ids length {len(station_ids)} != targets length {n}"
            )
        if len(months) != n:
            raise ValueError(
                f"months length {len(months)} != targets length {n}"
            )

        # Squeeze spatial dims if present: (N, C, 1, 1) -> (N, C)
        t = targets.view(n, targets.shape[1]) if targets.ndim == 4 else targets

        # Group samples by (station, month) and compute per-group mean
        groups: dict[tuple[str, int], list[torch.Tensor]] = defaultdict(list)
        for i in range(n):
            key = (station_ids[i], months[i])
            groups[key].append(t[i])

        self.climatology = {}
        for key, samples in groups.items():
            stacked = torch.stack(samples)
            self.climatology[key] = stacked.mean(dim=0)

        # Compute residuals and global std per channel
        residuals = torch.empty_like(t)
        for i in range(n):
            key = (station_ids[i], months[i])
            residuals[i] = t[i] - self.climatology[key]

        self.residual_std = residuals.std(dim=0, correction=0).clamp(min=1e-6)

    def _check_fitted(self) -> None:
        if self.climatology is None or self.residual_std is None:
            raise RuntimeError(
                "AnomalyNormalizer has not been fitted. Call fit() first."
            )

    def _get_climatology(
        self, station_id: str, month: int,
    ) -> torch.Tensor:
        """Look up climatology for a (station, month) pair."""
        key = (station_id, month)
        if key not in self.climatology:
            raise KeyError(
                f"No climatology for (station={station_id!r}, month={month}). "
                f"Known keys: {len(self.climatology)} entries."
            )
        return self.climatology[key]

    def normalize(
        self,
        targets: torch.Tensor,
        station_ids: list[str],
        months: list[int],
    ) -> torch.Tensor:
        """Normalize: anomaly = (obs - climatology) / residual_std.

        Args:
            targets: (N, C) or (N, C, 1, 1) tensor.
            station_ids: length-N station identifiers.
            months: length-N calendar months.

        Returns:
            Normalized anomalies with same shape as input.
        """
        self._check_fitted()
        original_shape = targets.shape
        n = targets.shape[0]
        t = targets.view(n, targets.shape[1]) if targets.ndim == 4 else targets

        result = torch.empty_like(t)
        std = self.residual_std.to(t.device)
        for i in range(n):
            clim = self._get_climatology(station_ids[i], months[i]).to(t.device)
            result[i] = (t[i] - clim) / std

        return result.view(original_shape)

    def denormalize(
        self,
        predictions: torch.Tensor,
        station_ids: list[str],
        months: list[int],
    ) -> torch.Tensor:
        """Denormalize: physical = anomaly * residual_std + climatology.

        Args:
            predictions: (N, C) or (N, C, 1, 1) normalized tensor.
            station_ids: length-N station identifiers.
            months: length-N calendar months.

        Returns:
            Tensor in original physical units.
        """
        self._check_fitted()
        original_shape = predictions.shape
        n = predictions.shape[0]
        p = (
            predictions.view(n, predictions.shape[1])
            if predictions.ndim == 4 else predictions
        )

        result = torch.empty_like(p)
        std = self.residual_std.to(p.device)
        for i in range(n):
            clim = self._get_climatology(station_ids[i], months[i]).to(p.device)
            result[i] = p[i] * std + clim

        return result.view(original_shape)

    def state_dict(self) -> dict:
        """Return climatology + std for checkpoint serialization.

        Climatology dict keys are (station_id, month) tuples. The state
        is fully serializable via torch.save.
        """
        self._check_fitted()
        return {
            "climatology": {
                k: v.clone() for k, v in self.climatology.items()
            },
            "residual_std": self.residual_std.clone(),
        }

    def load_state_dict(self, state: dict) -> None:
        """Load normalization parameters from a checkpoint."""
        self.climatology = {
            k: v.clone() for k, v in state["climatology"].items()
        }
        self.residual_std = state["residual_std"].clone()
