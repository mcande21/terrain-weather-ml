"""Z-score normalization for training targets."""

from __future__ import annotations

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
