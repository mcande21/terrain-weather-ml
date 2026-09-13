"""Terrain feature normalization.

Z-score normalization using per-channel mean and standard deviation
computed from training regions. Parameters are saved with model checkpoints.
"""

from __future__ import annotations

from pathlib import Path

import torch


class TerrainNormalizer:
    """Per-channel z-score normalization for terrain feature tensors.

    Usage:
        normalizer = TerrainNormalizer()
        normalizer.fit(training_tensors)
        normalized = normalizer.normalize(terrain_tensor)

        # Save/load with checkpoint
        normalizer.save("norm_params.pt")
        normalizer.load("norm_params.pt")

        # Or embed in model checkpoint
        checkpoint["terrain_normalizer"] = normalizer.state_dict()
        normalizer.load_state_dict(checkpoint["terrain_normalizer"])
    """

    def __init__(self):
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None

    @property
    def is_fitted(self) -> bool:
        return self.mean is not None and self.std is not None

    def fit(self, tensors: list[torch.Tensor]) -> None:
        """Compute per-channel mean and std from a list of terrain tensors.

        Args:
            tensors: List of tensors, each shape (C, H, W).
        """
        if not tensors:
            raise ValueError("Cannot fit on empty tensor list")

        n_channels = tensors[0].shape[0]

        # Collect all values per channel
        channel_sums = torch.zeros(n_channels, dtype=torch.float64)
        channel_sq_sums = torch.zeros(n_channels, dtype=torch.float64)
        total_pixels = 0

        for tensor in tensors:
            assert tensor.shape[0] == n_channels, (
                f"Channel mismatch: expected {n_channels}, got {tensor.shape[0]}"
            )
            n_pixels = tensor.shape[1] * tensor.shape[2]
            total_pixels += n_pixels

            for c in range(n_channels):
                vals = tensor[c].to(torch.float64)
                channel_sums[c] += vals.sum()
                channel_sq_sums[c] += (vals**2).sum()

        self.mean = (channel_sums / total_pixels).to(torch.float32)
        variance = (channel_sq_sums / total_pixels) - (channel_sums / total_pixels) ** 2
        # Clamp to avoid negative variance from floating point
        variance = variance.clamp(min=0)
        self.std = variance.sqrt().to(torch.float32)

        # Prevent division by zero for constant channels
        self.std = torch.where(self.std < 1e-8, torch.ones_like(self.std), self.std)

    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply z-score normalization per channel.

        Args:
            tensor: Terrain feature tensor, shape (C, H, W).

        Returns:
            Normalized tensor, same shape.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Normalizer not fitted. Call fit() or load() first."
            )
        # Reshape mean/std for broadcasting: (C,) -> (C, 1, 1)
        mean = self.mean[:, None, None].to(tensor.device)
        std = self.std[:, None, None].to(tensor.device)
        return (tensor - mean) / std

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Invert z-score normalization.

        Args:
            tensor: Normalized terrain feature tensor, shape (C, H, W).

        Returns:
            Denormalized tensor, same shape.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Normalizer not fitted. Call fit() or load() first."
            )
        mean = self.mean[:, None, None].to(tensor.device)
        std = self.std[:, None, None].to(tensor.device)
        return tensor * std + mean

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return normalization parameters as a state dict."""
        if not self.is_fitted:
            raise RuntimeError("Normalizer not fitted.")
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load normalization parameters from a state dict."""
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]

    def save(self, path: str | Path) -> None:
        """Save normalization parameters to a file."""
        torch.save(self.state_dict(), path)

    def load(self, path: str | Path) -> None:
        """Load normalization parameters from a file."""
        state = torch.load(path, weights_only=True)
        self.load_state_dict(state)
