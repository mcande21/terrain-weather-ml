"""PyTorch Dataset wrappers for each training phase.

Phase 1: CFD wind fields -> downscaling head training
Phase 2: ERA5 + Alpine station observations -> head-only training
Phase 3: HRRR + Colorado SNOTEL data + quantile mapping -> head fine-tuning
"""

from __future__ import annotations

import hashlib
import logging

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class CFDPhaseDataset(Dataset):
    """Phase 1 dataset: synthetic CFD wind fields for downscaling pre-training.

    Each sample provides terrain features and a target wind field (u, v).
    No StormCast involvement -- CFD data provides direct terrain->wind mappings.

    Expected sample dict keys:
        terrain: Tensor (C_terrain, H, W)
        wind_target: Tensor (2, H, W) -- u, v components
    """

    def __init__(self, samples: list[dict[str, torch.Tensor]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        return sample["terrain"], sample["wind_target"]

    def compute_data_hash(self) -> str:
        """Compute hash of dataset for checkpoint metadata."""
        h = hashlib.sha256()
        h.update(str(len(self)).encode())
        if len(self) > 0:
            terrain, target = self[0]
            h.update(str(terrain.shape).encode())
            h.update(str(target.shape).encode())
        return h.hexdigest()[:16]


class AlpinePhaseDataset(Dataset):
    """Phase 2 dataset: Alpine station data paired with HRRR weather input.

    Each sample provides:
        terrain: Tensor (C_terrain, H_fine, W_fine) -- static terrain features
        weather_in: Tensor (C_weather, H_coarse, W_coarse) -- HRRR coarse input
        weather_target: Tensor (C_out, H_fine, W_fine) -- station observations
        mask: Tensor (H_fine, W_fine) -- valid observation locations

    Expected sample dict keys:
        terrain, weather_in, weather_target, mask
    """

    def __init__(self, samples: list[dict[str, torch.Tensor]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s = self.samples[idx]
        return s["terrain"], s["weather_in"], s["weather_target"], s["mask"]

    def compute_data_hash(self) -> str:
        """Compute hash of dataset for checkpoint metadata."""
        h = hashlib.sha256()
        h.update(str(len(self)).encode())
        if len(self) > 0:
            terrain, weather_in, _target, _mask = self[0]
            h.update(str(terrain.shape).encode())
            h.update(str(weather_in.shape).encode())
            h.update(str(_target.shape).encode())
        return h.hexdigest()[:16]


class ERA5StationDataset(Dataset):
    """Phase 2 dataset: ERA5 grid cells paired with Alpine station observations.

    Each sample pairs ERA5-derived weather variables (via NWP passthrough) with
    terrain features and station observation targets. The weather tensor has
    the same spatial resolution as the terrain (interpolated to fine grid).

    Uses ERA5Conditioner from backbone/era5_conditioner.py for data access.

    Expected sample dict keys:
        terrain: Tensor (C_terrain, H, W) -- static terrain features
        weather: Tensor (C_weather, H, W) -- ERA5 surface variables (6 channels)
        target: Tensor (C_out, H, W) -- station observation targets
        mask: Tensor (H, W) -- valid observation locations
    """

    def __init__(self, samples: list[dict[str, torch.Tensor]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s = self.samples[idx]
        return s["terrain"], s["weather"], s["target"], s["mask"]

    def compute_data_hash(self) -> str:
        """Compute hash of dataset for checkpoint metadata."""
        h = hashlib.sha256()
        h.update(str(len(self)).encode())
        if len(self) > 0:
            terrain, weather, target, _mask = self[0]
            h.update(str(terrain.shape).encode())
            h.update(str(weather.shape).encode())
            h.update(str(target.shape).encode())
        return h.hexdigest()[:16]


class ColoradoPhaseDataset(Dataset):
    """Phase 3 dataset: Colorado SNOTEL with quantile-mapped HRRR input.

    Same tensor interface as AlpinePhaseDataset but the weather_in has been
    quantile-mapped from Colorado to Alpine distribution space.

    Expected sample dict keys:
        terrain, weather_in, weather_target, mask
    """

    def __init__(self, samples: list[dict[str, torch.Tensor]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s = self.samples[idx]
        return s["terrain"], s["weather_in"], s["weather_target"], s["mask"]

    def compute_data_hash(self) -> str:
        """Compute hash of dataset for checkpoint metadata."""
        h = hashlib.sha256()
        h.update(str(len(self)).encode())
        if len(self) > 0:
            terrain, weather_in, _target, _mask = self[0]
            h.update(str(terrain.shape).encode())
            h.update(str(weather_in.shape).encode())
        return h.hexdigest()[:16]
