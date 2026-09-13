"""Terrain feature encoder.

Assembles all terrain features into a single PyTorch tensor of shape (C, H, W).
Supports CPU/CUDA/MPS devices. Caches computed tensors to disk.

Channel ordering (C_TERRAIN = 17):
  0: elevation
  1: slope
  2: aspect_sin
  3: aspect_cos
  4: plan_curvature
  5: profile_curvature
  6: svf
  7-14: sx_n, sx_ne, sx_e, sx_se, sx_s, sx_sw, sx_w, sx_nw
  15: tpi
  16: roughness
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import torch

from terrain_weather_ml.terrain.dem import load_dem
from terrain_weather_ml.terrain.features import (
    compute_aspect,
    compute_curvature,
    compute_slope,
)
from terrain_weather_ml.terrain.roughness import compute_roughness
from terrain_weather_ml.terrain.svf import compute_svf
from terrain_weather_ml.terrain.sx import compute_sx
from terrain_weather_ml.terrain.tpi import compute_tpi

logger = logging.getLogger(__name__)

C_TERRAIN = 17

CHANNEL_NAMES = [
    "elevation",
    "slope",
    "aspect_sin",
    "aspect_cos",
    "plan_curvature",
    "profile_curvature",
    "svf",
    "sx_n",
    "sx_ne",
    "sx_e",
    "sx_se",
    "sx_s",
    "sx_sw",
    "sx_w",
    "sx_nw",
    "tpi",
    "roughness",
]


def _cache_key(dem_path: str | Path, resolution: float | None, crs_epsg: int | None) -> str:
    """Compute a deterministic cache key from DEM path and parameters."""
    key_str = f"{Path(dem_path).resolve()}|{resolution}|{crs_epsg}"
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


class TerrainFeatureEncoder:
    """Computes and caches static terrain features from DEM data.

    Args:
        cache_dir: Directory for cached tensor files.
        device: PyTorch device ('cpu', 'cuda', 'mps').
        svf_radius: SVF maximum search radius in meters.
        svf_azimuths: Number of SVF azimuth directions.
        sx_radius: Sx maximum search radius in meters.
        tpi_radius: TPI neighborhood radius in meters.
        default_roughness: Default z0 when NLCD is unavailable.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        device: str = "cpu",
        svf_radius: float = 5000.0,
        svf_azimuths: int = 36,
        sx_radius: float = 3000.0,
        tpi_radius: float = 500.0,
        default_roughness: float = 0.1,
    ):
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.device = device
        self.svf_radius = svf_radius
        self.svf_azimuths = svf_azimuths
        self.sx_radius = sx_radius
        self.tpi_radius = tpi_radius
        self.default_roughness = default_roughness

        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, dem_path: str | Path) -> Path | None:
        """Return cache file path, or None if caching is disabled."""
        if self.cache_dir is None:
            return None
        key = _cache_key(dem_path, None, None)
        return self.cache_dir / f"terrain_{key}.pt"

    def _load_cached(self, cache_path: Path | None) -> torch.Tensor | None:
        """Try to load a cached tensor."""
        if cache_path is not None and cache_path.exists():
            logger.info("Loading cached terrain features from %s", cache_path)
            return torch.load(cache_path, map_location=self.device, weights_only=True)
        return None

    def _save_cached(self, tensor: torch.Tensor, cache_path: Path | None) -> None:
        """Save tensor to cache."""
        if cache_path is not None:
            torch.save(tensor, cache_path)
            logger.info("Saved terrain features to cache: %s", cache_path)

    def encode(
        self,
        dem_path: str | Path,
        nlcd_raster: np.ndarray | None = None,
    ) -> torch.Tensor:
        """Compute terrain feature tensor from a DEM file.

        Args:
            dem_path: Path to GeoTIFF DEM file.
            nlcd_raster: Optional NLCD land cover raster. If None, uses default roughness.

        Returns:
            Terrain feature tensor of shape (C_TERRAIN, H, W) on the configured device.
        """
        cache_path = self._get_cache_path(dem_path)
        cached = self._load_cached(cache_path)
        if cached is not None:
            return cached

        # Load DEM
        dem_data = load_dem(dem_path)
        elevation = dem_data.elevation
        resolution = dem_data.resolution
        h, w = elevation.shape

        logger.info(
            "Computing terrain features for %s (%dx%d, res=%.1fm)",
            dem_path, h, w, resolution,
        )

        # Compute features
        slope = compute_slope(elevation, resolution)
        aspect_sin, aspect_cos = compute_aspect(elevation, resolution)
        plan_curv, profile_curv = compute_curvature(elevation, resolution)
        svf = compute_svf(
            elevation,
            resolution,
            max_radius=self.svf_radius,
            n_azimuths=self.svf_azimuths,
        )
        sx = compute_sx(elevation, resolution, max_radius=self.sx_radius)
        tpi = compute_tpi(elevation, resolution, radius=self.tpi_radius)
        roughness = compute_roughness(
            nlcd_raster,
            shape=(h, w),
            default_z0=self.default_roughness,
        )

        # Assemble tensor: (C_TERRAIN, H, W)
        channels = np.stack(
            [
                elevation,
                slope,
                aspect_sin,
                aspect_cos,
                plan_curv,
                profile_curv,
                svf,
                sx[0],  # N
                sx[1],  # NE
                sx[2],  # E
                sx[3],  # SE
                sx[4],  # S
                sx[5],  # SW
                sx[6],  # W
                sx[7],  # NW
                tpi,
                roughness,
            ],
            axis=0,
        )

        assert channels.shape[0] == C_TERRAIN, (
            f"Expected {C_TERRAIN} channels, got {channels.shape[0]}"
        )

        tensor = torch.from_numpy(channels).to(self.device)

        # Replace any remaining NaN with 0
        tensor = torch.nan_to_num(tensor, nan=0.0)

        self._save_cached(tensor, cache_path)

        return tensor
