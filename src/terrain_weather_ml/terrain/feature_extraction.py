"""Extract terrain features for geographic point locations.

Given a DEM and a set of lat/lon coordinates, extracts 64x64 patches
centered on each point and computes terrain features (point and area).
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform as warp_transform

from terrain_weather_ml.terrain import CHANNEL_NAMES
from terrain_weather_ml.terrain.encoder import TerrainFeatureEncoder

logger = logging.getLogger(__name__)

PATCH_SIZE = 64
AREA_RADIUS_M = 1000.0


class TerrainFeatureExtractor:
    """Extracts terrain features for station locations from a DEM."""

    def __init__(
        self,
        dem_path: str | Path,
        patch_size: int = PATCH_SIZE,
        area_radius_m: float = AREA_RADIUS_M,
    ):
        self.dem_path = Path(dem_path)
        self.patch_size = patch_size
        self.area_radius_m = area_radius_m
        self.encoder = TerrainFeatureEncoder(device="cpu")

        with rasterio.open(self.dem_path) as src:
            self.dem_crs = src.crs
            self.dem_transform = src.transform
            self.dem_height = src.height
            self.dem_width = src.width
            self.dem_resolution = abs(src.transform.a)

        self._area_mask = self._build_area_mask()

    def _build_area_mask(self) -> np.ndarray:
        """Build a circular mask for area statistics."""
        radius_px = self.area_radius_m / self.dem_resolution
        center = self.patch_size // 2
        y, x = np.ogrid[:self.patch_size, :self.patch_size]
        dist = np.sqrt((x - center) ** 2 + (y - center) ** 2)
        return dist <= radius_px

    def _latlon_to_pixel(
        self, lats: np.ndarray, lons: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert WGS84 lat/lon to DEM pixel coordinates."""
        xs, ys = warp_transform(
            CRS.from_epsg(4326), self.dem_crs, lons.tolist(), lats.tolist()
        )
        cols, rows = ~self.dem_transform @ (np.array(xs), np.array(ys))
        return np.round(rows).astype(int), np.round(cols).astype(int)

    def _extract_patch_tif(
        self, row: int, col: int, tmp_dir: str
    ) -> Path | None:
        """Extract a patch from the DEM and save as a temporary GeoTIFF."""
        half = self.patch_size // 2
        r0, r1 = row - half, row + half
        c0, c1 = col - half, col + half

        if r0 < 0 or r1 > self.dem_height or c0 < 0 or c1 > self.dem_width:
            return None

        with rasterio.open(self.dem_path) as src:
            window = rasterio.windows.Window(c0, r0, self.patch_size, self.patch_size)
            data = src.read(1, window=window).astype(np.float32)
            patch_transform = src.window_transform(window)

        patch_path = Path(tmp_dir) / f"patch_{row}_{col}.tif"
        with rasterio.open(
            patch_path,
            "w",
            driver="GTiff",
            height=self.patch_size,
            width=self.patch_size,
            count=1,
            dtype="float32",
            crs=self.dem_crs,
            transform=patch_transform,
        ) as dst:
            dst.write(data, 1)

        return patch_path

    def _compute_features_for_patch(
        self, patch_path: Path
    ) -> dict[str, float]:
        """Run the encoder on a patch and extract point + area features."""
        tensor = self.encoder.encode(str(patch_path))
        features = {}
        center = self.patch_size // 2

        for i, name in enumerate(CHANNEL_NAMES):
            channel = tensor[i].numpy()
            features[name] = float(channel[center, center])
            masked = channel[self._area_mask]
            features[f"{name}_mean"] = float(np.mean(masked))
            features[f"{name}_max"] = float(np.max(masked))
            features[f"{name}_std"] = float(np.std(masked))

        return features

    def extract_features(self, stations: pd.DataFrame) -> pd.DataFrame:
        """Extract terrain features for all stations.

        Args:
            stations: DataFrame with columns: name, station_id, latitude, longitude.

        Returns:
            DataFrame with station metadata + 68 terrain feature columns.
        """
        lats = stations["latitude"].values
        lons = stations["longitude"].values
        rows, cols = self._latlon_to_pixel(lats, lons)

        results = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            for idx in range(len(stations)):
                station = stations.iloc[idx]
                r, c = int(rows[idx]), int(cols[idx])
                logger.info(
                    "Processing %s (pixel %d, %d)", station["name"], r, c
                )

                patch_path = self._extract_patch_tif(r, c, tmp_dir)
                if patch_path is None:
                    logger.warning(
                        "Station %s at pixel (%d, %d) too close to DEM edge, skipping",
                        station["name"], r, c,
                    )
                    continue

                features = self._compute_features_for_patch(patch_path)
                features["name"] = station["name"]
                features["station_id"] = station["station_id"]
                results.append(features)
                patch_path.unlink()

        result_df = pd.DataFrame(results)
        col_order = ["station_id", "name"] + [
            c for c in result_df.columns if c not in ("station_id", "name")
        ]
        return result_df[col_order]
