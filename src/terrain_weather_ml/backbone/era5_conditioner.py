"""ERA5 reanalysis input conditioning for Alpine and global pre-training.

Tasks 2.1-2.4: CDS API download, variable extraction/mapping, temporal
alignment with station observations, and resumable download with checkpoints.

ERA5 enables the cross-domain transfer path that HRRR's US-only coverage
cannot support. Used in Phase 2 (Alpine pre-training) with ERA5 providing
coarse input conditioning for the terrain downscaling head.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

from terrain_weather_ml.data.reynolds_creek import DownloadCheckpoint

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Variable mapping: ERA5 short names -> project schema
# ---------------------------------------------------------------------------

# ERA5 short names (as they appear in downloaded NetCDF) mapped to the
# project's canonical variable names. These match the NWP passthrough
# adapter output so the downscaling head receives identically structured
# input regardless of NWP source.
CDS_VARIABLE_MAP: dict[str, str] = {
    "2t": "T2M",      # 2m temperature (K)
    "10u": "U10",      # 10m u-component of wind (m/s)
    "10v": "V10",      # 10m v-component of wind (m/s)
    "tp": "PRATE",     # Total precipitation (accumulated m -> mm/hr rate)
    "sp": "SP",        # Surface pressure (Pa)
    "blh": "BLH",      # Boundary layer height (m)
}

# CDS API long variable names (used in the retrieve request)
CDS_LONG_NAMES: dict[str, str] = {
    "2t": "2m_temperature",
    "10u": "10m_u_component_of_wind",
    "10v": "10m_v_component_of_wind",
    "tp": "total_precipitation",
    "sp": "surface_pressure",
    "blh": "boundary_layer_height",
}

# NetCDF variable names as they appear in downloaded ERA5 files.
# CDS may use either short or long forms depending on API version.
# The short-form keys (t2m, u10, v10, tp, sp, blh) are standard.
NETCDF_VAR_NAMES: dict[str, str] = {
    "2t": "t2m",
    "10u": "u10",
    "10v": "v10",
    "tp": "tp",
    "sp": "sp",
    "blh": "blh",
}

# ---------------------------------------------------------------------------
# Predefined spatial domains
# ---------------------------------------------------------------------------

PREDEFINED_DOMAINS: dict[str, dict[str, float]] = {
    "alpine": {
        "lat_min": 45.0,
        "lat_max": 48.0,
        "lon_min": 5.0,
        "lon_max": 11.0,
    },
    "colorado": {
        "lat_min": 37.0,
        "lat_max": 41.0,
        "lon_min": -109.0,
        "lon_max": -105.0,
    },
}


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class ERA5Data:
    """Container for processed ERA5 grid data.

    Args:
        fields: Dict mapping project variable names to 2D numpy arrays (H, W).
        grid_shape: Spatial dimensions (H, W).
        valid_time: ISO 8601 valid time string.
        variables: List of variable names present in fields.
    """

    fields: dict[str, np.ndarray]
    grid_shape: tuple[int, int]
    valid_time: str
    variables: list[str]


@dataclass
class ERA5Config:
    """Configuration for ERA5 data conditioning.

    Args:
        domain: Predefined domain name ("alpine", "colorado") or "custom".
        lat_min: Minimum latitude (south bound). Required for custom domain.
        lat_max: Maximum latitude (north bound). Required for custom domain.
        lon_min: Minimum longitude (west bound). Required for custom domain.
        lon_max: Maximum longitude (east bound). Required for custom domain.
        variables: List of project-schema variable names to extract.
        cds_api_version: CDS API version ("legacy" or "beta").
    """

    domain: str = "alpine"
    lat_min: float | None = None
    lat_max: float | None = None
    lon_min: float | None = None
    lon_max: float | None = None
    variables: list[str] = field(default_factory=lambda: [
        "T2M", "U10", "V10", "PRATE", "SP", "BLH",
    ])
    cds_api_version: str = "legacy"

    def __post_init__(self):
        if self.domain == "custom":
            if any(v is None for v in [
                self.lat_min, self.lat_max, self.lon_min, self.lon_max
            ]):
                raise ValueError(
                    "custom domain requires all bounds "
                    "(lat_min, lat_max, lon_min, lon_max)"
                )
        elif self.domain in PREDEFINED_DOMAINS:
            bounds = PREDEFINED_DOMAINS[self.domain]
            self.lat_min = bounds["lat_min"]
            self.lat_max = bounds["lat_max"]
            self.lon_min = bounds["lon_min"]
            self.lon_max = bounds["lon_max"]
        else:
            raise ValueError(
                f"Unknown domain '{self.domain}'. "
                f"Use one of {list(PREDEFINED_DOMAINS.keys())} or 'custom'."
            )


# ---------------------------------------------------------------------------
# 2.3 — Temporal alignment
# ---------------------------------------------------------------------------


class ERA5TemporalAligner:
    """Aligns ERA5 hourly timestamps with station observation timestamps.

    PeakWeather provides 10-min observations, IMIS provides 30-min.
    This aligner selects the nearest ERA5 timestep within a configurable
    tolerance window.

    Args:
        tolerance_minutes: Maximum allowed gap between observation and
            nearest ERA5 timestep. Default 30 minutes.
    """

    def __init__(self, tolerance_minutes: int = 30) -> None:
        self.tolerance = timedelta(minutes=tolerance_minutes)

    def find_nearest(
        self,
        obs_time: datetime,
        era5_times: list[datetime],
    ) -> datetime:
        """Find the nearest ERA5 timestep to an observation time.

        Args:
            obs_time: Station observation timestamp.
            era5_times: Available ERA5 hourly timestamps.

        Returns:
            The nearest ERA5 timestamp within tolerance.

        Raises:
            ValueError: If no ERA5 timestep falls within tolerance,
                or if era5_times is empty.
        """
        if not era5_times:
            raise ValueError("ERA5 times list is empty")

        best_match = None
        best_delta = None

        for era5_t in era5_times:
            delta = abs(obs_time - era5_t)
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_match = era5_t

        if best_delta > self.tolerance:
            raise ValueError(
                f"No ERA5 timestep within tolerance "
                f"({self.tolerance.total_seconds() / 60:.0f} min) of "
                f"{obs_time.isoformat()}. Nearest: {best_match.isoformat()} "
                f"(delta: {best_delta.total_seconds() / 60:.0f} min)"
            )

        return best_match

    def align_batch(
        self,
        obs_times: list[datetime],
        era5_times: list[datetime],
    ) -> dict[datetime, datetime]:
        """Align a batch of observation times to ERA5 timesteps.

        Args:
            obs_times: List of station observation timestamps.
            era5_times: Available ERA5 hourly timestamps.

        Returns:
            Dict mapping observation time -> matched ERA5 time.

        Raises:
            ValueError: If any observation has no ERA5 match in tolerance.
        """
        return {
            obs_t: self.find_nearest(obs_t, era5_times)
            for obs_t in obs_times
        }


# ---------------------------------------------------------------------------
# 2.1, 2.2, 2.4 — ERA5Conditioner
# ---------------------------------------------------------------------------


class ERA5Conditioner:
    """Downloads and conditions ERA5 data for the terrain downscaling head.

    Handles:
    - CDS API data download for configurable spatial domains
    - Variable extraction and mapping (ERA5 short names -> project schema)
    - Accumulated precipitation -> instantaneous rate conversion
    - Resumable download via DownloadCheckpoint (same pattern as Reynolds Creek)

    Supports both legacy and beta CDS API versions.
    """

    def __init__(
        self,
        config: ERA5Config,
        output_dir: Path | None = None,
        checkpoint_path: Path | None = None,
    ) -> None:
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else Path("data/era5")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if checkpoint_path is None:
            checkpoint_path = self.output_dir / ".era5_checkpoint.json"
        self.checkpoint = DownloadCheckpoint(checkpoint_path)

    # -- Download (2.1, 2.4) ------------------------------------------------

    def _make_timestep_key(self, timestamp: datetime) -> str:
        """Generate a canonical key for a timestep in the checkpoint."""
        return f"era5_{timestamp.strftime('%Y-%m-%dT%H')}"

    def is_timestep_complete(self, timestamp: datetime) -> bool:
        """Check whether a timestep has been downloaded."""
        key = self._make_timestep_key(timestamp)
        return self.checkpoint.is_complete(key)

    def _build_cds_request(
        self, timestamp: datetime
    ) -> tuple[str, dict]:
        """Build the CDS API retrieve request.

        Returns:
            (dataset_name, request_dict) tuple.
        """
        dataset = "reanalysis-era5-single-levels"

        # Area: [north, west, south, east] for CDS API
        area = [
            self.config.lat_max,
            self.config.lon_min,
            self.config.lat_min,
            self.config.lon_max,
        ]

        # CDS long variable names
        variables = [
            CDS_LONG_NAMES[short]
            for short in CDS_LONG_NAMES
        ]

        request = {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": variables,
            "year": str(timestamp.year),
            "month": f"{timestamp.month:02d}",
            "day": f"{timestamp.day:02d}",
            "time": f"{timestamp.hour:02d}:00",
            "area": area,
        }

        return dataset, request

    def download_timestep(
        self,
        timestamp: datetime,
        cds_client=None,
    ) -> Path | None:
        """Download ERA5 data for a single timestep.

        Skips if already downloaded (checkpoint tracking).

        Args:
            timestamp: UTC datetime for the ERA5 timestep.
            cds_client: CDS API client instance. If None, attempts to
                create one (requires cdsapi package and valid ~/.cdsapirc).

        Returns:
            Path to the downloaded NetCDF file, or None if skipped.
        """
        key = self._make_timestep_key(timestamp)

        if self.checkpoint.is_complete(key):
            logger.debug("Skipping completed timestep: %s", key)
            return None

        # Build output path
        dest = (
            self.output_dir
            / str(timestamp.year)
            / f"{timestamp.month:02d}"
            / f"era5_{timestamp.strftime('%Y%m%dT%H')}.nc"
        )
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Build request
        dataset, request = self._build_cds_request(timestamp)

        logger.info(
            "Downloading ERA5 %s (%d complete)",
            key,
            self.checkpoint.completed_count,
        )

        try:
            if cds_client is None:
                import cdsapi
                cds_client = cdsapi.Client()

            cds_client.retrieve(dataset, request, str(dest))

            # Compute file hash and mark complete
            file_hash = _compute_file_hash(dest)
            self.checkpoint.mark_complete(key, file_hash)
            self.checkpoint.save()

            logger.info("Downloaded %s -> %s", key, dest)
            return dest

        except Exception:
            logger.exception("Failed to download %s", key)
            raise

    # -- Variable extraction (2.2) ------------------------------------------

    def extract_variables(
        self,
        ds: xr.Dataset,
        time_idx: int = 0,
    ) -> ERA5Data:
        """Extract and map ERA5 variables to project schema.

        Handles:
        - Variable renaming (ERA5 short names -> project schema)
        - Accumulated precipitation -> instantaneous rate conversion
          (differencing consecutive timesteps, convert m -> mm/hr)

        Args:
            ds: xarray Dataset with ERA5 data (from downloaded NetCDF).
            time_idx: Index of the timestep to extract.

        Returns:
            ERA5Data with mapped variable names and converted values.
        """
        fields: dict[str, np.ndarray] = {}

        for era5_short, project_name in CDS_VARIABLE_MAP.items():
            nc_var = NETCDF_VAR_NAMES[era5_short]

            if nc_var not in ds.data_vars:
                logger.warning(
                    "Variable %s (ERA5: %s) not found in dataset",
                    project_name, nc_var,
                )
                continue

            if project_name == "PRATE":
                # ERA5 precipitation is accumulated over the forecast period.
                # Convert to instantaneous rate by differencing consecutive
                # timesteps.
                fields[project_name] = self._convert_precipitation(
                    ds[nc_var], time_idx
                )
            else:
                # Direct extraction
                data = ds[nc_var].isel(time=time_idx).values.astype(np.float32)
                fields[project_name] = data

        # Determine grid shape from first extracted field
        first_field = next(iter(fields.values()))
        grid_shape = first_field.shape

        # Build valid time string
        time_val = ds["time"].values[time_idx]
        if hasattr(time_val, "isoformat"):
            valid_time = time_val.isoformat()
        else:
            valid_time = str(np.datetime_as_string(time_val, unit="s"))

        return ERA5Data(
            fields=fields,
            grid_shape=grid_shape,
            valid_time=valid_time,
            variables=list(fields.keys()),
        )

    def _convert_precipitation(
        self,
        tp_var: xr.DataArray,
        time_idx: int,
    ) -> np.ndarray:
        """Convert ERA5 accumulated precipitation to instantaneous rate.

        ERA5 'tp' is accumulated total precipitation in meters since the
        start of the forecast step. To get hourly rate:
        - If time_idx > 0: diff = tp[time_idx] - tp[time_idx - 1]
        - If time_idx == 0: use the absolute value (first step accumulation)

        Then convert m -> mm (multiply by 1000).

        Returns:
            2D array of precipitation rate in mm/hr.
        """
        current = tp_var.isel(time=time_idx).values.astype(np.float32)

        if time_idx > 0:
            previous = tp_var.isel(time=time_idx - 1).values.astype(np.float32)
            rate_m = current - previous
        else:
            # First timestep: use absolute accumulated value
            rate_m = current

        # Convert m -> mm/hr (ERA5 hourly data, so 1 step = 1 hour)
        rate_mm_hr = rate_m * 1000.0

        # Ensure non-negative (accumulated precip should be monotonic,
        # but numerical precision can cause tiny negatives)
        return np.maximum(rate_mm_hr, 0.0)


def _compute_file_hash(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()
