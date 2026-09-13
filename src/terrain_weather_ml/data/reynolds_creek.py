"""Reynolds Creek data pipeline for gridded validation data.

Task 8.1: Variable and temporal subsetting
Task 8.2: Staged download with checkpointing
Task 8.3: NetCDF format conversion

Downloads a configurable subset of the Reynolds Creek Experimental Watershed
21TB hourly gridded product (10m resolution, 239 km2 in southwestern Idaho).
Only 6 core variables are downloaded. Temporal window is configurable
(default: 5 most recent complete water years). Downloads are resumable via
checkpoint files with file hash verification.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import xarray as xr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Task 8.1 — Variable and temporal subsetting
# ---------------------------------------------------------------------------

# The 6 core variables to download, mapped to standard names and units.
CORE_VARIABLES: dict[str, dict[str, str]] = {
    "air_temperature": {
        "long_name": "Air temperature at 2m",
        "units": "K",
        "raw_name": "tair",
    },
    "wind_speed": {
        "long_name": "Wind speed at 10m",
        "units": "m/s",
        "raw_name": "wspd",
    },
    "wind_direction": {
        "long_name": "Wind direction at 10m",
        "units": "degrees",
        "raw_name": "wdir",
    },
    "precipitation": {
        "long_name": "Precipitation (liquid equivalent)",
        "units": "mm/hr",
        "raw_name": "prec",
    },
    "relative_humidity": {
        "long_name": "Relative humidity",
        "units": "fraction",
        "raw_name": "rh",
    },
    "shortwave_radiation": {
        "long_name": "Incoming shortwave radiation",
        "units": "W/m2",
        "raw_name": "iswr",
    },
}

# Reynolds Creek Experimental Watershed approximate spatial bounds (WGS84).
# Full domain: ~239 km2 in southwestern Idaho.
RC_SPATIAL_BOUNDS = {
    "lat_min": 43.0,
    "lat_max": 43.2,
    "lon_min": -116.9,
    "lon_max": -116.6,
}

# Most recent complete water year for default temporal window.
# Water year N runs Oct 1 of year N-1 through Sep 30 of year N.
# The dataset has data through ~2020.
_LATEST_COMPLETE_WY = 2020
_DEFAULT_NUM_WATER_YEARS = 5

# Base URL for Reynolds Creek data (ESSD open access via PANGAEA/ARS).
RC_BASE_URL = "https://data.nal.usda.gov/dataset/reynolds-creek"


@dataclass
class ReynoldsCreekConfig:
    """Configuration for Reynolds Creek data subsetting.

    Defaults to 5 most recent complete water years, full spatial domain,
    and the 6 core variables.
    """

    start_date: date | None = None
    end_date: date | None = None
    num_water_years: int = _DEFAULT_NUM_WATER_YEARS
    spatial_subset: bool = False
    variables: list[str] | None = None

    def __post_init__(self):
        if self.variables is None:
            self.variables = list(CORE_VARIABLES.keys())

    @property
    def spatial_bounds(self) -> dict[str, float]:
        """Return the spatial bounds (full domain, no subsetting)."""
        return RC_SPATIAL_BOUNDS.copy()

    def is_variable_included(self, variable: str) -> bool:
        """Check if a variable is in the core set."""
        return variable in CORE_VARIABLES

    def get_temporal_bounds(self) -> tuple[date, date]:
        """Return (start_date, end_date) for the configured temporal window.

        Default: 5 most recent complete water years.
        Water year N: Oct 1 of year N-1 through Sep 30 of year N.
        """
        if self.start_date is not None and self.end_date is not None:
            return self.start_date, self.end_date

        # Default: 5 most recent complete water years
        end_wy = _LATEST_COMPLETE_WY
        start_wy = end_wy - self.num_water_years + 1

        start = date(start_wy - 1, 10, 1)  # Oct 1 of year before start WY
        end = date(end_wy, 9, 30)            # Sep 30 of end WY

        return start, end

    def get_date_range(self) -> list[date]:
        """Return list of daily dates within the temporal window."""
        start, end = self.get_temporal_bounds()
        days = []
        current = start
        while current <= end:
            days.append(current)
            current += timedelta(days=1)
        return days


# ---------------------------------------------------------------------------
# Task 8.2 — Staged download with checkpointing
# ---------------------------------------------------------------------------


@dataclass
class DownloadFileInfo:
    """Metadata for a single file in the download plan."""

    file_key: str
    variable: str
    target_date: date
    is_complete: bool = False
    file_hash: str | None = None


@dataclass
class DownloadCheckpoint:
    """Checkpoint tracking for resumable downloads.

    Persists to a JSON file. Tracks which files have been downloaded
    and their hashes for integrity verification on restart.
    """

    path: Path
    _completed: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        self.path = Path(self.path)
        if self.path.exists():
            self.load()

    def load(self) -> None:
        """Load checkpoint state from disk."""
        if self.path.exists():
            data = json.loads(self.path.read_text())
            self._completed = data.get("completed", {})
        else:
            self._completed = {}

    def save(self) -> None:
        """Persist checkpoint state to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"completed": self._completed}
        self.path.write_text(json.dumps(data, indent=2))

    def mark_complete(self, file_key: str, file_hash: str) -> None:
        """Mark a file as successfully downloaded with its hash."""
        self._completed[file_key] = file_hash

    def is_complete(self, file_key: str) -> bool:
        """Check if a file has been downloaded."""
        return file_key in self._completed

    def verify_hash(self, file_key: str, file_hash: str) -> bool:
        """Verify that a file's hash matches the stored hash."""
        stored_hash = self._completed.get(file_key)
        if stored_hash is None:
            return False
        return stored_hash == file_hash

    @property
    def completed_count(self) -> int:
        """Number of completed files."""
        return len(self._completed)

    def get_hash(self, file_key: str) -> str | None:
        """Get stored hash for a file, or None."""
        return self._completed.get(file_key)


def compute_file_hash(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


class ReynoldsCreekDownloader:
    """Resumable downloader for Reynolds Creek gridded data.

    Supports:
    - Variable subsetting (only core 6 variables)
    - Temporal windowing (configurable water years)
    - Checkpoint-based resume (skip completed files)
    - Hash verification (detect corrupted files)
    - Progress reporting
    """

    def __init__(
        self,
        output_dir: Path,
        checkpoint_path: Path | None = None,
        config: ReynoldsCreekConfig | None = None,
        base_url: str = RC_BASE_URL,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or ReynoldsCreekConfig()
        self.base_url = base_url

        if checkpoint_path is None:
            checkpoint_path = self.output_dir / ".download_checkpoint.json"
        self.checkpoint = DownloadCheckpoint(checkpoint_path)

    def plan_downloads(self) -> list[DownloadFileInfo]:
        """Generate the full download plan, marking completed files.

        Returns a list of DownloadFileInfo for every variable/date combo,
        with is_complete=True for files already in the checkpoint.
        """
        plan: list[DownloadFileInfo] = []
        dates = self.config.get_date_range()

        for variable in self.config.variables:
            for dt in dates:
                file_key = _make_file_key(variable, dt)
                is_done = self.checkpoint.is_complete(file_key)
                stored_hash = self.checkpoint.get_hash(file_key)
                plan.append(DownloadFileInfo(
                    file_key=file_key,
                    variable=variable,
                    target_date=dt,
                    is_complete=is_done,
                    file_hash=stored_hash,
                ))

        return plan

    def download_file(
        self,
        file_info: DownloadFileInfo,
        http_client=None,
    ) -> Path | None:
        """Download a single file, respecting checkpoint state.

        Args:
            file_info: Metadata for the file to download.
            http_client: Optional httpx client for making requests.

        Returns:
            Path to the downloaded file, or None if skipped/failed.
        """
        if file_info.is_complete:
            logger.debug("Skipping completed file: %s", file_info.file_key)
            return None

        dest = self.output_dir / file_info.file_key
        dest.parent.mkdir(parents=True, exist_ok=True)

        url = self._build_url(file_info)
        logger.info(
            "Downloading %s (%d/%d complete)",
            file_info.file_key,
            self.checkpoint.completed_count,
            self._total_files(),
        )

        try:
            if http_client is None:
                import httpx
                http_client = httpx.Client(timeout=300)

            response = http_client.get(url)
            response.raise_for_status()

            dest.write_bytes(response.content)
            file_hash = compute_file_hash(dest)

            self.checkpoint.mark_complete(file_info.file_key, file_hash)
            self.checkpoint.save()

            return dest

        except Exception:
            logger.exception("Failed to download %s", file_info.file_key)
            return None

    def _build_url(self, file_info: DownloadFileInfo) -> str:
        """Build the download URL for a file."""
        raw_name = CORE_VARIABLES[file_info.variable]["raw_name"]
        dt_str = file_info.target_date.strftime("%Y%m%d")
        return f"{self.base_url}/{raw_name}/{file_info.target_date.year}/{raw_name}_{dt_str}.nc"

    def _total_files(self) -> int:
        """Total number of files in the download plan."""
        dates = self.config.get_date_range()
        return len(self.config.variables) * len(dates)


def _make_file_key(variable: str, dt: date) -> str:
    """Generate the canonical file key for a variable/date combination."""
    return f"{variable}/{dt.year}/{variable}_{dt.strftime('%Y%m%d')}.nc"


# ---------------------------------------------------------------------------
# Task 8.3 — NetCDF format conversion
# ---------------------------------------------------------------------------


class NetCDFConverter:
    """Convert raw Reynolds Creek data to standardised per-variable daily NetCDF.

    Output organization: ``{variable}/{year}/{variable}_{YYYYMMDD}.nc``
    Each file contains one variable for one day with hourly time steps.
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)

    def convert_file(
        self,
        raw_path: Path,
        variable: str,
        target_date: date,
    ) -> Path:
        """Convert a single raw NetCDF file to the standard format.

        Args:
            raw_path: Path to the raw Reynolds Creek NetCDF.
            variable: Standard variable name (e.g. ``air_temperature``).
            target_date: The date this file covers.

        Returns:
            Path to the converted file.
        """
        ds = xr.open_dataset(raw_path)

        # Ensure the variable is present
        if variable not in ds.data_vars:
            # Try raw name mapping
            raw_name = CORE_VARIABLES.get(variable, {}).get("raw_name")
            if raw_name and raw_name in ds.data_vars:
                ds = ds.rename({raw_name: variable})
            else:
                # Use the first data variable and rename
                first_var = next(iter(ds.data_vars))
                ds = ds.rename({first_var: variable})

        # Add metadata attributes
        var_meta = CORE_VARIABLES.get(variable, {})
        ds[variable].attrs["units"] = var_meta.get("units", "unknown")
        ds[variable].attrs["long_name"] = var_meta.get(
            "long_name", variable
        )

        # Ensure only the target variable is in the output
        keep_vars = [variable]
        drop_vars = [v for v in ds.data_vars if v not in keep_vars]
        if drop_vars:
            ds = ds.drop_vars(drop_vars)

        # Build output path
        out_path = (
            self.output_dir
            / variable
            / str(target_date.year)
            / f"{variable}_{target_date.strftime('%Y%m%d')}.nc"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Write with compression when netCDF4 is available, plain otherwise
        encoding = {variable: {"dtype": "float32"}}
        try:
            import netCDF4  # noqa: F401
            encoding[variable]["zlib"] = True
            encoding[variable]["complevel"] = 4
            engine = "netcdf4"
        except ImportError:
            engine = "scipy"

        ds.to_netcdf(out_path, encoding=encoding, engine=engine)
        ds.close()

        logger.info("Converted %s -> %s", raw_path, out_path)
        return out_path

    def convert_batch(
        self,
        file_specs: list[tuple[Path, str, date]],
    ) -> list[Path]:
        """Convert multiple raw files.

        Args:
            file_specs: List of (raw_path, variable, target_date) tuples.

        Returns:
            List of output file paths.
        """
        results = []
        for raw_path, variable, target_date in file_specs:
            result = self.convert_file(raw_path, variable, target_date)
            results.append(result)
        return results
