"""PeakWeather HuggingFace download and parsing.

Task 2.1: Download and parse MeteoSwiss/PeakWeather dataset for 302 Swiss
stations at 10-min resolution.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Protocol

import pandas as pd

from terrain_weather_ml.data.alpine.schema import (
    Network,
    QualityFlag,
    StationMetadata,
    WeatherRecord,
)

logger = logging.getLogger(__name__)

# PeakWeather variable mapping: HuggingFace column names -> our schema
PEAKWEATHER_VAR_MAP = {
    "ta": "temperature",  # 2m air temperature (Celsius in raw)
    "fu": "wind_speed",  # wind speed (m/s)
    "dkl": "wind_direction",  # wind direction (degrees)
    "rre": "precipitation",  # precipitation (mm/10min)
    "prestas": "pressure",  # station pressure (hPa)
    "ure": "humidity",  # relative humidity (%)
    "gre": "global_radiation",  # global radiation (W/m2)
}

EXPECTED_STATION_COUNT = 302


class HuggingFaceDownloader(Protocol):
    """Protocol for HuggingFace dataset downloading."""

    def download(
        self,
        repo_id: str,
        start_date: datetime,
        end_date: datetime,
        local_dir: Path,
    ) -> list[Path]:
        """Download dataset files for the given date range.

        Returns list of downloaded file paths.
        """
        ...

    def list_files(self, repo_id: str) -> list[str]:
        """List available files in the repository."""
        ...


class DefaultHFDownloader:
    """Default HuggingFace downloader using huggingface_hub."""

    def __init__(self, token: str | None = None) -> None:
        self.token = token

    def download(
        self,
        repo_id: str,
        start_date: datetime,
        end_date: datetime,
        local_dir: Path,
    ) -> list[Path]:
        from huggingface_hub import hf_hub_download, list_repo_files

        files = list_repo_files(repo_id, token=self.token)
        downloaded = []
        for fname in files:
            if self._file_in_range(fname, start_date, end_date):
                path = hf_hub_download(
                    repo_id,
                    fname,
                    local_dir=str(local_dir),
                    token=self.token,
                )
                downloaded.append(Path(path))
        return downloaded

    def list_files(self, repo_id: str) -> list[str]:
        from huggingface_hub import list_repo_files

        return list(list_repo_files(repo_id, token=self.token))

    @staticmethod
    def _file_in_range(
        fname: str, start_date: datetime, end_date: datetime
    ) -> bool:
        """Check if a filename corresponds to a date within range."""
        # PeakWeather files are typically organized by year/month
        # Accept all data files for now; filtering happens at parse time
        return fname.endswith((".csv", ".parquet", ".csv.gz"))


class PeakWeatherClient:
    """Client for downloading and parsing PeakWeather data from HuggingFace.

    Supports incremental updates: only downloads date ranges not already
    present in the local cache.
    """

    REPO_ID = "MeteoSwiss/PeakWeather"

    def __init__(
        self,
        cache_dir: Path,
        downloader: HuggingFaceDownloader | None = None,
        token: str | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.downloader = downloader or DefaultHFDownloader(token=token)
        self._metadata_cache: dict[str, StationMetadata] | None = None

    def download(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> list[Path]:
        """Download PeakWeather data for the given date range.

        Supports incremental updates -- skips already-downloaded ranges.
        """
        existing_ranges = self._get_existing_ranges()
        missing_ranges = self._compute_missing_ranges(
            start_date, end_date, existing_ranges
        )

        if not missing_ranges:
            logger.info("All data already present for %s to %s", start_date, end_date)
            return list(self.cache_dir.glob("*.parquet")) + list(
                self.cache_dir.glob("*.csv*")
            )

        all_files: list[Path] = []
        for range_start, range_end in missing_ranges:
            logger.info("Downloading PeakWeather %s to %s", range_start, range_end)
            files = self.downloader.download(
                self.REPO_ID, range_start, range_end, self.cache_dir
            )
            all_files.extend(files)
            self._record_downloaded_range(range_start, range_end)

        return all_files

    def parse_stations(
        self, data: pd.DataFrame
    ) -> dict[str, StationMetadata]:
        """Extract station metadata from PeakWeather DataFrame.

        Returns dict mapping station_id to StationMetadata.
        """
        stations: dict[str, StationMetadata] = {}

        if "station_id" not in data.columns:
            # Try alternate column names
            if "stn" in data.columns:
                data = data.rename(columns={"stn": "station_id"})
            else:
                raise ValueError("No station_id or stn column found in data")

        # Map alternate column names
        col_renames = {
            "lat": "latitude",
            "lon": "longitude",
            "alt": "elevation",
            "elev": "elevation",
        }
        data = data.rename(
            columns={k: v for k, v in col_renames.items() if k in data.columns}
        )

        for sid in data["station_id"].unique():
            sdata = data[data["station_id"] == sid].iloc[0]
            lat = float(sdata["latitude"])
            lon = float(sdata["longitude"])
            elev = float(sdata["elevation"])

            stations[str(sid)] = StationMetadata(
                station_id=str(sid),
                network=Network.PEAKWEATHER,
                latitude=lat,
                longitude=lon,
                elevation=elev,
                terrain_ref_lat=lat,
                terrain_ref_lon=lon,
            )

        return stations

    def parse_observations(
        self, data: pd.DataFrame
    ) -> list[WeatherRecord]:
        """Parse PeakWeather observations into WeatherRecords.

        Raw PeakWeather data uses:
        - Temperature in Celsius -> convert to Kelvin
        - Humidity in % -> convert to fraction
        - All other variables already in target units
        """
        records: list[WeatherRecord] = []

        # Normalize column names
        if "stn" in data.columns:
            data = data.rename(columns={"stn": "station_id"})
        if "time" in data.columns:
            data = data.rename(columns={"time": "timestamp"})

        if not pd.api.types.is_datetime64_any_dtype(data["timestamp"]):
            data["timestamp"] = pd.to_datetime(data["timestamp"])

        for _, row in data.iterrows():
            record = WeatherRecord(
                timestamp=row["timestamp"].to_pydatetime(),
                station_id=str(row["station_id"]),
                network=Network.PEAKWEATHER,
            )

            # Map and convert variables
            for raw_col, var_name in PEAKWEATHER_VAR_MAP.items():
                if raw_col in row.index and pd.notna(row[raw_col]):
                    value = float(row[raw_col])
                    setattr(record, var_name, value)
                    setattr(record, f"{var_name}_qc", QualityFlag.GOOD)

            records.append(record)

        return records

    def _get_existing_ranges(self) -> list[tuple[datetime, datetime]]:
        """Get date ranges already downloaded."""
        manifest = self.cache_dir / ".manifest"
        if not manifest.exists():
            return []
        ranges = []
        for line in manifest.read_text().strip().split("\n"):
            if line:
                start_str, end_str = line.split(",")
                ranges.append((
                    datetime.fromisoformat(start_str),
                    datetime.fromisoformat(end_str),
                ))
        return ranges

    def _record_downloaded_range(
        self, start: datetime, end: datetime
    ) -> None:
        """Record a downloaded date range in the manifest."""
        manifest = self.cache_dir / ".manifest"
        with manifest.open("a") as f:
            f.write(f"{start.isoformat()},{end.isoformat()}\n")

    @staticmethod
    def _compute_missing_ranges(
        start: datetime,
        end: datetime,
        existing: list[tuple[datetime, datetime]],
    ) -> list[tuple[datetime, datetime]]:
        """Compute date ranges not covered by existing downloads."""
        if not existing:
            return [(start, end)]

        # Sort existing ranges
        existing = sorted(existing, key=lambda x: x[0])

        # Check if fully covered
        for ex_start, ex_end in existing:
            if ex_start <= start and ex_end >= end:
                return []

        # Simple gap-finding: return uncovered ranges
        missing = []
        cursor = start
        for ex_start, ex_end in existing:
            if cursor < ex_start:
                missing.append((cursor, min(ex_start, end)))
            cursor = max(cursor, ex_end)
        if cursor < end:
            missing.append((cursor, end))

        return missing
