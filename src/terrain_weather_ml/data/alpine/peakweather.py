"""PeakWeather HuggingFace download and parsing.

Task 2.1: Download and parse MeteoSwiss/PeakWeather dataset for 302 Swiss
stations at 10-min resolution.

Real HuggingFace schema:
- stations.parquet: nat_abbr index, station_height, 13 terrain features
- observations/YYYY.parquet: datetime index, MultiIndex columns (nat_abbr, param_short)
- Parameter codes: tre200s0, fkl010z0, dkl010z0, rre150z0, prestas0, ure200s0, sre000z0
"""

from __future__ import annotations

import logging
import re
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

# PeakWeather variable mapping: real MeteoSwiss parameter codes -> our schema
PEAKWEATHER_VAR_MAP = {
    "tre200s0": "temperature",    # 2m air temperature (Celsius in raw)
    "fkl010z0": "wind_speed",     # wind speed (m/s)
    "dkl010z0": "wind_direction", # wind direction (degrees)
    "rre150z0": "precipitation",  # precipitation (mm/10min)
    "prestas0": "pressure",       # station pressure (hPa)
    "ure200s0": "humidity",       # relative humidity (% in raw)
    "sre000z0": "sunshine",       # sunshine duration (min/10min)
    "fkl010z1": "wind_gust",      # wind gust peak (m/s), optional
}

# Pre-computed terrain feature columns in stations.parquet
TERRAIN_FEATURE_COLS = (
    "ASPECT_2000M_SIGRATIO1",
    "WE_DERIVATIVE_2000M_SIGRATIO1",
    "TPI_2000M",
    "SN_DERIVATIVE_10000M_SIGRATIO1",
    "dem",
    "SN_DERIVATIVE_2000M_SIGRATIO1",
    "SLOPE_10000M_SIGRATIO1",
    "ASPECT_10000M_SIGRATIO1",
    "SLOPE_2000M_SIGRATIO1",
    "STD_2000M",
    "STD_10000M",
    "TPI_10000M",
    "WE_DERIVATIVE_10000M_SIGRATIO1",
)

EXPECTED_STATION_COUNT = 302

# Metadata files that should always be downloaded
_METADATA_FILES = frozenset({
    "stations.parquet",
    "parameters.parquet",
    "installation.parquet",
})

# Pattern matching observation year files: data/observations/YYYY.parquet
_OBS_YEAR_RE = re.compile(r"observations/(\d{4})\.parquet$")


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
        """Check if a filename should be downloaded for the given date range.

        Metadata files (stations.parquet, parameters.parquet,
        installation.parquet) are always included. Observation files
        (data/observations/YYYY.parquet) are included only when the year
        overlaps the requested date range. All other files are skipped.
        """
        basename = fname.rsplit("/", 1)[-1] if "/" in fname else fname
        if basename in _METADATA_FILES:
            return True

        match = _OBS_YEAR_RE.search(fname)
        if match:
            year = int(match.group(1))
            return start_date.year <= year <= end_date.year

        return False


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

        Handles the real schema where station ID is the DataFrame index
        (nat_abbr), elevation is station_height, and 13 pre-computed
        terrain features are present as columns.

        Returns dict mapping station_id to StationMetadata.
        """
        # Handle index-as-ID: nat_abbr is the index, not a column
        if data.index.name == "nat_abbr":
            data = data.reset_index()

        # Normalize station ID column
        if "nat_abbr" in data.columns:
            data = data.rename(columns={"nat_abbr": "station_id"})
        elif "stn" in data.columns:
            data = data.rename(columns={"stn": "station_id"})
        elif "station_id" not in data.columns:
            raise ValueError(
                "No station_id, nat_abbr, or stn column found in data"
            )

        # Map alternate column names for coordinates and elevation
        col_renames = {
            "station_height": "elevation",
            "lat": "latitude",
            "lon": "longitude",
            "alt": "elevation",
            "elev": "elevation",
        }
        data = data.rename(
            columns={k: v for k, v in col_renames.items() if k in data.columns}
        )

        # Identify terrain feature columns present in data
        tf_cols = [c for c in TERRAIN_FEATURE_COLS if c in data.columns]

        stations: dict[str, StationMetadata] = {}
        for _, row in data.drop_duplicates("station_id").iterrows():
            sid = str(row["station_id"])
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            elev = float(row["elevation"])

            # Collect terrain features as immutable tuple of pairs
            terrain = tuple(
                (col, float(row[col]))
                for col in tf_cols
                if pd.notna(row[col])
            )

            # Preserve station name if available
            name = str(row["station_name"]) if "station_name" in row.index and pd.notna(row.get("station_name")) else None

            stations[sid] = StationMetadata(
                station_id=sid,
                network=Network.PEAKWEATHER,
                latitude=lat,
                longitude=lon,
                elevation=elev,
                terrain_ref_lat=lat,
                terrain_ref_lon=lon,
                station_name=name,
                terrain_features=terrain,
            )

        return stations

    def parse_observations(
        self, data: pd.DataFrame
    ) -> list[WeatherRecord]:
        """Parse PeakWeather observations into WeatherRecords.

        Handles the real schema with MultiIndex columns (nat_abbr, param_short)
        and datetime index. Uses vectorized operations for performance.

        Unit conversions applied:
        - Temperature: Celsius -> Kelvin (+ 273.15)
        - Humidity: percentage -> fraction (/ 100.0)
        """
        if isinstance(data.columns, pd.MultiIndex):
            return self._parse_multiindex_observations(data)
        return self._parse_flat_observations(data)

    def _parse_multiindex_observations(
        self, data: pd.DataFrame
    ) -> list[WeatherRecord]:
        """Parse observations with MultiIndex (nat_abbr, param_short) columns.

        Reshapes the wide MultiIndex DataFrame into flat records using
        vectorized pandas operations, then applies unit conversions.
        """
        # Stack the MultiIndex columns into a flat format:
        # datetime | nat_abbr | param_short | value
        flat = data.stack(level="nat_abbr", future_stack=True)
        # flat now has columns = param_short values, index = (datetime, nat_abbr)

        # Identify which param_short columns are in our var map
        mapped_params = {
            p: var for p, var in PEAKWEATHER_VAR_MAP.items()
            if p in flat.columns
        }

        # Rename param_short columns to our schema variable names
        flat = flat.rename(columns=mapped_params)

        # Apply unit conversions vectorized
        if "temperature" in flat.columns:
            flat["temperature"] = flat["temperature"] + 273.15
        if "humidity" in flat.columns:
            flat["humidity"] = flat["humidity"] / 100.0

        # Reset index to get datetime and nat_abbr as columns
        flat = flat.reset_index()

        # Rename index columns
        rename_map = {}
        if "datetime" in flat.columns:
            rename_map["datetime"] = "timestamp"
        if "nat_abbr" in flat.columns:
            rename_map["nat_abbr"] = "station_id"
        if rename_map:
            flat = flat.rename(columns=rename_map)

        # Ensure timestamps are proper datetime
        if "timestamp" in flat.columns:
            flat["timestamp"] = pd.to_datetime(flat["timestamp"], utc=True)

        # Build WeatherRecord objects from the flat DataFrame
        records: list[WeatherRecord] = []
        schema_vars = list(mapped_params.values())

        # Vectorized extraction: convert columns to numpy for speed
        timestamps = flat["timestamp"].values
        station_ids = flat["station_id"].values

        # Pre-extract variable arrays
        var_arrays: dict[str, pd.Series] = {}
        for var_name in schema_vars:
            if var_name in flat.columns:
                var_arrays[var_name] = flat[var_name]

        for i in range(len(flat)):
            ts = pd.Timestamp(timestamps[i]).to_pydatetime()
            record = WeatherRecord(
                timestamp=ts,
                station_id=str(station_ids[i]),
                network=Network.PEAKWEATHER,
            )

            for var_name, series in var_arrays.items():
                value = series.iloc[i]
                if pd.notna(value):
                    setattr(record, var_name, float(value))
                    setattr(record, f"{var_name}_qc", QualityFlag.GOOD)

            records.append(record)

        return records

    def _parse_flat_observations(
        self, data: pd.DataFrame
    ) -> list[WeatherRecord]:
        """Parse observations with flat column structure (legacy format).

        Kept for backward compatibility with pre-processed data.
        """
        if "stn" in data.columns:
            data = data.rename(columns={"stn": "station_id"})
        if "time" in data.columns:
            data = data.rename(columns={"time": "timestamp"})

        if not pd.api.types.is_datetime64_any_dtype(data["timestamp"]):
            data["timestamp"] = pd.to_datetime(data["timestamp"])

        # Apply unit conversions vectorized
        for raw_col, var_name in PEAKWEATHER_VAR_MAP.items():
            if raw_col in data.columns:
                data = data.rename(columns={raw_col: var_name})

        if "temperature" in data.columns:
            data["temperature"] = data["temperature"] + 273.15
        if "humidity" in data.columns:
            data["humidity"] = data["humidity"] / 100.0

        records: list[WeatherRecord] = []
        schema_vars = list(PEAKWEATHER_VAR_MAP.values())

        timestamps = data["timestamp"].values
        station_ids = data["station_id"].values

        var_arrays: dict[str, pd.Series] = {}
        for var_name in schema_vars:
            if var_name in data.columns:
                var_arrays[var_name] = data[var_name]

        for i in range(len(data)):
            ts = pd.Timestamp(timestamps[i]).to_pydatetime()
            record = WeatherRecord(
                timestamp=ts,
                station_id=str(station_ids[i]),
                network=Network.PEAKWEATHER,
            )

            for var_name, series in var_arrays.items():
                value = series.iloc[i]
                if pd.notna(value):
                    setattr(record, var_name, float(value))
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
