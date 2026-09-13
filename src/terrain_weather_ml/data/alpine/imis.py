"""Swiss IMIS ingestion from SLF API and EnviDat.

Task 2.2: SLF open API client for 180 high-alpine stations at 30-min
resolution, with EnviDat bulk file fallback for historical data.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from terrain_weather_ml.data.alpine.schema import (
    Network,
    QualityFlag,
    StationMetadata,
    WeatherRecord,
)

logger = logging.getLogger(__name__)

EXPECTED_STATION_COUNT = 180

# IMIS variable mapping: API field names -> our schema
IMIS_VAR_MAP = {
    "wind_speed": "wind_speed",  # m/s
    "wind_direction": "wind_direction",  # degrees
    "air_temperature": "temperature",  # Celsius in raw
    "snow_surface_temperature": "snow_surface_temp",  # Celsius in raw
    "snow_depth": "snow_depth",  # cm in raw -> meters
    "relative_humidity": "humidity",  # % in raw
}


class HTTPClient(Protocol):
    """Protocol for HTTP operations."""

    def get(self, url: str, params: dict[str, Any] | None = None) -> Any:
        """Make GET request. Returns parsed JSON or raises."""
        ...


class DefaultHTTPClient:
    """Default HTTP client using httpx."""

    def get(self, url: str, params: dict[str, Any] | None = None) -> Any:
        import httpx

        resp = httpx.get(url, params=params, timeout=60.0)
        resp.raise_for_status()
        return resp.json()


class IMISClient:
    """Client for Swiss IMIS data from SLF API with EnviDat fallback.

    The SLF API serves recent data (typically last 2 years). For historical
    data outside this window, falls back to EnviDat bulk files.
    """

    SLF_API_BASE = "https://measurement-api.slf.ch/imis/smart"
    ENVIDAT_BASE = "https://www.envidat.ch/dataset/imis"

    def __init__(
        self,
        cache_dir: Path,
        http_client: HTTPClient | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.http_client = http_client or DefaultHTTPClient()

    def fetch_stations(self) -> dict[str, StationMetadata]:
        """Fetch station metadata from SLF API.

        Returns dict mapping station_id to StationMetadata.
        """
        try:
            data = self.http_client.get(
                f"{self.SLF_API_BASE}/stations",
            )
        except (OSError, ValueError, KeyError):
            logger.warning("SLF API station fetch failed, trying EnviDat")
            data = self._envidat_stations_fallback()

        stations: dict[str, StationMetadata] = {}
        for entry in data:
            sid = str(entry["code"])
            stations[sid] = StationMetadata(
                station_id=sid,
                network=Network.IMIS,
                latitude=float(entry["latitude"]),
                longitude=float(entry["longitude"]),
                elevation=float(entry["elevation"]),
                terrain_ref_lat=float(entry["latitude"]),
                terrain_ref_lon=float(entry["longitude"]),
            )
        return stations

    def fetch_observations(
        self,
        start_date: datetime,
        end_date: datetime,
        station_ids: list[str] | None = None,
    ) -> list[WeatherRecord]:
        """Fetch IMIS observations for the given date range.

        Falls back to EnviDat bulk files for historical data when the
        SLF API returns 404 (data outside its temporal window).
        """
        try:
            records = self._fetch_from_api(start_date, end_date, station_ids)
        except APINotFoundError:
            logger.info(
                "SLF API returned 404 for %s to %s, falling back to EnviDat",
                start_date,
                end_date,
            )
            records = self._fetch_from_envidat(start_date, end_date, station_ids)

        return records

    def _fetch_from_api(
        self,
        start_date: datetime,
        end_date: datetime,
        station_ids: list[str] | None = None,
    ) -> list[WeatherRecord]:
        """Fetch observations from SLF API."""
        params: dict[str, Any] = {
            "from": start_date.isoformat(),
            "to": end_date.isoformat(),
        }
        if station_ids:
            params["station"] = ",".join(station_ids)

        try:
            data = self.http_client.get(
                f"{self.SLF_API_BASE}/data",
                params=params,
            )
        except HTTPNotFoundError as e:
            raise APINotFoundError(str(e)) from e

        return self._parse_api_response(data)

    def _fetch_from_envidat(
        self,
        start_date: datetime,
        end_date: datetime,
        station_ids: list[str] | None = None,
    ) -> list[WeatherRecord]:
        """Fetch historical observations from EnviDat bulk files."""
        data = self.http_client.get(
            f"{self.ENVIDAT_BASE}/bulk",
            params={
                "from": start_date.isoformat(),
                "to": end_date.isoformat(),
            },
        )
        return self._parse_api_response(data)

    def _parse_api_response(
        self, data: list[dict[str, Any]]
    ) -> list[WeatherRecord]:
        """Parse API/EnviDat response into WeatherRecords."""
        records: list[WeatherRecord] = []

        for entry in data:
            record = WeatherRecord(
                timestamp=datetime.fromisoformat(str(entry["timestamp"])),
                station_id=str(entry["station_code"]),
                network=Network.IMIS,
            )

            for api_field, var_name in IMIS_VAR_MAP.items():
                if api_field in entry and entry[api_field] is not None:
                    value = float(entry[api_field])
                    setattr(record, var_name, value)
                    setattr(record, f"{var_name}_qc", QualityFlag.GOOD)

            records.append(record)

        return records

    def _envidat_stations_fallback(self) -> list[dict[str, Any]]:
        """Fetch station metadata from EnviDat as fallback."""
        return self.http_client.get(f"{self.ENVIDAT_BASE}/stations")


class APINotFoundError(Exception):
    """Raised when SLF API returns 404 for a date range."""


class HTTPNotFoundError(Exception):
    """Raised for HTTP 404 responses."""
