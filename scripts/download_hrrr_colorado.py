"""Download HRRR analysis data for the Colorado subdomain.

Downloads 6-hourly HRRR surface analyses for Water Year 2024
(October 2023 through September 2024), crops to Colorado bounds,
saves as daily NetCDF files. Skips days that already exist.

Usage:
    .venv/bin/python scripts/download_hrrr_colorado.py
"""

from __future__ import annotations

import calendar
import logging
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import xarray as xr
from herbie import Herbie

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

COLORADO_BOUNDS = {
    "lat_min": 36.5, "lat_max": 41.5,
    "lon_min": 250.5, "lon_max": 255.5,
}

SEARCH = (
    ":TMP:2 m above ground:|:UGRD:10 m above ground:|:VGRD:10 m above ground:"
    "|:PRATE:surface:|:PRES:surface:|:HPBL:surface:"
)

EXPECTED_VARS = {"u10", "v10", "t2m", "sp", "blh", "prate"}


def crop_to_colorado(ds: xr.Dataset) -> xr.Dataset:
    lat = ds.latitude.values
    lon = ds.longitude.values

    mask = (
        (lat >= COLORADO_BOUNDS["lat_min"]) & (lat <= COLORADO_BOUNDS["lat_max"]) &
        (lon >= COLORADO_BOUNDS["lon_min"]) & (lon <= COLORADO_BOUNDS["lon_max"])
    )

    y_any = mask.any(axis=1)
    x_any = mask.any(axis=0)
    y_idx = np.where(y_any)[0]
    x_idx = np.where(x_any)[0]

    if len(y_idx) == 0 or len(x_idx) == 0:
        raise ValueError("No grid points found in Colorado bounds")

    return ds.isel(y=slice(y_idx[0], y_idx[-1] + 1), x=slice(x_idx[0], x_idx[-1] + 1))


def fetch_timestep(date_str: str, hour: int) -> xr.Dataset | None:
    try:
        H = Herbie(f"{date_str} {hour:02d}:00", model="hrrr", product="sfc", fxx=0, verbose=False)
        result = H.xarray(SEARCH, remove_grib=True)
        if isinstance(result, list):
            ds = xr.merge(result, compat="override")
        else:
            ds = result

        found = set(ds.data_vars) & EXPECTED_VARS
        if found != EXPECTED_VARS:
            logger.warning("Missing vars at %s %02dZ: %s", date_str, hour, EXPECTED_VARS - found)
            return None

        ds_co = crop_to_colorado(ds)

        keep_vars = list(EXPECTED_VARS & set(ds_co.data_vars))
        keep_coords = ["latitude", "longitude"]
        ds_out = ds_co[keep_vars]
        for c in keep_coords:
            if c in ds_co.coords:
                ds_out = ds_out.assign_coords({c: ds_co[c]})

        ds_out = ds_out.expand_dims("time")
        ds_out["time"] = [np.datetime64(f"{date_str}T{hour:02d}:00:00")]
        return ds_out
    except Exception as e:
        logger.warning("Failed %s %02dZ: %s", date_str, hour, e)
        return None


def download_day(date_str: str, output_dir: Path) -> Path | None:
    out_file = output_dir / f"hrrr_colorado_{date_str.replace('-', '')}.nc"
    if out_file.exists():
        logger.info("  Already exists: %s, skipping", out_file.name)
        return out_file

    datasets = []
    for hour in [0, 6, 12, 18]:
        logger.info("  %s %02dZ...", date_str, hour)
        ds = fetch_timestep(date_str, hour)
        if ds is not None:
            datasets.append(ds)

    if not datasets:
        logger.warning("  No data for %s", date_str)
        return None

    combined = xr.concat(datasets, dim="time")
    combined.to_netcdf(out_file)
    size_mb = out_file.stat().st_size / (1024 * 1024)
    logger.info("  Saved %s (%.1f MB, %d timesteps)", out_file.name, size_mb, len(datasets))
    return out_file


def wy2024_dates() -> list[date]:
    start = date(2023, 10, 1)
    end = date(2024, 9, 30)
    current = start
    dates = []
    while current <= end:
        dates.append(current)
        current += timedelta(days=1)
    return dates


def main():
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "data" / "hrrr" / "colorado"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_dates = wy2024_dates()
    total_days = len(all_dates)

    logger.info("Downloading HRRR Colorado subdomain: WY2024 (Oct 2023 - Sep 2024)")
    logger.info("Total days: %d", total_days)
    logger.info("Output: %s", output_dir)

    saved = []
    failed_dates = []
    skipped = 0

    for i, d in enumerate(all_dates, 1):
        date_str = d.isoformat()
        month_label = d.strftime("%b %Y")
        logger.info("Day %d/%d [%s]: %s", i, total_days, month_label, date_str)

        out_file = output_dir / f"hrrr_colorado_{date_str.replace('-', '')}.nc"
        if out_file.exists():
            logger.info("  Already exists: %s, skipping", out_file.name)
            saved.append(out_file)
            skipped += 1
            continue

        result = download_day(date_str, output_dir)
        if result:
            saved.append(result)
        else:
            failed_dates.append(date_str)

    if not saved:
        logger.error("No files downloaded!")
        sys.exit(1)

    total_size = sum(f.stat().st_size for f in saved) / (1024 * 1024)
    sample = xr.open_dataset(saved[0])
    logger.info("=" * 60)
    logger.info("Download complete — WY2024")
    logger.info("  Files: %d / %d days", len(saved), total_days)
    logger.info("  Skipped (pre-existing): %d", skipped)
    logger.info("  New downloads: %d", len(saved) - skipped)
    logger.info("  Failed/missing: %d", len(failed_dates))
    logger.info("  Total size: %.1f MB", total_size)
    logger.info("  Grid (Colorado): y=%d, x=%d",
                sample.sizes.get("y", 0), sample.sizes.get("x", 0))
    logger.info("  Variables: %s", sorted(sample.data_vars))
    logger.info("  Timesteps per file: 4 (00Z, 06Z, 12Z, 18Z)")
    logger.info("  Total timesteps: ~%d", len(saved) * 4)
    logger.info("  Lat range: %.2f - %.2f",
                float(sample.latitude.min()), float(sample.latitude.max()))
    logger.info("  Lon range: %.2f - %.2f",
                float(sample.longitude.min()), float(sample.longitude.max()))
    sample.close()

    if failed_dates:
        logger.warning("=" * 60)
        logger.warning("GAPS — %d dates with no data:", len(failed_dates))
        for fd in failed_dates:
            logger.warning("  %s", fd)


if __name__ == "__main__":
    main()
