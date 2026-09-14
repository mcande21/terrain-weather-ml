"""Download ERA5 reanalysis data for the Alpine domain via ARCO-ERA5.

Uses the free Google Cloud ARCO-ERA5 public mirror (no credentials needed).
Downloads surface variables for the Alpine domain (lat 45-48, lon 5-11),
saves as NetCDF files compatible with ERA5Conditioner.

Usage:
    python scripts/download_era5_alpine.py [--year 2024] [--month 1] [--hours-step 6]
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import gcsfs
import numpy as np
import xarray as xr
import zarr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ARCO_ERA5_STORE = "gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

VARIABLES = {
    "2m_temperature": "t2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "total_precipitation": "tp",
    "surface_pressure": "sp",
    "boundary_layer_height": "blh",
}

ALPINE_DOMAIN = {"lat_min": 45.0, "lat_max": 48.0, "lon_min": 5.0, "lon_max": 11.0}


def download_era5_alpine(
    output_dir: Path,
    year: int = 2024,
    month: int = 1,
    hours_step: int = 6,
) -> Path:
    """Download ERA5 Alpine data from ARCO-ERA5 public mirror.

    Args:
        output_dir: Directory to save NetCDF files.
        year: Year to download.
        month: Month to download.
        hours_step: Hours between timesteps (1=hourly, 6=6-hourly).

    Returns:
        Path to the saved NetCDF file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Connecting to ARCO-ERA5 public mirror (anonymous access)...")
    fs = gcsfs.GCSFileSystem(token="anon")
    store = gcsfs.GCSMap(ARCO_ERA5_STORE, gcs=fs)
    root = zarr.open(store, mode="r")

    # Time is stored as int64 "hours since 1900-01-01 00:00:00"
    time_arr = root["time"]
    epoch = np.datetime64("1900-01-01T00:00:00")

    start = np.datetime64(f"{year}-{month:02d}-01T00:00:00")
    if month == 12:
        end = np.datetime64(f"{year + 1}-01-01T00:00:00")
    else:
        end = np.datetime64(f"{year}-{month + 1:02d}-01T00:00:00")

    start_hour = int((start - epoch) / np.timedelta64(1, "h"))
    end_hour = int((end - epoch) / np.timedelta64(1, "h"))
    total_hours = time_arr.shape[0]

    if start_hour >= total_hours:
        raise ValueError(
            f"Requested period starts at hour {start_hour} but dataset has {total_hours}"
        )
    end_hour = min(end_hour, total_hours)

    time_indices = np.arange(start_hour, end_hour, hours_step)

    if len(time_indices) == 0:
        raise ValueError(f"No timesteps found for {year}-{month:02d}")

    logger.info(
        "Selected %d timesteps for %d-%02d (every %dh)",
        len(time_indices), year, month, hours_step,
    )

    lat_arr = np.array(root["latitude"][:])
    lon_arr = np.array(root["longitude"][:])

    lat_mask = (lat_arr >= ALPINE_DOMAIN["lat_min"]) & (
        lat_arr <= ALPINE_DOMAIN["lat_max"]
    )
    lon_mask = (lon_arr >= ALPINE_DOMAIN["lon_min"]) & (
        lon_arr <= ALPINE_DOMAIN["lon_max"]
    )
    lat_indices = np.where(lat_mask)[0]
    lon_indices = np.where(lon_mask)[0]

    lat_slice = slice(lat_indices[0], lat_indices[-1] + 1)
    lon_slice = slice(lon_indices[0], lon_indices[-1] + 1)
    selected_lats = lat_arr[lat_slice]
    selected_lons = lon_arr[lon_slice]

    logger.info(
        "Alpine domain: lat [%.2f, %.2f] (%d pts), lon [%.2f, %.2f] (%d pts)",
        selected_lats[0], selected_lats[-1], len(selected_lats),
        selected_lons[0], selected_lons[-1], len(selected_lons),
    )

    data_vars = {}
    for arco_name, nc_name in VARIABLES.items():
        logger.info("Reading %s (%s)...", arco_name, nc_name)
        var_arr = root[arco_name]

        chunks = []
        for idx in time_indices:
            chunk = var_arr[int(idx), lat_slice, lon_slice]
            chunks.append(chunk)

        stacked = np.stack(chunks, axis=0).astype(np.float32)
        data_vars[nc_name] = (["time", "latitude", "longitude"], stacked)
        logger.info("  %s: shape %s, range [%.2f, %.2f]",
                     nc_name, stacked.shape, float(np.nanmin(stacked)),
                     float(np.nanmax(stacked)))

    selected_times = epoch + time_indices.astype("timedelta64[h]")

    ds = xr.Dataset(
        data_vars,
        coords={
            "time": selected_times,
            "latitude": selected_lats,
            "longitude": selected_lons,
        },
        attrs={
            "source": "ARCO-ERA5 (Google Cloud public mirror)",
            "domain": "alpine",
            "lat_range": f"{ALPINE_DOMAIN['lat_min']}-{ALPINE_DOMAIN['lat_max']}",
            "lon_range": f"{ALPINE_DOMAIN['lon_min']}-{ALPINE_DOMAIN['lon_max']}",
        },
    )

    out_file = output_dir / f"era5_alpine_{year}{month:02d}.nc"
    logger.info("Saving to %s ...", out_file)
    ds.to_netcdf(out_file)

    file_size_mb = out_file.stat().st_size / (1024 * 1024)
    logger.info(
        "Done. File: %s (%.1f MB), %d timesteps, %d variables, grid %dx%d",
        out_file, file_size_mb, len(selected_times), len(VARIABLES),
        len(selected_lats), len(selected_lons),
    )

    return out_file


def validate_with_conditioner(nc_path: Path) -> None:
    """Validate downloaded data through ERA5Conditioner."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from terrain_weather_ml.backbone.era5_conditioner import ERA5Conditioner, ERA5Config

    logger.info("Validating through ERA5Conditioner...")
    config = ERA5Config(domain="alpine")
    conditioner = ERA5Conditioner(config)

    ds = xr.open_dataset(nc_path)
    result = conditioner.extract_variables(ds, time_idx=0)

    logger.info("Conditioner output:")
    logger.info("  Variables: %s", result.variables)
    logger.info("  Grid shape: %s", result.grid_shape)
    logger.info("  Valid time: %s", result.valid_time)
    for var_name, arr in result.fields.items():
        logger.info("  %s: shape=%s, min=%.2f, max=%.2f",
                     var_name, arr.shape, float(np.nanmin(arr)),
                     float(np.nanmax(arr)))

    expected_vars = {"T2M", "U10", "V10", "PRATE", "SP", "BLH"}
    found_vars = set(result.variables)
    if found_vars == expected_vars:
        logger.info("All 6 expected variables present — matches NWP passthrough format")
    else:
        missing = expected_vars - found_vars
        logger.warning("Missing variables: %s", missing)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ERA5 Alpine data")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--month", type=int, default=1)
    parser.add_argument("--hours-step", type=int, default=6,
                        help="Hours between timesteps (1=hourly, 6=6-hourly)")
    parser.add_argument("--output-dir", type=str, default="data/era5/alpine")
    parser.add_argument("--skip-validate", action="store_true")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=RuntimeWarning, module="gcsfs")

    out = download_era5_alpine(
        output_dir=Path(args.output_dir),
        year=args.year,
        month=args.month,
        hours_step=args.hours_step,
    )

    if not args.skip_validate:
        validate_with_conditioner(out)
