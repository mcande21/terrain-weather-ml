#!/usr/bin/env python3
"""Download Colorado mountain wind observations from Synoptic Data (MesoWest).

Queries the Synoptic API for mountain stations (>2500m) in Colorado with
wind speed and wind direction data. Downloads observations and saves station
metadata and time series to data/synoptic/.

Usage:
    # Recent data (free tier, last 5 days):
    python scripts/download_synoptic_wind.py

    # Historical WY2024 (requires paid Synoptic account):
    python scripts/download_synoptic_wind.py --start 2023-10-01 --end 2024-09-30

    # Custom date range:
    python scripts/download_synoptic_wind.py --start 2024-01-01 --end 2024-03-31

Environment:
    SYNOPTIC_TOKEN: API token (required)
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

try:
    import synoptic
except ImportError:
    print("synopticpy not installed. Run: pip install synopticpy")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "synoptic"

MIN_ELEVATION_FT = 8200  # ~2500m
MAX_ELEVATION_FT = 14500  # Colorado's highest peak is 14,440ft (Mt. Elbert)
VARIABLES = "wind_speed,wind_direction,air_temp"
# Max days per API request to stay within free-tier limits and avoid timeouts
CHUNK_DAYS = 5
# Networks with known bad elevation metadata
EXCLUDED_NETWORKS = {"APRSWXNET/CWOP"}


def get_mountain_stations(token: str) -> pl.DataFrame:
    """Query Synoptic metadata for CO mountain wind stations."""
    print("Querying station metadata...")
    meta = synoptic.Metadata(
        state="CO",
        vars="wind_speed,wind_direction",
        varsoperator="and",
        sensorvars=True,
        complete=True,
        token=token,
    )
    df = meta.df()

    mountain = df.filter(
        (pl.col("elevation") > MIN_ELEVATION_FT)
        & (pl.col("elevation") < MAX_ELEVATION_FT)
        & (pl.col("is_active") == True)
        & (~pl.col("shortname").is_in(EXCLUDED_NETWORKS))
    )

    print(f"  Total CO wind stations: {df.shape[0]}")
    print(f"  Active mountain stations (>{MIN_ELEVATION_FT}ft, <{MAX_ELEVATION_FT}ft): "
          f"{mountain.shape[0]}")
    return mountain


def save_station_metadata(stations: pl.DataFrame) -> Path:
    """Save station metadata CSV."""
    out = DATA_DIR / "stations.csv"
    cols = [
        "stid", "name", "latitude", "longitude", "elevation",
        "shortname", "mnet_id", "state", "elev_dem",
        "period_of_record_start", "period_of_record_end",
    ]
    available = [c for c in cols if c in stations.columns]
    export = stations.select(available).rename({"shortname": "network"})
    export = export.with_columns(
        (pl.col("elevation") * 0.3048).round(1).alias("elevation_m"),
    )
    export.write_csv(out)
    try:
        display_path = out.relative_to(PROJECT_ROOT)
    except ValueError:
        display_path = out
    print(f"  Saved {export.shape[0]} stations to {display_path}")
    return out


def download_observations(
    station_ids: list[str],
    start: datetime,
    end: datetime,
    token: str,
) -> pl.DataFrame:
    """Download wind/temp observations in chunks to respect API limits."""
    all_frames = []
    current = start
    chunk_num = 0

    # Batch stations in groups of 50 to avoid URL length limits
    batch_size = 50
    station_batches = [
        station_ids[i:i + batch_size]
        for i in range(0, len(station_ids), batch_size)
    ]

    total_chunks = 0
    delta = end - start
    chunks_per_batch = max(1, int(delta.total_seconds() / (CHUNK_DAYS * 86400)) + 1)
    total_chunks = chunks_per_batch * len(station_batches)

    print(f"  Downloading {len(station_ids)} stations in {len(station_batches)} batches "
          f"x {chunks_per_batch} time chunks = {total_chunks} API calls")

    for batch_idx, batch in enumerate(station_batches):
        stid_str = ",".join(batch)
        current = start

        while current < end:
            chunk_end = min(current + timedelta(days=CHUNK_DAYS), end)
            chunk_num += 1
            print(f"  [{chunk_num}/{total_chunks}] Batch {batch_idx + 1}, "
                  f"{current.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')} "
                  f"({len(batch)} stations)...", end="", flush=True)

            try:
                ts = synoptic.TimeSeries(
                    stid=stid_str,
                    start=current.strftime("%Y%m%d%H%M"),
                    end=chunk_end.strftime("%Y%m%d%H%M"),
                    vars=VARIABLES,
                    units="metric",
                    token=token,
                )
                df = ts.df()
                all_frames.append(df)
                print(f" {df.shape[0]} rows")
            except synoptic.SynopticAPIError as e:
                msg = str(e)
                if "does not have access to the requested history" in msg:
                    print(" SKIPPED (free tier: no historical access)")
                    if chunk_num == 1:
                        print("\n  ** Free tier detected. Falling back to recent data. **")
                        return download_recent(station_ids, token)
                else:
                    print(f" ERROR: {msg[:100]}")

            current = chunk_end

    if not all_frames:
        return pl.DataFrame()
    return pl.concat(all_frames)


def download_recent(station_ids: list[str], token: str) -> pl.DataFrame:
    """Download recent observations (free tier fallback, ~5 days)."""
    all_frames = []
    batch_size = 50
    batches = [
        station_ids[i:i + batch_size]
        for i in range(0, len(station_ids), batch_size)
    ]

    print(f"  Downloading recent data for {len(station_ids)} stations "
          f"in {len(batches)} batches...")

    for i, batch in enumerate(batches):
        stid_str = ",".join(batch)
        print(f"  [{i + 1}/{len(batches)}] {len(batch)} stations...", end="", flush=True)

        try:
            ts = synoptic.TimeSeries(
                stid=stid_str,
                recent="7200",  # 5 days = 7200 minutes
                vars=VARIABLES,
                units="metric",
                token=token,
            )
            df = ts.df()
            all_frames.append(df)
            print(f" {df.shape[0]} rows")
        except synoptic.SynopticAPIError as e:
            print(f" ERROR: {str(e)[:100]}")

    if not all_frames:
        return pl.DataFrame()
    return pl.concat(all_frames)


def save_observations(obs: pl.DataFrame, suffix: str = "") -> Path:
    """Save observations to parquet with a tidy schema."""
    if obs.is_empty():
        print("  No observations to save.")
        return DATA_DIR

    keep_cols = ["stid", "date_time", "variable", "value", "units",
                 "latitude", "longitude", "elevation", "qc_flagged"]
    available = [c for c in keep_cols if c in obs.columns]
    tidy = obs.select(available)

    # Deduplicate: some stations have multiple sensors per variable
    tidy = tidy.group_by(["stid", "date_time", "variable", "latitude",
                          "longitude", "elevation"]).agg(
        pl.col("value").first(),
    )

    # Pivot from long to wide format
    pivot = tidy.pivot(
        on="variable",
        index=["stid", "date_time", "latitude", "longitude", "elevation"],
        values="value",
    )

    fname = f"observations{suffix}.parquet"
    out = DATA_DIR / fname
    pivot.write_parquet(out)

    csv_out = DATA_DIR / fname.replace(".parquet", ".csv")
    pivot.write_csv(csv_out)

    try:
        display_pq = out.relative_to(PROJECT_ROOT)
        display_csv = csv_out.relative_to(PROJECT_ROOT)
    except ValueError:
        display_pq = out
        display_csv = csv_out
    print(f"  Saved {pivot.shape[0]} observations to {display_pq}")
    print(f"  Also saved CSV: {display_csv}")
    return out


def print_report(stations: pl.DataFrame, obs: pl.DataFrame):
    """Print summary report."""
    print("\n" + "=" * 60)
    print("SYNOPTIC WIND DATA DOWNLOAD REPORT")
    print("=" * 60)

    # Station summary
    print(f"\nStations: {stations.shape[0]} active mountain stations")
    elev = stations["elevation"]
    elev_m = elev * 0.3048
    print(f"Elevation range: {elev.min():.0f} - {elev.max():.0f} ft "
          f"({elev_m.min():.0f} - {elev_m.max():.0f} m)")

    nets = stations.group_by("shortname").len().sort("len", descending=True)
    print(f"\nNetworks ({nets.shape[0]} total):")
    for row in nets.iter_rows():
        print(f"  {row[0]:20s} {row[1]:4d} stations")

    if obs.is_empty():
        print("\nNo observations downloaded.")
        return

    # Observation summary
    print(f"\nObservations: {obs.shape[0]} rows")

    if "date_time" in obs.columns:
        dt = obs["date_time"]
        print(f"Time range: {dt.min()} to {dt.max()}")

    unique_stids = obs["stid"].n_unique()
    print(f"Stations with data: {unique_stids} / {stations.shape[0]}")

    # Variable coverage
    if "variable" in obs.columns:
        var_counts = obs.group_by("variable").len().sort("len", descending=True)
        print("\nVariable coverage:")
        for row in var_counts.iter_rows():
            print(f"  {row[0]:20s} {row[1]:>8,d} observations")

    # Per-station completeness
    if "date_time" in obs.columns and "stid" in obs.columns:
        wind_obs = obs.filter(pl.col("variable") == "wind_speed")
        if not wind_obs.is_empty():
            per_station = wind_obs.group_by("stid").agg(
                pl.col("date_time").count().alias("n_obs"),
            )
            mean_obs = per_station["n_obs"].mean()
            median_obs = per_station["n_obs"].median()
            print(f"\nWind speed obs per station: "
                  f"mean={mean_obs:.0f}, median={median_obs:.0f}")

    # Data volume
    parquet_files = list(DATA_DIR.glob("*.parquet"))
    csv_files = list(DATA_DIR.glob("*.csv"))
    total_mb = sum(f.stat().st_size for f in parquet_files + csv_files) / 1e6
    print(f"\nData volume: {total_mb:.1f} MB total on disk")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Download Synoptic wind data for Colorado")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD), default: recent 5 days")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--token", type=str, default=os.environ.get("SYNOPTIC_TOKEN"),
                        help="Synoptic API token (or set SYNOPTIC_TOKEN env var)")
    args = parser.parse_args()

    if not args.token:
        print("Error: SYNOPTIC_TOKEN environment variable or --token required")
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Get station metadata
    stations = get_mountain_stations(args.token)
    save_station_metadata(stations)
    station_ids = stations["stid"].to_list()

    # Step 2: Download observations
    if args.start and args.end:
        start = datetime.strptime(args.start, "%Y-%m-%d")  # noqa: DTZ007
        end = datetime.strptime(args.end, "%Y-%m-%d")  # noqa: DTZ007
        suffix = f"_{args.start.replace('-', '')}_{args.end.replace('-', '')}"
        print(f"\nDownloading observations: {args.start} to {args.end}")
        obs = download_observations(station_ids, start, end, args.token)
    else:
        suffix = "_recent"
        print("\nDownloading recent observations (last 5 days)...")
        obs = download_recent(station_ids, args.token)

    save_observations(obs, suffix)

    # Step 3: Report
    print_report(stations, obs)


if __name__ == "__main__":
    main()
