#!/usr/bin/env python3
"""Download Colorado mountain wind observations from IEM (Iowa Environmental Mesonet).

Queries the IEM ASOS archive for mountain stations (>2500m) in Colorado with
wind speed and wind direction data. Full historical access, free and unrestricted.

Usage:
    # WY2024 (default):
    python scripts/download_synoptic_wind.py

    # Custom date range:
    python scripts/download_synoptic_wind.py --start 2024-01-01 --end 2024-03-31

Data source: https://mesonet.agron.iastate.edu/request/download.phtml
"""

import argparse
import io
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "synoptic"

MIN_ELEVATION_M = 2500
IEM_BASE = "https://mesonet.agron.iastate.edu"
NETWORK = "CO_ASOS"


def get_mountain_stations() -> pl.DataFrame:
    """Fetch CO_ASOS station metadata from IEM GeoJSON, filter to mountain."""
    import json

    url = f"{IEM_BASE}/geojson/network/{NETWORK}.geojson"
    print(f"Querying station metadata from {NETWORK}...")
    with urlopen(url) as resp:
        data = json.loads(resp.read())

    rows = []
    for f in data["features"]:
        p = f["properties"]
        coords = f["geometry"]["coordinates"]
        elev = p.get("elevation") or 0
        rows.append({
            "station_id": p["sid"],
            "name": p["sname"],
            "latitude": coords[1],
            "longitude": coords[0],
            "elevation_m": float(elev),
            "network": NETWORK,
            "online": p.get("online", False),
            "archive_begin": p.get("archive_begin"),
            "archive_end": p.get("archive_end"),
        })

    df = pl.DataFrame(rows)
    mountain = df.filter(
        (pl.col("elevation_m") >= MIN_ELEVATION_M)
        & (pl.col("online") == True)
    )

    print(f"  Total {NETWORK} stations: {df.shape[0]}")
    print(f"  Active mountain stations (>={MIN_ELEVATION_M}m): {mountain.shape[0]}")
    return mountain


def save_station_metadata(stations: pl.DataFrame) -> Path:
    """Save station metadata CSV."""
    out = DATA_DIR / "stations.csv"
    stations.write_csv(out)
    try:
        display_path = out.relative_to(PROJECT_ROOT)
    except ValueError:
        display_path = out
    print(f"  Saved {stations.shape[0]} stations to {display_path}")
    return out


def download_station_obs(
    station_id: str,
    start: datetime,
    end: datetime,
) -> pl.DataFrame:
    """Download one station's observations from IEM ASOS archive."""
    url = (
        f"{IEM_BASE}/cgi-bin/request/asos.py?"
        f"station={station_id}"
        f"&data=tmpf&data=sknt&data=drct"
        f"&tz=Etc/UTC&format=onlycomma&latlon=yes&elev=yes"
        f"&year1={start.year}&month1={start.month}&day1={start.day}"
        f"&year2={end.year}&month2={end.month}&day2={end.day}"
    )
    with urlopen(url) as resp:
        csv_data = resp.read().decode("utf-8")

    if not csv_data.strip() or csv_data.startswith("Sorry"):
        return pl.DataFrame()

    df = pl.read_csv(
        io.StringIO(csv_data),
        schema_overrides={
            "tmpf": pl.Utf8,
            "sknt": pl.Utf8,
            "drct": pl.Utf8,
        },
    )
    return df


def download_all_observations(
    stations: pl.DataFrame,
    start: datetime,
    end: datetime,
) -> pl.DataFrame:
    """Download observations for all stations."""
    station_ids = stations["station_id"].to_list()
    all_frames = []

    print(f"\nDownloading observations for {len(station_ids)} stations "
          f"({start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')})...")

    for i, sid in enumerate(station_ids):
        print(f"  [{i + 1}/{len(station_ids)}] {sid}...", end="", flush=True)
        try:
            df = download_station_obs(sid, start, end)
            if df.is_empty():
                print(" no data")
            else:
                all_frames.append(df)
                print(f" {df.shape[0]} rows")
        except Exception as e:  # noqa: BLE001
            print(f" ERROR: {str(e)[:80]}")

    if not all_frames:
        return pl.DataFrame()
    return pl.concat(all_frames)


def convert_observations(obs: pl.DataFrame) -> pl.DataFrame:
    """Convert IEM ASOS format to standard units and column names.

    IEM returns: tmpf (Fahrenheit), sknt (knots), drct (degrees).
    We convert to: air_temp (Celsius), wind_speed (m/s), wind_direction (degrees).
    """
    if obs.is_empty():
        return obs

    M_FILTER = pl.lit("M")

    converted = obs.with_columns(
        pl.col("valid").str.to_datetime("%Y-%m-%d %H:%M", time_zone="UTC").alias("date_time"),
        pl.when(pl.col("sknt") != M_FILTER)
        .then(pl.col("sknt").cast(pl.Float64) * 0.514444)
        .otherwise(None)
        .alias("wind_speed"),
        pl.when(pl.col("drct") != M_FILTER)
        .then(pl.col("drct").cast(pl.Float64))
        .otherwise(None)
        .alias("wind_direction"),
        pl.when(pl.col("tmpf") != M_FILTER)
        .then((pl.col("tmpf").cast(pl.Float64) - 32.0) * 5.0 / 9.0)
        .otherwise(None)
        .alias("air_temp"),
    ).select([
        pl.col("station").alias("station_id"),
        "date_time",
        "lat", "lon",
        pl.col("elevation").cast(pl.Float64).alias("elevation_m"),
        "wind_speed",
        "wind_direction",
        "air_temp",
    ]).rename({"lat": "latitude", "lon": "longitude"})

    return converted


def save_observations(obs: pl.DataFrame, suffix: str = "") -> Path:
    """Save observations to parquet and CSV."""
    if obs.is_empty():
        print("  No observations to save.")
        return DATA_DIR

    fname = f"observations{suffix}"
    pq_out = DATA_DIR / f"{fname}.parquet"
    csv_out = DATA_DIR / f"{fname}.csv"

    obs.write_parquet(pq_out)
    obs.write_csv(csv_out)

    try:
        display_pq = pq_out.relative_to(PROJECT_ROOT)
        display_csv = csv_out.relative_to(PROJECT_ROOT)
    except ValueError:
        display_pq = pq_out
        display_csv = csv_out
    print(f"  Saved {obs.shape[0]} observations to {display_pq}")
    print(f"  Also saved CSV: {display_csv}")
    return pq_out


def print_report(stations: pl.DataFrame, obs: pl.DataFrame):
    """Print summary report."""
    print("\n" + "=" * 60)
    print("IEM WIND DATA DOWNLOAD REPORT")
    print("=" * 60)

    print(f"\nStations: {stations.shape[0]} active mountain stations")
    elev = stations["elevation_m"]
    print(f"Elevation range: {elev.min():.0f} - {elev.max():.0f} m "
          f"({elev.min() / 0.3048:.0f} - {elev.max() / 0.3048:.0f} ft)")

    print("\nStation list:")
    for row in stations.sort("elevation_m", descending=True).iter_rows(named=True):
        print(f"  {row['station_id']:8s} {row['name']:35s} {row['elevation_m']:7.0f}m")

    if obs.is_empty():
        print("\nNo observations downloaded.")
        return

    print(f"\nObservations: {obs.shape[0]:,} rows")

    if "date_time" in obs.columns:
        dt = obs["date_time"]
        print(f"Time range: {dt.min()} to {dt.max()}")

    unique_stations = obs["station_id"].n_unique()
    print(f"Stations with data: {unique_stations} / {stations.shape[0]}")

    for var_col in ["wind_speed", "wind_direction", "air_temp"]:
        if var_col in obs.columns:
            non_null = obs[var_col].drop_nulls().len()
            total = obs.shape[0]
            pct = non_null / total * 100 if total > 0 else 0
            print(f"  {var_col:20s} {non_null:>10,} / {total:,} ({pct:.1f}%)")

    per_station = obs.group_by("station_id").agg(
        pl.col("wind_speed").drop_nulls().len().alias("wind_obs"),
        pl.col("date_time").min().alias("first_obs"),
        pl.col("date_time").max().alias("last_obs"),
    ).sort("wind_obs", descending=True)

    print("\nPer-station wind obs:")
    print(f"  mean={per_station['wind_obs'].mean():.0f}, "
          f"median={per_station['wind_obs'].median():.0f}, "
          f"min={per_station['wind_obs'].min()}, max={per_station['wind_obs'].max()}")

    pq_files = list(DATA_DIR.glob("*.parquet"))
    csv_files = list(DATA_DIR.glob("*.csv"))
    total_mb = sum(f.stat().st_size for f in pq_files + csv_files) / 1e6
    print(f"\nData volume: {total_mb:.1f} MB total on disk")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Download Colorado mountain wind data from IEM ASOS archive"
    )
    parser.add_argument(
        "--start", type=str, default="2023-10-01",
        help="Start date YYYY-MM-DD (default: 2023-10-01 for WY2024)",
    )
    parser.add_argument(
        "--end", type=str, default="2024-09-30",
        help="End date YYYY-MM-DD (default: 2024-09-30 for WY2024)",
    )
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d")  # noqa: DTZ007
    end = datetime.strptime(args.end, "%Y-%m-%d")  # noqa: DTZ007
    suffix = f"_wy{end.year}" if args.start == "2023-10-01" else (
        f"_{args.start.replace('-', '')}_{args.end.replace('-', '')}"
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    stations = get_mountain_stations()
    save_station_metadata(stations)

    raw_obs = download_all_observations(stations, start, end)
    obs = convert_observations(raw_obs)
    save_observations(obs, suffix)

    print_report(stations, obs)


if __name__ == "__main__":
    main()
