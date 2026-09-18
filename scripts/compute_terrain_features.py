#!/usr/bin/env python3
"""Compute terrain features for station locations from a DEM.

Usage:
    python scripts/compute_terrain_features.py \\
        --stations data/snotel/stations.csv \\
        --dem data/dem/colorado_snotel_utm.tif \\
        --output data/integration/terrain_features_snotel.parquet

Input CSV must have columns: name, latitude, longitude, and optionally station_id.
Output is a Parquet file with 68 terrain features per station (17 point + 51 area).
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from terrain_weather_ml.terrain.feature_extraction import TerrainFeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stations", required=True, help="CSV with name, latitude, longitude columns"
    )
    parser.add_argument(
        "--dem", required=True, help="GeoTIFF DEM file (projected CRS, e.g. UTM)"
    )
    parser.add_argument(
        "--output", required=True, help="Output Parquet file path"
    )
    args = parser.parse_args()

    stations = pd.read_csv(args.stations)
    required = {"name", "latitude", "longitude"}
    missing = required - set(stations.columns)
    if missing:
        logger.error("Missing columns in stations CSV: %s", missing)
        sys.exit(1)

    if "station_id" not in stations.columns:
        stations["station_id"] = range(len(stations))

    logger.info("Loaded %d stations from %s", len(stations), args.stations)

    extractor = TerrainFeatureExtractor(args.dem)
    t0 = time.time()
    result = extractor.extract_features(stations)
    elapsed = time.time() - t0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)

    logger.info(
        "Saved %d stations x %d features to %s (%.1fs)",
        len(result), len(result.columns), output_path, elapsed,
    )

    print(f"\n{'='*80}")
    print(f"Terrain Features: {len(result)} stations, {len(result.columns)} columns")
    print(f"{'='*80}")
    print(result.to_string(index=False, max_colwidth=20))


if __name__ == "__main__":
    main()
