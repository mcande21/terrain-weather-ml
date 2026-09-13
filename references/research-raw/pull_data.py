"""Pull real SNOTEL + CAIC data into DuckDB."""
import asyncio
import logging
import sys
import time
from datetime import date

import duckdb

sys.path.insert(0, "/Users/cooperanderson/work/personal/code/colorado-avalanche-ml/src")

from avalanche_ml.db.schema import create_tables
from avalanche_ml.ingestion.snotel import SnotelClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = "/Users/cooperanderson/work/personal/code/colorado-avalanche-ml/data/avalanche.duckdb"
START_DATE = date(2014, 10, 1)
END_DATE = date(2025, 9, 1)
BATCH_SIZE = 10
BATCH_DELAY = 2.0


async def main():
    db = duckdb.connect(DB_PATH)
    create_tables(db)

    client = SnotelClient(db)

    # Step 1: Discover stations
    logger.info("Discovering SNOTEL stations...")
    station_count = await client.discover()
    logger.info("Discovered %d CO SNTL stations", station_count)

    sample = db.execute(
        "SELECT station_id, name, elevation, latitude, longitude FROM stations LIMIT 5"
    ).fetchdf()
    print(sample.to_string())

    # Step 2: Pull daily data for all stations
    stations = db.execute("SELECT station_id FROM stations ORDER BY station_id").fetchall()
    station_ids = [row[0] for row in stations]
    logger.info("Pulling daily data for %d stations from %s to %s", len(station_ids), START_DATE, END_DATE)

    total_readings = 0
    failed_stations = []
    t0 = time.time()

    for i in range(0, len(station_ids), BATCH_SIZE):
        batch = station_ids[i:i + BATCH_SIZE]
        batch_start = time.time()

        for sid in batch:
            try:
                count = await client.ingest(
                    station_id=sid,
                    start_date=START_DATE,
                    end_date=END_DATE,
                    resolution="daily",
                )
                total_readings += count
                if count > 0:
                    logger.info("  %s: %d readings", sid, count)
                else:
                    logger.warning("  %s: no data returned", sid)
            except Exception as e:
                logger.error("  %s: FAILED - %s", sid, e)
                failed_stations.append((sid, str(e)))

        batch_elapsed = time.time() - batch_start
        logger.info(
            "Batch %d-%d done in %.1fs, total readings so far: %d",
            i, i + len(batch) - 1, batch_elapsed, total_readings,
        )

        if i + BATCH_SIZE < len(station_ids):
            await asyncio.sleep(BATCH_DELAY)

    elapsed = time.time() - t0
    logger.info("SNOTEL pull complete: %d readings in %.1fs", total_readings, elapsed)

    if failed_stations:
        logger.warning("Failed stations (%d):", len(failed_stations))
        for sid, err in failed_stations:
            logger.warning("  %s: %s", sid, err)

    # Stats
    print("\n=== SNOTEL DAILY STATS ===")
    stats = db.execute("""
        SELECT
            COUNT(*) as total,
            COUNT(DISTINCT station_id) as stations,
            MIN(date) as earliest,
            MAX(date) as latest,
            COUNT(CASE WHEN swe_inches IS NULL THEN 1 END) as null_swe,
            COUNT(CASE WHEN air_temp_min_f IS NULL THEN 1 END) as null_temp_min,
            COUNT(CASE WHEN air_temp_max_f IS NULL THEN 1 END) as null_temp_max,
            COUNT(CASE WHEN precip_increment_inches IS NULL THEN 1 END) as null_precip
        FROM snotel_daily
    """).fetchdf()
    print(stats.to_string())

    db.close()
    print(f"\nTotal readings: {total_readings}")
    print(f"Failed stations: {len(failed_stations)}")


if __name__ == "__main__":
    asyncio.run(main())
