"""Temporal utilities for SNOTEL-HRRR date matching and WY splits."""

from __future__ import annotations

import pandas as pd


def snotel_date_to_hrrr_key(date_str: str) -> str:
    """Map a SNOTEL date to the HRRR WY2024 date key by month-day.

    HRRR data covers WY2024 (Oct 2023 - Sep 2024). This maps any SNOTEL
    date to the corresponding month-day in WY2024:
      Oct-Dec -> 2023-MM-DD
      Jan-Sep -> 2024-MM-DD

    Args:
        date_str: Date string in YYYY-MM-DD format.

    Returns:
        HRRR date key in YYYYMMDD format (matching HRRR file naming).
    """
    dt = pd.Timestamp(date_str)
    if dt.month >= 10:
        hrrr_year = 2023
    else:
        hrrr_year = 2024
    return f"{hrrr_year}{dt.month:02d}{dt.day:02d}"


def is_wy2024_date(ts: pd.Timestamp) -> bool:
    """Check if a timestamp falls within WY2024 (Oct 1 2023 - Sep 30 2024).

    Args:
        ts: Pandas Timestamp.

    Returns:
        True if the date is in WY2024.
    """
    wy_start = pd.Timestamp("2023-10-01")
    wy_end = pd.Timestamp("2024-09-30")
    return wy_start <= ts <= wy_end
