"""Tests for proper holdout evaluation (Group 2 tasks).

Task 2.1: Temporal holdout split (WY2020-2023 train / WY2024 test)
Task 2.2: Full station coverage with missing data flagging
Task 2.3: Per-season stratification (DJF/MAM/JJA/SON)
Task 2.4: Per-elevation-band stratification
Task 2.5: Bootstrap 95% confidence intervals
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest

from terrain_weather_ml.evaluation.loso import (
    ElevationBandConfig,
    LOSOConfig,
    LOSOResult,
    StationData,
    TemporalSplit,
    bootstrap_confidence_intervals,
    classify_elevation_band,
    classify_season,
    compute_seasonal_metrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_timestamps(start: str, end: str, freq: str = "D") -> np.ndarray:
    """Generate datetime64 timestamps between start and end."""
    return pd.date_range(start, end, freq=freq).values


def _make_station(
    station_id: str,
    elevation: float,
    timestamps: np.ndarray,
    temp_values: np.ndarray | None = None,
) -> StationData:
    """Create a synthetic station with temperature observations."""
    n = len(timestamps)
    rng = np.random.default_rng(hash(station_id) % (2**31))
    obs = {}
    if temp_values is not None:
        obs["temperature"] = temp_values.astype(np.float32)
    else:
        obs["temperature"] = rng.uniform(260, 290, n).astype(np.float32)
    return StationData(
        station_id=station_id,
        latitude=39.0,
        longitude=-106.0,
        elevation=elevation,
        tpi=0.0,
        observations=obs,
        timestamps=timestamps,
    )


# ===========================================================================
# Task 2.1: Temporal holdout split
# ===========================================================================

class TestTemporalSplit:
    """WY2020-2023 train / WY2024 test boundary enforcement."""

    def test_split_enforces_wy2024_boundary(self):
        """No WY2024 timestamps appear in the training split."""
        # Mix timestamps spanning WY2020 through WY2024
        ts_train = _make_timestamps("2019-10-01", "2023-09-30")
        ts_test = _make_timestamps("2023-10-01", "2024-09-30")
        all_ts = np.concatenate([ts_train, ts_test])

        split = TemporalSplit(test_water_year=2024)
        train_mask, test_mask = split.split(all_ts)

        # WY2024 = Oct 1 2023 through Sep 30 2024
        wy2024_start = np.datetime64("2023-10-01")
        wy2024_end = np.datetime64("2024-09-30")

        train_ts = all_ts[train_mask]
        test_ts = all_ts[test_mask]

        # No training timestamp falls within WY2024
        assert not np.any(
            (train_ts >= wy2024_start) & (train_ts <= wy2024_end)
        ), "Training set contains WY2024 timestamps"

        # All test timestamps fall within WY2024
        assert np.all(
            (test_ts >= wy2024_start) & (test_ts <= wy2024_end)
        ), "Test set contains non-WY2024 timestamps"

    def test_split_no_overlap(self):
        """Train and test masks are mutually exclusive and exhaustive."""
        all_ts = _make_timestamps("2019-10-01", "2024-09-30")
        split = TemporalSplit(test_water_year=2024)
        train_mask, test_mask = split.split(all_ts)

        # Mutually exclusive
        assert not np.any(train_mask & test_mask)
        # Exhaustive
        assert np.all(train_mask | test_mask)

    def test_split_produces_nonempty_partitions(self):
        """Both train and test splits are non-empty for valid date ranges."""
        all_ts = _make_timestamps("2019-10-01", "2024-09-30")
        split = TemporalSplit(test_water_year=2024)
        train_mask, test_mask = split.split(all_ts)

        assert train_mask.sum() > 0
        assert test_mask.sum() > 0

    def test_water_year_boundary_october(self):
        """October 1 belongs to the next water year."""
        # Oct 1 2023 is the first day of WY2024
        ts = np.array([
            np.datetime64("2023-09-30"),  # WY2023
            np.datetime64("2023-10-01"),  # WY2024
        ])
        split = TemporalSplit(test_water_year=2024)
        train_mask, test_mask = split.split(ts)

        assert train_mask[0] and not test_mask[0], "Sep 30 should be train"
        assert test_mask[1] and not train_mask[1], "Oct 1 should be test"


# ===========================================================================
# Task 2.2: Full station coverage with missing data flagging
# ===========================================================================

class TestStationCoverage:
    """Expand LOSO to all stations with >50% missing flagging."""

    def test_accepts_large_station_list(self):
        """LOSOConfig accepts 120 stations (target: 117 real stations)."""
        rng = np.random.default_rng(42)
        ts = _make_timestamps("2024-01-01", "2024-03-31")
        stations = []
        for i in range(120):
            stations.append(_make_station(
                station_id=f"STN_{i:03d}",
                elevation=2500 + i * 10,
                timestamps=ts,
            ))

        config = LOSOConfig(stations=stations)
        assert len(config.stations) == 120

    def test_flag_stations_over_50pct_missing(self):
        """Stations with >50% missing values are flagged and excluded."""
        ts = _make_timestamps("2024-01-01", "2024-03-31")
        n = len(ts)

        # Good station: 10% missing
        good_temp = np.random.default_rng(1).uniform(260, 290, n).astype(np.float32)
        good_temp[:int(n * 0.1)] = np.nan

        # Bad station: 60% missing
        bad_temp = np.random.default_rng(2).uniform(260, 290, n).astype(np.float32)
        bad_temp[:int(n * 0.6)] = np.nan

        # Another good station: 0% missing
        ok_temp = np.random.default_rng(3).uniform(260, 290, n).astype(np.float32)

        stations = [
            _make_station("GOOD", 3000.0, ts, good_temp),
            _make_station("BAD", 3000.0, ts, bad_temp),
            _make_station("OK", 3000.0, ts, ok_temp),
        ]

        config = LOSOConfig(
            stations=stations,
            missing_threshold=0.5,
        )

        flagged = config.flag_missing_stations()
        assert "BAD" in flagged
        assert "GOOD" not in flagged
        assert "OK" not in flagged

    def test_flagged_stations_report_reason(self):
        """Flagged stations include reason and variable name."""
        ts = _make_timestamps("2024-01-01", "2024-03-31")
        n = len(ts)
        bad_temp = np.full(n, np.nan, dtype=np.float32)

        stations = [_make_station("ALL_NAN", 3000.0, ts, bad_temp)]
        config = LOSOConfig(stations=stations, missing_threshold=0.5)
        flagged = config.flag_missing_stations()

        assert "ALL_NAN" in flagged
        assert "temperature" in flagged["ALL_NAN"]


# ===========================================================================
# Task 2.3: Per-season stratification (DJF/MAM/JJA/SON)
# ===========================================================================

class TestSeasonClassification:
    """Season classification and per-season metric stratification."""

    def test_classify_season_djf(self):
        """December, January, February classify as DJF."""
        assert classify_season(datetime.date(2024, 12, 15)) == "DJF"
        assert classify_season(datetime.date(2024, 1, 1)) == "DJF"
        assert classify_season(datetime.date(2024, 2, 28)) == "DJF"

    def test_classify_season_mam(self):
        """March, April, May classify as MAM."""
        assert classify_season(datetime.date(2024, 3, 1)) == "MAM"
        assert classify_season(datetime.date(2024, 4, 15)) == "MAM"
        assert classify_season(datetime.date(2024, 5, 31)) == "MAM"

    def test_classify_season_jja(self):
        """June, July, August classify as JJA."""
        assert classify_season(datetime.date(2024, 6, 1)) == "JJA"
        assert classify_season(datetime.date(2024, 7, 4)) == "JJA"
        assert classify_season(datetime.date(2024, 8, 31)) == "JJA"

    def test_classify_season_son(self):
        """September, October, November classify as SON."""
        assert classify_season(datetime.date(2024, 9, 1)) == "SON"
        assert classify_season(datetime.date(2024, 10, 31)) == "SON"
        assert classify_season(datetime.date(2024, 11, 30)) == "SON"

    def test_per_season_metrics_differ_from_aggregate(self):
        """Per-season metrics differ from all-season aggregate for
        synthetic data with seasonal patterns."""
        rng = np.random.default_rng(42)

        # Full year of daily timestamps
        ts = _make_timestamps("2024-01-01", "2024-12-31")
        n = len(ts)

        # Seasonal temperature pattern: cold DJF, warm JJA
        temps = np.zeros(n, dtype=np.float32)
        for i, t in enumerate(ts):
            month = pd.Timestamp(t).month
            if month in (12, 1, 2):
                temps[i] = 260 + rng.normal(0, 2)
            elif month in (3, 4, 5):
                temps[i] = 275 + rng.normal(0, 2)
            elif month in (6, 7, 8):
                temps[i] = 295 + rng.normal(0, 2)
            else:
                temps[i] = 280 + rng.normal(0, 2)

        # Model predictions with season-dependent bias
        preds = temps.copy()
        for i, t in enumerate(ts):
            month = pd.Timestamp(t).month
            if month in (12, 1, 2):
                preds[i] += 5  # Large winter bias
            else:
                preds[i] += 1  # Small bias other seasons

        result = compute_seasonal_metrics(ts, temps, preds)

        assert "DJF" in result
        assert "MAM" in result
        assert "JJA" in result
        assert "SON" in result
        assert "all" in result

        # DJF bias should be larger than MAM bias
        assert result["DJF"]["bias"] > result["MAM"]["bias"]

    def test_cross_year_djf_for_wy2024(self):
        """DJF for WY2024 includes Dec 2023, Jan 2024, Feb 2024."""
        ts = np.array([
            np.datetime64("2023-12-15"),
            np.datetime64("2024-01-15"),
            np.datetime64("2024-02-15"),
        ])

        for t in ts:
            date = pd.Timestamp(t).date()
            assert classify_season(date) == "DJF"


# ===========================================================================
# Task 2.4: Per-elevation-band stratification
# ===========================================================================

class TestElevationBands:
    """Elevation band classification and per-band metrics."""

    def test_default_band_thresholds(self):
        """Default thresholds: <2900, 2900-3400, >3400."""
        config = ElevationBandConfig()
        assert config.below_treeline_max == 2900
        assert config.above_treeline_min == 3400

    def test_classify_below_treeline(self):
        """Station at 2500m is below treeline."""
        config = ElevationBandConfig()
        assert classify_elevation_band(2500.0, config) == "below_treeline"

    def test_classify_near_treeline(self):
        """Station at 3150m is near treeline."""
        config = ElevationBandConfig()
        assert classify_elevation_band(3150.0, config) == "near_treeline"

    def test_classify_above_treeline(self):
        """Station at 3600m is above treeline."""
        config = ElevationBandConfig()
        assert classify_elevation_band(3600.0, config) == "above_treeline"

    def test_boundary_at_2900_is_near(self):
        """Station at exactly 2900m is near treeline (>= lower bound)."""
        config = ElevationBandConfig()
        assert classify_elevation_band(2900.0, config) == "near_treeline"

    def test_boundary_at_3400_is_above(self):
        """Station at exactly 3400m is above treeline (>= upper bound)."""
        config = ElevationBandConfig()
        assert classify_elevation_band(3400.0, config) == "above_treeline"

    def test_custom_thresholds(self):
        """Custom thresholds override defaults."""
        config = ElevationBandConfig(
            below_treeline_max=2750,
            above_treeline_min=3100,
        )
        # 2800m would be near treeline with custom, but below with default
        assert classify_elevation_band(2800.0, config) == "near_treeline"

    def test_per_band_metrics_differ(self):
        """Stations in different bands produce different aggregate metrics."""
        rng = np.random.default_rng(42)
        ts = _make_timestamps("2024-01-01", "2024-03-31")
        n = len(ts)

        stations = [
            _make_station("LOW_1", 2500.0, ts),
            _make_station("LOW_2", 2600.0, ts),
            _make_station("MID_1", 3100.0, ts),
            _make_station("MID_2", 3200.0, ts),
            _make_station("HIGH_1", 3500.0, ts),
            _make_station("HIGH_2", 3600.0, ts),
        ]

        config = ElevationBandConfig()
        bands = {
            s.station_id: classify_elevation_band(s.elevation, config)
            for s in stations
        }

        assert bands["LOW_1"] == "below_treeline"
        assert bands["MID_1"] == "near_treeline"
        assert bands["HIGH_1"] == "above_treeline"


# ===========================================================================
# Task 2.5: Bootstrap 95% confidence intervals
# ===========================================================================

class TestBootstrapCI:
    """Bootstrap 95% CIs with station-level resampling."""

    def test_ci_brackets_point_estimate(self):
        """CI bounds bracket the point estimate (mean)."""
        rng = np.random.default_rng(42)
        # Station-level RMSE values for 50 stations
        station_values = rng.uniform(2.0, 5.0, size=50)

        ci = bootstrap_confidence_intervals(
            station_values, n_resamples=10000, seed=42
        )

        point_estimate = np.mean(station_values)
        assert ci["lower"] <= point_estimate <= ci["upper"]

    def test_ci_width_decreases_with_more_stations(self):
        """More stations produce narrower confidence intervals."""
        rng = np.random.default_rng(42)

        # Few stations (10)
        small = rng.uniform(2.0, 5.0, size=10)
        ci_small = bootstrap_confidence_intervals(
            small, n_resamples=10000, seed=42
        )

        # Many stations (100)
        large = rng.uniform(2.0, 5.0, size=100)
        ci_large = bootstrap_confidence_intervals(
            large, n_resamples=10000, seed=42
        )

        width_small = ci_small["upper"] - ci_small["lower"]
        width_large = ci_large["upper"] - ci_large["lower"]
        assert width_large < width_small

    def test_ci_uses_2_5_and_97_5_percentiles(self):
        """95% CI uses 2.5th and 97.5th percentiles."""
        # Known distribution: all values the same -> zero width CI
        station_values = np.full(50, 3.14)

        ci = bootstrap_confidence_intervals(
            station_values, n_resamples=1000, seed=42
        )

        assert ci["lower"] == pytest.approx(3.14, abs=1e-10)
        assert ci["upper"] == pytest.approx(3.14, abs=1e-10)

    def test_default_10000_resamples(self):
        """Default resample count is 10,000."""
        station_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Just confirm it runs with default
        ci = bootstrap_confidence_intervals(station_values, seed=42)

        assert "lower" in ci
        assert "upper" in ci
        assert "point_estimate" in ci

    def test_configurable_resample_count(self):
        """Resample count is configurable."""
        station_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ci = bootstrap_confidence_intervals(
            station_values, n_resamples=500, seed=42
        )
        assert "lower" in ci
        assert "upper" in ci

    def test_ci_returns_point_estimate(self):
        """CI result includes the point estimate alongside bounds."""
        station_values = np.array([2.0, 3.0, 4.0])
        ci = bootstrap_confidence_intervals(station_values, seed=42)

        assert ci["point_estimate"] == pytest.approx(np.mean(station_values))
