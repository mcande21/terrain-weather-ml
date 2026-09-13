"""Tests for baseline model runners (task 7.4).

Evaluation wrappers for:
(a) Raw HRRR nearest-grid-cell (no downscaling)
(b) WindNinja solver with HRRR boundary conditions
(c) granite-wxc-downscaling (IBM)

All baselines produce metrics in the same format as the terrain-weather model.
"""

from __future__ import annotations

import numpy as np

from terrain_weather_ml.evaluation.baselines import (
    BaselineRunner,
    GraniteWxcRunner,
    HRRRBaselineRunner,
    WindNinjaRunner,
)
from terrain_weather_ml.evaluation.loso import StationData


def _make_station() -> StationData:
    """Create a single synthetic station."""
    return StationData(
        station_id="TEST_001",
        latitude=39.5,
        longitude=-106.0,
        elevation=3200,
        tpi=-20.0,
        observations={
            "wind_speed": np.random.default_rng(42).uniform(0, 15, 50).astype(np.float32),
            "wind_direction": np.random.default_rng(42).uniform(0, 360, 50).astype(np.float32),
            "temperature": np.random.default_rng(42).uniform(250, 300, 50).astype(np.float32),
            "precipitation": np.random.default_rng(42).exponential(0.5, 50).astype(np.float32),
        },
        timestamps=np.arange(50),
    )


REQUIRED_METRIC_KEYS = {"rmse", "mae", "bias", "r_squared"}
REQUIRED_VARIABLES = {"wind_speed", "temperature"}


class TestHRRRBaseline:
    """Raw HRRR nearest-grid-cell baseline."""

    def test_produces_predictions(self):
        """HRRR runner produces predictions for a station."""
        station = _make_station()
        runner = HRRRBaselineRunner()
        predictions = runner.predict_station(station)
        assert isinstance(predictions, dict)
        assert "wind_speed" in predictions
        assert "temperature" in predictions
        assert len(predictions["wind_speed"]) == len(station.timestamps)

    def test_metrics_format_matches(self):
        """HRRR metrics have all required keys."""
        station = _make_station()
        runner = HRRRBaselineRunner()
        metrics = runner.evaluate_station(station)
        for var in REQUIRED_VARIABLES:
            assert var in metrics
            assert REQUIRED_METRIC_KEYS.issubset(set(metrics[var].keys()))

    def test_baseline_name(self):
        """Runner reports its model name."""
        runner = HRRRBaselineRunner()
        assert runner.name == "hrrr_raw"

    def test_is_baseline_runner(self):
        """Implements BaselineRunner protocol."""
        runner = HRRRBaselineRunner()
        assert isinstance(runner, BaselineRunner)


class TestWindNinjaBaseline:
    """WindNinja solver baseline."""

    def test_produces_predictions(self):
        """WindNinja runner produces predictions."""
        station = _make_station()
        runner = WindNinjaRunner()
        predictions = runner.predict_station(station)
        assert isinstance(predictions, dict)
        assert "wind_speed" in predictions

    def test_metrics_format_matches(self):
        """WindNinja metrics match required format."""
        station = _make_station()
        runner = WindNinjaRunner()
        metrics = runner.evaluate_station(station)
        for var in REQUIRED_VARIABLES:
            assert var in metrics
            assert REQUIRED_METRIC_KEYS.issubset(set(metrics[var].keys()))

    def test_baseline_name(self):
        runner = WindNinjaRunner()
        assert runner.name == "windninja"


class TestGraniteWxcBaseline:
    """IBM granite-wxc-downscaling baseline."""

    def test_produces_predictions(self):
        """Granite runner produces predictions."""
        station = _make_station()
        runner = GraniteWxcRunner()
        predictions = runner.predict_station(station)
        assert isinstance(predictions, dict)
        assert "wind_speed" in predictions

    def test_metrics_format_matches(self):
        """Granite metrics match required format."""
        station = _make_station()
        runner = GraniteWxcRunner()
        metrics = runner.evaluate_station(station)
        for var in REQUIRED_VARIABLES:
            assert var in metrics
            assert REQUIRED_METRIC_KEYS.issubset(set(metrics[var].keys()))

    def test_baseline_name(self):
        runner = GraniteWxcRunner()
        assert runner.name == "granite_wxc"


class TestBaselineConsistency:
    """All baselines produce comparable output."""

    def test_all_baselines_same_format(self):
        """All three baselines produce metrics with identical keys."""
        station = _make_station()
        runners = [HRRRBaselineRunner(), WindNinjaRunner(), GraniteWxcRunner()]

        all_keys = []
        for runner in runners:
            metrics = runner.evaluate_station(station)
            keys = set()
            for var_name, var_metrics in metrics.items():
                for metric_name in var_metrics:
                    keys.add(f"{var_name}.{metric_name}")
            all_keys.append(keys)

        # All runners should have the same set of metric keys
        assert all_keys[0] == all_keys[1] == all_keys[2]
