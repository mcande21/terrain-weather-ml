"""Tests for LOSO cross-validation framework (task 7.1)."""

from __future__ import annotations

import numpy as np
import torch

from terrain_weather_ml.evaluation.loso import (
    LOSOConfig,
    LOSOEvaluator,
    LOSOResult,
    StationData,
)


def _make_mock_model():
    """Create a mock model that returns predictions matching input shape."""

    class MockModel:
        def predict(
            self, weather_input: torch.Tensor, terrain: torch.Tensor
        ) -> torch.Tensor:
            """Return predictions shaped (B, 4, H, W)."""
            b = weather_input.shape[0]
            h, w = terrain.shape[2], terrain.shape[3]
            torch.manual_seed(42)
            return torch.randn(b, 4, h, w)

    return MockModel()


def _make_synthetic_stations(n: int = 5) -> list[StationData]:
    """Create N synthetic SNOTEL stations with random data."""
    rng = np.random.default_rng(42)
    stations = []
    for i in range(n):
        n_timesteps = 100
        stations.append(
            StationData(
                station_id=f"SNOTEL_{i:03d}",
                latitude=39.0 + rng.uniform(-1, 1),
                longitude=-106.0 + rng.uniform(-1, 1),
                elevation=3000 + rng.uniform(-500, 500),
                tpi=rng.uniform(-100, 100),
                observations={
                    "wind_speed": rng.uniform(0, 15, n_timesteps).astype(np.float32),
                    "wind_direction": rng.uniform(0, 360, n_timesteps).astype(np.float32),
                    "temperature": rng.uniform(250, 300, n_timesteps).astype(np.float32),
                    "precipitation": rng.exponential(0.5, n_timesteps).astype(np.float32),
                },
                timestamps=np.arange(n_timesteps),
            )
        )
    return stations


class TestLOSOFramework:
    """Leave-one-station-out cross-validation framework."""

    def test_full_loso_produces_n_folds(self):
        """LOSO with N stations produces N folds."""
        stations = _make_synthetic_stations(5)
        model = _make_mock_model()
        config = LOSOConfig(stations=stations)

        evaluator = LOSOEvaluator(config)
        result = evaluator.evaluate(model)

        assert isinstance(result, LOSOResult)
        assert len(result.per_station) == 5

    def test_per_station_metrics_present(self):
        """Each fold produces per-station metrics with required keys."""
        stations = _make_synthetic_stations(5)
        model = _make_mock_model()
        config = LOSOConfig(stations=stations)

        evaluator = LOSOEvaluator(config)
        result = evaluator.evaluate(model)

        for metrics in result.per_station.values():
            assert "wind_speed" in metrics
            assert "temperature" in metrics

    def test_aggregate_statistics(self):
        """Aggregate statistics include mean, median, std across stations."""
        stations = _make_synthetic_stations(5)
        model = _make_mock_model()
        config = LOSOConfig(stations=stations)

        evaluator = LOSOEvaluator(config)
        result = evaluator.evaluate(model)

        agg = result.aggregate
        assert "mean" in agg
        assert "median" in agg
        assert "std" in agg

    def test_single_station_mode(self):
        """Single-station mode evaluates only the specified fold."""
        stations = _make_synthetic_stations(5)
        model = _make_mock_model()
        config = LOSOConfig(stations=stations)

        evaluator = LOSOEvaluator(config)
        result = evaluator.evaluate(model, station_id="SNOTEL_002")

        assert len(result.per_station) == 1
        assert "SNOTEL_002" in result.per_station

    def test_single_station_missing_raises(self):
        """Requesting a station not in the dataset raises ValueError."""
        stations = _make_synthetic_stations(5)
        model = _make_mock_model()
        config = LOSOConfig(stations=stations)

        evaluator = LOSOEvaluator(config)
        try:
            evaluator.evaluate(model, station_id="NONEXISTENT")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_result_serializable_to_dict(self):
        """LOSOResult can be serialized to a dict for JSON export."""
        stations = _make_synthetic_stations(5)
        model = _make_mock_model()
        config = LOSOConfig(stations=stations)

        evaluator = LOSOEvaluator(config)
        result = evaluator.evaluate(model)

        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert "per_station" in result_dict
        assert "aggregate" in result_dict
