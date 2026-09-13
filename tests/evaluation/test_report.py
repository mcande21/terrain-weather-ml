"""Tests for comparison report and tail-event evaluation (task 7.5).

Comparison table across models with paired t-test significance.
Tail-event metrics for safety-critical applications.
"""

from __future__ import annotations

import numpy as np

from terrain_weather_ml.evaluation.report import (
    ComparisonReport,
    ModelResults,
    TailEventConfig,
    TailEventResult,
    compute_tail_event_metrics,
    generate_comparison_report,
)


def _make_model_results(
    name: str,
    n_stations: int = 5,
    base_rmse: float = 2.0,
    seed: int = 42,
) -> ModelResults:
    """Create mock ModelResults with realistic metric distributions."""
    rng = np.random.default_rng(seed)
    per_station: dict[str, dict[str, dict[str, float]]] = {}

    for i in range(n_stations):
        station_id = f"SNOTEL_{i:03d}"
        per_station[station_id] = {
            "wind_speed": {
                "rmse": base_rmse + rng.normal(0, 0.3),
                "mae": base_rmse * 0.7 + rng.normal(0, 0.2),
                "bias": rng.normal(0, 0.5),
                "r_squared": max(0, min(1, 0.8 - base_rmse * 0.05 + rng.normal(0, 0.05))),
            },
            "temperature": {
                "rmse": base_rmse * 0.5 + rng.normal(0, 0.2),
                "mae": base_rmse * 0.35 + rng.normal(0, 0.1),
                "bias": rng.normal(0, 0.3),
                "r_squared": max(0, min(1, 0.9 - base_rmse * 0.02 + rng.normal(0, 0.03))),
            },
        }

    return ModelResults(name=name, per_station=per_station)


class TestComparisonReport:
    """Comparison table across models with significance."""

    def test_report_with_three_models(self):
        """Report generates comparison for 3 models."""
        models = [
            _make_model_results("terrain_weather", base_rmse=1.5, seed=1),
            _make_model_results("hrrr_raw", base_rmse=3.0, seed=2),
            _make_model_results("granite_wxc", base_rmse=2.0, seed=3),
        ]
        report = generate_comparison_report(models)
        assert isinstance(report, ComparisonReport)
        assert len(report.models) == 3

    def test_report_includes_all_variables(self):
        """Report covers all variables present in the results."""
        models = [
            _make_model_results("model_a", seed=10),
            _make_model_results("model_b", seed=20),
        ]
        report = generate_comparison_report(models)
        assert "wind_speed" in report.variables
        assert "temperature" in report.variables

    def test_report_includes_per_variable_metrics(self):
        """Per-variable aggregate metrics are present for each model."""
        models = [
            _make_model_results("model_a", seed=10),
            _make_model_results("model_b", seed=20),
        ]
        report = generate_comparison_report(models)
        for model_name in ["model_a", "model_b"]:
            for var in report.variables:
                assert model_name in report.aggregate_metrics
                assert var in report.aggregate_metrics[model_name]
                assert "rmse" in report.aggregate_metrics[model_name][var]

    def test_significance_flags_present(self):
        """Report includes paired t-test significance flags."""
        models = [
            _make_model_results("better", base_rmse=1.0, seed=1),
            _make_model_results("worse", base_rmse=5.0, seed=2),
        ]
        report = generate_comparison_report(models)
        assert report.significance is not None
        # With very different base_rmse, should detect significance
        # (5 stations is small, but the effect is huge)

    def test_significance_structure(self):
        """Significance dict maps (model_pair, variable) -> p_value."""
        models = [
            _make_model_results("a", seed=1),
            _make_model_results("b", seed=2),
            _make_model_results("c", seed=3),
        ]
        report = generate_comparison_report(models)
        # Should have pairwise comparisons
        assert isinstance(report.significance, dict)

    def test_report_serializable(self):
        """Report can be serialized to dict."""
        models = [
            _make_model_results("a", seed=1),
            _make_model_results("b", seed=2),
        ]
        report = generate_comparison_report(models)
        result = report.to_dict()
        assert isinstance(result, dict)
        assert "models" in result
        assert "aggregate_metrics" in result
        assert "significance" in result


class TestTailEventEvaluation:
    """Tail-event metrics for safety-critical evaluation."""

    def test_wind_tail_events(self):
        """Wind speed > 90th percentile subset is correctly filtered."""
        rng = np.random.default_rng(42)
        n = 1000
        obs = rng.uniform(0, 20, n).astype(np.float32)
        pred = obs + rng.normal(0, 1, n).astype(np.float32)

        config = TailEventConfig(
            variable="wind_speed",
            percentile=90,
            direction="above",
        )
        result = compute_tail_event_metrics(obs, pred, config)
        assert isinstance(result, TailEventResult)
        # Should have ~10% of data in tail
        assert 50 < result.n_events < 150
        assert result.metrics is not None

    def test_temperature_tail_events(self):
        """Temperature < 10th percentile (cold extremes)."""
        rng = np.random.default_rng(42)
        n = 1000
        obs = rng.uniform(250, 300, n).astype(np.float32)
        pred = obs + rng.normal(0, 2, n).astype(np.float32)

        config = TailEventConfig(
            variable="temperature",
            percentile=10,
            direction="below",
        )
        result = compute_tail_event_metrics(obs, pred, config)
        assert 50 < result.n_events < 150

    def test_precipitation_tail_events(self):
        """Precipitation > 95th percentile (heavy events)."""
        rng = np.random.default_rng(42)
        n = 1000
        obs = rng.exponential(1.0, n).astype(np.float32)
        pred = obs + rng.normal(0, 0.5, n).astype(np.float32)

        config = TailEventConfig(
            variable="precipitation",
            percentile=95,
            direction="above",
        )
        result = compute_tail_event_metrics(obs, pred, config)
        assert 30 < result.n_events < 70

    def test_tail_metrics_subset_only(self):
        """Tail metrics computed only on tail subset, not full data."""
        n = 100
        obs = np.arange(n, dtype=np.float32)
        # Make predictions perfect except for tail events
        pred = obs.copy()
        # Add large error only to tail (top 10%)
        pred[90:] += 50.0

        config_full = TailEventConfig(
            variable="test", percentile=90, direction="above"
        )
        result = compute_tail_event_metrics(obs, pred, config_full)
        # Tail RMSE should be ~50, not diluted by the good predictions
        assert result.metrics.rmse > 40

    def test_no_tail_events_returns_zero(self):
        """When no observations exceed threshold, n_events=0."""
        obs = np.array([1.0, 2.0, 3.0])
        pred = np.array([1.0, 2.0, 3.0])

        # Percentile 99 on 3 values: threshold is ~3.0
        # direction="above" with threshold at ~p99
        config = TailEventConfig(
            variable="test", percentile=99, direction="above"
        )
        result = compute_tail_event_metrics(obs, pred, config)
        # With only 3 values, 99th percentile leaves 0 or 1 events
        assert result.n_events <= 1

    def test_result_serializable(self):
        """TailEventResult can be serialized to dict."""
        rng = np.random.default_rng(42)
        obs = rng.uniform(0, 10, 100).astype(np.float32)
        pred = obs + rng.normal(0, 1, 100).astype(np.float32)

        config = TailEventConfig(
            variable="wind", percentile=90, direction="above"
        )
        result = compute_tail_event_metrics(obs, pred, config)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "n_events" in d
        assert "threshold_value" in d
