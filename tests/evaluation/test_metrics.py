"""Tests for evaluation metrics (task 7.2).

Per-variable metrics: RMSE, MAE, bias, R-squared for continuous variables.
Circular metrics for wind direction. Categorical metrics for precipitation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from terrain_weather_ml.evaluation.metrics import (
    CircularMetrics,
    ContinuousMetrics,
    PrecipitationMetrics,
    compute_circular_metrics,
    compute_continuous_metrics,
    compute_precipitation_metrics,
)


class TestContinuousMetrics:
    """RMSE, MAE, bias, R-squared for continuous variables."""

    def test_perfect_predictions(self):
        """Perfect predictions yield zero error and R²=1."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pred = obs.copy()
        m = compute_continuous_metrics(obs, pred)
        assert isinstance(m, ContinuousMetrics)
        assert m.rmse == pytest.approx(0.0, abs=1e-10)
        assert m.mae == pytest.approx(0.0, abs=1e-10)
        assert m.bias == pytest.approx(0.0, abs=1e-10)
        assert m.r_squared == pytest.approx(1.0, abs=1e-10)

    def test_known_bias(self):
        """Constant offset produces predictable bias."""
        obs = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        pred = obs + 2.0  # +2 bias
        m = compute_continuous_metrics(obs, pred)
        assert m.bias == pytest.approx(2.0, abs=1e-10)
        assert m.rmse == pytest.approx(2.0, abs=1e-10)
        assert m.mae == pytest.approx(2.0, abs=1e-10)
        # R² should be 1 for constant offset with perfect correlation
        assert m.r_squared == pytest.approx(1.0 - 4.0 / 200.0, abs=1e-6)

    def test_known_rmse(self):
        """Hand-computed RMSE matches expected value."""
        obs = np.array([1.0, 2.0, 3.0])
        pred = np.array([2.0, 2.0, 4.0])
        # errors = [1, 0, 1], MSE = 2/3, RMSE = sqrt(2/3)
        m = compute_continuous_metrics(obs, pred)
        assert m.rmse == pytest.approx(math.sqrt(2.0 / 3.0), abs=1e-10)

    def test_missing_observations_excluded(self):
        """NaN observations are excluded from metric computation."""
        obs = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        pred = np.array([1.0, 99.0, 3.0, 99.0, 5.0])
        m = compute_continuous_metrics(obs, pred)
        assert m.rmse == pytest.approx(0.0, abs=1e-10)
        assert m.mae == pytest.approx(0.0, abs=1e-10)

    def test_missing_predictions_excluded(self):
        """NaN predictions are also excluded."""
        obs = np.array([1.0, 2.0, 3.0])
        pred = np.array([1.0, np.nan, 3.0])
        m = compute_continuous_metrics(obs, pred)
        assert m.rmse == pytest.approx(0.0, abs=1e-10)

    def test_insufficient_data_returns_nan(self):
        """Fewer than 2 valid points returns NaN metrics."""
        obs = np.array([1.0, np.nan])
        pred = np.array([np.nan, 2.0])
        m = compute_continuous_metrics(obs, pred)
        assert np.isnan(m.rmse)


class TestCircularMetrics:
    """Circular RMSE and circular bias for wind direction."""

    def test_perfect_direction(self):
        """Identical directions yield zero circular error."""
        obs = np.array([0.0, 90.0, 180.0, 270.0])
        pred = obs.copy()
        m = compute_circular_metrics(obs, pred)
        assert isinstance(m, CircularMetrics)
        assert m.circular_rmse == pytest.approx(0.0, abs=1e-10)
        assert m.circular_bias == pytest.approx(0.0, abs=1e-10)

    def test_wrap_around(self):
        """Correctly handles 0/360 wrap-around."""
        obs = np.array([355.0, 5.0])
        pred = np.array([5.0, 355.0])
        m = compute_circular_metrics(obs, pred)
        # Each diff is 10 degrees (wrapping), RMSE = 10
        assert m.circular_rmse == pytest.approx(10.0, abs=1e-6)

    def test_known_circular_bias(self):
        """Constant clockwise offset produces positive circular bias."""
        obs = np.array([0.0, 90.0, 180.0, 270.0])
        pred = np.array([10.0, 100.0, 190.0, 280.0])
        m = compute_circular_metrics(obs, pred)
        assert m.circular_bias == pytest.approx(10.0, abs=1e-6)

    def test_opposite_directions(self):
        """180-degree errors produce circular RMSE = 180."""
        obs = np.array([0.0, 0.0])
        pred = np.array([180.0, 180.0])
        m = compute_circular_metrics(obs, pred)
        assert m.circular_rmse == pytest.approx(180.0, abs=1e-6)

    def test_missing_values_excluded(self):
        """NaN values excluded from circular metrics."""
        obs = np.array([0.0, np.nan, 90.0])
        pred = np.array([0.0, 180.0, 90.0])
        m = compute_circular_metrics(obs, pred)
        assert m.circular_rmse == pytest.approx(0.0, abs=1e-10)


class TestPrecipitationMetrics:
    """Categorical precipitation metrics at configurable thresholds."""

    def test_perfect_detection(self):
        """Perfect forecasts yield POD=1, FAR=0, CSI=1."""
        obs = np.array([0.0, 0.5, 0.0, 1.5, 0.0])
        pred = np.array([0.0, 0.5, 0.0, 1.5, 0.0])
        m = compute_precipitation_metrics(obs, pred, threshold=0.1)
        assert isinstance(m, PrecipitationMetrics)
        assert m.pod == pytest.approx(1.0)
        assert m.far == pytest.approx(0.0)
        assert m.csi == pytest.approx(1.0)

    def test_all_misses(self):
        """Forecast never predicts rain when observed -> POD=0."""
        obs = np.array([1.0, 2.0, 3.0])
        pred = np.array([0.0, 0.0, 0.0])
        m = compute_precipitation_metrics(obs, pred, threshold=0.1)
        assert m.pod == pytest.approx(0.0)
        assert m.csi == pytest.approx(0.0)

    def test_all_false_alarms(self):
        """Forecast always predicts rain when not observed -> FAR=1."""
        obs = np.array([0.0, 0.0, 0.0])
        pred = np.array([1.0, 1.0, 1.0])
        m = compute_precipitation_metrics(obs, pred, threshold=0.1)
        assert m.far == pytest.approx(1.0)

    def test_known_contingency(self):
        """Hand-computed contingency table produces expected metrics."""
        # hits=2, misses=1, false_alarms=1, correct_negatives=1
        obs = np.array([0.5, 1.0, 0.0, 2.0, 0.0])
        pred = np.array([0.5, 0.0, 0.5, 1.0, 0.0])
        m = compute_precipitation_metrics(obs, pred, threshold=0.1)
        # hits: obs>=0.1 AND pred>=0.1 -> idx 0, 3 -> 2 hits
        # misses: obs>=0.1 AND pred<0.1 -> idx 1 -> 1 miss
        # false alarms: obs<0.1 AND pred>=0.1 -> idx 2 -> 1 fa
        # POD = hits/(hits+misses) = 2/3
        # FAR = fa/(hits+fa) = 1/3
        # CSI = hits/(hits+misses+fa) = 2/4 = 0.5
        assert m.pod == pytest.approx(2.0 / 3.0, abs=1e-6)
        assert m.far == pytest.approx(1.0 / 3.0, abs=1e-6)
        assert m.csi == pytest.approx(0.5, abs=1e-6)

    def test_frequency_bias(self):
        """Frequency bias = (hits + false_alarms) / (hits + misses)."""
        obs = np.array([0.5, 1.0, 0.0, 2.0, 0.0])
        pred = np.array([0.5, 0.0, 0.5, 1.0, 0.0])
        m = compute_precipitation_metrics(obs, pred, threshold=0.1)
        # freq_bias = (2+1)/(2+1) = 1.0
        assert m.frequency_bias == pytest.approx(1.0, abs=1e-6)

    def test_ets_equitable_threat_score(self):
        """ETS accounts for hits expected by chance."""
        obs = np.array([0.5, 1.0, 0.0, 2.0, 0.0])
        pred = np.array([0.5, 0.0, 0.5, 1.0, 0.0])
        m = compute_precipitation_metrics(obs, pred, threshold=0.1)
        # hits_random = (hits+misses)*(hits+fa)/total = 3*3/5 = 1.8
        # ETS = (hits - hits_random) / (hits + misses + fa - hits_random)
        # ETS = (2 - 1.8) / (2 + 1 + 1 - 1.8) = 0.2 / 2.2
        assert m.ets == pytest.approx(0.2 / 2.2, abs=1e-6)

    def test_configurable_threshold(self):
        """Different threshold changes the contingency table."""
        obs = np.array([0.05, 0.2, 0.8, 2.0])
        pred = np.array([0.05, 0.2, 0.8, 2.0])
        m_low = compute_precipitation_metrics(obs, pred, threshold=0.1)
        m_high = compute_precipitation_metrics(obs, pred, threshold=1.0)
        # At 0.1 threshold: 3 events. At 1.0 threshold: 1 event.
        # Both perfect -> POD=1 for both, but different event counts.
        assert m_low.pod == pytest.approx(1.0)
        assert m_high.pod == pytest.approx(1.0)

    def test_missing_values_excluded(self):
        """NaN values excluded from precipitation metrics."""
        obs = np.array([0.5, np.nan, 1.0])
        pred = np.array([0.5, 0.5, 1.0])
        m = compute_precipitation_metrics(obs, pred, threshold=0.1)
        assert m.pod == pytest.approx(1.0)

    def test_no_events_returns_nan(self):
        """No observed events -> POD and CSI are NaN (undefined)."""
        obs = np.array([0.0, 0.0, 0.0])
        pred = np.array([0.0, 0.0, 0.0])
        m = compute_precipitation_metrics(obs, pred, threshold=0.1)
        assert np.isnan(m.pod)
        assert np.isnan(m.csi)
