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
    ConditionalMetricsResult,
    ContinuousMetrics,
    PrecipitationMetrics,
    ThresholdCrossingMetrics,
    compute_accumulation_accuracy,
    compute_circular_metrics,
    compute_conditional_metrics,
    compute_continuous_metrics,
    compute_precipitation_metrics,
    compute_skill_score,
    compute_threshold_crossing_accuracy,
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


class TestSkillScore:
    """Skill score: 1 - RMSE_model / RMSE_baseline (task 4.1)."""

    def test_known_skill_score(self):
        """Model RMSE 2.1, baseline RMSE 3.0 -> skill = 0.30."""
        ss = compute_skill_score(model_rmse=2.1, baseline_rmse=3.0)
        assert ss == pytest.approx(0.30, abs=1e-6)

    def test_equal_rmse_yields_zero(self):
        """Equal RMSE -> skill score is exactly 0."""
        ss = compute_skill_score(model_rmse=3.0, baseline_rmse=3.0)
        assert ss == pytest.approx(0.0, abs=1e-10)

    def test_model_worse_than_baseline(self):
        """Model RMSE > baseline -> negative skill score."""
        ss = compute_skill_score(model_rmse=4.0, baseline_rmse=2.0)
        # 1 - 4/2 = 1 - 2 = -1
        assert ss == pytest.approx(-1.0, abs=1e-10)

    def test_perfect_model(self):
        """Model RMSE 0 -> skill = 1.0 (perfect)."""
        ss = compute_skill_score(model_rmse=0.0, baseline_rmse=3.0)
        assert ss == pytest.approx(1.0, abs=1e-10)

    def test_zero_baseline_returns_nan(self):
        """Zero baseline RMSE -> undefined, return NaN."""
        ss = compute_skill_score(model_rmse=2.0, baseline_rmse=0.0)
        assert math.isnan(ss)

    def test_both_zero_returns_nan(self):
        """Both zero -> 0/0 -> NaN."""
        ss = compute_skill_score(model_rmse=0.0, baseline_rmse=0.0)
        assert math.isnan(ss)


class TestThresholdCrossingAccuracy:
    """Temperature threshold crossing accuracy at 0C (task 4.2)."""

    def test_perfect_crossing_detection(self):
        """Predicted crossings exactly match observed."""
        # obs: +1 -> -1 -> +2 (crossings at idx 0->1, 1->2)
        obs = np.array([1.0, -1.0, 2.0])
        pred = np.array([1.0, -1.0, 2.0])
        m = compute_threshold_crossing_accuracy(obs, pred, threshold=0.0)
        assert isinstance(m, ThresholdCrossingMetrics)
        assert m.pod == pytest.approx(1.0)
        assert m.far == pytest.approx(0.0)
        assert m.csi == pytest.approx(1.0)

    def test_all_missed_crossings(self):
        """Obs crosses 0C but pred stays above -> POD=0."""
        obs = np.array([1.0, -1.0, 1.0])
        pred = np.array([1.0, 1.0, 1.0])
        m = compute_threshold_crossing_accuracy(obs, pred, threshold=0.0)
        assert m.pod == pytest.approx(0.0)

    def test_false_alarm_crossings(self):
        """Pred crosses 0C but obs stays above -> FAR=1."""
        obs = np.array([1.0, 1.0, 1.0])
        pred = np.array([1.0, -1.0, 1.0])
        m = compute_threshold_crossing_accuracy(obs, pred, threshold=0.0)
        assert m.far == pytest.approx(1.0)

    def test_known_crossing_metrics(self):
        """Hand-computed crossing contingency table."""
        # Obs: 2, -1, 3, -2, 1 -> crossings at 0->1 (down), 1->2 (up),
        #                           2->3 (down), 3->4 (up) = 4 crossings
        # Pred: 2, -1, 3, 1, 1 -> crossings at 0->1 (down), 1->2 (up),
        #                           but NOT 2->3 (stays above), NOT 3->4
        # So: obs_cross = [True, True, True, True]
        #     pred_cross = [True, True, False, False]
        # hits=2, misses=2, false_alarms=0
        # POD = 2/4 = 0.5, FAR = 0/2 = 0, CSI = 2/(2+2+0) = 0.5
        obs = np.array([2.0, -1.0, 3.0, -2.0, 1.0])
        pred = np.array([2.0, -1.0, 3.0, 1.0, 1.0])
        m = compute_threshold_crossing_accuracy(obs, pred, threshold=0.0)
        assert m.pod == pytest.approx(0.5, abs=1e-6)
        assert m.far == pytest.approx(0.0, abs=1e-6)
        assert m.csi == pytest.approx(0.5, abs=1e-6)

    def test_no_crossings_returns_nan(self):
        """No observed crossings -> POD is NaN; CSI = 0 when false alarms exist."""
        obs = np.array([5.0, 5.0, 5.0])
        pred = np.array([5.0, -1.0, 5.0])
        m = compute_threshold_crossing_accuracy(obs, pred, threshold=0.0)
        assert math.isnan(m.pod)
        # CSI = 0/(0+0+2) = 0.0 (false alarms exist but no obs crossings)
        assert m.csi == pytest.approx(0.0, abs=1e-10)

    def test_no_crossings_at_all_returns_nan(self):
        """No crossings in either obs or pred -> POD and CSI are NaN."""
        obs = np.array([5.0, 5.0, 5.0])
        pred = np.array([5.0, 5.0, 5.0])
        m = compute_threshold_crossing_accuracy(obs, pred, threshold=0.0)
        assert math.isnan(m.pod)
        assert math.isnan(m.csi)

    def test_custom_threshold(self):
        """Crossing detection at custom threshold (e.g. -5C)."""
        obs = np.array([-3.0, -7.0, -3.0])
        pred = np.array([-3.0, -7.0, -3.0])
        m = compute_threshold_crossing_accuracy(obs, pred, threshold=-5.0)
        assert m.pod == pytest.approx(1.0)
        assert m.n_observed_crossings == 2


class TestAccumulationAccuracy:
    """Precipitation accumulation accuracy over multi-day windows (task 4.3)."""

    def test_perfect_accumulation(self):
        """Perfect predictions yield RMSE=0, R²=1."""
        # 72 hours (3 days), 3 complete windows with varying rates
        rng = np.random.default_rng(99)
        obs = np.concatenate([
            rng.uniform(0.5, 1.5, size=72),
            rng.uniform(1.0, 3.0, size=72),
            rng.uniform(0.0, 0.5, size=72),
        ])
        pred = obs.copy()
        result = compute_accumulation_accuracy(obs, pred, window_hours=72)
        assert result["rmse"] == pytest.approx(0.0, abs=1e-10)
        assert result["r_squared"] == pytest.approx(1.0, abs=1e-6)
        assert result["n_windows"] == 3

    def test_known_accumulation_rmse(self):
        """Known bias in accumulation produces predictable RMSE."""
        # 2 windows of 72 hours each = 144 hours
        obs = np.ones(144) * 1.0  # 1mm/hr -> 72mm per 3-day window
        pred = np.ones(144) * 1.5  # 1.5mm/hr -> 108mm per 3-day window
        result = compute_accumulation_accuracy(obs, pred, window_hours=72)
        # Each window: obs_accum=72, pred_accum=108, error=36
        # RMSE = 36.0 (both windows have same error)
        assert result["rmse"] == pytest.approx(36.0, abs=1e-6)
        assert result["bias"] == pytest.approx(36.0, abs=1e-6)

    def test_seven_day_windows(self):
        """7-day (168h) window accumulation."""
        obs = np.ones(336) * 2.0  # 2mm/hr, 2 complete windows
        pred = np.ones(336) * 2.0
        result = compute_accumulation_accuracy(obs, pred, window_hours=168)
        assert result["rmse"] == pytest.approx(0.0, abs=1e-10)
        assert result["n_windows"] == 2

    def test_incomplete_window_excluded(self):
        """Partial window at end is excluded."""
        # 200 hours with window=72: 2 full windows (144h), 56h remainder
        obs = np.ones(200)
        pred = obs.copy()
        result = compute_accumulation_accuracy(obs, pred, window_hours=72)
        assert result["n_windows"] == 2

    def test_insufficient_data_returns_nan(self):
        """Fewer hours than one window -> NaN metrics."""
        obs = np.ones(50)
        pred = obs.copy()
        result = compute_accumulation_accuracy(obs, pred, window_hours=72)
        assert math.isnan(result["rmse"])
        assert result["n_windows"] == 0

    def test_accumulation_correlation(self):
        """R-squared captures correlation of windowed totals."""
        rng = np.random.default_rng(42)
        obs = rng.uniform(0, 5, size=720)  # 30 days of hourly data
        # Pred = obs + small noise -> high correlation
        pred = obs + rng.normal(0, 0.1, size=720)
        pred = np.clip(pred, 0, None)
        result = compute_accumulation_accuracy(obs, pred, window_hours=72)
        assert result["r_squared"] > 0.9
        assert result["n_windows"] == 10


class TestConditionalMetrics:
    """Conditional metrics matrix: elevation bands x seasons (task 4.4)."""

    def _make_station_data(self):
        """Create synthetic station data spanning elevations and seasons.

        Returns:
            dict with keys: obs, pred, elevations, months, station_ids
        """
        rng = np.random.default_rng(123)
        n_stations = 12
        n_obs_per_station = 40

        elevations = np.array([
            2500, 2600, 2700, 2800,   # below treeline (<2750)
            2800, 2900, 3000, 3050,   # near treeline (2750-3100)
            3200, 3300, 3400, 3500,   # above treeline (>3100)
        ], dtype=float)

        station_ids = [f"STN{i:03d}" for i in range(n_stations)]

        all_obs = []
        all_pred = []
        all_elev = []
        all_months = []
        all_station = []

        # Each station gets obs in all 4 seasons
        season_months = [12, 3, 6, 9]  # DJF, MAM, JJA, SON
        for i, (elev, sid) in enumerate(zip(elevations, station_ids)):
            for month in season_months:
                n = n_obs_per_station // 4
                obs_vals = rng.normal(0, 5, size=n)
                # Higher elevation -> larger error to test stratification
                noise_scale = 1.0 + (elev - 2500) / 1000.0
                pred_vals = obs_vals + rng.normal(0, noise_scale, size=n)
                all_obs.extend(obs_vals)
                all_pred.extend(pred_vals)
                all_elev.extend([elev] * n)
                all_months.extend([month] * n)
                all_station.extend([sid] * n)

        return {
            "obs": np.array(all_obs),
            "pred": np.array(all_pred),
            "elevations": np.array(all_elev),
            "months": np.array(all_months),
            "station_ids": all_station,
        }

    def test_returns_structured_result(self):
        """Result contains 3 elevation bands x 4 seasons."""
        data = self._make_station_data()
        result = compute_conditional_metrics(
            obs=data["obs"],
            pred=data["pred"],
            elevations=data["elevations"],
            months=data["months"],
            station_ids=data["station_ids"],
        )
        assert isinstance(result, ConditionalMetricsResult)
        assert len(result.elevation_bands) == 3
        assert len(result.seasons) == 4
        # 12 cells total
        assert len(result.cells) == 12

    def test_elevation_band_assignment(self):
        """Stations assigned to correct elevation bands."""
        data = self._make_station_data()
        result = compute_conditional_metrics(
            obs=data["obs"],
            pred=data["pred"],
            elevations=data["elevations"],
            months=data["months"],
            station_ids=data["station_ids"],
        )
        # Each cell should have some observations
        for cell in result.cells.values():
            assert cell["metrics"].n_valid > 0

    def test_seasonal_stratification(self):
        """Different seasons produce different metrics."""
        data = self._make_station_data()
        result = compute_conditional_metrics(
            obs=data["obs"],
            pred=data["pred"],
            elevations=data["elevations"],
            months=data["months"],
            station_ids=data["station_ids"],
        )
        # Get metrics for two different seasons in the same elevation band
        band0 = result.elevation_bands[0]
        s1 = result.cells[(band0, "DJF")]
        s2 = result.cells[(band0, "JJA")]
        # They should have metrics (both populated)
        assert s1["metrics"].n_valid > 0
        assert s2["metrics"].n_valid > 0

    def test_low_confidence_flag(self):
        """Cells with <5 stations flagged as low confidence."""
        # Create data with very few stations in one band
        rng = np.random.default_rng(42)
        obs = rng.normal(0, 1, size=30)
        pred = obs + rng.normal(0, 0.5, size=30)
        # All 30 obs from 2 stations, all in one band+season
        elevations = np.full(30, 2500.0)
        months = np.full(30, 12)
        station_ids = ["A"] * 15 + ["B"] * 15

        result = compute_conditional_metrics(
            obs=obs,
            pred=pred,
            elevations=elevations,
            months=months,
            station_ids=station_ids,
        )
        # The cell with 2 stations should be low-confidence
        for cell in result.cells.values():
            if cell["n_stations"] > 0:
                if cell["n_stations"] < 5:
                    assert cell["low_confidence"] is True
                else:
                    assert cell["low_confidence"] is False

    def test_empty_cells_handled(self):
        """Cells with no data get NaN metrics and zero stations."""
        rng = np.random.default_rng(42)
        # Only DJF data at low elevation -> 11 cells empty
        obs = rng.normal(0, 1, size=50)
        pred = obs + 0.1
        elevations = np.full(50, 2500.0)
        months = np.full(50, 12)
        station_ids = [f"S{i}" for i in range(50)]

        result = compute_conditional_metrics(
            obs=obs,
            pred=pred,
            elevations=elevations,
            months=months,
            station_ids=station_ids,
        )
        # Check that empty cells have NaN RMSE
        empty_count = 0
        for cell in result.cells.values():
            if cell["n_stations"] == 0:
                assert math.isnan(cell["metrics"].rmse)
                empty_count += 1
        assert empty_count == 11  # Only 1 of 12 cells populated

    def test_custom_elevation_thresholds(self):
        """Custom elevation thresholds work correctly."""
        data = self._make_station_data()
        result = compute_conditional_metrics(
            obs=data["obs"],
            pred=data["pred"],
            elevations=data["elevations"],
            months=data["months"],
            station_ids=data["station_ids"],
            elevation_thresholds=(2600.0, 3200.0),
        )
        assert len(result.elevation_bands) == 3
        # Verify the band labels reflect custom thresholds
        assert "2600" in result.elevation_bands[0] or "2600" in result.elevation_bands[1]
