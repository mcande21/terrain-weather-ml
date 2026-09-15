"""Tests for baseline model runners.

Evaluation wrappers for:
(a) Raw HRRR nearest-grid-cell (no downscaling)
(b) WindNinja solver with HRRR boundary conditions
(c) granite-wxc-downscaling (IBM)
(d) Lapse-rate adjusted HRRR (Group 3)
(e) Station climatology (Group 3)
(f) Persistence (Group 3)
(g) Bias-corrected HRRR (Group 3)

All baselines produce metrics in the same format as the terrain-weather model.
"""

from __future__ import annotations

import numpy as np

from terrain_weather_ml.evaluation.baselines import (
    BaselineRunner,
    BiasCorrectedHRRRRunner,
    ClimatologyBaselineRunner,
    GraniteWxcRunner,
    HRRRBaselineRunner,
    LapseRateHRRRRunner,
    PersistenceBaselineRunner,
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


# ---------------------------------------------------------------------------
# Group 3: Extended baselines
# ---------------------------------------------------------------------------


class TestLapseRateHRRRBaseline:
    """Lapse-rate adjusted HRRR baseline (task 3.1)."""

    def test_temperature_correction_500m(self):
        """Station 500m above grid cell gets -3.25 K correction on temperature."""
        station = StationData(
            station_id="LAPSE_001",
            latitude=39.5,
            longitude=-106.0,
            elevation=3200.0,
            tpi=-10.0,
            observations={
                "temperature": np.array([280.0, 285.0, 270.0], dtype=np.float32),
                "wind_speed": np.array([5.0, 8.0, 3.0], dtype=np.float32),
            },
            timestamps=np.arange(3),
        )
        runner = LapseRateHRRRRunner(
            lapse_rate=6.5,
            grid_cell_elevations={"LAPSE_001": 2700.0},  # 500m below station
            noise_scale=0.0,  # isolate the correction
        )
        predictions = runner.predict_station(station)

        # Temperature should be corrected: obs + 0 noise - 3.25 K
        expected_correction = -(3200.0 - 2700.0) * 6.5 / 1000.0  # -3.25
        np.testing.assert_allclose(
            predictions["temperature"],
            station.observations["temperature"] + expected_correction,
            atol=1e-5,
        )

    def test_non_temperature_passthrough(self):
        """Non-temperature variables pass through without lapse correction."""
        station = StationData(
            station_id="LAPSE_001",
            latitude=39.5,
            longitude=-106.0,
            elevation=3200.0,
            tpi=-10.0,
            observations={
                "temperature": np.array([280.0], dtype=np.float32),
                "wind_speed": np.array([5.0], dtype=np.float32),
                "precipitation": np.array([1.0], dtype=np.float32),
            },
            timestamps=np.arange(1),
        )
        runner = LapseRateHRRRRunner(
            lapse_rate=6.5,
            grid_cell_elevations={"LAPSE_001": 2700.0},
            noise_scale=0.0,
        )
        predictions = runner.predict_station(station)

        # Wind and precip should be identical to obs (no correction, no noise)
        np.testing.assert_allclose(predictions["wind_speed"], [5.0], atol=1e-5)
        np.testing.assert_allclose(predictions["precipitation"], [1.0], atol=1e-5)

    def test_configurable_lapse_rate(self):
        """Lapse rate can be set to non-default value (e.g., 6.1 K/km)."""
        station = StationData(
            station_id="LAPSE_002",
            latitude=39.5,
            longitude=-106.0,
            elevation=3000.0,
            tpi=0.0,
            observations={
                "temperature": np.array([280.0], dtype=np.float32),
            },
            timestamps=np.arange(1),
        )
        runner = LapseRateHRRRRunner(
            lapse_rate=6.1,
            grid_cell_elevations={"LAPSE_002": 2000.0},  # 1000m diff
            noise_scale=0.0,
        )
        predictions = runner.predict_station(station)

        expected_correction = -(3000.0 - 2000.0) * 6.1 / 1000.0  # -6.1
        np.testing.assert_allclose(
            predictions["temperature"],
            station.observations["temperature"] + expected_correction,
            atol=1e-5,
        )

    def test_baseline_name(self):
        runner = LapseRateHRRRRunner()
        assert runner.name == "hrrr_lapse_rate"

    def test_is_baseline_runner(self):
        runner = LapseRateHRRRRunner()
        assert isinstance(runner, BaselineRunner)

    def test_metrics_format(self):
        station = _make_station()
        runner = LapseRateHRRRRunner(
            grid_cell_elevations={"TEST_001": 3000.0},
        )
        metrics = runner.evaluate_station(station)
        for var in REQUIRED_VARIABLES:
            assert var in metrics
            assert REQUIRED_METRIC_KEYS.issubset(set(metrics[var].keys()))


class TestClimatologyBaseline:
    """Station climatology baseline (task 3.2)."""

    def test_predictions_match_known_monthly_means(self):
        """Climatology predictions match fitted monthly means."""
        # Training data: station with known observations in months 1 and 7
        train_station = StationData(
            station_id="CLIM_001",
            latitude=39.5,
            longitude=-106.0,
            elevation=3000.0,
            tpi=0.0,
            observations={
                "temperature": np.array(
                    [260.0, 262.0, 264.0, 290.0, 292.0], dtype=np.float32,
                ),
            },
            timestamps=np.arange(5),
        )
        # First 3 obs are January (month 1), last 2 are July (month 7)
        train_months = {"CLIM_001": np.array([1, 1, 1, 7, 7])}

        runner = ClimatologyBaselineRunner()
        runner.fit([train_station], train_months)

        # Predict for a station in Jan and Jul
        eval_station = StationData(
            station_id="CLIM_001",
            latitude=39.5,
            longitude=-106.0,
            elevation=3000.0,
            tpi=0.0,
            observations={"temperature": np.zeros(2, dtype=np.float32)},
            timestamps=np.arange(2),
        )
        runner.set_predict_months({"CLIM_001": np.array([1, 7])})
        predictions = runner.predict_station(eval_station)

        expected_jan = np.mean([260.0, 262.0, 264.0])  # 262.0
        expected_jul = np.mean([290.0, 292.0])  # 291.0
        np.testing.assert_allclose(
            predictions["temperature"], [expected_jan, expected_jul], atol=1e-5,
        )

    def test_nan_for_missing_months(self):
        """NaN produced for months not present in training data."""
        train_station = StationData(
            station_id="CLIM_002",
            latitude=39.5,
            longitude=-106.0,
            elevation=3000.0,
            tpi=0.0,
            observations={
                "temperature": np.array([260.0, 262.0], dtype=np.float32),
            },
            timestamps=np.arange(2),
        )
        train_months = {"CLIM_002": np.array([1, 1])}  # only January

        runner = ClimatologyBaselineRunner()
        runner.fit([train_station], train_months)

        eval_station = StationData(
            station_id="CLIM_002",
            latitude=39.5,
            longitude=-106.0,
            elevation=3000.0,
            tpi=0.0,
            observations={"temperature": np.zeros(2, dtype=np.float32)},
            timestamps=np.arange(2),
        )
        runner.set_predict_months({"CLIM_002": np.array([1, 6])})
        predictions = runner.predict_station(eval_station)

        # January should have a value, June should be NaN
        assert not np.isnan(predictions["temperature"][0])
        assert np.isnan(predictions["temperature"][1])

    def test_baseline_name(self):
        runner = ClimatologyBaselineRunner()
        assert runner.name == "climatology"

    def test_is_baseline_runner(self):
        runner = ClimatologyBaselineRunner()
        assert isinstance(runner, BaselineRunner)


class TestPersistenceBaseline:
    """Persistence baseline (task 3.3)."""

    def test_24h_persistence(self):
        """Prediction equals observation 24 steps ago."""
        n = 48
        obs = np.arange(n, dtype=np.float32)
        station = StationData(
            station_id="PERS_001",
            latitude=39.5,
            longitude=-106.0,
            elevation=3000.0,
            tpi=0.0,
            observations={"temperature": obs},
            timestamps=np.arange(n),
        )
        runner = PersistenceBaselineRunner(lookback_steps=24)
        predictions = runner.predict_station(station)

        # Steps 24..47 should equal obs at steps 0..23
        np.testing.assert_allclose(
            predictions["temperature"][24:],
            obs[:24],
            atol=1e-5,
        )
        # First 24 steps have no 24h-prior data, should be NaN
        assert np.all(np.isnan(predictions["temperature"][:24]))

    def test_lookback_on_missing_value(self):
        """Falls back to t-48h when t-24h is NaN."""
        n = 72
        obs = np.arange(n, dtype=np.float32)
        # Make t=24 NaN so prediction at t=48 must look back to t=24 (NaN)
        # then to t=23 (not aligned). Actually: for t=48, lookback=24 -> t=24
        # which is NaN, so look at t=23, t=22... down to max_lookback
        # Better: specific test with known gap
        obs[24] = np.nan  # Missing value at step 24

        station = StationData(
            station_id="PERS_002",
            latitude=39.5,
            longitude=-106.0,
            elevation=3000.0,
            tpi=0.0,
            observations={"temperature": obs},
            timestamps=np.arange(n),
        )
        runner = PersistenceBaselineRunner(lookback_steps=24, max_lookback_steps=72)
        predictions = runner.predict_station(station)

        # At t=48, primary lookback t=24 is NaN -> fall back to t=23
        np.testing.assert_allclose(
            predictions["temperature"][48], 23.0, atol=1e-5,
        )

    def test_nan_when_no_valid_within_72h(self):
        """NaN when no valid observation exists within max lookback window."""
        n = 96
        obs = np.full(n, np.nan, dtype=np.float32)
        # Only first value is valid
        obs[0] = 280.0

        station = StationData(
            station_id="PERS_003",
            latitude=39.5,
            longitude=-106.0,
            elevation=3000.0,
            tpi=0.0,
            observations={"temperature": obs},
            timestamps=np.arange(n),
        )
        runner = PersistenceBaselineRunner(lookback_steps=24, max_lookback_steps=72)
        predictions = runner.predict_station(station)

        # At t=95, lookback window is t=71..t=95, all NaN -> NaN
        assert np.isnan(predictions["temperature"][95])
        # At t=24, primary lookback t=0 is valid -> 280.0
        np.testing.assert_allclose(
            predictions["temperature"][24], 280.0, atol=1e-5,
        )

    def test_baseline_name(self):
        runner = PersistenceBaselineRunner()
        assert runner.name == "persistence"

    def test_is_baseline_runner(self):
        runner = PersistenceBaselineRunner()
        assert isinstance(runner, BaselineRunner)


class TestBiasCorrectedHRRRBaseline:
    """Bias-corrected HRRR baseline (task 3.4)."""

    def test_removes_known_bias(self):
        """Bias correction removes systematic offset."""
        # Training: HRRR consistently 5K too warm in January
        train_station = StationData(
            station_id="BIAS_001",
            latitude=39.5,
            longitude=-106.0,
            elevation=3000.0,
            tpi=0.0,
            observations={
                "temperature": np.array(
                    [260.0, 262.0, 264.0], dtype=np.float32,
                ),
            },
            timestamps=np.arange(3),
        )
        # HRRR predictions during training (all 5K too warm)
        train_hrrr = {
            "BIAS_001": {
                "temperature": np.array(
                    [265.0, 267.0, 269.0], dtype=np.float32,
                ),
            },
        }
        train_months = {"BIAS_001": np.array([1, 1, 1])}

        runner = BiasCorrectedHRRRRunner()
        runner.fit([train_station], train_hrrr, train_months)

        # Evaluation: HRRR says 270 in January, bias is +5.0
        # Corrected = 270 - 5.0 = 265
        eval_station = StationData(
            station_id="BIAS_001",
            latitude=39.5,
            longitude=-106.0,
            elevation=3000.0,
            tpi=0.0,
            observations={
                "temperature": np.array([265.0], dtype=np.float32),
            },
            timestamps=np.arange(1),
        )
        runner.set_predict_months({"BIAS_001": np.array([1])})
        runner.set_predict_hrrr({
            "BIAS_001": {
                "temperature": np.array([270.0], dtype=np.float32),
            },
        })
        predictions = runner.predict_station(eval_station)

        # bias = mean(HRRR - obs) = mean([5, 5, 5]) = 5.0
        # corrected = 270 - 5 = 265
        np.testing.assert_allclose(
            predictions["temperature"], [265.0], atol=1e-5,
        )

    def test_per_month_correction(self):
        """Bias correction differs by month."""
        train_station = StationData(
            station_id="BIAS_002",
            latitude=39.5,
            longitude=-106.0,
            elevation=3000.0,
            tpi=0.0,
            observations={
                "temperature": np.array(
                    [260.0, 290.0], dtype=np.float32,
                ),
            },
            timestamps=np.arange(2),
        )
        train_hrrr = {
            "BIAS_002": {
                "temperature": np.array(
                    [263.0, 288.0], dtype=np.float32,  # +3 in Jan, -2 in Jul
                ),
            },
        }
        train_months = {"BIAS_002": np.array([1, 7])}

        runner = BiasCorrectedHRRRRunner()
        runner.fit([train_station], train_hrrr, train_months)

        eval_station = StationData(
            station_id="BIAS_002",
            latitude=39.5,
            longitude=-106.0,
            elevation=3000.0,
            tpi=0.0,
            observations={
                "temperature": np.array([261.0, 291.0], dtype=np.float32),
            },
            timestamps=np.arange(2),
        )
        runner.set_predict_months({"BIAS_002": np.array([1, 7])})
        runner.set_predict_hrrr({
            "BIAS_002": {
                "temperature": np.array([264.0, 289.0], dtype=np.float32),
            },
        })
        predictions = runner.predict_station(eval_station)

        # Jan: HRRR=264, bias=3.0 -> corrected = 261
        # Jul: HRRR=289, bias=-2.0 -> corrected = 291
        np.testing.assert_allclose(
            predictions["temperature"], [261.0, 291.0], atol=1e-5,
        )

    def test_baseline_name(self):
        runner = BiasCorrectedHRRRRunner()
        assert runner.name == "hrrr_bias_corrected"

    def test_is_baseline_runner(self):
        runner = BiasCorrectedHRRRRunner()
        assert isinstance(runner, BaselineRunner)


class TestExtendedBaselineConsistency:
    """All extended baselines conform to BaselineRunner interface."""

    def test_all_extended_baselines_are_baseline_runners(self):
        runners = [
            LapseRateHRRRRunner(),
            ClimatologyBaselineRunner(),
            PersistenceBaselineRunner(),
            BiasCorrectedHRRRRunner(),
        ]
        for runner in runners:
            assert isinstance(runner, BaselineRunner)
            assert hasattr(runner, "name")
            assert hasattr(runner, "predict_station")
            assert hasattr(runner, "evaluate_station")
