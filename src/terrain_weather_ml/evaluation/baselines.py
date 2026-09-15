"""Baseline model runners for evaluation comparison.

Evaluation wrappers for:
(a) Raw HRRR nearest-grid-cell (no downscaling)
(b) WindNinja solver with HRRR boundary conditions
(c) granite-wxc-downscaling (IBM)
(d) Lapse-rate adjusted HRRR (Group 3)
(e) Station climatology (Group 3)
(f) Persistence (Group 3)
(g) Bias-corrected HRRR (Group 3)

All baselines produce predictions and metrics in the same format as the
terrain-weather model, enabling direct per-station and aggregate comparison.

In mock/test mode, baselines generate synthetic predictions that add
calibrated noise to the observations. Real implementations would call
the actual model APIs or solvers.
"""

from __future__ import annotations

import logging

import numpy as np

from terrain_weather_ml.evaluation.loso import StationData
from terrain_weather_ml.evaluation.metrics import (
    compute_continuous_metrics,
)

logger = logging.getLogger(__name__)


class BaselineRunner:
    """Base class for baseline model evaluation runners.

    Subclasses must implement predict_station() to generate predictions
    for a given station. The evaluate_station() method computes metrics
    from predictions vs observations.

    Attributes:
        name: Short identifier for the baseline model.
    """

    name: str = "base"

    def predict_station(
        self, station: StationData
    ) -> dict[str, np.ndarray]:
        """Generate predictions for a station.

        Args:
            station: Station data with observations and metadata.

        Returns:
            Dict of variable name -> predicted values array,
            same length as station.timestamps.
        """
        raise NotImplementedError

    def evaluate_station(
        self, station: StationData
    ) -> dict[str, dict[str, float]]:
        """Evaluate baseline against a station's observations.

        Args:
            station: Station data with observations.

        Returns:
            Dict of variable -> {rmse, mae, bias, r_squared}.
        """
        predictions = self.predict_station(station)
        metrics: dict[str, dict[str, float]] = {}

        for var_name, obs in station.observations.items():
            if var_name not in predictions:
                continue
            pred = predictions[var_name]
            m = compute_continuous_metrics(obs, pred)
            metrics[var_name] = {
                "rmse": m.rmse,
                "mae": m.mae,
                "bias": m.bias,
                "r_squared": m.r_squared,
            }

        return metrics


class HRRRBaselineRunner(BaselineRunner):
    """Raw HRRR nearest-grid-cell baseline (no downscaling).

    In production, this would extract HRRR values at the nearest
    grid cell to each station. For testing, generates synthetic
    predictions with coarse-model-typical noise levels.
    """

    name = "hrrr_raw"

    def __init__(self, noise_scale: float = 3.0, seed: int = 100):
        self._noise_scale = noise_scale
        self._seed = seed

    def predict_station(
        self, station: StationData
    ) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(
            self._seed + hash(station.station_id) % (2**31)
        )
        n = len(station.timestamps)
        predictions: dict[str, np.ndarray] = {}

        for var_name, obs in station.observations.items():
            noise = rng.normal(0, self._noise_scale, n).astype(np.float32)
            predictions[var_name] = obs + noise

        return predictions


class WindNinjaRunner(BaselineRunner):
    """WindNinja solver baseline with HRRR boundary conditions.

    In production, this would run the WindNinja solver for each station
    location using HRRR fields as boundary conditions. For testing,
    generates synthetic predictions with intermediate noise levels
    (better than raw HRRR for wind, similar for other variables).
    """

    name = "windninja"

    def __init__(self, noise_scale: float = 2.0, seed: int = 200):
        self._noise_scale = noise_scale
        self._seed = seed

    def predict_station(
        self, station: StationData
    ) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(
            self._seed + hash(station.station_id) % (2**31)
        )
        n = len(station.timestamps)
        predictions: dict[str, np.ndarray] = {}

        for var_name, obs in station.observations.items():
            # WindNinja is better for wind variables
            scale = self._noise_scale
            if var_name in ("wind_speed", "wind_direction"):
                scale *= 0.6
            noise = rng.normal(0, scale, n).astype(np.float32)
            predictions[var_name] = obs + noise

        return predictions


class GraniteWxcRunner(BaselineRunner):
    """IBM granite-wxc-downscaling baseline.

    In production, this would run the granite-geospatial-wxc-downscaling
    model. For testing, generates synthetic predictions with
    downscaling-model-typical noise levels.
    """

    name = "granite_wxc"

    def __init__(self, noise_scale: float = 1.5, seed: int = 300):
        self._noise_scale = noise_scale
        self._seed = seed

    def predict_station(
        self, station: StationData
    ) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(
            self._seed + hash(station.station_id) % (2**31)
        )
        n = len(station.timestamps)
        predictions: dict[str, np.ndarray] = {}

        for var_name, obs in station.observations.items():
            noise = rng.normal(0, self._noise_scale, n).astype(np.float32)
            predictions[var_name] = obs + noise

        return predictions


# ---------------------------------------------------------------------------
# Group 3: Extended baselines
# ---------------------------------------------------------------------------


class LapseRateHRRRRunner(BaselineRunner):
    """Lapse-rate adjusted HRRR baseline.

    Corrects raw HRRR temperature at the nearest grid cell using the
    elevation difference between the HRRR grid cell and the station.
    Non-temperature variables pass through the raw HRRR value unchanged.

    The correction is: T_corrected = T_hrrr - (elev_station - elev_grid) * rate / 1000

    In mock/test mode, HRRR values are simulated as obs + noise (identical
    to HRRRBaselineRunner). The lapse correction is always applied.

    Attributes:
        name: "hrrr_lapse_rate"
    """

    name = "hrrr_lapse_rate"

    def __init__(
        self,
        lapse_rate: float = 6.5,
        grid_cell_elevations: dict[str, float] | None = None,
        noise_scale: float = 3.0,
        seed: int = 400,
    ):
        """Initialize lapse-rate HRRR baseline.

        Args:
            lapse_rate: Environmental lapse rate in K/km (default 6.5,
                Schwartzreich uses 6.1).
            grid_cell_elevations: Mapping of station_id to the HRRR grid
                cell elevation (meters). If a station is missing, its own
                elevation is used (zero correction).
            noise_scale: Std dev of synthetic HRRR noise (mock mode).
            seed: RNG seed for reproducibility.
        """
        self._lapse_rate = lapse_rate
        self._grid_cell_elevations = grid_cell_elevations or {}
        self._noise_scale = noise_scale
        self._seed = seed

    def predict_station(
        self, station: StationData
    ) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(
            self._seed + hash(station.station_id) % (2**31)
        )
        n = len(station.timestamps)
        predictions: dict[str, np.ndarray] = {}

        grid_elev = self._grid_cell_elevations.get(
            station.station_id, station.elevation
        )
        elev_diff_m = station.elevation - grid_elev

        for var_name, obs in station.observations.items():
            noise = rng.normal(0, self._noise_scale, n).astype(np.float32)
            pred = obs + noise

            if var_name == "temperature":
                correction = -(elev_diff_m * self._lapse_rate / 1000.0)
                pred = pred + np.float32(correction)

            predictions[var_name] = pred

        return predictions


class ClimatologyBaselineRunner(BaselineRunner):
    """Station climatology baseline.

    Predicts the monthly training-period mean for each station, variable,
    and calendar month. Requires a fit step on training data before
    prediction.

    Usage:
        runner = ClimatologyBaselineRunner()
        runner.fit(train_stations, train_months)
        runner.set_predict_months(eval_months)
        predictions = runner.predict_station(eval_station)

    Attributes:
        name: "climatology"
    """

    name = "climatology"

    def __init__(self) -> None:
        # station_id -> variable -> month (1-12) -> mean value
        self._climatology: dict[str, dict[str, dict[int, float]]] = {}
        # station_id -> month array for prediction timestamps
        self._predict_months: dict[str, np.ndarray] = {}

    def fit(
        self,
        stations: list[StationData],
        months: dict[str, np.ndarray],
    ) -> None:
        """Compute monthly climatology from training data.

        Args:
            stations: Training station data.
            months: Mapping station_id -> array of calendar months (1-12),
                same length as each station's timestamps.
        """
        for station in stations:
            station_months = months[station.station_id]
            var_clim: dict[str, dict[int, float]] = {}
            for var_name, obs in station.observations.items():
                monthly_means: dict[int, float] = {}
                for m in range(1, 13):
                    mask = station_months == m
                    if not mask.any():
                        continue
                    valid = obs[mask]
                    valid = valid[~np.isnan(valid)]
                    if len(valid) > 0:
                        monthly_means[m] = float(np.mean(valid))
                var_clim[var_name] = monthly_means
            self._climatology[station.station_id] = var_clim

    def set_predict_months(
        self, months: dict[str, np.ndarray]
    ) -> None:
        """Set month arrays for evaluation timestamps.

        Args:
            months: Mapping station_id -> array of calendar months (1-12)
                for the prediction period.
        """
        self._predict_months = months

    def predict_station(
        self, station: StationData
    ) -> dict[str, np.ndarray]:
        months = self._predict_months.get(station.station_id)
        if months is None:
            raise ValueError(
                f"No predict months set for station {station.station_id}. "
                "Call set_predict_months() before predict_station()."
            )

        clim = self._climatology.get(station.station_id, {})
        predictions: dict[str, np.ndarray] = {}

        for var_name in station.observations:
            var_clim = clim.get(var_name, {})
            pred = np.full(len(station.timestamps), np.nan, dtype=np.float32)
            for i, m in enumerate(months):
                mean_val = var_clim.get(int(m))
                if mean_val is not None:
                    pred[i] = mean_val
            predictions[var_name] = pred

        return predictions


class PersistenceBaselineRunner(BaselineRunner):
    """Persistence baseline (yesterday's observation as today's prediction).

    For each timestep, predicts the value from ``lookback_steps`` steps
    prior (default 24 for hourly data = 24-hour persistence). When the
    lookback value is NaN, searches backward up to ``max_lookback_steps``
    for the most recent valid observation.

    Attributes:
        name: "persistence"
    """

    name = "persistence"

    def __init__(
        self,
        lookback_steps: int = 24,
        max_lookback_steps: int = 72,
    ):
        """Initialize persistence baseline.

        Args:
            lookback_steps: Primary lookback offset in timesteps
                (default 24 for hourly -> 24h persistence).
            max_lookback_steps: Maximum lookback window when the primary
                offset value is missing (default 72 for 72h).
        """
        self._lookback = lookback_steps
        self._max_lookback = max_lookback_steps

    def predict_station(
        self, station: StationData
    ) -> dict[str, np.ndarray]:
        predictions: dict[str, np.ndarray] = {}

        for var_name, obs in station.observations.items():
            n = len(obs)
            pred = np.full(n, np.nan, dtype=np.float32)

            for i in range(n):
                primary = i - self._lookback
                if primary < 0:
                    # No data point at primary offset -> NaN
                    continue

                if not np.isnan(obs[primary]):
                    pred[i] = obs[primary]
                    continue

                # Primary is NaN -> search backward from primary
                earliest = max(0, i - self._max_lookback)
                for j in range(primary - 1, earliest - 1, -1):
                    if not np.isnan(obs[j]):
                        pred[i] = obs[j]
                        break

            predictions[var_name] = pred

        return predictions


class BiasCorrectedHRRRRunner(BaselineRunner):
    """Bias-corrected HRRR baseline.

    Subtracts the per-station, per-variable, per-month mean difference
    (HRRR - observation) computed over the training period from each
    HRRR prediction. The simplest form of model output statistics (MOS).

    Usage:
        runner = BiasCorrectedHRRRRunner()
        runner.fit(train_stations, train_hrrr, train_months)
        runner.set_predict_months(eval_months)
        runner.set_predict_hrrr(eval_hrrr)
        predictions = runner.predict_station(eval_station)

    Attributes:
        name: "hrrr_bias_corrected"
    """

    name = "hrrr_bias_corrected"

    def __init__(self) -> None:
        # station_id -> variable -> month (1-12) -> mean bias
        self._bias: dict[str, dict[str, dict[int, float]]] = {}
        self._predict_months: dict[str, np.ndarray] = {}
        # station_id -> variable -> hrrr prediction array for eval period
        self._predict_hrrr: dict[str, dict[str, np.ndarray]] = {}

    def fit(
        self,
        stations: list[StationData],
        hrrr_predictions: dict[str, dict[str, np.ndarray]],
        months: dict[str, np.ndarray],
    ) -> None:
        """Compute per-station, per-variable, per-month bias from training.

        Args:
            stations: Training station data with observations.
            hrrr_predictions: station_id -> variable -> HRRR prediction
                array for the training period.
            months: station_id -> array of calendar months (1-12).
        """
        for station in stations:
            sid = station.station_id
            station_months = months[sid]
            station_hrrr = hrrr_predictions.get(sid, {})
            var_bias: dict[str, dict[int, float]] = {}

            for var_name, obs in station.observations.items():
                hrrr_vals = station_hrrr.get(var_name)
                if hrrr_vals is None:
                    continue
                monthly_bias: dict[int, float] = {}
                for m in range(1, 13):
                    mask = station_months == m
                    if not mask.any():
                        continue
                    h = hrrr_vals[mask]
                    o = obs[mask]
                    valid = ~(np.isnan(h) | np.isnan(o))
                    if valid.any():
                        monthly_bias[m] = float(np.mean(h[valid] - o[valid]))
                var_bias[var_name] = monthly_bias

            self._bias[sid] = var_bias

    def set_predict_months(
        self, months: dict[str, np.ndarray]
    ) -> None:
        """Set month arrays for evaluation timestamps.

        Args:
            months: Mapping station_id -> array of calendar months (1-12).
        """
        self._predict_months = months

    def set_predict_hrrr(
        self, hrrr_predictions: dict[str, dict[str, np.ndarray]],
    ) -> None:
        """Set raw HRRR predictions for the evaluation period.

        Args:
            hrrr_predictions: station_id -> variable -> HRRR prediction array.
        """
        self._predict_hrrr = hrrr_predictions

    def predict_station(
        self, station: StationData
    ) -> dict[str, np.ndarray]:
        sid = station.station_id
        months = self._predict_months.get(sid)
        if months is None:
            raise ValueError(
                f"No predict months set for station {sid}. "
                "Call set_predict_months() before predict_station()."
            )
        station_hrrr = self._predict_hrrr.get(sid, {})
        station_bias = self._bias.get(sid, {})
        predictions: dict[str, np.ndarray] = {}

        for var_name in station.observations:
            hrrr_vals = station_hrrr.get(var_name)
            if hrrr_vals is None:
                predictions[var_name] = np.full(
                    len(station.timestamps), np.nan, dtype=np.float32,
                )
                continue

            var_bias = station_bias.get(var_name, {})
            pred = hrrr_vals.copy().astype(np.float32)
            for i, m in enumerate(months):
                bias = var_bias.get(int(m))
                if bias is not None:
                    pred[i] = pred[i] - np.float32(bias)

            predictions[var_name] = pred

        return predictions
