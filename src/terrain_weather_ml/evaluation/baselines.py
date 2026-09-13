"""Baseline model runners for evaluation comparison (task 7.4).

Evaluation wrappers for:
(a) Raw HRRR nearest-grid-cell (no downscaling)
(b) WindNinja solver with HRRR boundary conditions
(c) granite-wxc-downscaling (IBM)

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
