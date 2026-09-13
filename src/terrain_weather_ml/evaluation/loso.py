"""Leave-one-station-out cross-validation framework (task 7.1).

Implements LOSO cross-validation over SNOTEL stations. For each fold,
one station is held out and the model is evaluated on its observations.
Supports single-station mode and produces per-station + aggregate metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StationData:
    """Station observation data for evaluation.

    Attributes:
        station_id: Unique station identifier.
        latitude: Station latitude (WGS84).
        longitude: Station longitude (WGS84).
        elevation: Station elevation in meters.
        tpi: Topographic position index (meters). Negative = valley.
        observations: Dict of variable name -> 1D array of observations.
        timestamps: 1D array of timestamp indices or values.
    """

    station_id: str
    latitude: float
    longitude: float
    elevation: float
    tpi: float
    observations: dict[str, np.ndarray]
    timestamps: np.ndarray


class PredictionModel(Protocol):
    """Protocol for models that can produce station predictions."""

    def predict(self, weather_input: Any, terrain: Any) -> Any: ...


@dataclass
class LOSOConfig:
    """Configuration for LOSO cross-validation.

    Attributes:
        stations: List of station data for evaluation.
    """

    stations: list[StationData]


@dataclass
class LOSOResult:
    """Result of LOSO cross-validation.

    Attributes:
        per_station: Dict of station_id -> metrics dict.
        aggregate: Aggregate statistics (mean, median, std) across stations.
    """

    per_station: dict[str, dict[str, Any]] = field(default_factory=dict)
    aggregate: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON export."""
        return {
            "per_station": {
                sid: _metrics_to_serializable(m)
                for sid, m in self.per_station.items()
            },
            "aggregate": {
                stat_name: _metrics_to_serializable(m)
                for stat_name, m in self.aggregate.items()
            },
        }


def _metrics_to_serializable(metrics: dict[str, Any]) -> dict[str, Any]:
    """Convert metric values to JSON-serializable types."""
    result = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            result[k] = _metrics_to_serializable(v)
        elif isinstance(v, (np.floating, np.integer)):
            result[k] = float(v)
        elif isinstance(v, np.ndarray):
            result[k] = v.tolist()
        else:
            result[k] = v
    return result


def _compute_station_metrics(
    observations: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Compute per-variable metrics for a single station.

    For each variable present in both observations and predictions,
    computes RMSE, MAE, bias, and R-squared.

    Missing observations (NaN) are excluded from metric computation.
    """
    metrics: dict[str, dict[str, float]] = {}

    for var_name, obs in observations.items():
        if var_name not in predictions:
            continue

        pred = predictions[var_name]

        # Exclude missing values
        valid = ~(np.isnan(obs) | np.isnan(pred))
        if valid.sum() < 2:
            continue

        obs_v = obs[valid]
        pred_v = pred[valid]

        errors = pred_v - obs_v
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        mae = float(np.mean(np.abs(errors)))
        bias = float(np.mean(errors))

        # R-squared
        ss_res = float(np.sum(errors ** 2))
        ss_tot = float(np.sum((obs_v - np.mean(obs_v)) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        metrics[var_name] = {
            "rmse": rmse,
            "mae": mae,
            "bias": bias,
            "r_squared": r_squared,
        }

    return metrics


class LOSOEvaluator:
    """Leave-one-station-out cross-validation evaluator.

    For each fold, one station is held out and the model is evaluated
    against that station's observations. The remaining stations define
    the training context (available but not necessarily used by the
    evaluator -- the model is assumed pre-trained).

    Args:
        config: LOSO configuration with station data.
    """

    def __init__(self, config: LOSOConfig):
        self.config = config
        self._station_index = {
            s.station_id: s for s in config.stations
        }

    def evaluate(
        self,
        model: Any,
        station_id: str | None = None,
    ) -> LOSOResult:
        """Run LOSO evaluation.

        Args:
            model: Model with a predict() method.
            station_id: If provided, evaluate only this station's fold.

        Returns:
            LOSOResult with per-station and aggregate metrics.

        Raises:
            ValueError: If station_id is not found in the dataset.
        """
        if station_id is not None:
            if station_id not in self._station_index:
                raise ValueError(
                    f"Station '{station_id}' not found. "
                    f"Available: {list(self._station_index.keys())}"
                )
            stations_to_eval = [self._station_index[station_id]]
        else:
            stations_to_eval = self.config.stations

        per_station: dict[str, dict[str, Any]] = {}

        for station in stations_to_eval:
            logger.info("Evaluating fold: hold out %s", station.station_id)
            predictions = self._predict_station(model, station)
            metrics = _compute_station_metrics(
                station.observations, predictions
            )
            per_station[station.station_id] = metrics

        aggregate = self._compute_aggregate(per_station)

        return LOSOResult(
            per_station=per_station,
            aggregate=aggregate,
        )

    def _predict_station(
        self,
        model: Any,
        station: StationData,
    ) -> dict[str, np.ndarray]:
        """Generate predictions for a held-out station.

        The model's predict method is called and raw output is mapped
        to per-variable arrays matching the station's observation
        variables.

        For now, predictions are generated as random values with the
        same shape as observations. Real implementation will call the
        model with terrain/weather inputs constructed from the station
        location.
        """
        import torch

        n_timesteps = len(station.timestamps)

        # Create synthetic terrain/weather inputs for the model
        terrain = torch.randn(1, 17, 16, 16)
        weather = torch.randn(1, 6, 4, 4)

        # Get model output
        with torch.no_grad():
            output = model.predict(weather, terrain)

        # Map output channels to variables
        # Output: (B, 4, H, W) -> u_wind, v_wind, temperature, precipitation
        # Extract point prediction (center of the grid)
        h_mid = output.shape[2] // 2
        w_mid = output.shape[3] // 2

        u_wind = output[0, 0, h_mid, w_mid].item()
        v_wind = output[0, 1, h_mid, w_mid].item()
        temp = output[0, 2, h_mid, w_mid].item()
        precip = output[0, 3, h_mid, w_mid].item()

        # Repeat point prediction across timesteps (simplistic for now)
        rng = np.random.default_rng(hash(station.station_id) % (2**32))
        predictions: dict[str, np.ndarray] = {}

        if "wind_speed" in station.observations:
            speed = np.sqrt(u_wind**2 + v_wind**2)
            predictions["wind_speed"] = (
                np.full(n_timesteps, speed)
                + rng.normal(0, 1, n_timesteps).astype(np.float32)
            )

        if "wind_direction" in station.observations:
            direction = (np.degrees(np.arctan2(u_wind, v_wind)) + 180) % 360
            predictions["wind_direction"] = (
                np.full(n_timesteps, direction)
                + rng.normal(0, 10, n_timesteps).astype(np.float32)
            ) % 360

        if "temperature" in station.observations:
            predictions["temperature"] = (
                np.full(n_timesteps, temp * 10 + 275)
                + rng.normal(0, 2, n_timesteps).astype(np.float32)
            )

        if "precipitation" in station.observations:
            predictions["precipitation"] = np.abs(
                np.full(n_timesteps, abs(precip))
                + rng.normal(0, 0.3, n_timesteps).astype(np.float32)
            )

        return predictions

    def _compute_aggregate(
        self,
        per_station: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Compute aggregate statistics across all stations.

        Returns mean, median, and std of each metric across stations.
        """
        if not per_station:
            return {"mean": {}, "median": {}, "std": {}}

        # Collect all (variable, metric) pairs
        all_vars: set[str] = set()
        all_metrics: set[str] = set()
        for station_metrics in per_station.values():
            for var_name, var_metrics in station_metrics.items():
                all_vars.add(var_name)
                if isinstance(var_metrics, dict):
                    all_metrics.update(var_metrics.keys())

        mean_agg: dict[str, dict[str, float]] = {}
        median_agg: dict[str, dict[str, float]] = {}
        std_agg: dict[str, dict[str, float]] = {}

        for var_name in sorted(all_vars):
            mean_agg[var_name] = {}
            median_agg[var_name] = {}
            std_agg[var_name] = {}

            for metric_name in sorted(all_metrics):
                values = []
                for station_metrics in per_station.values():
                    if var_name in station_metrics:
                        var_m = station_metrics[var_name]
                        if isinstance(var_m, dict) and metric_name in var_m:
                            values.append(var_m[metric_name])

                if values:
                    arr = np.array(values)
                    mean_agg[var_name][metric_name] = float(np.mean(arr))
                    median_agg[var_name][metric_name] = float(np.median(arr))
                    std_agg[var_name][metric_name] = float(np.std(arr))

        return {
            "mean": mean_agg,
            "median": median_agg,
            "std": std_agg,
        }
