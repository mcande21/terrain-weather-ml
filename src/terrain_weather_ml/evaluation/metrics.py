"""Evaluation metrics for terrain-weather models (task 7.2).

Per-variable metrics:
- Continuous: RMSE, MAE, bias, R-squared
- Circular: circular RMSE, circular bias (for wind direction)
- Categorical: frequency bias, POD, FAR, CSI, ETS (for precipitation)

Missing observations (NaN) are excluded from all computations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ContinuousMetrics:
    """Metrics for continuous variables (temperature, wind speed, pressure).

    Attributes:
        rmse: Root mean square error.
        mae: Mean absolute error.
        bias: Mean error (prediction - observation).
        r_squared: Coefficient of determination.
        n_valid: Number of valid (non-NaN) data points used.
    """

    rmse: float
    mae: float
    bias: float
    r_squared: float
    n_valid: int

    def to_dict(self) -> dict[str, float]:
        return {
            "rmse": self.rmse,
            "mae": self.mae,
            "bias": self.bias,
            "r_squared": self.r_squared,
            "n_valid": self.n_valid,
        }


@dataclass
class CircularMetrics:
    """Metrics for circular variables (wind direction in degrees).

    Attributes:
        circular_rmse: Root mean square of circular differences.
        circular_bias: Mean circular difference (signed, degrees).
        n_valid: Number of valid data points used.
    """

    circular_rmse: float
    circular_bias: float
    n_valid: int

    def to_dict(self) -> dict[str, float]:
        return {
            "circular_rmse": self.circular_rmse,
            "circular_bias": self.circular_bias,
            "n_valid": self.n_valid,
        }


@dataclass
class PrecipitationMetrics:
    """Categorical metrics for precipitation at a given threshold.

    Attributes:
        threshold: Precipitation threshold (mm/hr).
        pod: Probability of detection (hit rate).
        far: False alarm ratio.
        csi: Critical success index (threat score).
        ets: Equitable threat score (Gilbert skill score).
        frequency_bias: Frequency bias (forecast events / observed events).
        hits: Number of correct positive forecasts.
        misses: Number of missed events.
        false_alarms: Number of false alarm events.
        correct_negatives: Number of correct negative forecasts.
    """

    threshold: float
    pod: float
    far: float
    csi: float
    ets: float
    frequency_bias: float
    hits: int
    misses: int
    false_alarms: int
    correct_negatives: int

    def to_dict(self) -> dict[str, float]:
        return {
            "threshold": self.threshold,
            "pod": self.pod,
            "far": self.far,
            "csi": self.csi,
            "ets": self.ets,
            "frequency_bias": self.frequency_bias,
            "hits": self.hits,
            "misses": self.misses,
            "false_alarms": self.false_alarms,
            "correct_negatives": self.correct_negatives,
        }


def _get_valid_mask(
    obs: np.ndarray, pred: np.ndarray
) -> np.ndarray:
    """Return boolean mask of valid (non-NaN) pairs."""
    return ~(np.isnan(obs) | np.isnan(pred))


def compute_continuous_metrics(
    obs: np.ndarray,
    pred: np.ndarray,
) -> ContinuousMetrics:
    """Compute continuous metrics for a single variable.

    Args:
        obs: Observed values, 1D array.
        pred: Predicted values, 1D array (same length as obs).

    Returns:
        ContinuousMetrics with RMSE, MAE, bias, R-squared.
        Returns NaN metrics if fewer than 2 valid points.
    """
    valid = _get_valid_mask(obs, pred)
    n_valid = int(valid.sum())

    if n_valid < 2:
        return ContinuousMetrics(
            rmse=float("nan"),
            mae=float("nan"),
            bias=float("nan"),
            r_squared=float("nan"),
            n_valid=n_valid,
        )

    o = obs[valid]
    p = pred[valid]
    errors = p - o

    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))
    bias = float(np.mean(errors))

    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((o - np.mean(o)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return ContinuousMetrics(
        rmse=rmse,
        mae=mae,
        bias=bias,
        r_squared=r_squared,
        n_valid=n_valid,
    )


def _circular_difference(angle1: np.ndarray, angle2: np.ndarray) -> np.ndarray:
    """Signed circular difference in degrees, range [-180, 180].

    Returns angle1 - angle2 wrapped to [-180, 180].
    """
    diff = angle1 - angle2
    return (diff + 180.0) % 360.0 - 180.0


def compute_circular_metrics(
    obs: np.ndarray,
    pred: np.ndarray,
) -> CircularMetrics:
    """Compute circular metrics for wind direction.

    Args:
        obs: Observed directions in degrees [0, 360).
        pred: Predicted directions in degrees [0, 360).

    Returns:
        CircularMetrics with circular RMSE and circular bias.
        Returns NaN metrics if fewer than 2 valid points.
    """
    valid = _get_valid_mask(obs, pred)
    n_valid = int(valid.sum())

    if n_valid < 2:
        return CircularMetrics(
            circular_rmse=float("nan"),
            circular_bias=float("nan"),
            n_valid=n_valid,
        )

    o = obs[valid]
    p = pred[valid]

    # Signed circular differences: pred - obs
    diffs = _circular_difference(p, o)

    circular_rmse = float(np.sqrt(np.mean(diffs ** 2)))

    # Circular mean of the signed differences
    circular_bias = float(np.mean(diffs))

    return CircularMetrics(
        circular_rmse=circular_rmse,
        circular_bias=circular_bias,
        n_valid=n_valid,
    )


def compute_precipitation_metrics(
    obs: np.ndarray,
    pred: np.ndarray,
    threshold: float = 0.1,
) -> PrecipitationMetrics:
    """Compute categorical precipitation metrics at a threshold.

    Builds a 2x2 contingency table (observed yes/no vs predicted yes/no)
    and derives POD, FAR, CSI, ETS, and frequency bias.

    Args:
        obs: Observed precipitation values.
        pred: Predicted precipitation values.
        threshold: Precipitation threshold (mm/hr) for event detection.

    Returns:
        PrecipitationMetrics at the given threshold.
        POD and CSI are NaN if no observed events. FAR is NaN if no
        forecast events.
    """
    valid = _get_valid_mask(obs, pred)
    o = obs[valid]
    p = pred[valid]
    n_total = len(o)

    obs_event = o >= threshold
    pred_event = p >= threshold

    hits = int(np.sum(obs_event & pred_event))
    misses = int(np.sum(obs_event & ~pred_event))
    false_alarms = int(np.sum(~obs_event & pred_event))
    correct_negatives = int(np.sum(~obs_event & ~pred_event))

    # POD = hits / (hits + misses)
    pod = hits / (hits + misses) if (hits + misses) > 0 else float("nan")

    # FAR = false_alarms / (hits + false_alarms)
    far = (
        false_alarms / (hits + false_alarms)
        if (hits + false_alarms) > 0
        else float("nan")
    )

    # CSI = hits / (hits + misses + false_alarms)
    denom_csi = hits + misses + false_alarms
    csi = hits / denom_csi if denom_csi > 0 else float("nan")

    # ETS (Equitable Threat Score)
    # hits_random = (hits + misses) * (hits + false_alarms) / total
    if n_total > 0 and denom_csi > 0:
        hits_random = (
            (hits + misses) * (hits + false_alarms) / n_total
        )
        denom_ets = hits + misses + false_alarms - hits_random
        ets = (
            (hits - hits_random) / denom_ets
            if denom_ets > 0
            else float("nan")
        )
    else:
        ets = float("nan")

    # Frequency bias = (hits + false_alarms) / (hits + misses)
    frequency_bias = (
        (hits + false_alarms) / (hits + misses)
        if (hits + misses) > 0
        else float("nan")
    )

    return PrecipitationMetrics(
        threshold=threshold,
        pod=pod,
        far=far,
        csi=csi,
        ets=ets,
        frequency_bias=frequency_bias,
        hits=hits,
        misses=misses,
        false_alarms=false_alarms,
        correct_negatives=correct_negatives,
    )
