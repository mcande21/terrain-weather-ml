"""Evaluation metrics for terrain-weather models (task 7.2).

Per-variable metrics:
- Continuous: RMSE, MAE, bias, R-squared
- Circular: circular RMSE, circular bias (for wind direction)
- Categorical: frequency bias, POD, FAR, CSI, ETS (for precipitation)
- Skill scores: relative improvement over baselines
- Threshold crossing: POD/FAR/CSI for temperature zero-crossing events
- Accumulation: multi-day precipitation window accuracy
- Conditional: metrics stratified by elevation band and season

Missing observations (NaN) are excluded from all computations.
"""

from __future__ import annotations

from collections.abc import Sequence
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


# ---------------------------------------------------------------------------
# Group 4: Extended metrics
# ---------------------------------------------------------------------------


@dataclass
class ThresholdCrossingMetrics:
    """Metrics for temperature threshold crossing detection.

    A crossing occurs when the temperature transitions across the threshold
    between consecutive timesteps (either direction).

    Attributes:
        pod: Probability of detection for crossings.
        far: False alarm ratio for crossings.
        csi: Critical success index for crossings.
        n_observed_crossings: Number of observed crossing events.
        n_predicted_crossings: Number of predicted crossing events.
        hits: Crossings detected by both obs and pred.
        misses: Crossings in obs not captured by pred.
        false_alarms: Crossings in pred not present in obs.
    """

    pod: float
    far: float
    csi: float
    n_observed_crossings: int
    n_predicted_crossings: int
    hits: int
    misses: int
    false_alarms: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "pod": self.pod,
            "far": self.far,
            "csi": self.csi,
            "n_observed_crossings": self.n_observed_crossings,
            "n_predicted_crossings": self.n_predicted_crossings,
            "hits": self.hits,
            "misses": self.misses,
            "false_alarms": self.false_alarms,
        }


# Season lookup: month -> season abbreviation
_MONTH_TO_SEASON: dict[int, str] = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}

_SEASONS = ("DJF", "MAM", "JJA", "SON")

# Default elevation thresholds (meters) for Colorado treeline stratification
_DEFAULT_ELEV_THRESHOLDS = (2750.0, 3100.0)


@dataclass
class ConditionalMetricsResult:
    """Result of conditional metrics computation (elevation x season matrix).

    Attributes:
        elevation_bands: Ordered list of elevation band labels.
        seasons: Ordered list of season labels.
        cells: Dict mapping (band_label, season) -> cell dict with keys:
            'metrics' (ContinuousMetrics), 'n_stations' (int),
            'low_confidence' (bool).
        elevation_thresholds: The (low, high) thresholds used.
    """

    elevation_bands: list[str]
    seasons: list[str]
    cells: dict[tuple[str, str], dict]
    elevation_thresholds: tuple[float, float]


def compute_skill_score(
    model_rmse: float,
    baseline_rmse: float,
) -> float:
    """Compute skill score: 1 - RMSE_model / RMSE_baseline.

    A positive value means the model outperforms the baseline.
    Zero means equal. Negative means the model is worse.

    Args:
        model_rmse: RMSE of the model predictions.
        baseline_rmse: RMSE of the baseline predictions.

    Returns:
        Skill score as a float. NaN if baseline_rmse is zero.
    """
    if baseline_rmse == 0.0:
        return float("nan")
    return 1.0 - model_rmse / baseline_rmse


def _detect_crossings(
    series: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Detect threshold crossings between consecutive timesteps.

    A crossing occurs when consecutive values are on opposite sides of
    the threshold. Returns a boolean array of length len(series) - 1.
    """
    above = series > threshold
    below = series < threshold
    # Crossing: one above and next below, or one below and next above
    cross = (above[:-1] & below[1:]) | (below[:-1] & above[1:])
    return cross


def compute_threshold_crossing_accuracy(
    obs: np.ndarray,
    pred: np.ndarray,
    threshold: float = 0.0,
) -> ThresholdCrossingMetrics:
    """Evaluate accuracy of predicting threshold crossings (e.g. 0C).

    Detects when consecutive observed/predicted values cross the threshold
    and computes POD, FAR, CSI on crossing events.

    Args:
        obs: Observed temperature series, 1D array.
        pred: Predicted temperature series, 1D array (same length).
        threshold: Temperature threshold (default 0.0 for freezing).

    Returns:
        ThresholdCrossingMetrics with POD, FAR, CSI.
    """
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)

    if len(obs) < 2:
        return ThresholdCrossingMetrics(
            pod=float("nan"), far=float("nan"), csi=float("nan"),
            n_observed_crossings=0, n_predicted_crossings=0,
            hits=0, misses=0, false_alarms=0,
        )

    obs_cross = _detect_crossings(obs, threshold)
    pred_cross = _detect_crossings(pred, threshold)

    hits = int(np.sum(obs_cross & pred_cross))
    misses = int(np.sum(obs_cross & ~pred_cross))
    false_alarms = int(np.sum(~obs_cross & pred_cross))

    n_obs_cross = int(np.sum(obs_cross))
    n_pred_cross = int(np.sum(pred_cross))

    pod = hits / (hits + misses) if (hits + misses) > 0 else float("nan")
    far = (
        false_alarms / (hits + false_alarms)
        if (hits + false_alarms) > 0
        else float("nan")
    )
    denom_csi = hits + misses + false_alarms
    csi = hits / denom_csi if denom_csi > 0 else float("nan")

    return ThresholdCrossingMetrics(
        pod=pod,
        far=far,
        csi=csi,
        n_observed_crossings=n_obs_cross,
        n_predicted_crossings=n_pred_cross,
        hits=hits,
        misses=misses,
        false_alarms=false_alarms,
    )


def compute_accumulation_accuracy(
    obs: np.ndarray,
    pred: np.ndarray,
    window_hours: int = 72,
) -> dict[str, float | int]:
    """Compute precipitation accumulation accuracy over multi-day windows.

    Sums hourly precipitation into non-overlapping windows and computes
    RMSE, bias, and R-squared of the windowed totals.

    Args:
        obs: Observed hourly precipitation, 1D array.
        pred: Predicted hourly precipitation, 1D array (same length).
        window_hours: Window size in hours (72 for 3-day, 168 for 7-day).

    Returns:
        Dict with keys: 'rmse', 'bias', 'r_squared', 'n_windows'.
        NaN metrics if fewer than 1 complete window.
    """
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)

    n = len(obs)
    n_windows = n // window_hours

    if n_windows < 1:
        return {
            "rmse": float("nan"),
            "bias": float("nan"),
            "r_squared": float("nan"),
            "n_windows": 0,
        }

    # Reshape into (n_windows, window_hours) and sum each window
    usable = n_windows * window_hours
    obs_windows = obs[:usable].reshape(n_windows, window_hours).sum(axis=1)
    pred_windows = pred[:usable].reshape(n_windows, window_hours).sum(axis=1)

    errors = pred_windows - obs_windows
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    bias = float(np.mean(errors))

    if n_windows < 2:
        r_squared = float("nan")
    else:
        ss_res = float(np.sum(errors ** 2))
        ss_tot = float(np.sum((obs_windows - np.mean(obs_windows)) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "rmse": rmse,
        "bias": bias,
        "r_squared": r_squared,
        "n_windows": n_windows,
    }


def _assign_elevation_band(
    elevation: float,
    thresholds: tuple[float, float],
) -> str:
    """Assign an elevation to a labeled band.

    Returns a string label like 'below_2750', '2750_3100', 'above_3100'.
    """
    low, high = thresholds
    if elevation < low:
        return f"below_{low:.0f}"
    elif elevation <= high:
        return f"{low:.0f}_{high:.0f}"
    else:
        return f"above_{high:.0f}"


def compute_conditional_metrics(
    obs: np.ndarray,
    pred: np.ndarray,
    elevations: np.ndarray,
    months: np.ndarray,
    station_ids: Sequence[str],
    elevation_thresholds: tuple[float, float] = _DEFAULT_ELEV_THRESHOLDS,
    min_stations_confidence: int = 5,
) -> ConditionalMetricsResult:
    """Compute metrics stratified by elevation band and season.

    Produces a 3x4 matrix (3 elevation bands x 4 seasons) of
    ContinuousMetrics, with low-confidence flags for sparse cells.

    Args:
        obs: Observed values, 1D array.
        pred: Predicted values, 1D array (same length).
        elevations: Per-observation station elevation (meters).
        months: Per-observation month (1-12).
        station_ids: Per-observation station identifier.
        elevation_thresholds: (low, high) boundaries for 3 bands.
        min_stations_confidence: Minimum unique stations for confidence.

    Returns:
        ConditionalMetricsResult with the cell matrix.
    """
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    elevations = np.asarray(elevations, dtype=float)
    months = np.asarray(months, dtype=int)

    low, high = elevation_thresholds
    band_labels = [
        f"below_{low:.0f}",
        f"{low:.0f}_{high:.0f}",
        f"above_{high:.0f}",
    ]

    cells: dict[tuple[str, str], dict] = {}

    for band_label in band_labels:
        for season in _SEASONS:
            # Build mask for this cell
            if band_label.startswith("below_"):
                band_mask = elevations < low
            elif band_label.startswith("above_"):
                band_mask = elevations > high
            else:
                band_mask = (elevations >= low) & (elevations <= high)

            season_months = [
                m for m, s in _MONTH_TO_SEASON.items() if s == season
            ]
            season_mask = np.isin(months, season_months)

            cell_mask = band_mask & season_mask

            if not np.any(cell_mask):
                cells[(band_label, season)] = {
                    "metrics": ContinuousMetrics(
                        rmse=float("nan"),
                        mae=float("nan"),
                        bias=float("nan"),
                        r_squared=float("nan"),
                        n_valid=0,
                    ),
                    "n_stations": 0,
                    "low_confidence": True,
                }
                continue

            cell_obs = obs[cell_mask]
            cell_pred = pred[cell_mask]
            cell_stations = [
                sid for sid, m in zip(station_ids, cell_mask) if m
            ]
            n_unique_stations = len(set(cell_stations))

            metrics = compute_continuous_metrics(cell_obs, cell_pred)
            cells[(band_label, season)] = {
                "metrics": metrics,
                "n_stations": n_unique_stations,
                "low_confidence": n_unique_stations < min_stations_confidence,
            }

    return ConditionalMetricsResult(
        elevation_bands=band_labels,
        seasons=list(_SEASONS),
        cells=cells,
        elevation_thresholds=elevation_thresholds,
    )
