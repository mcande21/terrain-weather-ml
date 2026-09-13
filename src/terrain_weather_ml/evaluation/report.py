"""Comparison report and tail-event evaluation (task 7.5).

Generates comparison tables across models with paired t-test
significance flags. Evaluates tail-event performance for
safety-critical applications (avalanche, wildfire).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
from scipy import stats

from terrain_weather_ml.evaluation.metrics import (
    ContinuousMetrics,
    compute_continuous_metrics,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelResults:
    """Evaluation results for a single model.

    Attributes:
        name: Model identifier.
        per_station: Dict of station_id -> variable -> metric -> value.
    """

    name: str
    per_station: dict[str, dict[str, dict[str, float]]]


@dataclass
class ComparisonReport:
    """Comparison report across multiple models.

    Attributes:
        models: List of model names in comparison order.
        variables: Set of variable names evaluated.
        aggregate_metrics: model_name -> variable -> metric -> value.
        significance: Pairwise significance results.
    """

    models: list[str]
    variables: set[str]
    aggregate_metrics: dict[str, dict[str, dict[str, float]]]
    significance: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "models": self.models,
            "variables": sorted(self.variables),
            "aggregate_metrics": self.aggregate_metrics,
            "significance": self.significance,
        }


@dataclass
class TailEventConfig:
    """Configuration for tail-event evaluation.

    Attributes:
        variable: Variable name for tail selection.
        percentile: Percentile threshold (0-100).
        direction: "above" for upper tail, "below" for lower tail.
    """

    variable: str
    percentile: float
    direction: str  # "above" or "below"


@dataclass
class TailEventResult:
    """Result of tail-event evaluation.

    Attributes:
        config: The tail-event configuration used.
        n_events: Number of tail events identified.
        threshold_value: The percentile threshold value.
        metrics: ContinuousMetrics computed on the tail subset.
    """

    config: TailEventConfig
    n_events: int
    threshold_value: float
    metrics: ContinuousMetrics | None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "variable": self.config.variable,
            "percentile": self.config.percentile,
            "direction": self.config.direction,
            "n_events": self.n_events,
            "threshold_value": float(self.threshold_value),
        }
        if self.metrics is not None:
            result["metrics"] = self.metrics.to_dict()
        return result


def generate_comparison_report(
    model_results: list[ModelResults],
    alpha: float = 0.05,
) -> ComparisonReport:
    """Generate a comparison report across multiple models.

    Computes aggregate metrics (mean across stations) for each model
    and performs pairwise paired t-tests on RMSE values to identify
    statistically significant differences.

    Args:
        model_results: List of ModelResults for each model to compare.
        alpha: Significance level for paired t-test (default 0.05).

    Returns:
        ComparisonReport with aggregates and significance flags.
    """
    models = [m.name for m in model_results]

    # Collect all variables
    variables: set[str] = set()
    for m in model_results:
        for station_metrics in m.per_station.values():
            variables.update(station_metrics.keys())

    # Compute aggregate metrics per model per variable
    aggregate_metrics: dict[str, dict[str, dict[str, float]]] = {}
    for m in model_results:
        aggregate_metrics[m.name] = {}
        for var in variables:
            metric_values: dict[str, list[float]] = {}
            for station_metrics in m.per_station.values():
                if var in station_metrics:
                    for metric_name, value in station_metrics[var].items():
                        if metric_name not in metric_values:
                            metric_values[metric_name] = []
                        metric_values[metric_name].append(value)

            var_agg: dict[str, float] = {}
            for metric_name, values in metric_values.items():
                arr = np.array(values)
                var_agg[metric_name] = float(np.mean(arr))
            aggregate_metrics[m.name][var] = var_agg

    # Pairwise significance tests (paired t-test on RMSE)
    significance: dict[str, dict[str, Any]] = {}
    for m1, m2 in combinations(model_results, 2):
        pair_key = f"{m1.name}_vs_{m2.name}"
        significance[pair_key] = {}

        for var in variables:
            # Collect paired RMSE values across stations
            common_stations = (
                set(m1.per_station.keys()) & set(m2.per_station.keys())
            )
            rmse1 = []
            rmse2 = []
            for sid in common_stations:
                if (
                    var in m1.per_station[sid]
                    and var in m2.per_station[sid]
                    and "rmse" in m1.per_station[sid][var]
                    and "rmse" in m2.per_station[sid][var]
                ):
                    rmse1.append(m1.per_station[sid][var]["rmse"])
                    rmse2.append(m2.per_station[sid][var]["rmse"])

            if len(rmse1) >= 2:
                t_stat, p_value = stats.ttest_rel(rmse1, rmse2)
                significance[pair_key][var] = {
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                    "significant": p_value < alpha,
                    "n_pairs": len(rmse1),
                    "better_model": (
                        m1.name if np.mean(rmse1) < np.mean(rmse2)
                        else m2.name
                    ),
                }
            else:
                significance[pair_key][var] = {
                    "t_statistic": float("nan"),
                    "p_value": float("nan"),
                    "significant": False,
                    "n_pairs": len(rmse1),
                    "better_model": None,
                }

    return ComparisonReport(
        models=models,
        variables=variables,
        aggregate_metrics=aggregate_metrics,
        significance=significance,
    )


def compute_tail_event_metrics(
    obs: np.ndarray,
    pred: np.ndarray,
    config: TailEventConfig,
) -> TailEventResult:
    """Compute metrics on tail events only.

    Identifies tail events based on the observed distribution and
    computes metrics only on those timesteps.

    Args:
        obs: Observed values.
        pred: Predicted values.
        config: Tail event configuration (variable, percentile, direction).

    Returns:
        TailEventResult with tail-only metrics.
    """
    # Exclude NaN values for percentile computation
    valid = ~(np.isnan(obs) | np.isnan(pred))
    obs_valid = obs[valid]
    pred_valid = pred[valid]

    if len(obs_valid) == 0:
        return TailEventResult(
            config=config,
            n_events=0,
            threshold_value=float("nan"),
            metrics=None,
        )

    # Compute threshold from observed distribution
    threshold_value = float(np.percentile(obs_valid, config.percentile))

    # Select tail events
    if config.direction == "above":
        tail_mask = obs_valid > threshold_value
    elif config.direction == "below":
        tail_mask = obs_valid < threshold_value
    else:
        raise ValueError(
            f"direction must be 'above' or 'below', got '{config.direction}'"
        )

    n_events = int(tail_mask.sum())

    if n_events < 2:
        return TailEventResult(
            config=config,
            n_events=n_events,
            threshold_value=threshold_value,
            metrics=None,
        )

    # Compute metrics on tail subset
    tail_obs = obs_valid[tail_mask]
    tail_pred = pred_valid[tail_mask]
    metrics = compute_continuous_metrics(tail_obs, tail_pred)

    return TailEventResult(
        config=config,
        n_events=n_events,
        threshold_value=threshold_value,
        metrics=metrics,
    )
