## Why

Current evaluation has three critical gaps identified by literature review (DEVINE, Wind-Topo, WeatherBench 2, VALUE): (1) global z-score normalization causes +7K seasonal bias because it collapses seasonal structure, (2) only 13 of 117 stations are evaluated, making results unrepresentative, and (3) training data is used as the test set with no temporal holdout, invalidating all reported skill metrics. Fixing these is prerequisite to any publishable result.

## What Changes

- Replace global z-score normalization with anomaly-based target normalization (subtract monthly climatology before z-scoring) to eliminate seasonal bias
- Expand evaluation from 13 to all 117 SNOTEL stations for representative coverage
- Implement proper temporal holdout: train on WY2020-2023, test on WY2024
- Add baselines: lapse-rate adjusted HRRR, station climatology, persistence, bias-corrected HRRR
- Add skill scores (1 - RMSE_model/RMSE_baseline), per-season metrics (DJF/MAM/JJA/SON), per-elevation-band reporting (below/near/above treeline), and bootstrap 95% confidence intervals
- Retrain Phases 2 and 3 with corrected anomaly normalization

## Capabilities

### New Capabilities

- `training/anomaly-normalization`: Monthly climatology-based target normalization that removes seasonal signal before z-scoring
- `evaluation/proper-holdout`: Temporal holdout split with full station coverage, per-season and per-elevation reporting, and bootstrap CIs
- `evaluation/baselines-extended`: Lapse-rate HRRR, climatology, persistence, and bias-corrected HRRR baselines
- `evaluation/metrics-extended`: Skill scores, threshold crossing accuracy, accumulation accuracy, and conditional metrics

### Modified Capabilities

- `training/transfer-pipeline`: Phases 2 and 3 use anomaly normalization instead of global z-score for target preprocessing
- `evaluation/cross-validation`: Expand from 13 to 117 stations, add per-season and per-elevation-band reporting dimensions

## Impact

- `src/terrain_weather_ml/training/normalizer.py` — rewrite TargetNormalizer to AnomalyNormalizer with monthly climatology
- `src/terrain_weather_ml/training/phases.py` — integrate anomaly normalization into Phase 2 and Phase 3 pipelines
- `src/terrain_weather_ml/evaluation/baselines.py` — add four new baseline runners
- `src/terrain_weather_ml/evaluation/metrics.py` — add skill score computation and conditional metrics
- `src/terrain_weather_ml/evaluation/loso.py` — expand station coverage, add holdout split, per-season/elevation grouping, bootstrap CIs
- `src/terrain_weather_ml/evaluation/report.py` — update comparison tables for new baselines and metrics
- `scripts/run_phase2.py`, `scripts/run_phase3.py`, `scripts/run_validation.py` — update for new normalization and evaluation
- Existing checkpoints become incompatible — retrain required
