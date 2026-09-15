## 1. Anomaly Normalization

- [ ] 1.1 Implement `AnomalyNormalizer` class in `normalizer.py` with `fit_climatology()` that computes per-station, per-variable, per-month mean from training data (WY2020-2023), with 30-observation minimum and all-month fallback. Verify: unit test that fit produces 12 monthly means per station-variable and handles sparse months correctly.
- [ ] 1.2 Implement `normalize()` and `denormalize()` methods that perform two-step anomaly normalization (subtract climatology, then z-score) and its inverse. Verify: round-trip test that `denormalize(normalize(x))` recovers original values within 1e-5 tolerance.
- [ ] 1.3 Implement `state_dict()` and `load_state_dict()` that serialize/deserialize climatology table, anomaly mean, and anomaly std. Verify: test that save-then-load produces a normalizer that gives identical normalize/denormalize results.
- [ ] 1.4 Update `TargetNormalizer` imports and usages across the codebase to use `AnomalyNormalizer`. Verify: `ruff check src/` passes and `grep -r "TargetNormalizer" src/` returns zero hits outside of backward-compat aliases if any.

## 2. Evaluation Framework Extensions

- [ ] 2.1 Implement temporal holdout split in `loso.py` that enforces WY2020-2023 train / WY2024 test boundary with no timestamp overlap. Verify: test that constructs a mixed dataset and confirms zero WY2024 timestamps in training split.
- [ ] 2.2 Expand LOSO evaluator to accept all available stations (117 target) and flag stations with >50% missing values. Verify: test with synthetic 120-station data that excludes 3 stations with 60% missing and reports them.
- [ ] 2.3 Add per-season stratification (DJF/MAM/JJA/SON) to `LOSOResult` and `_compute_aggregate()`. Verify: test with synthetic data spanning 4 seasons that confirms per-season metrics differ from all-season aggregate.
- [ ] 2.4 Add per-elevation-band stratification (below/near/above treeline with configurable thresholds) to aggregate metrics. Verify: test with stations at 2500m, 3100m, 3600m that confirms correct band assignment and band-specific metrics.
- [ ] 2.5 Implement bootstrap 95% CIs with station-level resampling (10,000 default, configurable). Verify: test that CI bounds bracket the point estimate and width decreases with more stations.

## 3. Extended Baselines

- [ ] 3.1 Implement `LapseRateHRRRRunner` in `baselines.py` that adjusts HRRR temperature by `-(elevation_diff * lapse_rate)` with configurable lapse rate (default 6.5 K/km) and passes non-temperature variables through. Verify: test that a station 500m above the grid cell gets a -3.25 K correction.
- [ ] 3.2 Implement `ClimatologyBaselineRunner` that predicts the monthly training-period mean for each station-variable-month. Verify: test with known monthly means that confirms predictions match and NaN is produced for missing months.
- [ ] 3.3 Implement `PersistenceBaselineRunner` that predicts each variable as the observation 24 hours prior, with 72-hour lookback for missing values. Verify: test with a gap at t-24h that confirms lookback to t-48h and NaN when no valid observation exists within 72h.
- [ ] 3.4 Implement `BiasCorrectedHRRRRunner` that subtracts per-station, per-variable, per-month mean (HRRR - obs) from HRRR predictions. Verify: test that correction removes known systematic bias.

## 4. Extended Metrics

- [ ] 4.1 Implement `compute_skill_score()` in `metrics.py` that computes `1 - RMSE_model/RMSE_baseline` for each model-baseline pair. Verify: test with known RMSE values that confirms correct skill score (e.g., 2.1/3.0 yields 0.30).
- [ ] 4.2 Implement `compute_threshold_crossing_accuracy()` for 0 degrees C temperature crossings, returning POD, FAR, CSI. Verify: test with synthetic temperature series containing known crossings.
- [ ] 4.3 Implement `compute_accumulation_accuracy()` for 3-day and 7-day precipitation windows, returning RMSE, bias, R-squared of windowed totals. Verify: test with known precipitation series and expected 3-day sums.
- [ ] 4.4 Implement conditional metrics matrix (3 elevation bands x 4 seasons) that computes full metric set per cell with low-confidence flag for cells with <5 stations. Verify: test with synthetic station data that confirms correct cell assignment and flagging.

## 5. Retrain Phases 2 and 3

- [ ] 5.1 Update `scripts/run_phase2.py` to use `AnomalyNormalizer` instead of `TargetNormalizer`, fitting climatology on Alpine training data before training. Verify: script runs without error and checkpoint contains climatology parameters.
- [ ] 5.2 Update `scripts/run_phase3.py` to use `AnomalyNormalizer`, fitting climatology on WY2020-2023 SNOTEL data only. Verify: script runs without error and no WY2024 data appears in climatology computation.
- [ ] 5.3 Retrain Phase 2 with anomaly normalization and verify validation loss is comparable or improved versus prior global z-score training. Verify: Phase 2 checkpoint saved with anomaly normalization parameters.

## 6. Full Validation Run and Report

- [ ] 6.1 Update `scripts/run_validation.py` to run evaluation with temporal holdout, all 117 stations, all 7 baselines, extended metrics, per-season/elevation reporting, bootstrap CIs, and skill scores. Verify: script runs end-to-end and produces a complete JSON results file.
- [ ] 6.2 Update `evaluation/report.py` to generate comparison tables including skill scores against each baseline, per-season and per-elevation breakdowns, bootstrap CIs, and threshold/accumulation metrics. Verify: report includes all sections and no section is empty.
- [ ] 6.3 Run full validation and produce updated `reports/validation_phase3_jan2024.md` with corrected evaluation results. Verify: report contains metrics for 100+ stations, 4 seasons, 3 elevation bands, 7 baselines, with CIs on all aggregate metrics.
