## Context

See proposal.md for motivation. The current codebase has a `TargetNormalizer` that computes global z-score across all samples (normalizer.py), a LOSO evaluator hardcoded to 13 stations (loso.py), metrics limited to RMSE/MAE/bias/R2 (metrics.py), and baselines limited to raw HRRR, WindNinja, and granite-wxc (baselines.py). The evaluation has no temporal holdout — the same data used for training is evaluated.

## Goals / Non-Goals

**Goals:**

- Eliminate the +7K seasonal temperature bias from normalization
- Establish a statistically rigorous evaluation framework (proper holdout, full station coverage, CIs)
- Provide baselines that bracket task difficulty (trivial through competitive)
- Enable publishable results with standard reporting (per-season, per-elevation, skill scores)

**Non-Goals:**

- Changing the model architecture (StormCast backbone, U-Net head)
- Changing the data pipeline or data sources
- Spatial holdout (entire regions held out) — only temporal holdout in this change
- Hyperparameter re-optimization — retrain with corrected normalization at existing settings
- Adding new evaluation variables beyond the existing four (wind speed, direction, temperature, precipitation)

## Decisions

### D1: Anomaly-based normalization over per-month z-score

**Choice:** Subtract monthly station climatology, then z-score the residual anomalies globally.

**Alternative considered:** Per-month z-score (separate mean/std for each calendar month). Rejected because it eliminates inter-month variability signal that the model should learn, and WeatherBench 2 uses the anomaly approach as standard practice.

**Alternative considered:** No normalization (raw physical units). Rejected because temperature (~250-310 K) and precipitation (~0-50 mm) have vastly different scales, causing loss imbalance.

### D2: Nested temporal split for evaluation

**Choice:** WY2020-2023 for training, WY2024 for testing. Within training, use LOSO (leave-one-station-out) for model selection.

**Alternative considered:** k-fold temporal split (e.g., each water year as a fold). Rejected because it would require 4x training runs and WY2024 is the most complete recent year.

**Alternative considered:** Random temporal split. Rejected because weather is temporally autocorrelated — random splits leak information through nearby timesteps.

### D3: Lapse-rate adjusted HRRR as primary baseline to beat

**Choice:** 6.5 K/km standard environmental lapse rate, matching meteorological convention.

**Alternative considered:** 6.1 K/km (Schwartzreich 2024 uses this for Colorado). Keeping 6.5 as default for comparability with broader literature, but making it configurable so 6.1 can be tested.

**Rationale:** Lapse-rate HRRR is the strongest "no-ML" baseline — if the learned model cannot beat simple elevation correction, the downscaling head adds no value. Raw HRRR and climatology are easier baselines that establish the floor.

### D4: Bootstrap with 10,000 resamples at station level

**Choice:** Resample stations (not individual timesteps) with 10,000 draws, report 2.5th/97.5th percentiles.

**Alternative considered:** Block bootstrap on time blocks. Rejected because station-level resampling is simpler, preserves within-station temporal correlation, and matches Schwartzreich's protocol for direct comparability.

**Alternative considered:** 1,000 resamples. 10,000 gives stable CI bounds at the cost of ~10 seconds of computation — negligible versus model evaluation time.

### D5: Extend existing BaselineRunner pattern

**Choice:** Add four new `BaselineRunner` subclasses to `baselines.py` alongside the existing three, keeping the same `predict_station()` / `evaluate_station()` interface.

**Alternative considered:** New baseline framework with a registry pattern. Rejected — the existing pattern works, there are only 7 baselines total, and a registry adds complexity without benefit at this scale.

## Risks / Trade-offs

**[SNOTEL station placement bias]** SNOTEL stations are biased toward sheltered, forested positions for snow measurement. Results may not generalize to exposed ridgelines or alpine bowls. → Mitigation: Acknowledge as a known limitation in reporting. Per-elevation-band metrics partially expose this.

**[Monthly climatology edge effects]** Stations with short records may have noisy monthly climatologies. → Mitigation: Require minimum 30 observations per station-month; fall back to all-month mean for sparse months.

**[Existing checkpoint incompatibility]** All Phase 2 and Phase 3 checkpoints become invalid after normalization change. → Mitigation: This is expected and intentional. Full retrain is part of the task plan.

**[WY2024 as sole test year]** Single test year limits assessment of inter-annual variability. → Mitigation: Acceptable for initial publication. Multi-year evaluation is a future extension once WY2025 data is available.

## Open Questions

None — all design decisions are resolved.
