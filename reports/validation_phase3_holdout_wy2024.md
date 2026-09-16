# Phase 3 Validation Report: WY2024 Holdout Evaluation

## Summary

- **Evaluation period:** WY2024 (Oct 2023 - Sep 2024, holdout)
- **Training period:** WY2020-2023
- **Normalizer:** AnomalyNormalizer (per-station per-month climatology)
- **Stations evaluated:** 117
- **Runtime:** 295.6s
- **Variables:** temperature (K), precipitation (kg/m^2/s)

## Model vs All Baselines (Aggregate)

| Variable | Metric | Model | hrrr_raw | hrrr_lapse_rate | climatology | persistence | hrrr_bias_corrected |
|----------|--------|-------|----------|----------|----------|----------|----------|
| temperature | rmse | nan | 2.7919 | 2.7782 | 4.0745 | 3.8449 | 2.5581 |
| temperature | mae | nan | 2.2594 | 2.2419 | 3.1615 | 2.8707 | 2.0330 |
| temperature | bias | nan | -0.6891 | -0.4651 | -0.1984 | -0.0090 | -0.2665 |
| precipitation | rmse | nan | nan | nan | 0.0001 | nan | nan |
| precipitation | mae | nan | nan | nan | 0.0000 | nan | nan |
| precipitation | bias | nan | nan | nan | -0.0000 | nan | nan |

## Skill Scores (1 - RMSE_model / RMSE_baseline)

| Variable | hrrr_raw | hrrr_lapse_rate | climatology | persistence | hrrr_bias_corrected |
|----------|----------|----------|----------|----------|----------|
| temperature | +nan | +nan | +nan | +nan | +nan |
| precipitation | +nan | +nan | +nan | +nan | +nan |

## Per-Season Temperature Metrics (Model)

| Season | RMSE (K) | MAE (K) | Bias (K) | R^2 | N |
|--------|----------|---------|----------|-----|---|
| DJF | 10.33 | 5.73 | -2.87 | -3.3281 | 10360 |
| MAM | 8.77 | 4.49 | -1.20 | -2.1041 | 10444 |
| JJA | 9.87 | 4.78 | -2.91 | -7.9792 | 10369 |
| SON | 12.46 | 6.16 | -3.50 | -2.6947 | 10353 |

## Per-Elevation-Band Temperature Metrics (Model)

| Elevation Band | Stations | Mean RMSE (K) | Mean Bias (K) | 95% CI |
|----------------|----------|---------------|---------------|--------|
| below_treeline | 31 | 7.74 | -2.57 | [6.30, 9.33] |
| near_treeline | 70 | 7.74 | -2.70 | [6.06, 9.76] |
| above_treeline | 13 | 7.30 | -2.27 | [4.85, 11.36] |

## Bootstrap 95% Confidence Intervals

- **temperature RMSE:** nan [nan, nan]
- **precipitation RMSE:** nan [nan, nan]

## Per-Station Temperature Metrics (Selected)

| Station | Elev (m) | Model RMSE | HRRR RMSE | Model Bias | R^2 |
|---------|----------|-----------|-----------|------------|-----|
| 1160 | 3271 | 3.89 | 1.81 | -0.43 | 0.7628 |
| 580 | 3374 | 3.92 | 2.11 | -0.81 | 0.7182 |
| 1100 | 3176 | 3.94 | 2.31 | -1.33 | 0.6820 |
| 970 | 3179 | 3.96 | 3.53 | -0.07 | 0.7148 |
| 632 | 3234 | 3.97 | 2.88 | -1.09 | 0.7122 |
| 762 | 3523 | 3.97 | 2.12 | -0.86 | 0.7452 |
| 1188 | 3386 | 4.04 | 2.53 | -0.64 | 0.7187 |
| 327 | 3530 | 4.06 | 3.05 | -0.50 | 0.7055 |
| 415 | 3200 | 4.07 | 2.81 | -1.23 | 0.6925 |
| 713 | 3371 | 4.07 | 2.75 | -0.47 | 0.6855 |
| 935 | 3362 | 22.06 | 3.77 | -10.36 | -9.1659 |
| 1186 | 3392 | 25.47 | 2.08 | -10.93 | -8.7882 |
| 682 | 3045 | 27.80 | 3.17 | -14.45 | -10.3750 |
| 938 | 3402 | 28.99 | 4.86 | -14.83 | -15.0451 |
| 556 | 2929 | 30.27 | 3.50 | -14.35 | -14.0983 |
| 547 | 3213 | 31.29 | 2.35 | -14.96 | -14.8287 |
| 1123 | 3042 | 49.56 | 3.96 | -32.00 | -39.3785 |
| 1324 | 3560 | nan | 2.38 | +nan | nan |
| 1325 | 3560 | nan | 2.67 | +nan | nan |
| 1326 | 3508 | nan | 4.56 | +nan | nan |

## Statistical Significance (Paired t-test on RMSE)

- terrain_downscaling_vs_hrrr_raw (temperature): t=nan, p=nan, not significant, better=hrrr_raw
- terrain_downscaling_vs_hrrr_raw (precipitation): t=nan, p=nan, not significant, better=hrrr_raw
- terrain_downscaling_vs_hrrr_lapse_rate (temperature): t=nan, p=nan, not significant, better=hrrr_lapse_rate
- terrain_downscaling_vs_hrrr_lapse_rate (precipitation): t=nan, p=nan, not significant, better=hrrr_lapse_rate
- terrain_downscaling_vs_climatology (temperature): t=5.504, p=0.0000, **significant**, better=climatology
- terrain_downscaling_vs_climatology (precipitation): t=14.200, p=0.0000, **significant**, better=climatology
- terrain_downscaling_vs_persistence (temperature): t=nan, p=nan, not significant, better=persistence
- terrain_downscaling_vs_persistence (precipitation): t=nan, p=nan, not significant, better=persistence
- terrain_downscaling_vs_hrrr_bias_corrected (temperature): t=nan, p=nan, not significant, better=hrrr_bias_corrected
- terrain_downscaling_vs_hrrr_bias_corrected (precipitation): t=nan, p=nan, not significant, better=hrrr_bias_corrected
- hrrr_raw_vs_hrrr_lapse_rate (temperature): t=0.629, p=0.5305, not significant, better=hrrr_lapse_rate
- hrrr_raw_vs_hrrr_lapse_rate (precipitation): t=nan, p=nan, not significant, better=hrrr_lapse_rate
- hrrr_raw_vs_climatology (temperature): t=-20.142, p=0.0000, **significant**, better=hrrr_raw
- hrrr_raw_vs_climatology (precipitation): t=33.576, p=0.0000, **significant**, better=climatology
- hrrr_raw_vs_persistence (temperature): t=-16.117, p=0.0000, **significant**, better=hrrr_raw
- hrrr_raw_vs_persistence (precipitation): t=nan, p=nan, not significant, better=persistence
- hrrr_raw_vs_hrrr_bias_corrected (temperature): t=4.774, p=0.0000, **significant**, better=hrrr_bias_corrected
- hrrr_raw_vs_hrrr_bias_corrected (precipitation): t=nan, p=nan, not significant, better=hrrr_bias_corrected
- hrrr_lapse_rate_vs_climatology (temperature): t=-21.566, p=0.0000, **significant**, better=hrrr_lapse_rate
- hrrr_lapse_rate_vs_climatology (precipitation): t=33.576, p=0.0000, **significant**, better=climatology
- hrrr_lapse_rate_vs_persistence (temperature): t=-18.000, p=0.0000, **significant**, better=hrrr_lapse_rate
- hrrr_lapse_rate_vs_persistence (precipitation): t=nan, p=nan, not significant, better=persistence
- hrrr_lapse_rate_vs_hrrr_bias_corrected (temperature): t=4.950, p=0.0000, **significant**, better=hrrr_bias_corrected
- hrrr_lapse_rate_vs_hrrr_bias_corrected (precipitation): t=nan, p=nan, not significant, better=hrrr_bias_corrected
- climatology_vs_persistence (temperature): t=7.188, p=0.0000, **significant**, better=persistence
- climatology_vs_persistence (precipitation): t=-36.169, p=0.0000, **significant**, better=climatology
- climatology_vs_hrrr_bias_corrected (temperature): t=39.964, p=0.0000, **significant**, better=hrrr_bias_corrected
- climatology_vs_hrrr_bias_corrected (precipitation): t=nan, p=nan, not significant, better=hrrr_bias_corrected
- persistence_vs_hrrr_bias_corrected (temperature): t=31.043, p=0.0000, **significant**, better=hrrr_bias_corrected
- persistence_vs_hrrr_bias_corrected (precipitation): t=nan, p=nan, not significant, better=hrrr_bias_corrected

## Physics Checks

### Cold Air Pool Reproduction

- Valley stations (TPI < -20m): 0
- Ridge stations (TPI > 50m): 0
- Insufficient valley/ridge station pairs for analysis

## Interpretation

This is a **proper holdout evaluation** using WY2024 data
that was not used during any phase of model training.
The AnomalyNormalizer removes per-station per-month climatology
before z-scoring, preventing seasonal cycle conflation.

Key questions answered:
1. Does the terrain downscaling head generalize to unseen time? (temporal holdout)
2. Does terrain correction beat all baselines? (skill scores)
3. How does performance vary by season? (DJF/MAM/JJA/SON)
4. How does performance vary by elevation? (treeline bands)
5. Does the model respect cold air pooling physics? (CAP check)
