## MODIFIED Requirements

### Requirement: Leave-one-station-out cross-validation

The system SHALL implement leave-one-station-out (LOSO) cross-validation on all available Colorado SNOTEL stations (target: 117 stations, expanded from the prior 13-station subset). For each fold, one station is held out as the test station and the model is evaluated on its observations. The system SHALL report per-station and aggregate metrics, stratified by season (DJF, MAM, JJA, SON) and elevation band (below treeline < 2900m, near treeline 2900-3400m, above treeline > 3400m).

#### Scenario: Full LOSO evaluation

- **WHEN** LOSO evaluation is run on the Colorado SNOTEL network
- **THEN** the system produces evaluation folds for all available stations (target 117), one per station, with per-station metrics and aggregate statistics (mean, median, standard deviation across stations)

#### Scenario: Single-station evaluation

- **WHEN** LOSO evaluation is run for a specific station ID
- **THEN** the system evaluates only that fold and returns the per-station metrics

#### Scenario: Per-season LOSO reporting

- **WHEN** LOSO evaluation completes
- **THEN** the results include aggregate metrics broken down by meteorological season (DJF, MAM, JJA, SON) in addition to the all-season aggregate

#### Scenario: Per-elevation LOSO reporting

- **WHEN** LOSO evaluation completes
- **THEN** the results include aggregate metrics broken down by elevation band (below/near/above treeline) in addition to the all-elevation aggregate

### Requirement: Baseline comparisons

The system SHALL compare the terrain-weather model against the following baselines, using the same LOSO evaluation protocol and metrics:

1. **Raw HRRR:** Nearest-grid-cell HRRR forecast at each station location (no downscaling)
2. **Lapse-rate HRRR:** HRRR with temperature corrected by standard environmental lapse rate (6.5 K/km)
3. **Station climatology:** Monthly mean from the training period
4. **Persistence:** 24-hour persistence (yesterday's value at the same hour)
5. **Bias-corrected HRRR:** Station-specific monthly bias correction from training period
6. **WindNinja:** WindNinja solver run at each station location using HRRR boundary conditions
7. **granite-wxc-downscaling:** IBM Granite geospatial weather downscaling model

#### Scenario: Baseline evaluation

- **WHEN** a baseline model is run
- **THEN** the system produces metrics in the same format as the terrain-weather model, enabling direct per-station and aggregate comparison

#### Scenario: Comparison report

- **WHEN** all models and baselines have been evaluated
- **THEN** the system produces a comparison table with per-variable aggregate metrics for each model, skill scores against each baseline, and statistical significance flags (paired t-test, p < 0.05)
