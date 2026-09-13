# evaluation/cross-validation Specification

## Purpose
Evaluates model skill via leave-one-station-out cross-validation on Colorado SNOTEL, physics consistency checks, and comparison against established baselines.

## Requirements

### Requirement: Leave-one-station-out cross-validation

The system SHALL implement leave-one-station-out (LOSO) cross-validation on the 118 Colorado SNOTEL stations. For each fold, one station is held out as the test station and the model is evaluated on its observations. The system SHALL report per-station and aggregate metrics.

#### Scenario: Full LOSO evaluation

- **WHEN** LOSO evaluation is run on the Colorado SNOTEL network
- **THEN** the system produces 118 evaluation folds, one per station, with per-station metrics and aggregate statistics (mean, median, standard deviation across stations)

#### Scenario: Single-station evaluation

- **WHEN** LOSO evaluation is run for a specific station ID
- **THEN** the system evaluates only that fold and returns the per-station metrics

### Requirement: Evaluation metrics

The system SHALL compute the following metrics per variable (wind speed, wind direction, temperature, precipitation):

- RMSE (root mean square error)
- MAE (mean absolute error)
- Bias (mean error)
- R-squared (coefficient of determination)
- For wind direction: circular RMSE and circular bias
- For precipitation: frequency bias, probability of detection (POD), false alarm ratio (FAR), critical success index (CSI) at configurable thresholds (default: 0.1, 1.0, 5.0 mm/hr)

#### Scenario: Metric computation for a station

- **WHEN** predictions and observations for one station and time range are provided
- **THEN** the system returns all metrics per variable, handling missing observations by excluding those timesteps

### Requirement: Physics consistency checks

The system SHALL evaluate physical consistency of model output through:

1. **Wind mass conservation:** Compute the 2D divergence of the predicted wind field at each timestep. Report the mean and max absolute divergence across the evaluation domain.
2. **Cold air pool reproduction:** For SNOTEL stations in valley locations (TPI < -50m), compute the mean temperature offset between valley and ridgeline stations during nocturnal inversions (identified by positive temperature lapse rate in the lowest 500m). The model SHALL reproduce cold air pooling with a bias below 2K.
3. **Orographic precipitation consistency:** Compare the model's elevation-precipitation gradient against PRISM 30-year normals. The predicted annual precipitation vs. elevation regression slope SHALL agree with PRISM within 20%.

#### Scenario: Wind divergence check

- **WHEN** physics consistency checks are run
- **THEN** the report includes mean and max absolute divergence across all evaluation timesteps and a pass/fail flag against the threshold (max divergence < 0.01 s^-1)

#### Scenario: Cold air pool check

- **WHEN** nocturnal inversions are identified in the evaluation period
- **THEN** the report includes valley-ridgeline temperature bias and a pass/fail flag against the 2K threshold

#### Scenario: Orographic precipitation check

- **WHEN** the evaluation period spans at least one water year
- **THEN** the report includes the predicted and PRISM precipitation-elevation slopes with percent agreement

### Requirement: Baseline comparisons

The system SHALL compare the terrain-weather model against the following baselines, using the same LOSO evaluation protocol and metrics:

1. **Raw HRRR:** Nearest-grid-cell HRRR forecast at each station location (no downscaling)
2. **WindNinja:** WindNinja solver run at each station location using HRRR boundary conditions
3. **granite-wxc-downscaling:** IBM Granite geospatial weather downscaling model (ibm-granite/granite-geospatial-wxc-downscaling)

#### Scenario: Baseline evaluation

- **WHEN** a baseline model is run
- **THEN** the system produces metrics in the same format as the terrain-weather model, enabling direct per-station and aggregate comparison

#### Scenario: Comparison report

- **WHEN** all models and baselines have been evaluated
- **THEN** the system produces a comparison table with per-variable aggregate metrics for each model, highlighting statistically significant improvements (paired t-test, p < 0.05)

### Requirement: Safety-critical tail event evaluation

The system SHALL separately evaluate performance on tail events relevant to avalanche and wildfire applications: wind speed above 90th percentile, temperature below 10th percentile, and precipitation above 95th percentile. Tail event metrics SHALL be reported alongside full-distribution metrics.

#### Scenario: Tail event reporting

- **WHEN** evaluation completes
- **THEN** the results include a tail-event section with metrics computed only on timesteps where the observed value exceeds the specified percentile thresholds

**Optional evaluation baselines note:** SNOWstorm model (Zenodo) — wind-driven snow redistribution U-Nets, directly relevant to avalanche wind-slab loading prediction. Can serve as an optional evaluation baseline for wind-terrain interaction skill, particularly for snow transport and deposition patterns driven by terrain-modified wind fields. WFIP2 Columbia Gorge dataset — 184 sonic anemometers deployed during a 6-week intensive observation period in complex terrain. Provides an optional physics-validation dataset with extremely dense spatial coverage for evaluating wind field accuracy in channeled flow through complex topography.
