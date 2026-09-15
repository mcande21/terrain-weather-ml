## Purpose

Extends the metric suite with skill scores relative to each baseline, temperature threshold crossing accuracy, precipitation accumulation accuracy, and conditional metrics stratified by elevation and season.

## ADDED Requirements

### Requirement: Skill scores vs each baseline

The system SHALL compute a skill score for each model-baseline pair defined as `SS = 1 - (RMSE_model / RMSE_baseline)`. A positive skill score indicates the model outperforms the baseline; zero means equal; negative means worse. Skill scores SHALL be computed per-variable for every baseline in the evaluation.

#### Scenario: Skill score computation

- **WHEN** the model has RMSE 2.1 K for temperature and the lapse-rate HRRR baseline has RMSE 3.0 K
- **THEN** the skill score is `1 - 2.1/3.0 = 0.30`

#### Scenario: Perfect baseline match

- **WHEN** the model RMSE equals the baseline RMSE
- **THEN** the skill score is 0.0

#### Scenario: Skill score reported per baseline

- **WHEN** the evaluation report is generated
- **THEN** it includes skill scores against each of: raw HRRR, lapse-rate HRRR, climatology, persistence, bias-corrected HRRR

### Requirement: Temperature threshold crossing accuracy

The system SHALL evaluate the model's ability to predict freezing-level crossings at the 0 degrees C threshold. Metrics SHALL include: probability of detection (POD), false alarm ratio (FAR), and critical success index (CSI) for correctly predicting when temperature crosses 0 degrees C between consecutive timesteps.

#### Scenario: Freezing crossing detection

- **WHEN** the observed temperature transitions from +1 C to -1 C between consecutive hours
- **THEN** this counts as a freezing crossing event

#### Scenario: Crossing metrics computation

- **WHEN** threshold crossing evaluation completes
- **THEN** the report includes POD, FAR, and CSI for 0 degrees C crossings, computed across all stations and timesteps in the test set

### Requirement: Precipitation accumulation accuracy

The system SHALL evaluate precipitation accumulation accuracy over 3-day and 7-day windows. For each window, the system computes the total predicted and observed accumulation and reports RMSE, bias, and R-squared of the windowed totals across all station-windows in the test period.

#### Scenario: 3-day accumulation

- **WHEN** precipitation accumulation is evaluated over 3-day windows
- **THEN** each window sums hourly precipitation into a 72-hour total, and metrics are computed across all non-overlapping windows for all stations

#### Scenario: 7-day accumulation

- **WHEN** precipitation accumulation is evaluated over 7-day windows
- **THEN** each window sums hourly precipitation into a 168-hour total, and metrics are computed across all non-overlapping windows for all stations

### Requirement: Conditional metrics by elevation band and season

The system SHALL compute all standard metrics (RMSE, MAE, bias, R-squared) conditioned on both elevation band and season simultaneously, producing a 3x4 matrix (3 elevation bands x 4 seasons) of metric sets per variable. This enables identification of conditions where the model excels or struggles.

#### Scenario: Conditional matrix generation

- **WHEN** conditional metrics are computed
- **THEN** the report includes a matrix with 12 cells (3 elevation bands x 4 seasons), each containing the full metric set for that condition

#### Scenario: Sparse cell handling

- **WHEN** an elevation-season cell has fewer than 5 stations contributing
- **THEN** the cell is reported with a low-confidence flag and the station count is included
