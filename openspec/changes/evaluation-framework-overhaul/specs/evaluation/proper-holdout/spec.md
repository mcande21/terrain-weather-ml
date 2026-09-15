## Purpose

Implements proper temporal holdout evaluation with full station coverage, stratified reporting by season and elevation band, and bootstrap confidence intervals for statistically rigorous model assessment.

## ADDED Requirements

### Requirement: Temporal holdout split

The system SHALL split data by water year: WY2020-2023 for training and WY2024 for testing. No WY2024 data SHALL appear in training, validation, or model selection. The split SHALL be enforced at the dataset level before any model training begins.

#### Scenario: Holdout enforcement

- **WHEN** the evaluation framework constructs train and test sets
- **THEN** all timestamps in the test set fall within WY2024 (October 1, 2023 through September 30, 2024) and no training sample shares a timestamp with any test sample

#### Scenario: No temporal leakage

- **WHEN** model selection (hyperparameter tuning, early stopping) is performed
- **THEN** it uses only WY2020-2023 data via nested cross-validation or a held-out validation split within that period, never WY2024

### Requirement: Full station coverage

The system SHALL evaluate on all SNOTEL stations available in the Colorado dataset (target: 117 stations). The evaluation SHALL report per-station metrics and flag stations excluded due to data quality issues (> 50% missing values in WY2024).

#### Scenario: Station coverage reporting

- **WHEN** evaluation completes
- **THEN** the report lists total stations attempted, stations successfully evaluated, and stations excluded with reason

#### Scenario: Data quality exclusion

- **WHEN** a station has more than 50% missing values for any evaluated variable in WY2024
- **THEN** that station-variable pair is excluded from aggregate metrics and flagged in the report

### Requirement: Per-season reporting

The system SHALL report metrics stratified by meteorological season: DJF (December-January-February), MAM (March-April-May), JJA (June-July-August), SON (September-October-November). Aggregate metrics SHALL also be reported across all seasons.

#### Scenario: Seasonal stratification

- **WHEN** evaluation metrics are computed
- **THEN** the report includes a metrics table for each of the four seasons plus an all-season aggregate, each with the same metric set

#### Scenario: Cross-year DJF

- **WHEN** DJF season is computed for WY2024
- **THEN** it includes December 2023, January 2024, and February 2024

### Requirement: Per-elevation-band reporting

The system SHALL report metrics stratified by elevation band: below treeline (< 2900m), near treeline (2900-3400m), and above treeline (> 3400m). Band thresholds SHALL be configurable. Station assignment to bands uses the station's recorded elevation.

#### Scenario: Elevation band stratification

- **WHEN** evaluation metrics are computed
- **THEN** the report includes a metrics table for each elevation band plus an all-elevation aggregate

#### Scenario: Band membership

- **WHEN** a station at 3150m elevation is evaluated with default thresholds
- **THEN** it is assigned to the near-treeline band (2900-3400m)

### Requirement: Bootstrap 95% confidence intervals

The system SHALL compute 95% confidence intervals on aggregate metrics using bootstrap resampling with 10,000 resamples (configurable). Resampling SHALL operate at the station level (resample which stations are included) to preserve within-station correlation structure.

#### Scenario: Bootstrap CI computation

- **WHEN** aggregate metrics are computed across N stations
- **THEN** 10,000 bootstrap samples of size N are drawn with replacement from the station-level metric values, and the 2.5th and 97.5th percentiles define the 95% CI

#### Scenario: CI reporting

- **WHEN** the evaluation report is generated
- **THEN** each aggregate metric includes its point estimate and 95% CI bounds
