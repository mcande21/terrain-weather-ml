## Purpose

Provides four additional evaluation baselines — lapse-rate adjusted HRRR, station climatology, persistence, and bias-corrected HRRR — that bracket the difficulty of the downscaling task and establish the minimum bar the learned model must beat.

## ADDED Requirements

### Requirement: Lapse-rate adjusted HRRR baseline

The system SHALL compute a lapse-rate adjusted HRRR baseline that corrects HRRR temperature at the nearest grid cell using the elevation difference between the HRRR grid cell and the station. The standard environmental lapse rate SHALL default to 6.5 K/km and be configurable. Non-temperature variables SHALL pass through the raw HRRR value without adjustment.

#### Scenario: Temperature lapse-rate correction

- **WHEN** a lapse-rate HRRR prediction is generated for a station 500m above the nearest HRRR grid cell
- **THEN** the temperature is adjusted by -3.25 K (500m x 6.5 K/km) relative to the HRRR grid cell value

#### Scenario: Non-temperature passthrough

- **WHEN** wind speed or precipitation predictions are requested from the lapse-rate HRRR baseline
- **THEN** the raw HRRR nearest-grid-cell value is returned without adjustment

### Requirement: Station climatology baseline

The system SHALL compute a station climatology baseline that predicts the monthly mean of each variable from the training period (WY2020-2023) for each station. The prediction for any timestep is the training-period monthly mean for that station, variable, and calendar month.

#### Scenario: Climatology prediction

- **WHEN** a climatology prediction is requested for station X, variable temperature, in January 2024
- **THEN** the predicted value is the mean temperature at station X during all January months in WY2020-2023

#### Scenario: Missing training months

- **WHEN** a station has no training data for a given month
- **THEN** the climatology baseline produces NaN for that month and the station-variable-month is excluded from skill score computation

### Requirement: Persistence baseline

The system SHALL compute a persistence baseline that predicts each variable's value as the most recent valid observation. For hourly data, persistence predicts today's value equals yesterday's value at the same hour (24-hour persistence). Persistence SHALL handle missing values by using the most recent non-NaN observation within a 72-hour lookback window.

#### Scenario: 24-hour persistence

- **WHEN** a persistence prediction is generated for hour H on day D
- **THEN** the predicted value equals the observation at hour H on day D-1

#### Scenario: Missing value lookback

- **WHEN** the observation 24 hours ago is missing
- **THEN** the system uses the most recent valid observation within the prior 72 hours, or NaN if none exists

### Requirement: Bias-corrected HRRR baseline

The system SHALL compute a bias-corrected HRRR baseline by subtracting the station-specific mean difference (HRRR - observation) computed over the training period from each HRRR prediction. Bias correction parameters SHALL be computed per-station, per-variable, and per-month to capture seasonal bias patterns.

#### Scenario: Bias correction application

- **WHEN** a bias-corrected HRRR prediction is generated for station X in January
- **THEN** the output equals the raw HRRR value minus the mean (HRRR - observation) difference at station X during all January months in WY2020-2023

#### Scenario: Bias correction parameter storage

- **WHEN** the bias-corrected HRRR baseline is fitted
- **THEN** it stores a correction table of station x variable x 12 months, reusable across evaluation runs without recomputation
