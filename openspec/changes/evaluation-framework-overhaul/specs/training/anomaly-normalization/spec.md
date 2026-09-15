## Purpose

Removes seasonal signal from training targets by subtracting monthly station climatology before z-scoring, eliminating the +7K bias caused by global normalization collapsing seasonal structure.

## ADDED Requirements

### Requirement: Compute monthly station climatology from training period

The system SHALL compute per-station, per-variable, per-month mean and standard deviation from the training period (WY2020-2023). Months with fewer than 30 valid observations for a station-variable pair SHALL be flagged as insufficient and fall back to the all-month mean for that station-variable.

#### Scenario: Climatology computation

- **WHEN** the anomaly normalizer is fitted on the training dataset
- **THEN** it produces a climatology table with 12 monthly means and standard deviations per station per variable, computed only from WY2020-2023 data

#### Scenario: Insufficient monthly data

- **WHEN** a station-variable-month combination has fewer than 30 valid observations in the training period
- **THEN** the system uses the all-month mean for that station-variable pair and logs a warning identifying the gap

### Requirement: Normalize targets as anomalies then z-score

The system SHALL normalize targets in two steps: (1) subtract the monthly climatology for the target's station, variable, and calendar month to produce an anomaly, then (2) apply per-variable z-score normalization (subtract anomaly mean, divide by anomaly standard deviation) computed across all training anomalies. The z-score statistics SHALL be computed after the climatology subtraction, not before.

#### Scenario: Two-step normalization

- **WHEN** a training target value is normalized
- **THEN** the output equals `(observation - monthly_climatology_mean) / anomaly_std` where `anomaly_std` is the standard deviation of all training anomalies for that variable

#### Scenario: Normalization preserves sample count

- **WHEN** a batch of N target values is normalized
- **THEN** exactly N normalized values are produced with no samples dropped

### Requirement: Denormalize predictions back to physical units

The system SHALL reverse the normalization by applying inverse z-score (multiply by anomaly std, add anomaly mean) then adding back the monthly climatology. The denormalized output SHALL be in the original physical units (K for temperature, m/s for wind, mm for precipitation).

#### Scenario: Round-trip consistency

- **WHEN** a target value is normalized and then denormalized
- **THEN** the result matches the original value within floating-point tolerance (< 1e-5 absolute error)

#### Scenario: Inference-time denormalization

- **WHEN** model predictions are denormalized during inference
- **THEN** the system uses the stored climatology and z-score parameters from the training checkpoint, not recomputed values

### Requirement: Climatology and normalization params saved in checkpoint

The system SHALL serialize the monthly climatology table and the anomaly z-score parameters (per-variable mean and std of anomalies) into the model checkpoint. Loading a checkpoint SHALL restore the normalizer to a fully functional state without access to the original training data.

#### Scenario: Checkpoint serialization

- **WHEN** a training checkpoint is saved
- **THEN** it contains `climatology` (station x variable x 12 months), `anomaly_mean` (per variable), and `anomaly_std` (per variable)

#### Scenario: Checkpoint loading

- **WHEN** a checkpoint is loaded for inference or continued training
- **THEN** the normalizer can normalize and denormalize without refitting
