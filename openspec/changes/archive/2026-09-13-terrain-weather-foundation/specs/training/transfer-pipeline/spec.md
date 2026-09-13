## Purpose

Orchestrates the three-phase training protocol — CFD pre-train, Alpine fine-tune, Colorado LoRA adaptation — with phase-gated checkpointing and cross-domain alignment.

## ADDED Requirements

### Requirement: Phase 1 — CFD pre-training

The system SHALL pre-train the downscaling head on synthetic CFD data from the cfd-pipeline. This phase trains the U-Net to learn terrain-wind interaction physics from idealized simulations.

- **Data source:** wind-cfd-trial + WindNinja synthetic cases
- **Loss function:** MSE on wind field + mass-conservation penalty (lambda=0.1)
- **Trainable parameters:** Full downscaling head (U-Net encoder, decoder, projection layer)
- **StormCast:** Not used in this phase — CFD data provides direct terrain→wind mappings

#### Scenario: Phase 1 completion

- **WHEN** Phase 1 training reaches the configured epoch count or early-stopping criterion
- **THEN** the system saves a Phase 1 checkpoint and logs validation loss on a held-out CFD test set

#### Scenario: Phase 1 checkpoint contains no StormCast weights

- **WHEN** a Phase 1 checkpoint is saved
- **THEN** it contains only downscaling head parameters, not StormCast or LoRA weights

### Requirement: Phase 2 — Alpine fine-tuning

The system SHALL fine-tune the pre-trained downscaling head on PeakWeather + Swiss IMIS station observations paired with HRRR/ERA5 coarse inputs.

- **Data source:** alpine-pipeline (302 PeakWeather + 180 IMIS stations, temporally aligned)
- **Loss function:** MSE on all output variables (wind u/v, temperature, precipitation) + conservation penalty
- **Trainable parameters:** Full downscaling head + LoRA adapters on StormCast
- **Initialization:** Downscaling head from Phase 1 checkpoint; StormCast frozen with freshly injected LoRA adapters
- **DEVINE weights:** Wind channels initialized from DEVINE; temperature and precipitation channels begin training from random initialization with wind channels optionally frozen for a configurable warmup period

#### Scenario: Phase 2 initialization from Phase 1

- **WHEN** Phase 2 begins
- **THEN** the system loads the Phase 1 downscaling head checkpoint, injects LoRA adapters into StormCast, and verifies the combined model produces valid output before training starts

#### Scenario: Alpine station pairing with coarse input

- **WHEN** a training sample is constructed
- **THEN** it pairs the nearest HRRR grid cell's weather state at the station's timestamp with the station's observed values and the station's pre-computed terrain features

### Requirement: Phase 3 — Colorado LoRA adaptation

The system SHALL adapt the Alpine-trained model to the Colorado domain using SNOTEL station data (118 stations) via LoRA fine-tuning and quantile-mapping domain alignment.

- **Data source:** Colorado SNOTEL stations (118 stations, hourly)
- **Loss function:** MSE on available variables + conservation penalty
- **Trainable parameters:** LoRA adapters only (downscaling head frozen from Phase 2)
- **Domain alignment:** Quantile mapping between Alpine and Colorado variable distributions following the RainShift protocol — per-variable empirical CDF matching applied to the coarse input before the adapted model

#### Scenario: Quantile mapping application

- **WHEN** Phase 3 training processes a Colorado SNOTEL sample
- **THEN** the coarse weather input is quantile-mapped from the Colorado distribution to the Alpine distribution before being fed to the model, bridging the domain gap

#### Scenario: Phase 3 checkpoint

- **WHEN** Phase 3 completes
- **THEN** the saved checkpoint contains Phase 2 downscaling head weights (frozen), Phase 3 LoRA weights, and the quantile mapping parameters

### Requirement: Phase-gated checkpointing

The system SHALL enforce phase ordering: Phase 2 requires a Phase 1 checkpoint, Phase 3 requires a Phase 2 checkpoint. Each phase SHALL produce a checkpoint with metadata indicating the phase, epoch, validation metrics, and a hash of the input checkpoint it was initialized from.

#### Scenario: Attempt Phase 3 without Phase 2

- **WHEN** Phase 3 is launched without a valid Phase 2 checkpoint
- **THEN** the system raises an error specifying which checkpoint is missing

#### Scenario: Checkpoint metadata

- **WHEN** any phase saves a checkpoint
- **THEN** the checkpoint file includes: phase number, total epochs trained, best validation loss, training data hash, and parent checkpoint hash

### Requirement: Data flow specification

The training data flow across phases SHALL be:

1. Phase 1: `CFD(DEM patch, boundary conditions) → wind field` (terrain→wind only)
2. Phase 2: `StormCast(HRRR) + terrain features → station observations` (full pipeline)
3. Phase 3: `quantile_map(StormCast(HRRR)) + terrain features → SNOTEL observations` (domain-adapted)

#### Scenario: Data flow validation

- **WHEN** a training run is configured
- **THEN** the system validates that the specified data source matches the expected phase data flow before beginning training
