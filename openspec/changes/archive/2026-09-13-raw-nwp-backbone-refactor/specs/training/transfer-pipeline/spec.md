## MODIFIED Requirements

### Requirement: Phase 2 — Alpine fine-tuning

The system SHALL fine-tune the pre-trained downscaling head on PeakWeather + Swiss IMIS station observations paired with ERA5 reanalysis coarse inputs (not StormCast, which is US-only and cannot process Alpine data).

- **Data source:** alpine-pipeline (302 PeakWeather + 180 IMIS stations, temporally aligned)
- **Coarse weather input:** ERA5 reanalysis via the ERA5 conditioner (not StormCast — HRRR has no Alpine coverage)
- **Loss function:** MSE on all output variables (wind u/v, temperature, precipitation) + conservation penalty
- **Trainable parameters:** Full downscaling head (no LoRA in Phase 2 — LoRA is StormCast-specific and StormCast is not used here)
- **Initialization:** Downscaling head from Phase 1 checkpoint
- **DEVINE weights:** Wind channels initialized from DEVINE; temperature and precipitation channels begin training from random initialization with wind channels optionally frozen for a configurable warmup period

#### Scenario: Phase 2 initialization from Phase 1

- **WHEN** Phase 2 begins
- **THEN** the system loads the Phase 1 downscaling head checkpoint and verifies the combined model produces valid output before training starts

#### Scenario: Alpine station pairing with coarse input

- **WHEN** a training sample is constructed
- **THEN** it pairs the nearest ERA5 grid cell's weather state at the station's timestamp with the station's observed values and the station's pre-computed terrain features

#### Scenario: ERA5 temporal alignment

- **WHEN** a PeakWeather observation at 10-min resolution is paired with ERA5
- **THEN** the system selects the nearest ERA5 hourly timestep within 30-minute tolerance

### Requirement: Phase 3 — Colorado LoRA adaptation

The system SHALL adapt the Alpine-trained model to the Colorado domain using SNOTEL station data (118 stations). Phase 3 uses HRRR as coarse weather input via the NWP passthrough adapter.

- **Data source:** Colorado SNOTEL stations (118 stations, hourly)
- **Coarse weather input:** HRRR via the NWP passthrough adapter (3km, all 6 surface variables natively available)
- **Loss function:** MSE on available variables + conservation penalty
- **Trainable parameters:** Downscaling head fine-tuning (optionally with reduced learning rate on layers initialized from Phase 2)
- **Domain alignment:** Quantile mapping between Alpine ERA5 and Colorado HRRR variable distributions following the RainShift protocol — per-variable empirical CDF matching applied to the coarse input

#### Scenario: Quantile mapping application

- **WHEN** Phase 3 training processes a Colorado SNOTEL sample
- **THEN** the coarse HRRR weather input is quantile-mapped from the Colorado distribution to the Alpine distribution before being fed to the model, bridging the domain gap

#### Scenario: Phase 3 checkpoint

- **WHEN** Phase 3 completes
- **THEN** the saved checkpoint contains Phase 2 downscaling head weights (fine-tuned), the quantile mapping parameters, and optionally LoRA weights if the experimental StormCast backbone was active

#### Scenario: Phase 3 uses HRRR not StormCast

- **WHEN** Phase 3 processes a training sample
- **THEN** the coarse weather input comes from the NWP passthrough adapter reading raw HRRR data, not from StormCast inference
