## ADDED Requirements

### Requirement: Default backbone selection

The inference service SHALL use the NWP passthrough adapter as its default backbone, reading raw HRRR data for coarse weather input. StormCast SHALL be available as an optional experimental backbone selectable via configuration.

#### Scenario: Default inference uses HRRR passthrough

- **WHEN** the inference service starts with default configuration
- **THEN** it initializes the NWP passthrough adapter and serves predictions from raw HRRR data without running neural backbone inference

#### Scenario: Experimental StormCast backbone

- **WHEN** the configuration specifies `backbone: stormcast`
- **THEN** the inference service loads the StormCast adapter with LoRA weights and uses neural inference for the 4 refined variables, supplementing with HRRR passthrough for the remaining 2

#### Scenario: Single data source at inference

- **WHEN** the service is running with the default NWP passthrough backbone
- **THEN** HRRR is the only external data dependency — no ERA5 access is required, and the service's update latency matches HRRR's cycle frequency (~hourly)
