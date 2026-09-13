## Purpose

Thin NWP variable extractor that reads raw HRRR or ERA5 surface fields and provides the 6-channel weather tensor to the downscaling head without neural inference.

## ADDED Requirements

### Requirement: NWP source abstraction

The system SHALL accept raw NWP data from either HRRR (3km, CONUS) or ERA5 (31km, global) and extract the same 6 surface variables regardless of source. The adapter SHALL implement the same interface as StormCastAdapter so the downscaling head is source-agnostic.

#### Scenario: HRRR source extraction

- **WHEN** a HRRR analysis or forecast grid is provided
- **THEN** the adapter extracts T2M, U10, V10, PRATE, SP, BLH and returns a `(6, H, W)` tensor at HRRR's native 3km resolution

#### Scenario: ERA5 source extraction

- **WHEN** an ERA5 reanalysis field is provided
- **THEN** the adapter extracts the equivalent 6 variables (2t, 10u, 10v, tp, sp, blh), maps them to the project variable schema, and returns a `(6, H_era5, W_era5)` tensor at ERA5's native resolution

#### Scenario: Unknown NWP source

- **WHEN** input data does not match HRRR or ERA5 expected format
- **THEN** the system raises a clear error identifying the expected formats

### Requirement: Variable mapping and unit normalization

The system SHALL map NWP-native variable names to the project's canonical 6-variable schema and normalize units to a consistent convention.

| Project Name | HRRR Name | ERA5 Name | Output Unit |
|---|---|---|---|
| T2M | TMP:2m / t2m | 2t | Kelvin |
| U10 | UGRD:10m / u10 | 10u | m/s |
| V10 | VGRD:10m / v10 | 10v | m/s |
| PRATE | PRATE / prate | tp (accumulated → rate) | kg/m²/s |
| SP | PRES:surface / sp | sp | Pa |
| BLH | HPBL / blh | blh | m |

#### Scenario: ERA5 precipitation conversion

- **WHEN** ERA5 provides total precipitation as an accumulated quantity (meters)
- **THEN** the adapter converts to an instantaneous rate (kg/m²/s) by differencing consecutive timesteps and dividing by the interval

#### Scenario: HRRR variable name variants

- **WHEN** HRRR data uses cfgrib ECMWF-style short names (t2m, u10, v10, sp, blh, prate) instead of GRIB2 parameter names
- **THEN** the adapter accepts both naming conventions

### Requirement: Interface contract preservation

The adapter output SHALL match the existing StormCast adapter interface contract: C_WEATHER=6 channels, producing a tensor of shape `(batch, 6, H, W)` consumable by the downscaling head without modification. The `validate_interface_contract()` function SHALL pass with the NWP passthrough adapter.

#### Scenario: Drop-in replacement for StormCast

- **WHEN** the NWP passthrough adapter replaces StormCastAdapter in the pipeline configuration
- **THEN** the downscaling head, training pipeline, and inference service function without code changes

#### Scenario: Interface validation at initialization

- **WHEN** the adapter is initialized with a target downscaling head
- **THEN** `validate_interface_contract(c_weather=6, c_terrain=17)` passes

### Requirement: No neural inference

The adapter SHALL NOT run any neural network forward pass. It SHALL only perform data extraction, variable mapping, unit conversion, and optional spatial interpolation. Compute cost SHALL be negligible compared to the downscaling head forward pass.

#### Scenario: Compute overhead

- **WHEN** the adapter processes a single HRRR timestep
- **THEN** wall-clock time for extraction is under 100ms on CPU (excluding I/O)
