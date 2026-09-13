## Purpose

Integrates frozen NVIDIA StormCast as the coarse weather backbone with LoRA adapters for regional specialization, providing the dynamic weather input to the downscaling head.

## ADDED Requirements

### Requirement: StormCast checkpoint loading

The system SHALL load StormCast weights from the nvidia/stormcast-v1-era5-hrrr checkpoint. The model SHALL be loaded in inference-only mode with all original parameters frozen (requires_grad=False).

#### Scenario: Successful model load

- **WHEN** the StormCast checkpoint path or HuggingFace identifier is provided
- **THEN** the system loads the full model with all parameters frozen and reports the parameter count and memory footprint

#### Scenario: Checkpoint not found

- **WHEN** the checkpoint path does not exist and the HuggingFace download fails
- **THEN** the system raises a clear error with instructions for obtaining the checkpoint

### Requirement: Surface variable subset extraction

StormCast outputs 99 state variables at 3km resolution. The system SHALL extract only the surface subset consumed by the downscaling head:

1. 2m temperature (T2m)
2. 10m u-wind component (U10)
3. 10m v-wind component (V10)
4. Precipitation rate
5. Surface pressure (SP)
6. Boundary layer height (BLH)

The extraction SHALL produce a tensor of shape `(6, H, W)` from the full StormCast output.

#### Scenario: Extract surface variables

- **WHEN** StormCast produces a full 99-variable output
- **THEN** the system extracts the 6 surface variables by index and returns a `(6, H, W)` tensor

#### Scenario: Variable index validation

- **WHEN** the StormCast checkpoint is loaded
- **THEN** the system verifies that the expected variable indices map to the correct physical quantities by checking the model's variable metadata

### Requirement: LoRA adapter injection

The system SHALL inject LoRA (Low-Rank Adaptation) adapters into StormCast's attention layers following the WeatherPEFT pattern. Adapter configuration:

- **Target modules:** Attention layers only (query, key, value projections)
- **Rank:** r=8 (configurable)
- **Alpha:** 16 (configurable, scaling factor)
- **Dropout:** 0.0 (configurable)

Only LoRA parameters SHALL be trainable. The original StormCast parameters SHALL remain frozen.

#### Scenario: LoRA parameter count

- **WHEN** LoRA adapters with rank r=8 are injected
- **THEN** the number of trainable parameters is less than 1% of total StormCast parameters

#### Scenario: Save and load LoRA weights

- **WHEN** LoRA adapters are trained and saved
- **THEN** the saved checkpoint contains only the LoRA parameters (not the full StormCast weights), and loading a LoRA checkpoint onto a fresh StormCast model reproduces the adapted model exactly

### Requirement: Interface contract with downscaling head

The StormCast adapter output tensor shape SHALL be documented and validated at initialization. The contract: StormCast adapter outputs `(6, H_coarse, W_coarse)` at 3km resolution, which the downscaling head consumes as its dynamic input branch. The spatial dimensions SHALL be determined by the HRRR grid subset for the target region.

#### Scenario: Shape validation at initialization

- **WHEN** the StormCast adapter and downscaling head are initialized together
- **THEN** the system verifies that the adapter's output channel count matches the downscaling head's expected dynamic input channels, raising an error on mismatch

### Requirement: HRRR input conditioning

The system SHALL accept HRRR analysis or forecast grids as input to StormCast for inference. The input SHALL be formatted to match StormCast's expected input schema (variable ordering, normalization, grid projection).

#### Scenario: HRRR grid to StormCast input

- **WHEN** a HRRR GRIB2 file is provided
- **THEN** the system extracts the required input variables, reprojects to StormCast's expected grid, normalizes, and produces the input tensor for a forward pass

**Alternative backbone note:** ECMWF AIFS (CC BY 4.0) is the most permissive-license alternative backbone with the lowest distribution-shift risk relative to StormCast. It operates on the same HRRR-scale resolution and produces comparable surface variables. If StormCast proves unsuitable (memory, weight compatibility, or licensing concerns), AIFS is the recommended fallback with minimal architecture changes required.
