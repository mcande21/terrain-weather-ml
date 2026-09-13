## MODIFIED Requirements

### Requirement: StormCast checkpoint loading

The system SHALL load StormCast weights from the nvidia/stormcast-v1-era5-hrrr checkpoint. The model SHALL be loaded in inference-only mode with all original parameters frozen (requires_grad=False). The loader SHALL handle NVIDIA Modulus `.mdlus` file format by repacking the zip archive to strip the `model/` prefix before loading the state dict.

StormCast is an **experimental** backbone, not the default. The default backbone is the NWP passthrough adapter.

#### Scenario: Successful model load

- **WHEN** the StormCast checkpoint path or HuggingFace identifier is provided
- **THEN** the system loads the full model with all parameters frozen and reports the parameter count and memory footprint

#### Scenario: Checkpoint not found

- **WHEN** the checkpoint path does not exist and the HuggingFace download fails
- **THEN** the system raises a clear error with instructions for obtaining the checkpoint

#### Scenario: NVIDIA Modulus format loading

- **WHEN** the checkpoint file has a `.mdlus` extension
- **THEN** the system repacks the zip archive (stripping the `model/` entry prefix) and loads the state dict via `torch.load`

### Requirement: Surface variable subset extraction

StormCast outputs 99 state variables at 3km resolution. The system SHALL extract only the 4 surface variables that StormCast actually provides:

1. 2m temperature (T2m) — output index 2
2. 10m u-wind component (U10) — output index 0
3. 10m v-wind component (V10) — output index 1
4. Mean sea level pressure (MSLP) — output index 3

The extraction SHALL produce a tensor of shape `(4, H, W)`. Note: precipitation rate, surface pressure, and boundary layer height are NOT in StormCast's 99-channel output — they must be supplied from raw HRRR via passthrough.

#### Scenario: Extract surface variables

- **WHEN** StormCast produces a full 99-variable output
- **THEN** the system extracts the 4 surface variables by their correct indices [2, 0, 1, 3] and returns a `(4, H, W)` tensor

#### Scenario: Variable index validation

- **WHEN** the StormCast checkpoint is loaded
- **THEN** the system verifies that the expected variable indices map to the correct physical quantities by checking the model's variable metadata from `metadata.zarr.zip`

### Requirement: LoRA adapter injection

The system SHALL inject LoRA (Low-Rank Adaptation) adapters into StormCast's `affine` linear layers, not attention QKV (which is a fused Conv2d in the real architecture). Adapter configuration:

- **Target modules:** `affine` layers (nn.Linear projections in each UNet block)
- **Rank:** r=8 (configurable)
- **Alpha:** 16 (configurable, scaling factor)
- **Dropout:** 0.0 (configurable)

Only LoRA parameters SHALL be trainable. The original StormCast parameters SHALL remain frozen.

#### Scenario: LoRA parameter count

- **WHEN** LoRA adapters with rank r=8 are injected into the real StormCast model
- **THEN** the number of trainable parameters is less than 1% of total StormCast parameters

#### Scenario: Save and load LoRA weights

- **WHEN** LoRA adapters are trained and saved
- **THEN** the saved checkpoint contains only the LoRA parameters (not the full StormCast weights), and loading a LoRA checkpoint onto a fresh StormCast model reproduces the adapted model exactly

### Requirement: Interface contract with downscaling head

When used as the backbone, StormCast provides 4 weather channels. To match the C_WEATHER=6 interface contract, the remaining 2 channels (PRATE, SP) SHALL be supplemented from raw HRRR passthrough, and BLH derived or omitted. The combined output SHALL be `(6, H_coarse, W_coarse)`.

#### Scenario: Shape validation at initialization

- **WHEN** the StormCast adapter and downscaling head are initialized together
- **THEN** the system verifies that the combined output (4 StormCast + passthrough) matches the downscaling head's expected 6 dynamic input channels

### Requirement: HRRR input conditioning

The system SHALL accept HRRR analysis or forecast grids as input to StormCast for inference. The input SHALL be formatted to match StormCast's expected input schema (variable ordering, normalization, grid projection).

#### Scenario: HRRR grid to StormCast input

- **WHEN** a HRRR GRIB2 file is provided
- **THEN** the system extracts the required input variables, reprojects to StormCast's expected grid, normalizes, and produces the input tensor for a forward pass
