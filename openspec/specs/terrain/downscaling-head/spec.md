# terrain/downscaling-head Specification

## Purpose
Downscales coarse 3km weather grids to sub-km point predictions using a DEVINE-inspired U-Net conditioned on static terrain features, with mass-conservation enforcement on wind output.

## Requirements

### Requirement: Dual-branch input architecture

The downscaling head SHALL accept two separate input branches:

1. **Static branch:** Terrain feature tensor `(C_terrain, H_fine, W_fine)` from the terrain encoder — computed once per region, does not change per timestep.
2. **Dynamic branch:** Coarse weather tensor `(C_weather, H_coarse, W_coarse)` from StormCast or HRRR — changes every timestep.

The system SHALL upsample the coarse weather tensor to match the fine terrain grid resolution before concatenation in the U-Net encoder.

#### Scenario: Forward pass with both branches

- **WHEN** a terrain tensor at 100m resolution and a weather tensor at 3km resolution are provided
- **THEN** the system upsamples the weather tensor to 100m, concatenates with the terrain tensor along the channel dimension, and produces sub-km output at the fine grid resolution

#### Scenario: Terrain tensor resolution mismatch

- **WHEN** the terrain tensor and target output resolution differ
- **THEN** the system raises an error indicating the terrain tensor must be pre-computed at the target output resolution

### Requirement: Multi-variable output

The downscaling head SHALL produce predictions for: wind speed (m/s), wind direction (degrees), 2m temperature (K), and precipitation rate (mm/hr). The output tensor shape SHALL be `(C_out, H_fine, W_fine)` where C_out includes u-wind, v-wind, temperature, and precipitation channels.

Wind output SHALL use u/v component representation internally and convert to speed/direction at the output interface.

#### Scenario: Multi-variable prediction

- **WHEN** a forward pass completes
- **THEN** the output contains four physical variables (wind-u, wind-v, temperature, precipitation) at the fine grid resolution

### Requirement: U-Net architecture with skip connections

The downscaling head SHALL use an encoder-decoder U-Net architecture with skip connections. The encoder SHALL have at least 4 downsampling stages. Skip connections SHALL concatenate encoder features with decoder features at each resolution level.

#### Scenario: Architecture preserves spatial resolution

- **WHEN** an input of shape `(C, H, W)` is processed
- **THEN** the output has the same spatial dimensions `(H, W)`

### Requirement: Mass-conservation constraint on wind

The system SHALL enforce mass conservation on wind output via a divergence-free projection layer applied after the U-Net decoder. The projection SHALL compute the 2D divergence of the (u, v) wind field and subtract the gradient of a pressure-like correction field obtained by solving a Poisson equation, producing a divergence-free wind field.

This constraint SHALL be applied as a hard architectural constraint (projection layer), not a soft penalty.

#### Scenario: Wind output is divergence-free

- **WHEN** the downscaling head produces a wind prediction
- **THEN** the 2D divergence of the output wind field is numerically zero (absolute max divergence below 1e-5 s^-1)

#### Scenario: Non-wind variables unaffected

- **WHEN** the divergence-free projection is applied
- **THEN** temperature and precipitation channels pass through unchanged

### Requirement: DEVINE weight initialization for wind channels

The system SHALL support initializing wind output channels from pre-trained DEVINE weights (louisletoumelin/wind_downscaling_cnn). DEVINE weights apply to wind channels ONLY. Temperature and precipitation channels SHALL be initialized randomly. During initial fine-tuning, the system SHALL support freezing wind channels (using DEVINE weights) while training temperature and precipitation channels.

#### Scenario: Load DEVINE pretrained weights

- **WHEN** a DEVINE checkpoint path is provided at initialization
- **THEN** the wind-related encoder/decoder weights are loaded from DEVINE, and temperature/precipitation channels are randomly initialized

#### Scenario: Freeze wind channels during fine-tuning

- **WHEN** the freeze-wind configuration flag is set
- **THEN** gradients for wind channel parameters loaded from DEVINE are disabled, while temperature/precipitation parameters train normally

### Requirement: Loss function

The training loss SHALL be: `L = MSE(predicted, target) + lambda_div * conservation_penalty + lambda_oro * orographic_penalty`, where the conservation penalty is the mean squared divergence of the wind field before projection, and the orographic penalty (optional) enforces that predicted precipitation increases on windward slopes and decreases on leeward slopes relative to terrain gradient and wind direction (formulated as a correlation penalty). Lambda_div SHALL be configurable (default: 0.1). Lambda_oro SHALL be configurable (default: 0.0, disabled until empirically validated). The MSE term SHALL be computed per-variable with configurable per-variable weights (default: equal weighting).

#### Scenario: Loss computation

- **WHEN** a training step completes
- **THEN** the loss includes the per-variable MSE, the divergence penalty (computed on the pre-projection wind field), and optionally the orographic precipitation penalty when lambda_oro > 0
