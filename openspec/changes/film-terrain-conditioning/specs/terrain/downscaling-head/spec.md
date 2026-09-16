## MODIFIED Requirements

### Requirement: Dual-branch input architecture

The downscaling head SHALL accept two separate input branches:

1. **Static branch:** Terrain feature tensor `(C_terrain, H_fine, W_fine)` from the terrain encoder — computed once per region, does not change per timestep. Terrain features are processed by a FiLM generator to produce per-decoder-block modulation parameters.
2. **Dynamic branch:** Coarse weather tensor `(C_weather, H_coarse, W_coarse)` from StormCast or HRRR — changes every timestep.

The system SHALL upsample the coarse weather tensor to match the fine terrain grid resolution. The U-Net encoder SHALL receive only the weather tensor (`C_weather` channels). Terrain influence SHALL enter through FiLM conditioning at each decoder stage, not through input concatenation.

#### Scenario: Forward pass with FiLM conditioning

- **WHEN** a terrain tensor at 100m resolution and a weather tensor at 3km resolution are provided
- **THEN** the system upsamples the weather tensor to 100m, passes it through the U-Net encoder (6 input channels), applies FiLM terrain conditioning at each decoder block, and produces sub-km output at the fine grid resolution

#### Scenario: Terrain tensor resolution mismatch

- **WHEN** the terrain tensor and target output resolution differ
- **THEN** the system raises an error indicating the terrain tensor must be pre-computed at the target output resolution

### Requirement: U-Net architecture with skip connections

The downscaling head SHALL use an encoder-decoder U-Net architecture with skip connections. The encoder SHALL have at least 4 downsampling stages. Skip connections SHALL concatenate encoder features with decoder features at each resolution level. Each decoder block SHALL accept FiLM modulation parameters (gamma, beta) and apply them after normalization within the block.

#### Scenario: Architecture preserves spatial resolution

- **WHEN** an input of shape `(C_weather, H, W)` is processed with terrain FiLM conditioning
- **THEN** the output has the same spatial dimensions `(H, W)`

#### Scenario: FiLM conditioning at each decoder stage

- **WHEN** a forward pass is executed
- **THEN** each of the 4 decoder blocks receives and applies its corresponding (gamma, beta) pair from the FiLM generator

### Requirement: DEVINE weight initialization for wind channels

The system SHALL support initializing wind output channels from pre-trained DEVINE weights (louisletoumelin/wind_downscaling_cnn). Due to the change in input channels from `C_terrain + C_weather` to `C_weather` only, DEVINE encoder weights for the initial convolution layer SHALL be adapted or skipped when input channel dimensions do not match. Decoder and final output weights SHALL be loaded where shapes are compatible. The FiLM generator SHALL be initialized independently (gamma=1, beta=0) and SHALL NOT load DEVINE weights.

#### Scenario: Load DEVINE pretrained weights with reduced input channels

- **WHEN** a DEVINE checkpoint trained with 23 input channels is provided and the new head has 6 input channels
- **THEN** the initial convolution layer weights are skipped (shape mismatch), compatible decoder weights are loaded, and wind output channels are initialized from DEVINE

#### Scenario: FiLM generator independent of DEVINE

- **WHEN** DEVINE weights are loaded
- **THEN** the FiLM generator parameters remain at their identity initialization (gamma=1, beta=0) and are fully trainable
