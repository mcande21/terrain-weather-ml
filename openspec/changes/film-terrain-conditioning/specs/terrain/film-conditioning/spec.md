## Purpose

Generates per-decoder-block affine modulation parameters (gamma, beta) from static terrain features, enabling terrain to condition weather processing at multiple spatial scales via Feature-wise Linear Modulation.

## ADDED Requirements

### Requirement: FiLM generator terrain-to-modulation mapping

The FiLM generator SHALL accept a terrain feature tensor of shape `(B, C_terrain, H, W)` and produce a list of `(gamma, beta)` parameter pairs, one pair per U-Net decoder block. Each gamma and beta tensor SHALL have shape `(B, C_block, 1, 1)` where `C_block` is the feature channel count of the corresponding decoder block.

#### Scenario: FiLM parameter generation for 4-block U-Net

- **WHEN** a terrain tensor of shape `(B, 17, H, W)` is provided and the U-Net has 4 decoder blocks with feature counts `[8f, 4f, 2f, f]`
- **THEN** the generator produces 4 `(gamma, beta)` pairs with shapes `[(B, 8f, 1, 1), (B, 4f, 1, 1), (B, 2f, 1, 1), (B, f, 1, 1)]` respectively

#### Scenario: Batch dimension preserved

- **WHEN** a batch of N terrain tensors is provided
- **THEN** all gamma and beta tensors have batch dimension N, allowing per-sample terrain conditioning

### Requirement: Affine modulation application

Each FiLM modulation block SHALL apply element-wise affine transformation to decoder activations after normalization: `output = gamma * normalized_activation + beta`. The modulation SHALL be applied after GroupNorm and before the activation function within each decoder block.

#### Scenario: Identity initialization

- **WHEN** the FiLM generator is freshly initialized (no training)
- **THEN** gamma parameters are initialized to 1.0 and beta parameters to 0.0, so the initial modulated output equals the unmodulated output

#### Scenario: Modulation changes decoder output

- **WHEN** gamma != 1.0 or beta != 0.0
- **THEN** the decoder block output differs from the unmodulated case by the affine transform applied after normalization

### Requirement: Static terrain caching

The FiLM generator SHALL compute terrain modulation parameters once per region and cache them. The cached parameters SHALL be reusable across multiple timesteps without recomputation, since terrain features are static.

#### Scenario: Temporal reuse of FiLM parameters

- **WHEN** the same terrain tensor is used for multiple weather timesteps
- **THEN** the FiLM parameters are computed once and the same (gamma, beta) pairs are applied to each timestep's decoder activations

### Requirement: Lightweight generator architecture

The FiLM generator SHALL use a compact CNN architecture (no more than 5 convolutional layers) to map terrain features to modulation parameters. The generator SHALL NOT replicate the complexity of the main U-Net encoder. The generator's parameter count SHALL be less than 10% of the main U-Net's parameter count.

#### Scenario: Parameter efficiency

- **WHEN** the FiLM generator is instantiated alongside a U-Net with base_features=64
- **THEN** the generator's total parameter count is less than 10% of the U-Net's total parameter count
