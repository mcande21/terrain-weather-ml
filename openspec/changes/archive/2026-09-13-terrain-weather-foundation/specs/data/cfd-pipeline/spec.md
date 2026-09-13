## Purpose

Prepares synthetic CFD training data from WindNinja solver runs and the HuggingFace wind-cfd-trial dataset for pre-training the terrain downscaling head.

## ADDED Requirements

### Requirement: wind-cfd-trial dataset ingestion

The system SHALL download and parse the wind-cfd-trial dataset from HuggingFace. The dataset contains approximately 10,000 RANS (Reynolds-Averaged Navier-Stokes) simulation cases. Each case consists of a DEM patch with specified wind boundary conditions and the resulting 3D wind field solution.

#### Scenario: Successful wind-cfd-trial download

- **WHEN** the pipeline is run with a HuggingFace token
- **THEN** the system downloads and indexes all available cases from the wind-cfd-trial dataset

### Requirement: Input/output tensor format

Each CFD training sample SHALL be structured as:

**Input:**
- DEM patch tensor: `(1, H, W)` — elevation grid for the simulation domain
- Terrain feature tensor: `(C_terrain, H, W)` — derived features from the terrain encoder
- Boundary wind conditions: `(3,)` — wind speed (m/s), wind direction (degrees), reference height (m)

**Output:**
- Wind field tensor: `(3, H, W)` — u, v, w wind components at surface level across the domain

The spatial dimensions (H, W) SHALL match the DEM patch grid.

#### Scenario: Load a single CFD case

- **WHEN** a training dataloader requests case N from the CFD dataset
- **THEN** the system returns the input tuple (DEM patch, terrain features, boundary conditions) and the target wind field tensor with matching spatial dimensions

### Requirement: WindNinja synthetic generation

The system SHALL support generating additional synthetic training cases by running the WindNinja solver on arbitrary DEM patches with sampled wind conditions. Generated cases SHALL follow the same input/output tensor format as wind-cfd-trial data.

#### Scenario: Generate synthetic case

- **WHEN** a DEM patch and wind boundary conditions are provided
- **THEN** the system runs WindNinja and returns the solved wind field in the standard tensor format

#### Scenario: WindNinja not installed

- **WHEN** the WindNinja binary is not available on the system
- **THEN** the pipeline skips synthetic generation and operates on wind-cfd-trial data only, logging a warning

### Requirement: Domain randomization

The system SHALL apply domain randomization to CFD training samples: random rotation of wind direction, random scaling of wind speed within physical bounds (0.5-30 m/s), random cropping of DEM patches (minimum 64x64 grid cells), and random additive elevation noise (Gaussian, sigma configurable, default 5m) to improve generalization.

#### Scenario: Augmented training batch

- **WHEN** a training batch of size B is requested
- **THEN** each sample in the batch has independently randomized wind direction, speed scaling, crop region, and elevation noise applied, with the target wind field transformed consistently

### Requirement: Mass conservation labeling

Each CFD sample SHALL include a divergence field computed from the target wind field, allowing the training pipeline to verify that CFD solutions satisfy mass conservation. Samples with maximum absolute divergence exceeding a configurable threshold (default: 0.01 s^-1) SHALL be flagged for exclusion.

#### Scenario: Divergence check on wind field

- **WHEN** a CFD case is loaded
- **THEN** the divergence of the wind field is computed and stored alongside the sample, with a boolean flag indicating whether it passes the conservation threshold
