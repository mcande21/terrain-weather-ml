"""Transfer training pipeline for terrain-weather model.

3-phase training protocol:
  Phase 1: CFD pre-training of the downscaling head (wind-only)
  Phase 2: Alpine fine-tuning with ERA5 passthrough + DEVINE init
  Phase 3: Colorado adaptation with HRRR passthrough + quantile mapping
"""

from terrain_weather_ml.training.checkpoint import (
    CheckpointMetadata,
    PhaseCheckpoint,
    load_phase_checkpoint,
    save_phase_checkpoint,
    validate_phase_prerequisites,
)
from terrain_weather_ml.training.datasets import (
    AlpinePhaseDataset,
    CFDPhaseDataset,
    ColoradoPhaseDataset,
    ERA5StationDataset,
)
from terrain_weather_ml.training.phases import (
    Phase1Trainer,
    Phase2Trainer,
    Phase3Trainer,
    PhaseConfig,
)
from terrain_weather_ml.training.quantile_mapping import (
    QuantileMapper,
    Season,
    compute_empirical_cdf,
)

__all__ = [
    "AlpinePhaseDataset",
    "CFDPhaseDataset",
    "CheckpointMetadata",
    "ColoradoPhaseDataset",
    "ERA5StationDataset",
    "Phase1Trainer",
    "Phase2Trainer",
    "Phase3Trainer",
    "PhaseCheckpoint",
    "PhaseConfig",
    "QuantileMapper",
    "Season",
    "compute_empirical_cdf",
    "load_phase_checkpoint",
    "save_phase_checkpoint",
    "validate_phase_prerequisites",
]
