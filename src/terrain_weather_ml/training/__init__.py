"""Transfer training pipeline for terrain-weather model.

3-phase training protocol:
  Phase 1: CFD pre-training of the downscaling head (wind-only)
  Phase 2: Alpine fine-tuning with StormCast + LoRA + DEVINE init
  Phase 3: Colorado LoRA adaptation with quantile mapping
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
