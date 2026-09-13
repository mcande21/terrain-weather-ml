"""StormCast backbone adapter module.

Frozen NVIDIA StormCast with LoRA adapters for regional specialization.
Extracts 6 surface weather variables for the downscaling head.
"""

from terrain_weather_ml.backbone.era5_conditioner import (
    CDS_VARIABLE_MAP,
    PREDEFINED_DOMAINS,
    ERA5Conditioner,
    ERA5Config,
    ERA5Data,
    ERA5TemporalAligner,
)
from terrain_weather_ml.backbone.hrrr import HRRRConditioner, HRRRConfig, HRRRData
from terrain_weather_ml.backbone.interface import (
    InterfaceContractError,
    validate_interface_contract,
)
from terrain_weather_ml.backbone.stormcast import (
    C_WEATHER,
    SURFACE_VARIABLE_NAMES,
    LoRALinear,
    StormCastAdapter,
    StormCastConfig,
)

__all__ = [
    "CDS_VARIABLE_MAP",
    "C_WEATHER",
    "PREDEFINED_DOMAINS",
    "SURFACE_VARIABLE_NAMES",
    "ERA5Conditioner",
    "ERA5Config",
    "ERA5Data",
    "ERA5TemporalAligner",
    "HRRRConditioner",
    "HRRRConfig",
    "HRRRData",
    "InterfaceContractError",
    "LoRALinear",
    "StormCastAdapter",
    "StormCastConfig",
    "validate_interface_contract",
]
