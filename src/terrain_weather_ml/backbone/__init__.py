"""StormCast backbone adapter module.

Frozen NVIDIA StormCast with LoRA adapters for regional specialization.
Extracts 6 surface weather variables for the downscaling head.
"""

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
    "C_WEATHER",
    "SURFACE_VARIABLE_NAMES",
    "HRRRConditioner",
    "HRRRConfig",
    "HRRRData",
    "InterfaceContractError",
    "LoRALinear",
    "StormCastAdapter",
    "StormCastConfig",
    "validate_interface_contract",
]
