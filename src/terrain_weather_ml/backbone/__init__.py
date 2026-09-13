"""Weather backbone adapter module.

Provides weather adapters that extract 6 surface variables for the
downscaling head. NWPPassthroughAdapter is the default (raw NWP field
extraction); StormCastAdapter is available as an experimental option.
"""

from terrain_weather_ml.backbone.adapter_protocol import WeatherAdapter
from terrain_weather_ml.backbone.hrrr import HRRRConditioner, HRRRConfig, HRRRData
from terrain_weather_ml.backbone.interface import (
    InterfaceContractError,
    validate_interface_contract,
)
from terrain_weather_ml.backbone.nwp_passthrough import NWPPassthroughAdapter
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
    "NWPPassthroughAdapter",
    "StormCastAdapter",
    "StormCastConfig",
    "WeatherAdapter",
    "validate_interface_contract",
]
