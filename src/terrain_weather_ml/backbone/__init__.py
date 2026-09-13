"""Weather backbone adapter module.

Provides weather adapters that extract 6 surface variables for the
downscaling head. NWPPassthroughAdapter is the default (raw NWP field
extraction); StormCastAdapter is available as an experimental option.
"""

from terrain_weather_ml.backbone.adapter_protocol import WeatherAdapter
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
from terrain_weather_ml.backbone.nwp_passthrough import NWPPassthroughAdapter
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
    "NWPPassthroughAdapter",
    "StormCastAdapter",
    "StormCastConfig",
    "WeatherAdapter",
    "validate_interface_contract",
]
