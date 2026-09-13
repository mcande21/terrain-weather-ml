"""Weather backbone adapter module.

Default backbone is NWP passthrough (raw HRRR/ERA5 field extraction).
StormCast is available as an experimental option via create_adapter().

Use create_adapter(backbone="passthrough") for default behavior, or
create_adapter(backbone="stormcast", ...) for experimental StormCast.
"""

from __future__ import annotations

from typing import Any

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
    HRRR_PASSTHROUGH_NAMES,
    STORMCAST_SURFACE_INDICES,
    STORMCAST_SURFACE_NAMES,
    SURFACE_VARIABLE_NAMES,
    LoRALinear,
    StormCastAdapter,
    StormCastConfig,
)


class _NWPPassthroughStub:
    """Minimal passthrough stub for backbone selection.

    Returns raw NWP fields as a (B, C_WEATHER, H, W) tensor.
    This is a temporary stub until the full NWPPassthroughAdapter
    is available from the parallel passthrough work.
    """

    experimental = False

    def extract(self, nwp_data: dict) -> Any:
        """Extract weather variables from raw NWP data dict.

        Expects nwp_data to contain keys for each surface variable
        (T2M, U10, V10, MSLP, PRATE, BLH) as (B, 1, H, W) tensors.
        """
        import torch

        channels = []
        for name in SURFACE_VARIABLE_NAMES:
            channels.append(nwp_data[name])
        return torch.cat(channels, dim=1)


def create_adapter(
    backbone: str = "passthrough",
    **kwargs: Any,
) -> WeatherAdapter | StormCastAdapter | _NWPPassthroughStub:
    """Factory function to create a weather backbone adapter.

    Args:
        backbone: Adapter type. "passthrough" (default) for raw NWP
            field extraction. "stormcast" for experimental StormCast.
        **kwargs: Passed to the adapter's config (StormCastConfig for
            stormcast backbone).

    Returns:
        An adapter satisfying the WeatherAdapter protocol.

    Raises:
        ValueError: If backbone name is not recognized.
    """
    if backbone == "passthrough":
        return NWPPassthroughAdapter()

    if backbone == "stormcast":
        config = StormCastConfig(**kwargs)
        return StormCastAdapter(config)

    raise ValueError(
        f"Unknown backbone: {backbone!r}. "
        f"Valid options: 'passthrough', 'stormcast'"
    )


__all__ = [
    "CDS_VARIABLE_MAP",
    "C_WEATHER",
    "HRRR_PASSTHROUGH_NAMES",
    "PREDEFINED_DOMAINS",
    "STORMCAST_SURFACE_INDICES",
    "STORMCAST_SURFACE_NAMES",
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
    "create_adapter",
    "validate_interface_contract",
]
