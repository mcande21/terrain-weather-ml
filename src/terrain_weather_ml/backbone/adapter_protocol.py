"""WeatherAdapter protocol for backbone adapters.

Defines the structural interface that all weather backbone adapters must
satisfy. Both NWPPassthroughAdapter and StormCastAdapter implement this
protocol, allowing the downscaling head, training pipeline, and inference
service to be adapter-agnostic.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class WeatherAdapter(Protocol):
    """Protocol for weather backbone adapters.

    Any adapter that extracts surface weather variables from NWP data
    must implement this interface. The extract method accepts a dict
    mapping variable names to tensors and returns a batched
    (batch, C_WEATHER, H, W) tensor with the 6 canonical surface
    variables: T2M, U10, V10, PRATE, SP, BLH.
    """

    def extract(self, nwp_data: dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract surface weather variables from NWP data.

        Args:
            nwp_data: Dict mapping NWP variable names to tensors.
                Contents depend on the adapter:
                - NWPPassthrough: raw HRRR/ERA5 fields
                - StormCast: 'stormcast_input' tensor + HRRR passthrough vars

        Returns:
            Weather tensor of shape (batch, 6, H, W) with channels ordered
            as: T2M, U10, V10, PRATE, SP, BLH.
        """
        ...
