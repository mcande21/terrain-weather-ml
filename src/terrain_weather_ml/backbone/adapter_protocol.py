"""Weather adapter protocol (Design Decision D4).

Defines the structural interface that all backbone adapters must satisfy.
Both NWPPassthroughAdapter and StormCastAdapter implement this protocol.

NOTE: This is a minimal stub created alongside the StormCast experimental
refactor. The passthrough adapter work may extend this protocol. If
adapter_protocol.py already exists from that parallel work, reconcile
by keeping the superset of method signatures.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class WeatherAdapter(Protocol):
    """Protocol for weather backbone adapters.

    All adapters must provide an extract() method that produces
    a (batch, C_WEATHER, H, W) tensor from NWP data.

    C_WEATHER = 6: T2M, U10, V10, MSLP, PRATE, BLH.
    """

    def extract(self, nwp_data: dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract weather variables from NWP data.

        Args:
            nwp_data: Dictionary mapping variable names to tensors.
                Contents depend on the adapter:
                - NWPPassthrough: raw HRRR/ERA5 fields
                - StormCast: 'stormcast_input' tensor + HRRR passthrough vars

        Returns:
            Weather tensor of shape (batch, 6, H, W).
        """
        ...
