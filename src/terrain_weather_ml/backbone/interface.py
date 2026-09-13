"""Interface contract validation between StormCast adapter and downscaling head.

Validates that the adapter's output channel count matches the downscaling
head's expected dynamic input channels, and that terrain channel count
matches the expected static input channels.
"""

from __future__ import annotations


class InterfaceContractError(ValueError):
    """Raised when adapter/head interface contract is violated."""


def validate_interface_contract(
    weather_channels: int,
    terrain_channels: int,
    expected_weather: int = 6,
    expected_terrain: int = 17,
) -> None:
    """Validate the interface contract between backbone and downscaling head.

    The StormCast adapter produces weather_channels (expected: 6) and the
    terrain encoder produces terrain_channels (expected: 17). These are
    concatenated to form the 23-channel input to the downscaling head.

    Args:
        weather_channels: Actual number of weather output channels.
        terrain_channels: Actual number of terrain feature channels.
        expected_weather: Expected weather channels (default 6, C_WEATHER).
        expected_terrain: Expected terrain channels (default 17, C_TERRAIN).

    Raises:
        InterfaceContractError: If channel counts don't match expectations.
    """
    if weather_channels != expected_weather:
        raise InterfaceContractError(
            f"Weather channel mismatch: adapter produces {weather_channels} "
            f"weather channels, but downscaling head expects {expected_weather}. "
            f"Check StormCast surface variable extraction indices."
        )

    if terrain_channels != expected_terrain:
        raise InterfaceContractError(
            f"Terrain channel mismatch: encoder produces {terrain_channels} "
            f"terrain channels, but downscaling head expects {expected_terrain}. "
            f"Check terrain feature encoder configuration."
        )
