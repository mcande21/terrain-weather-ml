"""NWP passthrough adapter.

Thin extractor that reads raw HRRR or ERA5 surface fields and provides
the 6-channel weather tensor to the downscaling head without neural
inference. No model weights, no GPU required.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical variable schema (order matters -- matches C_WEATHER indexing)
# ---------------------------------------------------------------------------

CANONICAL_VARIABLES = ["T2M", "U10", "V10", "PRATE", "SP", "BLH"]

# HRRR cfgrib-style short names -> canonical
HRRR_VARIABLE_MAP: dict[str, str] = {
    "t2m": "T2M",
    "u10": "U10",
    "v10": "V10",
    "prate": "PRATE",
    "sp": "SP",
    "blh": "BLH",
}

# ERA5 CDS short names -> canonical
ERA5_VARIABLE_MAP: dict[str, str] = {
    "2t": "T2M",
    "10u": "U10",
    "10v": "V10",
    "tp": "PRATE",  # requires accumulation -> rate conversion
    "sp": "SP",
    "blh": "BLH",
}

# Default ERA5 accumulation interval (seconds) for tp -> rate conversion
_DEFAULT_ERA5_DT_SECONDS = 3600


def _detect_source(
    keys: set[str],
) -> tuple[str, dict[str, str]]:
    """Detect NWP source from variable names.

    Returns:
        Tuple of (source_name, variable_map).

    Raises:
        ValueError: If variable names don't match HRRR or ERA5 conventions,
            or if required variables are missing.
    """
    hrrr_keys = set(HRRR_VARIABLE_MAP.keys())
    era5_keys = set(ERA5_VARIABLE_MAP.keys())

    if hrrr_keys.issubset(keys):
        return "hrrr", HRRR_VARIABLE_MAP
    if era5_keys.issubset(keys):
        return "era5", ERA5_VARIABLE_MAP

    # Check for partial matches to give better error messages
    hrrr_overlap = keys & hrrr_keys
    era5_overlap = keys & era5_keys

    if hrrr_overlap and len(hrrr_overlap) >= len(era5_overlap):
        missing = hrrr_keys - keys
        raise ValueError(
            f"Missing required HRRR variables: {sorted(missing)}. "
            f"Found: {sorted(hrrr_overlap)}."
        )
    if era5_overlap:
        missing = era5_keys - keys
        raise ValueError(
            f"Missing required ERA5 variables: {sorted(missing)}. "
            f"Found: {sorted(era5_overlap)}."
        )

    raise ValueError(
        f"Cannot recognize NWP source from variable names: {sorted(keys)}. "
        f"Expected HRRR keys {sorted(hrrr_keys)} or ERA5 keys {sorted(era5_keys)}."
    )


class NWPPassthroughAdapter:
    """Passthrough adapter extracting surface variables from raw NWP data.

    Accepts a dict mapping NWP variable names (HRRR cfgrib-style or ERA5
    short names) to 2D tensors. Auto-detects the source, maps variables to
    the canonical 6-variable schema, and returns a (1, 6, H, W) tensor.

    For ERA5 data, accumulated total precipitation (tp, meters) is converted
    to an instantaneous rate (kg/m^2/s) using the configured accumulation
    interval.

    Args:
        era5_dt_seconds: ERA5 accumulation interval in seconds for tp -> rate
            conversion. Default 3600 (hourly ERA5).
    """

    def __init__(self, era5_dt_seconds: int = _DEFAULT_ERA5_DT_SECONDS) -> None:
        self.era5_dt_seconds = era5_dt_seconds

    def extract(self, nwp_data: dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract 6 surface weather variables from NWP data.

        Args:
            nwp_data: Dict mapping variable names to 2D tensors (H, W).

        Returns:
            Weather tensor of shape (1, 6, H, W) with channels:
            T2M, U10, V10, PRATE, SP, BLH.

        Raises:
            ValueError: If source unrecognized, variables missing, or
                shapes inconsistent.
        """
        keys = set(nwp_data.keys())
        source, var_map = _detect_source(keys)

        # Check all required variables are present
        missing = set(var_map.keys()) - keys
        if missing:
            raise ValueError(
                f"Missing required {source.upper()} variables: {sorted(missing)}"
            )

        # Validate consistent spatial shapes
        shapes = {name: tensor.shape for name, tensor in nwp_data.items()
                  if name in var_map}
        unique_shapes = set(shapes.values())
        if len(unique_shapes) > 1:
            raise ValueError(
                f"Inconsistent spatial shapes across variables: {shapes}"
            )

        # Extract and stack in canonical order
        channels: list[torch.Tensor] = []
        for canonical_name in CANONICAL_VARIABLES:
            # Find source key for this canonical variable
            source_key = next(
                k for k, v in var_map.items() if v == canonical_name
            )
            field = nwp_data[source_key].float()

            # ERA5 tp: accumulated meters -> kg/m^2/s rate
            if source == "era5" and source_key == "tp":
                field = field * 1000.0 / self.era5_dt_seconds

            channels.append(field)

        # Stack: (6, H, W) -> (1, 6, H, W)
        stacked = torch.stack(channels, dim=0).unsqueeze(0)

        logger.debug(
            "Extracted %d variables from %s: shape %s",
            len(channels), source.upper(), stacked.shape,
        )
        return stacked
