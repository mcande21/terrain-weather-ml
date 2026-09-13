"""Surface roughness from NLCD land cover classification.

Maps NLCD land cover classes to aerodynamic roughness length z0 (meters)
via a lookup table. Falls back to a configurable default when NLCD is unavailable.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# NLCD land cover class -> aerodynamic roughness length z0 (meters)
# Based on standard roughness length classifications
NLCD_ROUGHNESS_TABLE: dict[int, float] = {
    11: 0.0002,   # Open Water
    12: 0.005,    # Perennial Ice/Snow
    21: 0.1,      # Developed, Open Space
    22: 0.5,      # Developed, Low Intensity
    23: 1.0,      # Developed, Medium Intensity
    24: 2.0,      # Developed, High Intensity
    31: 0.005,    # Barren Land
    41: 1.0,      # Deciduous Forest
    42: 1.2,      # Evergreen Forest
    43: 1.1,      # Mixed Forest
    51: 0.2,      # Dwarf Scrub (Alaska only)
    52: 0.15,     # Shrub/Scrub
    71: 0.03,     # Grassland/Herbaceous
    72: 0.02,     # Sedge/Herbaceous (Alaska only)
    73: 0.01,     # Lichens (Alaska only)
    74: 0.005,    # Moss (Alaska only)
    81: 0.05,     # Pasture/Hay
    82: 0.05,     # Cultivated Crops
    90: 1.0,      # Woody Wetlands
    95: 0.1,      # Emergent Herbaceous Wetlands
}


def compute_roughness(
    nlcd: np.ndarray | None,
    shape: tuple[int, int] | None = None,
    default_z0: float = 0.1,
) -> np.ndarray:
    """Map NLCD land cover to aerodynamic roughness length z0.

    Args:
        nlcd: NLCD classification raster (H, W), int values.
            If None, falls back to default roughness.
        shape: Output shape when nlcd is None. Required if nlcd is None.
        default_z0: Default roughness length in meters for missing/unknown classes.

    Returns:
        Roughness z0 array in meters, shape (H, W), float32.
    """
    if nlcd is None:
        if shape is None:
            raise ValueError("shape is required when nlcd is None")
        logger.warning(
            "No NLCD data available; using default roughness z0=%.3f m", default_z0
        )
        return np.full(shape, default_z0, dtype=np.float32)

    roughness = np.full(nlcd.shape, default_z0, dtype=np.float32)
    known_classes = set()

    for class_id, z0 in NLCD_ROUGHNESS_TABLE.items():
        mask = nlcd == class_id
        if mask.any():
            roughness[mask] = z0
            known_classes.add(class_id)

    # Check for unknown classes
    unique_classes = set(np.unique(nlcd).tolist())
    unknown = unique_classes - set(NLCD_ROUGHNESS_TABLE.keys())
    if unknown:
        logger.warning(
            "Unknown NLCD classes %s mapped to default z0=%.3f m",
            sorted(unknown),
            default_z0,
        )

    return roughness
