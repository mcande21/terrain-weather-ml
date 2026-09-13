"""CFD data pipeline for synthetic wind field pre-training data.

Downloads and indexes the wind-cfd-trial HuggingFace dataset (zarr stores of
RANS simulations), optionally generates synthetic cases via WindNinja, applies
domain randomization, and computes mass-conservation labels.

The real dataset (souravsud/wind-cfd-trial) stores each case as a zarr store
with 3D wind fields (Ux, Uy, Uz) on a terrain-following mesh. This pipeline
extracts near-surface 2D slices for training.
"""

from __future__ import annotations

import logging
import math
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)

WIND_CFD_TRIAL_REPO = "souravsud/wind-cfd-trial"

# Zarr variable name mapping: real dataset -> pipeline internal names
ZARR_WIND_VAR_MAP = {
    "Ux": "u_wind",
    "Uy": "v_wind",
    "Uz": "w_wind",
}

# Zarr attr name mapping: real dataset attrs -> pipeline metadata keys
ZARR_ATTR_MAP = {
    "reference_velocity_ms": "wind_speed",
    "rotation_deg": "wind_direction",
    "reference_height_m": "reference_height",
}

# Default vertical level index for 3D -> 2D extraction.
# k=0 is the nearest-to-surface level (~2m AGL in real data).
DEFAULT_VERTICAL_LEVEL = 0


def download_zarr_cases(
    repo_id: str = WIND_CFD_TRIAL_REPO,
    cache_dir: str | Path | None = None,
) -> list[Path]:
    """Download zarr stores from a HuggingFace dataset repo.

    Uses ``huggingface_hub.snapshot_download`` to fetch the repo, then
    discovers all ``.zarr`` directories within it.

    Args:
        repo_id: HuggingFace repo ID (e.g. ``souravsud/wind-cfd-trial``).
        cache_dir: Local cache directory. Defaults to HF cache.

    Returns:
        List of paths to zarr store directories.
    """
    from huggingface_hub import snapshot_download

    kwargs: dict[str, Any] = {"repo_id": repo_id, "repo_type": "dataset"}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)

    local_dir = Path(snapshot_download(**kwargs))

    # Discover all .zarr directories (may be nested under data/)
    zarr_paths: list[Path] = []
    for candidate in sorted(local_dir.rglob("*.zarr")):
        if candidate.is_dir():
            zarr_paths.append(candidate)

    if not zarr_paths:
        logger.warning("No .zarr stores found in %s", local_dir)

    return zarr_paths


def load_zarr_case(zarr_path: str | Path) -> dict[str, Any]:
    """Load a single zarr store and return a pipeline-ready dict.

    Reads the zarr store via xarray, maps variable names (Ux->u_wind etc.)
    and attribute names (reference_velocity_ms->wind_speed etc.), and returns
    numpy arrays with metadata.

    Args:
        zarr_path: Path to a ``.zarr`` directory.

    Returns:
        Dict with keys: ``case_id``, ``Ux``, ``Uy``, ``Uz``, ``dem``,
        ``h_agl`` (if present), ``wind_speed``, ``wind_direction``,
        ``reference_height``, and ``zarr_path``.
    """
    import xarray as xr

    ds = xr.open_zarr(str(zarr_path))

    result: dict[str, Any] = {}

    # Extract case_id from attrs
    result["case_id"] = ds.attrs.get("case_id", Path(zarr_path).stem)

    # Map attrs to pipeline metadata keys
    for zarr_attr, pipeline_key in ZARR_ATTR_MAP.items():
        if zarr_attr in ds.attrs:
            result[pipeline_key] = float(ds.attrs[zarr_attr])

    # Extract wind field arrays (keep original Ux/Uy/Uz names)
    for var_name in ("Ux", "Uy", "Uz"):
        if var_name in ds:
            result[var_name] = ds[var_name].values

    # Extract DEM (2D)
    if "dem" in ds:
        result["dem"] = ds["dem"].values

    # Extract height above ground level (3D, for vertical level selection)
    if "h_agl" in ds:
        result["h_agl"] = ds["h_agl"].values

    # Store zarr path for reference
    result["zarr_path"] = str(zarr_path)

    ds.close()

    return result


# ---------------------------------------------------------------------------
# Task 3.1 -- Dataset index
# ---------------------------------------------------------------------------


@dataclass
class CFDDatasetIndex:
    """Index over wind-cfd-trial (or any CFD) dataset cases.

    Stores per-case metadata for O(1) lookup by case_id.
    """

    _cases: dict[str, dict[str, Any]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_huggingface(
        cls,
        repo_id: str = WIND_CFD_TRIAL_REPO,
        cache_dir: str | Path | None = None,
    ) -> CFDDatasetIndex:
        """Download and index all zarr cases from the HuggingFace dataset.

        Each zarr store's ``.attrs`` provide the case metadata
        (reference_velocity_ms, rotation_deg, reference_height_m).
        """
        import xarray as xr

        zarr_paths = download_zarr_cases(repo_id=repo_id, cache_dir=cache_dir)

        index = cls()
        for zarr_path in zarr_paths:
            ds = xr.open_zarr(str(zarr_path))
            attrs = dict(ds.attrs)
            case_id = attrs.get("case_id", Path(zarr_path).stem)

            index._cases[case_id] = {
                "case_id": case_id,
                "wind_speed": float(attrs.get("reference_velocity_ms", 0.0)),
                "wind_direction": float(attrs.get("rotation_deg", 0.0)),
                "reference_height": float(attrs.get("reference_height_m", 0.0)),
                "zarr_path": str(zarr_path),
            }
            ds.close()

        logger.info("Indexed %d CFD cases from %s", len(index), repo_id)
        return index

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._cases)

    def get_case(self, case_id: str) -> dict[str, Any] | None:
        """Return metadata for a case, or None if not found."""
        return self._cases.get(case_id)

    def case_ids(self) -> list[str]:
        """Return all indexed case IDs."""
        return list(self._cases.keys())


# ---------------------------------------------------------------------------
# Task 3.2 -- Input/output tensor formatting
# ---------------------------------------------------------------------------


@dataclass
class CFDSample:
    """A single formatted CFD training sample.

    Attributes:
        dem: Elevation grid, shape ``(1, H, W)``.
        terrain_features: Derived terrain features, shape ``(C_terrain, H, W)``.
            Placeholder until the terrain encoder is integrated.
        boundary_conditions: Wind boundary vector ``(3,)`` --
            (speed m/s, direction degrees, reference height m).
        wind_field: Target wind field ``(3, H, W)`` -- u, v, w components.
        case_id: Original dataset case identifier.
    """

    dem: Tensor
    terrain_features: Tensor
    boundary_conditions: Tensor
    wind_field: Tensor
    case_id: str


def format_cfd_tensors(
    raw: dict[str, Any],
    vertical_level: int = DEFAULT_VERTICAL_LEVEL,
) -> CFDSample:
    """Convert a raw zarr case dict to standardised tensors.

    Handles both the real zarr schema (3D wind fields with Ux/Uy/Uz names)
    and pre-extracted 2D data. When wind fields are 3D (ni, nj, nk), extracts
    a 2D slice at the given vertical level.

    Args:
        raw: Dict with keys ``Ux``, ``Uy``, ``Uz`` (or legacy ``u_wind`` etc.),
             ``dem``, ``wind_speed``, ``wind_direction``, ``reference_height``,
             ``case_id``.
        vertical_level: Vertical level index for 3D->2D extraction.
            Default 0 = nearest surface (~2m AGL in real data).

    Returns:
        A :class:`CFDSample` with correctly shaped 2D tensors.
    """
    dem_np = np.asarray(raw["dem"], dtype=np.float32)
    dem = torch.from_numpy(dem_np).unsqueeze(0)  # (1, H, W)

    # Placeholder terrain features -- just the normalised DEM for now.
    terrain_features = dem.clone()  # (1, H, W)

    boundary_conditions = torch.tensor(
        [raw["wind_speed"], raw["wind_direction"], raw["reference_height"]],
        dtype=torch.float32,
    )

    # Resolve variable names: accept both Ux/Uy/Uz and legacy u_wind/v_wind/w_wind
    u_key = "Ux" if "Ux" in raw else "u_wind"
    v_key = "Uy" if "Uy" in raw else "v_wind"
    w_key = "Uz" if "Uz" in raw else "w_wind"

    u_np = np.asarray(raw[u_key], dtype=np.float32)
    v_np = np.asarray(raw[v_key], dtype=np.float32)
    w_np = np.asarray(raw[w_key], dtype=np.float32)

    # 3D -> 2D extraction: if wind fields have 3 dims, take a vertical slice
    if u_np.ndim == 3:
        u_np = u_np[:, :, vertical_level]
        v_np = v_np[:, :, vertical_level]
        w_np = w_np[:, :, vertical_level]

    u = torch.from_numpy(u_np)
    v = torch.from_numpy(v_np)
    w = torch.from_numpy(w_np)
    wind_field = torch.stack([u, v, w], dim=0)  # (3, H, W)

    return CFDSample(
        dem=dem,
        terrain_features=terrain_features,
        boundary_conditions=boundary_conditions,
        wind_field=wind_field,
        case_id=raw["case_id"],
    )


# ---------------------------------------------------------------------------
# Task 3.3 -- WindNinja synthetic generation
# ---------------------------------------------------------------------------

WINDNINJA_BINARY = "WindNinja_cli"


class WindNinjaGenerator:
    """Generate synthetic CFD training data via the WindNinja solver.

    Gracefully handles environments where WindNinja is not installed.
    """

    def __init__(self, work_dir: str | Path | None = None) -> None:
        self._work_dir = Path(work_dir) if work_dir else None

    def is_available(self) -> bool:
        """Return True if the WindNinja binary is on PATH."""
        return shutil.which(WINDNINJA_BINARY) is not None

    def generate(
        self,
        dem: Tensor,
        boundary_conditions: Tensor,
    ) -> CFDSample | None:
        """Run WindNinja on a DEM patch with given boundary conditions.

        Args:
            dem: Elevation grid tensor ``(1, H, W)``.
            boundary_conditions: ``(3,)`` -- speed, direction, reference height.

        Returns:
            A :class:`CFDSample` with the solved wind field, or ``None``
            if WindNinja is not available.
        """
        if not self.is_available():
            logger.warning(
                "WindNinja binary not found on PATH. "
                "Synthetic generation skipped -- operating on wind-cfd-trial data only."
            )
            return None

        work_dir = self._work_dir or Path("/tmp/windninja_work")
        work_dir.mkdir(parents=True, exist_ok=True)

        _, _h, _w = dem.shape
        speed = boundary_conditions[0].item()
        direction = boundary_conditions[1].item()
        ref_height = boundary_conditions[2].item()

        # Write DEM to temporary file for WindNinja
        dem_path = work_dir / "dem_input.npy"
        np.save(dem_path, dem.squeeze(0).numpy())

        # Run WindNinja CLI
        cmd = [
            WINDNINJA_BINARY,
            "--dem", str(dem_path),
            "--wind_speed", str(speed),
            "--wind_direction", str(direction),
            "--height", str(ref_height),
            "--output_dir", str(work_dir),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode != 0:
            logger.error("WindNinja failed: %s", result.stderr)
            return None

        # Load output wind fields
        u = np.load(work_dir / "u_wind.npy")
        v = np.load(work_dir / "v_wind.npy")
        w_wind = np.load(work_dir / "w_wind.npy")

        wind_field = torch.stack([
            torch.from_numpy(u.astype(np.float32)),
            torch.from_numpy(v.astype(np.float32)),
            torch.from_numpy(w_wind.astype(np.float32)),
        ], dim=0)

        terrain_features = dem.clone()

        return CFDSample(
            dem=dem.clone(),
            terrain_features=terrain_features,
            boundary_conditions=boundary_conditions.clone(),
            wind_field=wind_field,
            case_id="windninja_synthetic",
        )


# ---------------------------------------------------------------------------
# Task 3.4 -- Domain randomization
# ---------------------------------------------------------------------------

MIN_WIND_SPEED = 0.5   # m/s
MAX_WIND_SPEED = 30.0   # m/s
DEFAULT_ELEVATION_NOISE_SIGMA = 5.0   # metres
DEFAULT_MIN_CROP_SIZE = 64


def _copy_sample(sample: CFDSample) -> CFDSample:
    """Deep-copy a CFDSample (all tensors cloned)."""
    return CFDSample(
        dem=sample.dem.clone(),
        terrain_features=sample.terrain_features.clone(),
        boundary_conditions=sample.boundary_conditions.clone(),
        wind_field=sample.wind_field.clone(),
        case_id=sample.case_id,
    )


def augment_wind_direction(sample: CFDSample, rotation_deg: float) -> CFDSample:
    """Rotate wind direction and transform (u, v) components consistently.

    Uses standard 2-D rotation matrix on the horizontal wind components.
    """
    out = _copy_sample(sample)
    rad = math.radians(rotation_deg)
    cos_r = math.cos(rad)
    sin_r = math.sin(rad)

    u = sample.wind_field[0]
    v = sample.wind_field[1]

    out.wind_field[0] = cos_r * u - sin_r * v
    out.wind_field[1] = sin_r * u + cos_r * v

    old_dir = sample.boundary_conditions[1].item()
    new_dir = (old_dir + rotation_deg) % 360.0
    out.boundary_conditions[1] = new_dir

    return out


def augment_wind_speed(sample: CFDSample, scale_factor: float) -> CFDSample:
    """Scale wind speed and wind field components, clamped to physical bounds."""
    out = _copy_sample(sample)
    old_speed = sample.boundary_conditions[0].item()
    new_speed = float(np.clip(old_speed * scale_factor, MIN_WIND_SPEED, MAX_WIND_SPEED))

    # Effective scale after clamping
    effective_scale = new_speed / old_speed if old_speed > 0 else 1.0
    out.boundary_conditions[0] = new_speed
    out.wind_field = sample.wind_field * effective_scale

    return out


def augment_random_crop(
    sample: CFDSample,
    min_size: int = DEFAULT_MIN_CROP_SIZE,
) -> CFDSample:
    """Randomly crop the spatial dimensions of DEM, terrain features, and wind field.

    The crop is at least ``min_size x min_size``. If the input is already
    that small, returns an unmodified copy.
    """
    _, h, w = sample.dem.shape
    if h <= min_size or w <= min_size:
        return _copy_sample(sample)

    crop_h = torch.randint(min_size, h + 1, (1,)).item()
    crop_w = torch.randint(min_size, w + 1, (1,)).item()
    top = torch.randint(0, h - crop_h + 1, (1,)).item()
    left = torch.randint(0, w - crop_w + 1, (1,)).item()

    out = _copy_sample(sample)
    out.dem = sample.dem[:, top : top + crop_h, left : left + crop_w].clone()
    out.terrain_features = sample.terrain_features[
        :, top : top + crop_h, left : left + crop_w
    ].clone()
    out.wind_field = sample.wind_field[
        :, top : top + crop_h, left : left + crop_w
    ].clone()

    return out


def augment_elevation_noise(
    sample: CFDSample,
    sigma: float = DEFAULT_ELEVATION_NOISE_SIGMA,
) -> CFDSample:
    """Add Gaussian noise to the DEM (and terrain features derived from DEM).

    Wind field is NOT modified -- the noise tests model robustness to DEM
    uncertainty, not physics consistency.
    """
    out = _copy_sample(sample)
    noise = torch.randn_like(sample.dem) * sigma
    out.dem = sample.dem + noise
    out.terrain_features = sample.terrain_features + noise
    return out


def augment_sample(
    sample: CFDSample,
    elevation_noise_sigma: float = DEFAULT_ELEVATION_NOISE_SIGMA,
    min_crop_size: int = DEFAULT_MIN_CROP_SIZE,
) -> CFDSample:
    """Apply all domain randomisation augmentations to a sample.

    Each augmentation is independently randomised.
    """
    # 1) Random wind direction rotation [0, 360)
    rotation = torch.rand(1).item() * 360.0
    out = augment_wind_direction(sample, rotation)

    # 2) Random wind speed scaling -- uniform log-scale within physical bounds
    current_speed = out.boundary_conditions[0].item()
    if current_speed > 0:
        min_scale = MIN_WIND_SPEED / current_speed
        max_scale = MAX_WIND_SPEED / current_speed
        scale = min_scale + torch.rand(1).item() * (max_scale - min_scale)
    else:
        scale = 1.0
    out = augment_wind_speed(out, scale)

    # 3) Random crop
    out = augment_random_crop(out, min_size=min_crop_size)

    # 4) Elevation noise
    out = augment_elevation_noise(out, sigma=elevation_noise_sigma)

    return out


# ---------------------------------------------------------------------------
# Task 3.5 -- Mass conservation labeling
# ---------------------------------------------------------------------------

# Default threshold for near-surface 2D slices of 3D terrain-following RANS.
# Real data shows max |div| of 7-10 s^-1 for k=0 slices because the 2D
# divergence (du/dx + dv/dy) omits the vertical dw/dz term that balances
# the 3D continuity equation in terrain-following coordinates. A threshold
# of 10.0 s^-1 filters only extreme numerical artifacts while accepting
# physically valid near-surface flow.
DEFAULT_DIVERGENCE_THRESHOLD = 10.0  # s^-1


def compute_divergence(wind_field: Tensor) -> Tensor:
    """Compute 2-D divergence of a wind field via central finite differences.

    Args:
        wind_field: ``(3, H, W)`` tensor with (u, v, w) components.

    Returns:
        ``(H, W)`` divergence field: ``du/dx + dv/dy``.
        Boundary pixels use forward/backward differences; interior uses central.
    """
    u = wind_field[0]  # (H, W)
    v = wind_field[1]  # (H, W)

    # du/dx via central differences (interior), forward/backward at edges
    du_dx = torch.zeros_like(u)
    du_dx[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / 2.0
    du_dx[:, 0] = u[:, 1] - u[:, 0]
    du_dx[:, -1] = u[:, -1] - u[:, -2]

    # dv/dy via central differences (interior), forward/backward at edges
    dv_dy = torch.zeros_like(v)
    dv_dy[1:-1, :] = (v[2:, :] - v[:-2, :]) / 2.0
    dv_dy[0, :] = v[1, :] - v[0, :]
    dv_dy[-1, :] = v[-1, :] - v[-2, :]

    return du_dx + dv_dy


@dataclass
class LabeledCFDSample(CFDSample):
    """A CFDSample enriched with mass-conservation diagnostics.

    Attributes:
        divergence_field: ``(H, W)`` divergence of the wind field.
        passes_conservation: ``True`` if max |divergence| < threshold.
    """

    divergence_field: Tensor = field(default_factory=lambda: torch.empty(0))
    passes_conservation: bool = True


def label_mass_conservation(
    sample: CFDSample,
    threshold: float = DEFAULT_DIVERGENCE_THRESHOLD,
) -> LabeledCFDSample:
    """Compute divergence and label a CFD sample for mass conservation.

    Args:
        sample: The input sample.
        threshold: Maximum absolute divergence (s^-1). Samples exceeding
            this are flagged ``passes_conservation=False``. Default is
            10.0 s^-1, suitable for near-surface 2D slices of 3D RANS data.
            For synthetic 2D-native data (e.g. WindNinja), use a stricter
            threshold like 0.01 s^-1.

    Returns:
        A :class:`LabeledCFDSample` with divergence field and flag.
    """
    div_field = compute_divergence(sample.wind_field)
    max_abs_div = div_field.abs().max().item()

    return LabeledCFDSample(
        dem=sample.dem,
        terrain_features=sample.terrain_features,
        boundary_conditions=sample.boundary_conditions,
        wind_field=sample.wind_field,
        case_id=sample.case_id,
        divergence_field=div_field,
        passes_conservation=(max_abs_div < threshold),
    )


def filter_by_conservation(
    samples: list[LabeledCFDSample],
) -> list[LabeledCFDSample]:
    """Filter out samples that fail mass conservation check."""
    return [s for s in samples if s.passes_conservation]
