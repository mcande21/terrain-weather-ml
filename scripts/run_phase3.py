#!/usr/bin/env python3
"""Phase 3 Colorado adaptation: fine-tune the downscaling head on HRRR + SNOTEL.

Loads the Phase 2 Alpine-trained checkpoint and adapts it to the Colorado domain
using full WY2024 HRRR weather (366 days) paired with SNOTEL station observations
from WY2020-2024 (5 water years). Applies quantile mapping (RainShift protocol)
to bridge the ERA5 -> HRRR distribution gap.

Key constraints:
- SNOTEL has NO wind data: wind target weight is set to 0, preserving Phase 2
  wind corrections from PeakWeather Alpine training.
- SNOTEL is daily, HRRR is 6-hourly: matches SNOTEL daily obs to the 12Z HRRR
  analysis per day.
- SNOTEL precipitation is accumulated: differences consecutive days to get daily
  precipitation, converted to rate matching HRRR units (kg/m^2/s).
- Only SNOTEL dates with matching HRRR data produce training samples.
"""

import hashlib
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import xarray as xr
from rasterio.warp import transform as warp_transform

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from terrain_weather_ml.terrain.features import (
    compute_aspect,
    compute_curvature,
    compute_slope,
)
from terrain_weather_ml.terrain.roughness import compute_roughness
from terrain_weather_ml.terrain.svf import compute_svf
from terrain_weather_ml.terrain.sx import compute_sx
from terrain_weather_ml.terrain.tpi import compute_tpi
from terrain_weather_ml.training.checkpoint import load_phase_checkpoint
from terrain_weather_ml.training.normalizer import TargetNormalizer
from terrain_weather_ml.training.phases import Phase3Trainer, PhaseConfig
from terrain_weather_ml.training.quantile_mapping import QuantileMapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# --- Paths ---
DEM_PATH = PROJECT_ROOT / "data" / "dem" / "colorado_snotel_utm.tif"
HRRR_DIR = PROJECT_ROOT / "data" / "hrrr" / "colorado"
ERA5_PATH = PROJECT_ROOT / "data" / "era5" / "alpine" / "era5_alpine_202401.nc"
STATIONS_PATH = PROJECT_ROOT / "data" / "snotel" / "stations.csv"
SNOTEL_DIR = PROJECT_ROOT / "data" / "snotel"
PHASE2_CKPT = PROJECT_ROOT / "checkpoints" / "phase2" / "phase2_checkpoint.pt"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "phase3"
TERRAIN_CACHE = PROJECT_ROOT / "data" / "cache" / "terrain_patches_colorado"

# --- Hyperparameters ---
PATCH_SIZE = 64
EPOCHS = 30
BATCH_SIZE = 8
MAX_SAMPLES = 30000
LEARNING_RATE = 1e-4
LAMBDA_DIV = 0.05
BASE_FEATURES = 32
C_TERRAIN = 17
C_WEATHER = 6
C_OUT = 4

SVF_RADIUS = 500.0
SVF_AZIMUTHS = 8
SX_RADIUS = 500.0
TPI_RADIUS = 300.0

HRRR_VAR_ORDER = ["t2m", "u10", "v10", "prate", "sp", "blh"]
HRRR_12Z_INDEX = 2


def extract_dem_patches(stations_df, dem_path, patch_size):
    """Extract DEM elevation patches centered on each station."""
    with rasterio.open(dem_path) as src:
        dem_crs = src.crs
        dem_transform = src.transform
        dem_data = src.read(1).astype(np.float32)
        dem_h, dem_w = dem_data.shape

    lats = stations_df["latitude"].values
    lons = stations_df["longitude"].values
    xs, ys = warp_transform("EPSG:4326", dem_crs, lons, lats)

    half = patch_size // 2
    patches = {}
    skipped = 0

    for i, sid in enumerate(stations_df["station_id"].values):
        col, row = ~dem_transform * (xs[i], ys[i])
        row, col = round(row), round(col)

        r0 = row - half
        r1 = row + half
        c0 = col - half
        c1 = col + half

        if r0 < 0 or r1 > dem_h or c0 < 0 or c1 > dem_w:
            skipped += 1
            continue

        patch = dem_data[r0:r1, c0:c1].copy()
        if patch.shape != (patch_size, patch_size):
            skipped += 1
            continue
        if np.isnan(patch).all():
            skipped += 1
            continue

        patches[sid] = patch

    if skipped:
        logger.info("Skipped %d stations (out of bounds or invalid DEM)", skipped)
    return patches


def compute_terrain_tensor(elevation, resolution=30.0):
    """Compute 17-channel terrain feature tensor from an elevation patch."""
    h, w = elevation.shape

    slope = compute_slope(elevation, resolution)
    aspect_sin, aspect_cos = compute_aspect(elevation, resolution)
    plan_curv, profile_curv = compute_curvature(elevation, resolution)
    svf = compute_svf(
        elevation, resolution, max_radius=SVF_RADIUS, n_azimuths=SVF_AZIMUTHS
    )
    sx = compute_sx(elevation, resolution, max_radius=SX_RADIUS)
    tpi = compute_tpi(elevation, resolution, radius=TPI_RADIUS)
    roughness = compute_roughness(None, shape=(h, w), default_z0=0.1)

    channels = np.stack(
        [
            elevation, slope, aspect_sin, aspect_cos,
            plan_curv, profile_curv, svf,
            sx[0], sx[1], sx[2], sx[3], sx[4], sx[5], sx[6], sx[7],
            tpi, roughness,
        ],
        axis=0,
    )
    tensor = torch.from_numpy(channels).float()
    return torch.nan_to_num(tensor, nan=0.0)


def compute_all_terrain(patches, cache_dir):
    """Compute and cache terrain features for all station patches."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    terrain = {}
    computed = 0
    cached = 0

    for sid, elevation in patches.items():
        cache_path = cache_dir / f"{sid}_{PATCH_SIZE}.pt"

        if cache_path.exists():
            terrain[sid] = torch.load(
                cache_path, map_location="cpu", weights_only=True
            )
            cached += 1
            continue

        tensor = compute_terrain_tensor(elevation)
        torch.save(tensor, cache_path)
        terrain[sid] = tensor
        computed += 1

        if computed % 10 == 0:
            logger.info("  Computed terrain for %d/%d stations...", computed, len(patches))

    logger.info("Terrain: %d computed, %d from cache", computed, cached)
    return terrain


def load_hrrr_12z(hrrr_dir):
    """Load 12Z HRRR analysis from daily NetCDF files.

    Returns:
        tensors: dict[str, Tensor] -- date string -> (6, 200, 162) weather tensor
        lat: 2D latitude array
        lon: 2D longitude array (0-360 convention)
    """
    hrrr_dir = Path(hrrr_dir)
    files = sorted(hrrr_dir.glob("hrrr_colorado_*.nc"))
    logger.info("Found %d HRRR daily files", len(files))

    tensors = {}
    lat = None
    lon = None

    for f in files:
        ds = xr.open_dataset(f)
        date_str = f.stem.split("_")[-1]

        if lat is None:
            lat = ds.latitude.values
            lon = ds.longitude.values

        n_times = len(ds.time)
        t_idx = min(HRRR_12Z_INDEX, n_times - 1)

        channels = []
        for var in HRRR_VAR_ORDER:
            field = ds[var].isel(time=t_idx).values.astype(np.float32)
            channels.append(torch.from_numpy(field))

        tensors[date_str] = torch.stack(channels, dim=0)
        ds.close()

    logger.info("Loaded %d HRRR 12Z tensors, shape %s", len(tensors),
                list(tensors[next(iter(tensors))].shape))
    return tensors, lat, lon


def find_nearest_hrrr_cell(station_lat, station_lon, hrrr_lat, hrrr_lon):
    """Find nearest HRRR grid cell for a station.

    HRRR uses 0-360 longitude convention, stations use -180 to 180.
    """
    stn_lon_360 = station_lon % 360
    dist = (hrrr_lat - station_lat) ** 2 + (hrrr_lon - stn_lon_360) ** 2
    return np.unravel_index(np.argmin(dist), dist.shape)


def extract_hrrr_patch(weather_full, center_row, center_col, patch_size=8):
    """Extract a small HRRR patch around a station.

    Uses a smaller patch than terrain (8x8 at 3km ~ 24km) since the
    head upsamples weather to terrain resolution.
    """
    h, w = weather_full.shape[1], weather_full.shape[2]
    half = patch_size // 2

    r0 = max(0, center_row - half)
    r1 = min(h, center_row + half)
    c0 = max(0, center_col - half)
    c1 = min(w, center_col + half)

    patch = weather_full[:, r0:r1, c0:c1]

    if patch.shape[1] < patch_size or patch.shape[2] < patch_size:
        padded = torch.zeros(weather_full.shape[0], patch_size, patch_size)
        ph, pw = patch.shape[1], patch.shape[2]
        padded[:, :ph, :pw] = patch
        return padded

    return patch


def load_snotel_observations(snotel_dir, station_ids):
    """Load all SNOTEL water year observations (WY2020-2024).

    Temperature: TOBS (degF) -> Kelvin
    Precipitation: accumulated inches -> daily rate in kg/m^2/s
    """
    snotel_dir = Path(snotel_dir)
    wy_files = sorted(snotel_dir.glob("observations_wy*.csv"))
    logger.info("Found %d SNOTEL water year files", len(wy_files))

    all_obs = []
    for f in wy_files:
        df = pd.read_csv(f)
        df["date"] = pd.to_datetime(df["date"])
        all_obs.append(df)
        logger.info("  Loaded %s: %d rows", f.name, len(df))

    obs = pd.concat(all_obs, ignore_index=True)
    logger.info("Total SNOTEL observations: %d rows across %d WYs", len(obs), len(wy_files))

    records = {}

    for sid in station_ids:
        stn_obs = obs[obs["station_id"] == sid].sort_values("date")
        if len(stn_obs) < 2:
            continue

        result = pd.DataFrame(index=stn_obs["date"].values)

        tobs_col = "Air Temperature Observed (degF) Start of Day Values"
        if tobs_col in stn_obs.columns:
            tobs_f = stn_obs[tobs_col].values.astype(float)
            tobs_k = (tobs_f - 32.0) * 5.0 / 9.0 + 273.15
            result["temperature"] = tobs_k

        prec_col = "Precipitation Accumulation (in) Start of Day Values"
        if prec_col in stn_obs.columns:
            prec_acc = stn_obs[prec_col].values.astype(float)
            daily_prec_in = np.diff(prec_acc, prepend=prec_acc[0])
            daily_prec_in = np.maximum(daily_prec_in, 0.0)
            daily_prec_mm = daily_prec_in * 25.4
            daily_prec_rate = daily_prec_mm / 86400.0
            result["precipitation"] = daily_prec_rate

        if "temperature" not in result.columns and "precipitation" not in result.columns:
            continue

        records[sid] = result

    logger.info("SNOTEL observations loaded for %d stations", len(records))
    return records


def calibrate_quantile_mapping(era5_path, hrrr_tensors):
    """Fit quantile mapping between ERA5 (Alpine) and HRRR (Colorado).

    Both datasets cover January 2024. The mapper learns CDF transfer
    functions for each weather variable to bridge the domain gap.
    """
    ds = xr.open_dataset(era5_path)
    era5_var_map = {"t2m": "t2m", "u10": "u10", "v10": "v10",
                    "tp": "prate", "sp": "sp", "blh": "blh"}

    source_data = {}
    for era5_var, canonical in era5_var_map.items():
        vals = ds[era5_var].values.flatten().astype(np.float32)
        vals = vals[~np.isnan(vals)]
        if era5_var == "tp":
            vals = vals * 1000.0 / 3600.0
        source_data[canonical] = torch.from_numpy(vals)

    ds.close()

    target_data = {}
    all_hrrr = torch.stack(list(hrrr_tensors.values()), dim=0)
    for i, var in enumerate(HRRR_VAR_ORDER):
        vals = all_hrrr[:, i].flatten()
        target_data[var] = vals

    mapper = QuantileMapper(n_quantiles=500)
    mapper.fit(source_data=source_data, target_data=target_data)
    return mapper


def apply_quantile_mapping(weather_tensor, mapper):
    """Apply per-variable quantile mapping to a HRRR weather tensor.

    Maps HRRR values to the ERA5 (Alpine) distribution space so the
    Alpine-trained head sees familiar input ranges.

    Input/output shape: (C, H, W)
    """
    mapped = weather_tensor.clone()
    for i, var in enumerate(HRRR_VAR_ORDER):
        try:
            mapped[i] = mapper.transform(weather_tensor[i], variable=var)
        except KeyError:
            pass
    return mapped


class CompactColoradoDataset(torch.utils.data.Dataset):
    """Memory-efficient dataset storing scalar targets, expanding on access."""

    def __init__(self, terrain_list, weather_list, target_scalars, h, w):
        self.terrain_list = terrain_list
        self.weather_list = weather_list
        self.target_scalars = target_scalars
        self.h = h
        self.w = w

    def __len__(self):
        return len(self.terrain_list)

    def __getitem__(self, idx):
        terrain = self.terrain_list[idx]
        weather = self.weather_list[idx]
        target = self.target_scalars[idx].view(-1, 1, 1).expand(-1, self.h, self.w)
        mask = torch.ones(self.h, self.w)
        return terrain, weather, target.contiguous(), mask

    def compute_data_hash(self):
        h = hashlib.sha256()
        h.update(str(len(self)).encode())
        if len(self) > 0:
            terrain, weather, target, _ = self[0]
            h.update(str(terrain.shape).encode())
            h.update(str(weather.shape).encode())
            h.update(str(target.shape).encode())
        return h.hexdigest()[:16]


def build_samples(station_obs, hrrr_tensors, hrrr_lat, hrrr_lon,
                  terrain_tensors, stations_df, mapper):
    """Build compact training samples matching SNOTEL daily obs to 12Z HRRR."""
    stn_meta = stations_df.set_index("station_id")
    terrain_list = []
    weather_list = []
    target_list = []
    n_matched = 0
    n_skipped = 0

    for sid, obs_df in station_obs.items():
        if sid not in terrain_tensors:
            n_skipped += 1
            continue
        if sid not in stn_meta.index:
            n_skipped += 1
            continue

        terrain = terrain_tensors[sid]

        stn_lat = stn_meta.loc[sid, "latitude"]
        stn_lon = stn_meta.loc[sid, "longitude"]
        hrrr_row, hrrr_col = find_nearest_hrrr_cell(
            stn_lat, stn_lon, hrrr_lat, hrrr_lon
        )

        for i, date_val in enumerate(obs_df.index):
            date_ts = pd.Timestamp(date_val)
            date_str = date_ts.strftime("%Y%m%d")

            if date_str not in hrrr_tensors:
                n_skipped += 1
                continue

            weather_full = hrrr_tensors[date_str]
            weather_patch = extract_hrrr_patch(weather_full, hrrr_row, hrrr_col)
            weather_mapped = apply_quantile_mapping(weather_patch, mapper)

            row = obs_df.iloc[i]
            has_temp = "temperature" in obs_df.columns and pd.notna(row.get("temperature"))
            has_prec = "precipitation" in obs_df.columns and pd.notna(row.get("precipitation"))

            if not has_temp and not has_prec:
                n_skipped += 1
                continue

            target_scalar = torch.zeros(C_OUT)
            if has_temp:
                target_scalar[2] = float(row["temperature"])
            if has_prec:
                target_scalar[3] = float(row["precipitation"])

            terrain_list.append(terrain)
            weather_list.append(weather_mapped)
            target_list.append(target_scalar)
            n_matched += 1

    logger.info("Samples: %d matched, %d skipped", n_matched, n_skipped)
    target_scalars = torch.stack(target_list) if target_list else torch.zeros(0, C_OUT)
    return terrain_list, weather_list, target_scalars


def main():
    device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info("Device: %s", device_name)

    # 1. Verify Phase 2 checkpoint
    if not PHASE2_CKPT.exists():
        logger.error("Phase 2 checkpoint not found: %s", PHASE2_CKPT)
        sys.exit(1)
    p2 = load_phase_checkpoint(PHASE2_CKPT)
    logger.info(
        "Phase 2 checkpoint: phase=%d, epoch=%d, val_loss=%.4f",
        p2.metadata.phase, p2.metadata.epoch, p2.metadata.best_val_loss,
    )

    # 2. Load SNOTEL station metadata
    stations_df = pd.read_csv(STATIONS_PATH)
    logger.info("Loaded %d SNOTEL stations", len(stations_df))

    # 3. Extract Colorado DEM patches per station
    logger.info("Extracting %dx%d DEM patches...", PATCH_SIZE, PATCH_SIZE)
    patches = extract_dem_patches(stations_df, DEM_PATH, PATCH_SIZE)
    logger.info("Extracted %d valid patches", len(patches))

    # 4. Compute terrain features
    logger.info("Computing 17-channel terrain features...")
    t0 = time.time()
    terrain_tensors = compute_all_terrain(patches, TERRAIN_CACHE)
    logger.info("Terrain features ready in %.1fs", time.time() - t0)

    sample_t = next(iter(terrain_tensors.values()))
    logger.info("Terrain tensor shape: %s", list(sample_t.shape))

    # 5. Load HRRR 12Z data
    logger.info("Loading HRRR 12Z data...")
    hrrr_tensors, hrrr_lat, hrrr_lon = load_hrrr_12z(HRRR_DIR)

    # 6. Calibrate quantile mapping (ERA5 Alpine -> HRRR Colorado)
    logger.info("Calibrating quantile mapping (ERA5 Alpine -> HRRR Colorado)...")
    mapper = calibrate_quantile_mapping(ERA5_PATH, hrrr_tensors)

    # 7. Load SNOTEL observations (all water years)
    logger.info("Loading SNOTEL observations (WY2020-2024)...")
    station_obs = load_snotel_observations(SNOTEL_DIR, list(patches.keys()))
    logger.info("Observations loaded for %d stations", len(station_obs))

    # 8. Build training samples (compact scalar targets)
    logger.info("Building training samples...")
    terrain_list, weather_list, target_scalars = build_samples(
        station_obs, hrrr_tensors, hrrr_lat, hrrr_lon,
        terrain_tensors, stations_df, mapper,
    )
    logger.info("Total training samples: %d", len(terrain_list))

    if len(terrain_list) == 0:
        logger.error("No valid training samples. Check data alignment.")
        sys.exit(1)

    if len(terrain_list) > MAX_SAMPLES:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(terrain_list), size=MAX_SAMPLES, replace=False)
        idx.sort()
        terrain_list = [terrain_list[i] for i in idx]
        weather_list = [weather_list[i] for i in idx]
        target_scalars = target_scalars[idx]
        logger.info("Subsampled to %d samples", len(terrain_list))

    # 9. Fit target normalizer on scalar targets
    all_targets_4d = target_scalars.unsqueeze(-1).unsqueeze(-1)
    normalizer = TargetNormalizer()
    normalizer.fit(all_targets_4d)
    logger.info(
        "Target normalizer: mean=%s, std=%s",
        normalizer.mean.tolist(), normalizer.std.tolist(),
    )

    # Normalize scalar targets in-place
    target_scalars = (target_scalars - normalizer.mean) / normalizer.std

    # 10. Create compact dataset and trainer
    h, w = terrain_list[0].shape[1], terrain_list[0].shape[2]
    dataset = CompactColoradoDataset(terrain_list, weather_list, target_scalars, h, w)

    config = PhaseConfig(
        phase=3,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        lambda_div=LAMBDA_DIV,
        checkpoint_dir=str(CHECKPOINT_DIR),
        device=device_name,
        base_features=BASE_FEATURES,
        c_terrain=C_TERRAIN,
        c_weather=C_WEATHER,
        c_out=C_OUT,
        prerequisite_checkpoint=str(PHASE2_CKPT),
    )

    trainer = Phase3Trainer(config, quantile_mapper=mapper)

    # Zero out wind target weights: SNOTEL has no wind data
    from terrain_weather_ml.terrain.downscaling.loss import DownscalingLoss
    trainer.loss_fn = DownscalingLoss(
        lambda_div=LAMBDA_DIV,
        lambda_oro=0.0,
        variable_weights=[0.0, 0.0, 1.0, 1.0],
    )
    trainer.loss_fn.to(trainer.device)

    param_count = sum(p.numel() for p in trainer.head.parameters())
    logger.info(
        "Model: %d parameters (%.2f MB)", param_count, param_count * 4 / 1e6
    )

    # 11. Train
    logger.info(
        "Training: %d epochs, batch_size=%d, lr=%s, samples=%d",
        EPOCHS, BATCH_SIZE, LEARNING_RATE, len(dataset),
    )
    t0 = time.time()
    checkpoint_path = trainer.train(dataset)
    elapsed = time.time() - t0

    # 12. Save normalizer state in checkpoint
    ckpt_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_data["extra_data"]["normalizer"] = normalizer.state_dict()
    torch.save(ckpt_data, checkpoint_path)
    logger.info("Saved normalizer state to checkpoint")

    # 13. Report
    print()
    print("=" * 60)
    print("Phase 3 Colorado Adaptation Complete")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Time: {elapsed:.1f}s ({elapsed / EPOCHS:.1f}s/epoch)")
    print(f"Samples: {len(dataset)}")
    print(f"Stations: {len(terrain_tensors)}")
    print(f"HRRR timesteps: {len(hrrr_tensors)}")
    print()
    print("Loss curve (first 10, then every 5th):")
    for i, loss in enumerate(trainer.epoch_losses):
        if i < 10 or (i + 1) % 5 == 0 or i == len(trainer.epoch_losses) - 1:
            marker = " <-- best" if loss == min(trainer.epoch_losses) else ""
            print(f"  Epoch {i + 1:2d}: {loss:.6f}{marker}")

    initial = trainer.epoch_losses[0]
    final = trainer.epoch_losses[-1]
    print()
    print(f"Initial loss: {initial:.6f}")
    print(f"Final loss:   {final:.6f}")
    if initial > 0:
        print(f"Reduction:    {(initial - final) / initial * 100:.1f}%")

    ckpt = load_phase_checkpoint(checkpoint_path)
    print()
    print(
        f"Checkpoint verified: phase={ckpt.metadata.phase}, "
        f"epoch={ckpt.metadata.epoch}, "
        f"val_loss={ckpt.metadata.best_val_loss:.6f}"
    )
    print(f"Checkpoint hash: {ckpt.compute_hash()[:16]}")
    print(f"Normalizer mean: {normalizer.mean.tolist()}")
    print(f"Normalizer std:  {normalizer.std.tolist()}")

    if ckpt.extra_data.get("quantile_params"):
        qp = ckpt.extra_data["quantile_params"]
        print(f"Quantile mapping: {len(qp)} variables ({', '.join(qp.keys())})")


if __name__ == "__main__":
    main()
