#!/usr/bin/env python3
"""Phase 2 Alpine fine-tuning: train the downscaling head on ERA5 + PeakWeather.

Loads all 12 months of ERA5 reanalysis (WY2024: Oct 2023 - Sep 2024) paired with
PeakWeather Swiss station observations and DEM terrain patches. The downscaling
head learns terrain-weather corrections:
(ERA5 coarse weather + terrain features) -> station observations.
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
import torch.utils.data
import xarray as xr
from rasterio.warp import transform as warp_transform

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from terrain_weather_ml.backbone.era5_conditioner import (
    ERA5Conditioner,
    ERA5Config,
)
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
from terrain_weather_ml.training.normalizer import AnomalyNormalizer
from terrain_weather_ml.training.phases import Phase2Trainer, PhaseConfig

PROGRESS_FILE = Path(__file__).resolve().parent.parent / "phase2_progress.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def log_progress(msg):
    """Write progress to both logger and a file with explicit flush."""
    logger.info(msg)
    with open(PROGRESS_FILE, "a") as f:
        f.write(msg + "\n")

# --- Paths ---
DEM_PATH = PROJECT_ROOT / "data" / "dem" / "alpine_stations_utm.tif"
ERA5_DIR = PROJECT_ROOT / "data" / "era5" / "alpine"
STATIONS_PATH = (
    PROJECT_ROOT / "data" / "raw" / "peakweather" / "data" / "stations.parquet"
)
OBS_PATH = (
    PROJECT_ROOT / "data" / "raw" / "peakweather" / "data" / "observations"
    / "2024.parquet"
)
PHASE1_CKPT = PROJECT_ROOT / "checkpoints" / "phase1" / "phase1_checkpoint.pt"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "phase2"
TERRAIN_CACHE = PROJECT_ROOT / "data" / "cache" / "terrain_patches"

# --- Hyperparameters ---
PATCH_SIZE = 64
EPOCHS = 50
BATCH_SIZE = 8
MAX_SAMPLES = 30000
LEARNING_RATE = 1e-3
LAMBDA_DIV = 0.05
BASE_FEATURES = 32  # Must match Phase 1
C_TERRAIN = 17
C_WEATHER = 6
C_OUT = 4

# Reduced terrain feature params for speed on small patches
SVF_RADIUS = 500.0
SVF_AZIMUTHS = 8
SX_RADIUS = 500.0
TPI_RADIUS = 300.0


def extract_dem_patches(stations_df, dem_path, patch_size):
    """Extract DEM elevation patches centered on each station.

    Projects station lat/lon (EPSG:4326) to the DEM's UTM CRS, finds the
    pixel coordinates, and crops a patch_size x patch_size window.

    Returns dict mapping station_id -> numpy elevation array (H, W).
    """
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

    for i, sid in enumerate(stations_df.index):
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
    """Compute and cache terrain features for all station patches.

    Returns dict mapping station_id -> tensor (17, H, W).
    """
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


def load_era5(era5_dir):
    """Load all ERA5 monthly files and return per-timestep weather tensors + times.

    Returns:
        tensors: dict[int, Tensor] — timestep index -> (6, 13, 25) tensor
        times: list[datetime] — UTC-aware datetimes
    """
    era5_dir = Path(era5_dir)
    files = sorted(era5_dir.glob("era5_alpine_*.nc"))
    logger.info("Found %d ERA5 monthly files", len(files))

    conditioner = ERA5Conditioner(ERA5Config(domain="alpine"))
    var_order = ["T2M", "U10", "V10", "PRATE", "SP", "BLH"]

    tensors = {}
    times = []
    global_idx = 0

    for f in files:
        ds = xr.open_dataset(f)
        for t_idx in range(len(ds.time)):
            era5_data = conditioner.extract_variables(ds, time_idx=t_idx)
            channels = [
                torch.from_numpy(era5_data.fields[v]).float() for v in var_order
            ]
            tensors[global_idx] = torch.stack(channels, dim=0)

            ts = pd.Timestamp(ds.time.values[t_idx])
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            times.append(ts.to_pydatetime())
            global_idx += 1
        ds.close()

    logger.info("Loaded %d total ERA5 timesteps from %d files", len(tensors), len(files))
    return tensors, times


def load_observations(obs_path, station_ids):
    """Load all PeakWeather 2024 observations (full year).

    Extracts temperature (K), u/v wind (m/s), and precipitation (mm/10min)
    per station across all available months.

    Returns dict[station_id, DataFrame] with columns:
        temperature, u_wind, v_wind, precipitation
    """
    obs = pd.read_parquet(obs_path)
    logger.info("PeakWeather 2024: %d timesteps x %d columns", len(obs), len(obs.columns))

    available_stations = obs.columns.get_level_values(0).unique()
    records = {}

    for sid in station_ids:
        if sid not in available_stations:
            continue

        stn = obs[sid]
        result = pd.DataFrame(index=stn.index)

        if "tre200s0" in stn.columns:
            result["temperature"] = stn["tre200s0"] + 273.15

        if "fkl010z0" in stn.columns and "dkl010z0" in stn.columns:
            speed = stn["fkl010z0"]
            direction_rad = np.radians(stn["dkl010z0"])
            result["u_wind"] = -speed * np.sin(direction_rad)
            result["v_wind"] = -speed * np.cos(direction_rad)

        if "rre150z0" in stn.columns:
            result["precipitation"] = stn["rre150z0"]

        needed = {"temperature", "u_wind", "v_wind", "precipitation"}
        if result.columns.intersection(needed).empty:
            continue

        records[sid] = result

    return records


class CompactStationDataset(torch.utils.data.Dataset):
    """Memory-efficient dataset storing scalar targets, expanding on access.

    Point observations are stored as (C_OUT,) scalars instead of (C_OUT, H, W)
    spatial tensors, reducing memory by ~4000x per sample. Targets are expanded
    to full spatial resolution in __getitem__.
    """

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


def build_samples(station_obs, era5_times, weather_tensors, terrain_tensors):
    """Align observations to ERA5 timesteps and build compact training samples.

    Stores scalar target values instead of full spatial tensors to avoid
    O(N * H * W) memory for point observations.

    Returns:
        terrain_list, weather_list, target_scalars, station_ids_list, months_list
    """
    terrain_list = []
    weather_list = []
    target_list = []
    station_ids_list = []
    months_list = []
    n_matched = 0
    n_skipped = 0

    for sid, obs_df in station_obs.items():
        if sid not in terrain_tensors:
            continue

        terrain = terrain_tensors[sid]

        obs_index = obs_df.index.sort_values()
        if obs_index.tz is None:
            obs_index = obs_index.tz_localize("UTC")

        for t_idx, era5_t in enumerate(era5_times):
            era5_ts = pd.Timestamp(era5_t)
            if era5_ts.tzinfo is None:
                era5_ts = era5_ts.tz_localize("UTC")
            pos = obs_index.searchsorted(era5_ts)

            best_obs_t = None
            best_delta = float("inf")
            for candidate in [pos - 1, pos]:
                if 0 <= candidate < len(obs_index):
                    delta = abs((obs_index[candidate] - era5_ts).total_seconds())
                    if delta < best_delta:
                        best_delta = delta
                        best_obs_t = obs_index[candidate]

            if best_obs_t is None or best_delta > 1800:
                n_skipped += 1
                continue

            row = obs_df.loc[best_obs_t]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]

            has_temp = "temperature" in row.index and pd.notna(row.get("temperature"))
            has_wind = (
                "u_wind" in row.index
                and "v_wind" in row.index
                and pd.notna(row.get("u_wind"))
                and pd.notna(row.get("v_wind"))
            )
            if not has_temp and not has_wind:
                n_skipped += 1
                continue

            target_scalar = torch.zeros(C_OUT)
            if has_wind:
                target_scalar[0] = float(row["u_wind"])
                target_scalar[1] = float(row["v_wind"])
            if has_temp:
                target_scalar[2] = float(row["temperature"])
            if "precipitation" in row.index and pd.notna(row.get("precipitation")):
                target_scalar[3] = float(row["precipitation"])

            terrain_list.append(terrain)
            weather_list.append(weather_tensors[t_idx])
            target_list.append(target_scalar)
            station_ids_list.append(str(sid))
            months_list.append(era5_ts.month)
            n_matched += 1

    logger.info("Samples: %d matched, %d skipped", n_matched, n_skipped)
    target_scalars = torch.stack(target_list) if target_list else torch.zeros(0, C_OUT)
    return terrain_list, weather_list, target_scalars, station_ids_list, months_list


def main():
    PROGRESS_FILE.write_text("")
    device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info("Device: %s", device_name)

    # 1. Verify Phase 1 checkpoint
    if not PHASE1_CKPT.exists():
        logger.error("Phase 1 checkpoint not found: %s", PHASE1_CKPT)
        sys.exit(1)
    p1 = load_phase_checkpoint(PHASE1_CKPT)
    logger.info(
        "Phase 1 checkpoint: phase=%d, epoch=%d, val_loss=%.4f",
        p1.metadata.phase, p1.metadata.epoch, p1.metadata.best_val_loss,
    )

    # 2. Load station metadata
    stations_df = pd.read_parquet(STATIONS_PATH)
    logger.info("Loaded %d stations", len(stations_df))

    # 3. Extract DEM patches around each station
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

    # 5. Load ERA5 (all months)
    logger.info("Loading ERA5 data (all months)...")
    weather_tensors, era5_times = load_era5(ERA5_DIR)
    logger.info(
        "ERA5: %d timesteps, weather shape %s",
        len(weather_tensors), list(weather_tensors[0].shape),
    )

    # 6. Load PeakWeather observations (full year)
    logger.info("Loading PeakWeather 2024 observations (all months)...")
    station_obs = load_observations(OBS_PATH, list(patches.keys()))
    logger.info("Observations loaded for %d stations", len(station_obs))

    # 7. Build training samples (compact scalar targets)
    logger.info("Building training samples...")
    terrain_list, weather_list, target_scalars, sample_sids, sample_months = (
        build_samples(station_obs, era5_times, weather_tensors, terrain_tensors)
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
        sample_sids = [sample_sids[i] for i in idx]
        sample_months = [sample_months[i] for i in idx]
        logger.info("Subsampled to %d samples (from full year)", len(terrain_list))

    # 8. Fit anomaly normalizer on scalar targets
    normalizer = AnomalyNormalizer()
    normalizer.fit(target_scalars, sample_sids, sample_months)
    logger.info(
        "Anomaly normalizer: %d climatology entries, residual_std=%s",
        len(normalizer.climatology), normalizer.residual_std.tolist(),
    )

    # Normalize scalar targets
    target_scalars = normalizer.normalize(target_scalars, sample_sids, sample_months)

    # 9. Create compact dataset and trainer
    h, w = terrain_list[0].shape[1], terrain_list[0].shape[2]
    dataset = CompactStationDataset(terrain_list, weather_list, target_scalars, h, w)

    config = PhaseConfig(
        phase=2,
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
        prerequisite_checkpoint=str(PHASE1_CKPT),
    )

    trainer = Phase2Trainer(config)

    # LR schedule: 1e-3 for first 10 epochs, then cosine decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer, T_max=EPOCHS - 10, eta_min=1e-5,
    )

    param_count = sum(p.numel() for p in trainer.head.parameters())
    logger.info(
        "Model: %d parameters (%.2f MB)", param_count, param_count * 4 / 1e6
    )

    # 10. Train with LR schedule
    logger.info(
        "Training: %d epochs, batch_size=%d, lr=%s, samples=%d",
        EPOCHS, BATCH_SIZE, LEARNING_RATE, len(dataset),
    )
    t0 = time.time()

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False,
    )
    best_loss = float("inf")

    for epoch in range(EPOCHS):
        trainer.head.train()
        epoch_loss = 0.0
        n_batches = 0

        for terrain, weather, weather_target, mask in loader:
            terrain = terrain.to(trainer.device)
            weather = weather.to(trainer.device)
            weather_target = weather_target.to(trainer.device)

            trainer.optimizer.zero_grad()
            pred = trainer.head(terrain, weather)
            loss_dict = trainer.loss_fn(pred, weather_target, terrain)
            loss = loss_dict["total"]
            loss.backward()
            trainer.optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        trainer.epoch_losses.append(avg_loss)
        best_loss = min(best_loss, avg_loss)
        trainer.best_val_loss = best_loss

        if epoch >= 10:
            scheduler.step()

        current_lr = trainer.optimizer.param_groups[0]["lr"]
        log_progress(
            f"Phase 2 epoch {epoch + 1}/{EPOCHS}: loss={avg_loss:.6f} lr={current_lr:.2e}"
        )

    elapsed = time.time() - t0

    # 11. Save checkpoint with normalizer state
    checkpoint_path = trainer._save_checkpoint(dataset)
    ckpt_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_data["extra_data"]["normalizer"] = normalizer.state_dict()
    ckpt_data["extra_data"]["normalizer_type"] = "anomaly"
    torch.save(ckpt_data, checkpoint_path)
    logger.info("Saved AnomalyNormalizer state to checkpoint")

    # 12. Report
    print()
    print("=" * 60)
    print("Phase 2 Alpine Fine-tuning Complete (AnomalyNormalizer)")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Time: {elapsed:.1f}s ({elapsed / EPOCHS:.1f}s/epoch)")
    print(f"Samples: {len(dataset)}")
    print(f"Stations: {len(terrain_tensors)}")
    print(f"ERA5 timesteps: {len(era5_times)}")
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
    print("Normalizer: AnomalyNormalizer")
    print(f"  Climatology entries: {len(normalizer.climatology)}")
    print(f"  Residual std: {normalizer.residual_std.tolist()}")


if __name__ == "__main__":
    main()
