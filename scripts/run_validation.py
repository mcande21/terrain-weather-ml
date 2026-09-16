"""Phase 3 validation: comprehensive holdout evaluation with all baselines.

Loads the Phase 3 checkpoint (with AnomalyNormalizer) and evaluates on
WY2024 SNOTEL observations (temporal holdout). Compares model against
five baselines: raw HRRR, lapse-rate adjusted HRRR, climatology,
persistence, and bias-corrected HRRR.

Reports per-season (DJF/MAM/JJA/SON), per-elevation band, skill scores,
bootstrap 95% CIs, and cold air pool physics checks.
"""

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

from terrain_weather_ml.evaluation.loso import (
    ElevationBandConfig,
    bootstrap_confidence_intervals,
    classify_elevation_band,
)
from terrain_weather_ml.evaluation.metrics import (
    compute_continuous_metrics,
    compute_skill_score,
)
from terrain_weather_ml.evaluation.physics import check_cold_air_pool
from terrain_weather_ml.evaluation.report import (
    ModelResults,
    generate_comparison_report,
)
from terrain_weather_ml.terrain.downscaling.head import TerrainDownscalingHead
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
from terrain_weather_ml.training.quantile_mapping import QuantileMapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# --- Paths ---
PHASE3_CKPT = PROJECT_ROOT / "checkpoints" / "phase3" / "phase3_checkpoint.pt"
DEM_PATH = PROJECT_ROOT / "data" / "dem" / "colorado_snotel_utm.tif"
HRRR_DIR = PROJECT_ROOT / "data" / "hrrr" / "colorado"
STATIONS_PATH = PROJECT_ROOT / "data" / "snotel" / "stations.csv"
SNOTEL_DIR = PROJECT_ROOT / "data" / "snotel"
TERRAIN_CACHE = PROJECT_ROOT / "data" / "cache" / "terrain_patches_colorado"
REPORT_DIR = PROJECT_ROOT / "reports"

# --- Config ---
PATCH_SIZE = 64
BASE_FEATURES = 32
C_TERRAIN = 17
C_WEATHER = 6
C_OUT = 4
HRRR_VAR_ORDER = ["t2m", "u10", "v10", "prate", "sp", "blh"]
HRRR_12Z_INDEX = 2
SVF_RADIUS = 500.0
SVF_AZIMUTHS = 8
SX_RADIUS = 500.0
TPI_RADIUS = 300.0

IDX_TEMPERATURE = 2
IDX_PRECIPITATION = 3

LAPSE_RATE = 6.5  # K/km


def load_model(checkpoint_path):
    """Load Phase 3 model head, quantile mapper, and normalizer from checkpoint."""
    ckpt = load_phase_checkpoint(checkpoint_path)
    logger.info(
        "Loaded Phase %d checkpoint: epoch=%d, val_loss=%.4f",
        ckpt.metadata.phase, ckpt.metadata.epoch, ckpt.metadata.best_val_loss,
    )

    head = TerrainDownscalingHead(
        c_terrain=C_TERRAIN,
        c_weather=C_WEATHER,
        c_out=C_OUT,
        base_features=BASE_FEATURES,
        apply_divergence_free=False,
    )

    head_state = {}
    for key, value in ckpt.model_state_dict.items():
        if key.startswith("head."):
            head_state[key[5:]] = value
    head.load_state_dict(head_state)
    head.eval()

    mapper = None
    if ckpt.extra_data.get("quantile_params"):
        mapper = QuantileMapper()
        mapper.load_params(ckpt.extra_data["quantile_params"])
        logger.info("Loaded quantile mapper for %d variables",
                     len(ckpt.extra_data["quantile_params"]))

    normalizer = None
    if ckpt.extra_data.get("normalizer"):
        norm_type = ckpt.extra_data.get("normalizer_type", "target")
        if norm_type == "anomaly":
            normalizer = AnomalyNormalizer()
            normalizer.load_state_dict(ckpt.extra_data["normalizer"])
            logger.info(
                "Loaded AnomalyNormalizer: %d climatology entries",
                len(normalizer.climatology),
            )
        else:
            from terrain_weather_ml.training.normalizer import TargetNormalizer
            normalizer = TargetNormalizer()
            normalizer.load_state_dict(ckpt.extra_data["normalizer"])
            logger.info("Loaded TargetNormalizer (legacy)")

    return head, mapper, normalizer


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

    for i, sid in enumerate(stations_df["station_id"].values):
        col, row = ~dem_transform * (xs[i], ys[i])
        row, col = round(row), round(col)

        r0 = row - half
        r1 = row + half
        c0 = col - half
        c1 = col + half

        if r0 < 0 or r1 > dem_h or c0 < 0 or c1 > dem_w:
            continue
        patch = dem_data[r0:r1, c0:c1].copy()
        if patch.shape != (patch_size, patch_size):
            continue
        if np.isnan(patch).all():
            continue

        patches[sid] = patch

    return patches


def compute_terrain_tensor(elevation, resolution=30.0):
    """Compute 17-channel terrain feature tensor from an elevation patch."""
    h, w = elevation.shape
    slope = compute_slope(elevation, resolution)
    aspect_sin, aspect_cos = compute_aspect(elevation, resolution)
    plan_curv, profile_curv = compute_curvature(elevation, resolution)
    svf = compute_svf(elevation, resolution, max_radius=SVF_RADIUS, n_azimuths=SVF_AZIMUTHS)
    sx = compute_sx(elevation, resolution, max_radius=SX_RADIUS)
    tpi = compute_tpi(elevation, resolution, radius=TPI_RADIUS)
    roughness = compute_roughness(None, shape=(h, w), default_z0=0.1)

    channels = np.stack([
        elevation, slope, aspect_sin, aspect_cos,
        plan_curv, profile_curv, svf,
        sx[0], sx[1], sx[2], sx[3], sx[4], sx[5], sx[6], sx[7],
        tpi, roughness,
    ], axis=0)
    tensor = torch.from_numpy(channels).float()
    return torch.nan_to_num(tensor, nan=0.0)


def compute_all_terrain(patches, cache_dir):
    """Compute and cache terrain features for all station patches."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    terrain = {}
    for sid, elevation in patches.items():
        cache_path = cache_dir / f"{sid}_{PATCH_SIZE}.pt"
        if cache_path.exists():
            terrain[sid] = torch.load(cache_path, map_location="cpu", weights_only=True)
        else:
            tensor = compute_terrain_tensor(elevation)
            torch.save(tensor, cache_path)
            terrain[sid] = tensor

    return terrain


def load_hrrr_12z(hrrr_dir):
    """Load 12Z HRRR analysis from daily NetCDF files."""
    hrrr_dir = Path(hrrr_dir)
    files = sorted(hrrr_dir.glob("hrrr_colorado_*.nc"))

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

    return tensors, lat, lon


def find_nearest_hrrr_cell(station_lat, station_lon, hrrr_lat, hrrr_lon):
    """Find nearest HRRR grid cell for a station (0-360 lon convention)."""
    stn_lon_360 = station_lon % 360
    dist = (hrrr_lat - station_lat) ** 2 + (hrrr_lon - stn_lon_360) ** 2
    return np.unravel_index(np.argmin(dist), dist.shape)


def extract_hrrr_patch(weather_full, center_row, center_col, patch_size=8):
    """Extract a small HRRR patch around a station."""
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


def load_snotel_wy2024(snotel_dir, station_ids):
    """Load full WY2024 SNOTEL observations (Oct 2023 - Sep 2024).

    Returns dict[station_id, DataFrame] with temperature (K) and
    precipitation rate (kg/m^2/s).
    """
    snotel_dir = Path(snotel_dir)
    obs_path = snotel_dir / "observations_wy2024.csv"
    obs = pd.read_csv(obs_path)
    obs["date"] = pd.to_datetime(obs["date"])
    logger.info("WY2024 observations: %d rows", len(obs))

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

    return records


def load_snotel_training(snotel_dir, station_ids):
    """Load WY2020-2023 SNOTEL observations for baseline fitting."""
    snotel_dir = Path(snotel_dir)
    all_obs = []
    for wy in range(2020, 2024):
        f = snotel_dir / f"observations_wy{wy}.csv"
        if f.exists():
            df = pd.read_csv(f)
            df["date"] = pd.to_datetime(df["date"])
            all_obs.append(df)

    if not all_obs:
        return {}

    obs = pd.concat(all_obs, ignore_index=True)
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

    return records


def apply_quantile_mapping(weather_tensor, mapper):
    """Apply per-variable quantile mapping to a HRRR weather tensor."""
    mapped = weather_tensor.clone()
    for i, var in enumerate(HRRR_VAR_ORDER):
        try:
            mapped[i] = mapper.transform(weather_tensor[i], variable=var)
        except KeyError:
            pass
    return mapped


def run_model_predictions(head, mapper, normalizer, terrain_tensors,
                          hrrr_tensors, hrrr_lat, hrrr_lon, stations_df,
                          station_obs):
    """Run model predictions for each station on each day.

    Handles AnomalyNormalizer denormalization with per-sample station_id
    and month context.
    """
    stn_meta = stations_df.set_index("station_id")
    predictions = {}
    is_anomaly = isinstance(normalizer, AnomalyNormalizer)

    for sid, obs_df in station_obs.items():
        if sid not in terrain_tensors or sid not in stn_meta.index:
            continue

        terrain = terrain_tensors[sid].unsqueeze(0)
        stn_lat = stn_meta.loc[sid, "latitude"]
        stn_lon = stn_meta.loc[sid, "longitude"]
        hrrr_row, hrrr_col = find_nearest_hrrr_cell(
            stn_lat, stn_lon, hrrr_lat, hrrr_lon
        )

        temp_preds = []
        prec_preds = []

        for date_val in obs_df.index:
            date_ts = pd.Timestamp(date_val)
            date_str = date_ts.strftime("%Y%m%d")
            if date_str not in hrrr_tensors:
                temp_preds.append(np.nan)
                prec_preds.append(np.nan)
                continue

            weather_full = hrrr_tensors[date_str]
            weather_patch = extract_hrrr_patch(weather_full, hrrr_row, hrrr_col)
            if mapper is not None:
                weather_patch = apply_quantile_mapping(weather_patch.cpu(), mapper)
                weather_patch = weather_patch.to(terrain.device)

            with torch.no_grad():
                out = head(terrain, weather_patch.unsqueeze(0))

                h_mid = out.shape[2] // 2
                w_mid = out.shape[3] // 2
                center = out[0, :, h_mid, w_mid].unsqueeze(0)  # (1, C)

                if normalizer is not None:
                    if is_anomaly:
                        try:
                            center = normalizer.denormalize(
                                center, [str(sid)], [date_ts.month]
                            )
                        except KeyError:
                            center = center * 0 + float("nan")
                    else:
                        center_4d = center.unsqueeze(-1).unsqueeze(-1)
                        center_4d = normalizer.denormalize(center_4d)
                        center = center_4d.squeeze(-1).squeeze(-1)

            temp_preds.append(center[0, IDX_TEMPERATURE].item())
            prec_preds.append(max(0.0, center[0, IDX_PRECIPITATION].item()))

        stn_preds = {}
        if "temperature" in obs_df.columns:
            stn_preds["temperature"] = np.array(temp_preds)
        if "precipitation" in obs_df.columns:
            stn_preds["precipitation"] = np.array(prec_preds)

        predictions[sid] = stn_preds

    return predictions


def run_hrrr_baseline(hrrr_tensors, hrrr_lat, hrrr_lon,
                      stations_df, station_obs):
    """Extract raw HRRR value at each station location as baseline."""
    stn_meta = stations_df.set_index("station_id")
    baseline = {}

    for sid, obs_df in station_obs.items():
        if sid not in stn_meta.index:
            continue

        stn_lat = stn_meta.loc[sid, "latitude"]
        stn_lon = stn_meta.loc[sid, "longitude"]
        hrrr_row, hrrr_col = find_nearest_hrrr_cell(
            stn_lat, stn_lon, hrrr_lat, hrrr_lon
        )

        temp_vals = []
        prec_vals = []

        for date_val in obs_df.index:
            date_str = pd.Timestamp(date_val).strftime("%Y%m%d")
            if date_str not in hrrr_tensors:
                temp_vals.append(np.nan)
                prec_vals.append(np.nan)
                continue

            weather = hrrr_tensors[date_str]
            temp_vals.append(weather[0, hrrr_row, hrrr_col].item())
            prec_vals.append(max(0.0, weather[3, hrrr_row, hrrr_col].item()))

        stn_baseline = {}
        if "temperature" in obs_df.columns:
            stn_baseline["temperature"] = np.array(temp_vals)
        if "precipitation" in obs_df.columns:
            stn_baseline["precipitation"] = np.array(prec_vals)

        baseline[sid] = stn_baseline

    return baseline


def run_lapse_rate_baseline(hrrr_tensors, hrrr_lat, hrrr_lon,
                            stations_df, station_obs, dem_patches):
    """Lapse-rate adjusted HRRR baseline.

    Corrects HRRR temperature using elevation difference between the
    station and the mean DEM patch elevation (proxy for grid cell).
    """
    stn_meta = stations_df.set_index("station_id")
    baseline = {}

    for sid, obs_df in station_obs.items():
        if sid not in stn_meta.index or sid not in dem_patches:
            continue

        stn_elev = stn_meta.loc[sid, "elevation"]
        grid_elev = float(np.nanmean(dem_patches[sid]))
        elev_diff_m = stn_elev - grid_elev

        stn_lat = stn_meta.loc[sid, "latitude"]
        stn_lon = stn_meta.loc[sid, "longitude"]
        hrrr_row, hrrr_col = find_nearest_hrrr_cell(
            stn_lat, stn_lon, hrrr_lat, hrrr_lon
        )

        temp_vals = []
        prec_vals = []

        for date_val in obs_df.index:
            date_str = pd.Timestamp(date_val).strftime("%Y%m%d")
            if date_str not in hrrr_tensors:
                temp_vals.append(np.nan)
                prec_vals.append(np.nan)
                continue

            weather = hrrr_tensors[date_str]
            raw_temp = weather[0, hrrr_row, hrrr_col].item()
            correction = -(elev_diff_m * LAPSE_RATE / 1000.0)
            temp_vals.append(raw_temp + correction)
            prec_vals.append(max(0.0, weather[3, hrrr_row, hrrr_col].item()))

        stn_baseline = {}
        if "temperature" in obs_df.columns:
            stn_baseline["temperature"] = np.array(temp_vals)
        if "precipitation" in obs_df.columns:
            stn_baseline["precipitation"] = np.array(prec_vals)

        baseline[sid] = stn_baseline

    return baseline


def run_climatology_baseline(station_obs, train_obs):
    """Station climatology baseline: monthly mean from WY2020-2023."""
    climatology = {}
    for sid, train_df in train_obs.items():
        months_arr = pd.DatetimeIndex(train_df.index).month
        var_clim = {}
        for var in ["temperature", "precipitation"]:
            if var not in train_df.columns:
                continue
            monthly = {}
            for m in range(1, 13):
                mask = months_arr == m
                if mask.any():
                    vals = train_df[var].values[mask]
                    valid = vals[~np.isnan(vals)]
                    if len(valid) > 0:
                        monthly[m] = float(np.mean(valid))
            var_clim[var] = monthly
        climatology[sid] = var_clim

    baseline = {}
    for sid, obs_df in station_obs.items():
        if sid not in climatology:
            continue

        stn_clim = climatology[sid]
        stn_baseline = {}

        for var in ["temperature", "precipitation"]:
            if var not in obs_df.columns or var not in stn_clim:
                continue
            preds = np.full(len(obs_df), np.nan, dtype=np.float32)
            for i, date_val in enumerate(obs_df.index):
                m = pd.Timestamp(date_val).month
                if m in stn_clim[var]:
                    preds[i] = stn_clim[var][m]
            stn_baseline[var] = preds

        if stn_baseline:
            baseline[sid] = stn_baseline

    return baseline


def run_persistence_baseline(station_obs):
    """Persistence baseline: previous day's observation."""
    baseline = {}

    for sid, obs_df in station_obs.items():
        stn_baseline = {}

        for var in ["temperature", "precipitation"]:
            if var not in obs_df.columns:
                continue
            vals = obs_df[var].values.astype(np.float32)
            pred = np.full(len(vals), np.nan, dtype=np.float32)
            for i in range(1, len(vals)):
                if not np.isnan(vals[i - 1]):
                    pred[i] = vals[i - 1]
            stn_baseline[var] = pred

        if stn_baseline:
            baseline[sid] = stn_baseline

    return baseline


def run_bias_corrected_baseline(hrrr_tensors, hrrr_lat, hrrr_lon,
                                stations_df, station_obs, train_obs):
    """Bias-corrected HRRR: subtract per-station per-month training bias.

    Computes training-period mean (HRRR - obs) per station per month
    from WY2020-2023 data (using month-day matching to WY2024 HRRR),
    then subtracts that bias from WY2024 HRRR predictions.
    """
    from terrain_weather_ml.training.temporal_utils import snotel_date_to_hrrr_key

    stn_meta = stations_df.set_index("station_id")

    # Compute per-station per-month bias from training data
    bias_table = {}  # sid -> var -> month -> mean_bias
    for sid, train_df in train_obs.items():
        if sid not in stn_meta.index:
            continue

        stn_lat = stn_meta.loc[sid, "latitude"]
        stn_lon = stn_meta.loc[sid, "longitude"]
        hrrr_row, hrrr_col = find_nearest_hrrr_cell(
            stn_lat, stn_lon, hrrr_lat, hrrr_lon
        )

        var_bias = {}
        for var, hrrr_idx in [("temperature", 0), ("precipitation", 3)]:
            if var not in train_df.columns:
                continue
            monthly_diffs = {}  # month -> list of (hrrr - obs) diffs
            for i, date_val in enumerate(train_df.index):
                date_ts = pd.Timestamp(date_val)
                hrrr_key = snotel_date_to_hrrr_key(str(date_ts.date()))
                if hrrr_key not in hrrr_tensors:
                    continue
                obs_val = train_df[var].values[i]
                if np.isnan(obs_val):
                    continue
                hrrr_val = hrrr_tensors[hrrr_key][hrrr_idx, hrrr_row, hrrr_col].item()
                m = date_ts.month
                if m not in monthly_diffs:
                    monthly_diffs[m] = []
                monthly_diffs[m].append(hrrr_val - obs_val)

            month_bias = {}
            for m, diffs in monthly_diffs.items():
                if diffs:
                    month_bias[m] = float(np.mean(diffs))
            var_bias[var] = month_bias
        bias_table[sid] = var_bias

    # Apply bias correction to WY2024 predictions
    baseline = {}
    for sid, obs_df in station_obs.items():
        if sid not in stn_meta.index:
            continue

        stn_lat = stn_meta.loc[sid, "latitude"]
        stn_lon = stn_meta.loc[sid, "longitude"]
        hrrr_row, hrrr_col = find_nearest_hrrr_cell(
            stn_lat, stn_lon, hrrr_lat, hrrr_lon
        )
        stn_bias = bias_table.get(sid, {})

        stn_baseline = {}
        for var, hrrr_idx in [("temperature", 0), ("precipitation", 3)]:
            if var not in obs_df.columns:
                continue
            preds = np.full(len(obs_df), np.nan, dtype=np.float32)
            var_month_bias = stn_bias.get(var, {})

            for i, date_val in enumerate(obs_df.index):
                date_str = pd.Timestamp(date_val).strftime("%Y%m%d")
                if date_str not in hrrr_tensors:
                    continue
                m = pd.Timestamp(date_val).month
                hrrr_val = hrrr_tensors[date_str][hrrr_idx, hrrr_row, hrrr_col].item()
                bias = var_month_bias.get(m, 0.0)
                corrected = hrrr_val - bias
                if var == "precipitation":
                    corrected = max(0.0, corrected)
                preds[i] = corrected

            stn_baseline[var] = preds

        if stn_baseline:
            baseline[sid] = stn_baseline

    return baseline


def evaluate_predictions(station_obs, predictions, model_name="model"):
    """Compute per-station metrics and return ModelResults."""
    per_station = {}

    for sid, obs_df in station_obs.items():
        if sid not in predictions:
            continue

        stn_preds = predictions[sid]
        stn_metrics = {}

        for var in ["temperature", "precipitation"]:
            if var not in obs_df.columns or var not in stn_preds:
                continue
            obs_arr = obs_df[var].values.astype(np.float64)
            pred_arr = stn_preds[var].astype(np.float64)
            m = compute_continuous_metrics(obs_arr, pred_arr)
            stn_metrics[var] = {
                "rmse": m.rmse, "mae": m.mae, "bias": m.bias,
                "r_squared": m.r_squared,
            }

        if stn_metrics:
            per_station[sid] = stn_metrics

    return ModelResults(name=model_name, per_station=per_station)


def compute_station_tpi(terrain_tensors):
    """Extract mean TPI from terrain tensor for each station."""
    tpi_idx = 15
    tpi_values = {}
    for sid, tensor in terrain_tensors.items():
        tpi_values[sid] = tensor[tpi_idx].mean().item()
    return tpi_values


def compute_per_season_metrics(station_obs, predictions, stations_df):
    """Compute per-season (DJF/MAM/JJA/SON) aggregate metrics."""
    all_obs_temp = []
    all_pred_temp = []
    all_months = []

    for sid, obs_df in station_obs.items():
        if sid not in predictions:
            continue
        preds = predictions[sid]
        if "temperature" not in obs_df.columns or "temperature" not in preds:
            continue

        obs_arr = obs_df["temperature"].values.astype(np.float64)
        pred_arr = preds["temperature"].astype(np.float64)
        months_arr = pd.DatetimeIndex(obs_df.index).month

        all_obs_temp.extend(obs_arr)
        all_pred_temp.extend(pred_arr)
        all_months.extend(months_arr)

    if not all_obs_temp:
        return {}

    obs_all = np.array(all_obs_temp)
    pred_all = np.array(all_pred_temp)
    months_all = np.array(all_months)

    month_to_season = {
        12: "DJF", 1: "DJF", 2: "DJF",
        3: "MAM", 4: "MAM", 5: "MAM",
        6: "JJA", 7: "JJA", 8: "JJA",
        9: "SON", 10: "SON", 11: "SON",
    }

    results = {}
    for season in ("DJF", "MAM", "JJA", "SON"):
        season_months = [m for m, s in month_to_season.items() if s == season]
        mask = np.isin(months_all, season_months)
        if mask.sum() < 2:
            continue
        m = compute_continuous_metrics(obs_all[mask], pred_all[mask])
        results[season] = {
            "rmse": m.rmse, "mae": m.mae, "bias": m.bias,
            "r_squared": m.r_squared, "n_valid": m.n_valid,
        }

    return results


def compute_per_elevation_metrics(station_obs, predictions, stations_df):
    """Compute per-elevation-band aggregate metrics."""
    stn_meta = stations_df.set_index("station_id")
    elev_config = ElevationBandConfig()

    band_rmses = {"below_treeline": [], "near_treeline": [], "above_treeline": []}
    band_biases = {"below_treeline": [], "near_treeline": [], "above_treeline": []}
    band_counts = {"below_treeline": 0, "near_treeline": 0, "above_treeline": 0}

    for sid, metrics_dict in station_obs.items():
        if sid not in predictions or sid not in stn_meta.index:
            continue
        preds = predictions[sid]
        if "temperature" not in metrics_dict.columns or "temperature" not in preds:
            continue

        elev = stn_meta.loc[sid, "elevation"]
        band = classify_elevation_band(elev, elev_config)

        obs_arr = metrics_dict["temperature"].values.astype(np.float64)
        pred_arr = preds["temperature"].astype(np.float64)
        m = compute_continuous_metrics(obs_arr, pred_arr)

        if not np.isnan(m.rmse):
            band_rmses[band].append(m.rmse)
            band_biases[band].append(m.bias)
            band_counts[band] += 1

    results = {}
    for band in ("below_treeline", "near_treeline", "above_treeline"):
        if band_rmses[band]:
            arr = np.array(band_rmses[band])
            ci = bootstrap_confidence_intervals(arr, seed=42)
            results[band] = {
                "n_stations": band_counts[band],
                "mean_rmse": float(np.mean(arr)),
                "mean_bias": float(np.mean(band_biases[band])),
                "ci_lower": ci["lower"],
                "ci_upper": ci["upper"],
            }
        else:
            results[band] = {"n_stations": 0}

    return results


def generate_comprehensive_report(
    model_results, all_baseline_results, comparison,
    model_season, baseline_seasons, model_elev, baseline_elevs,
    skill_scores, cap_result, valley_stations, ridge_stations,
    n_stations, elapsed, stations_df, station_obs, model_preds,
):
    """Generate markdown validation report with all metrics."""
    lines = []
    lines.append("# Phase 3 Validation Report: WY2024 Holdout Evaluation")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("- **Evaluation period:** WY2024 (Oct 2023 - Sep 2024, holdout)")
    lines.append("- **Training period:** WY2020-2023")
    lines.append("- **Normalizer:** AnomalyNormalizer (per-station per-month climatology)")
    lines.append(f"- **Stations evaluated:** {n_stations}")
    lines.append(f"- **Runtime:** {elapsed:.1f}s")
    lines.append("- **Variables:** temperature (K), precipitation (kg/m^2/s)")
    lines.append("")

    # Baselines comparison
    baseline_names = [r.name for r in all_baseline_results]
    lines.append("## Model vs All Baselines (Aggregate)")
    lines.append("")
    header = "| Variable | Metric | Model |"
    divider = "|----------|--------|-------|"
    for bn in baseline_names:
        header += f" {bn} |"
        divider += "----------|"
    lines.append(header)
    lines.append(divider)

    model_agg = comparison.aggregate_metrics.get("terrain_downscaling", {})
    for var in ["temperature", "precipitation"]:
        if var not in model_agg:
            continue
        for metric in ["rmse", "mae", "bias"]:
            m_val = model_agg[var].get(metric, float("nan"))
            row = f"| {var} | {metric} | {m_val:.4f} |"
            for br in all_baseline_results:
                b_agg = comparison.aggregate_metrics.get(br.name, {})
                b_val = b_agg.get(var, {}).get(metric, float("nan"))
                row += f" {b_val:.4f} |"
            lines.append(row)

    # Skill scores vs each baseline
    lines.append("")
    lines.append("## Skill Scores (1 - RMSE_model / RMSE_baseline)")
    lines.append("")
    lines.append("| Variable | " + " | ".join(baseline_names) + " |")
    lines.append("|----------|" + "|".join(["----------"] * len(baseline_names)) + "|")
    for var in ["temperature", "precipitation"]:
        row = f"| {var} |"
        for bn in baseline_names:
            key = f"{var}_vs_{bn}"
            ss = skill_scores.get(key, float("nan"))
            row += f" {ss:+.4f} |"
        lines.append(row)

    # Per-season metrics
    lines.append("")
    lines.append("## Per-Season Temperature Metrics (Model)")
    lines.append("")
    lines.append("| Season | RMSE (K) | MAE (K) | Bias (K) | R^2 | N |")
    lines.append("|--------|----------|---------|----------|-----|---|")
    for season in ("DJF", "MAM", "JJA", "SON"):
        sm = model_season.get(season, {})
        lines.append(
            f"| {season} | {sm.get('rmse', float('nan')):.2f} "
            f"| {sm.get('mae', float('nan')):.2f} "
            f"| {sm.get('bias', float('nan')):+.2f} "
            f"| {sm.get('r_squared', float('nan')):.4f} "
            f"| {sm.get('n_valid', 0)} |"
        )

    # Per-elevation metrics
    lines.append("")
    lines.append("## Per-Elevation-Band Temperature Metrics (Model)")
    lines.append("")
    lines.append("| Elevation Band | Stations | Mean RMSE (K) | Mean Bias (K) | 95% CI |")
    lines.append("|----------------|----------|---------------|---------------|--------|")
    for band in ("below_treeline", "near_treeline", "above_treeline"):
        em = model_elev.get(band, {})
        n = em.get("n_stations", 0)
        if n > 0:
            lines.append(
                f"| {band} | {n} "
                f"| {em['mean_rmse']:.2f} "
                f"| {em['mean_bias']:+.2f} "
                f"| [{em['ci_lower']:.2f}, {em['ci_upper']:.2f}] |"
            )
        else:
            lines.append(f"| {band} | 0 | N/A | N/A | N/A |")

    # Bootstrap CIs on aggregate
    lines.append("")
    lines.append("## Bootstrap 95% Confidence Intervals")
    lines.append("")
    for var in ["temperature", "precipitation"]:
        station_rmses = []
        for sid, metrics in model_results.per_station.items():
            if var in metrics and "rmse" in metrics[var]:
                station_rmses.append(metrics[var]["rmse"])
        if station_rmses:
            arr = np.array(station_rmses)
            ci = bootstrap_confidence_intervals(arr, seed=42)
            lines.append(
                f"- **{var} RMSE:** {ci['point_estimate']:.4f} "
                f"[{ci['lower']:.4f}, {ci['upper']:.4f}]"
            )

    # Per-station table (top 20 + bottom 20)
    lines.append("")
    lines.append("## Per-Station Temperature Metrics (Selected)")
    lines.append("")
    lines.append("| Station | Elev (m) | Model RMSE | HRRR RMSE | Model Bias | R^2 |")
    lines.append("|---------|----------|-----------|-----------|------------|-----|")

    stn_meta = stations_df.set_index("station_id")
    station_rmse_pairs = []
    hrrr_results = next(
        (r for r in all_baseline_results if r.name == "hrrr_raw"), None
    )
    for sid in sorted(model_results.per_station.keys()):
        m_t = model_results.per_station[sid].get("temperature", {})
        if "rmse" in m_t:
            station_rmse_pairs.append((sid, m_t["rmse"]))

    station_rmse_pairs.sort(key=lambda x: x[1])
    selected = station_rmse_pairs[:10] + station_rmse_pairs[-10:]
    for sid, _ in selected:
        m_t = model_results.per_station[sid].get("temperature", {})
        h_t = {}
        if hrrr_results and sid in hrrr_results.per_station:
            h_t = hrrr_results.per_station[sid].get("temperature", {})
        elev = stn_meta.loc[sid, "elevation"] if sid in stn_meta.index else 0
        lines.append(
            f"| {sid} | {elev:.0f} | {m_t.get('rmse', float('nan')):.2f} "
            f"| {h_t.get('rmse', float('nan')):.2f} "
            f"| {m_t.get('bias', float('nan')):+.2f} "
            f"| {m_t.get('r_squared', float('nan')):.4f} |"
        )

    # Significance tests
    lines.append("")
    lines.append("## Statistical Significance (Paired t-test on RMSE)")
    lines.append("")
    for pair_key, pair_results in comparison.significance.items():
        for var, sig_data in pair_results.items():
            t_stat = sig_data.get("t_statistic", float("nan"))
            p_val = sig_data.get("p_value", float("nan"))
            sig = sig_data.get("significant", False)
            better = sig_data.get("better_model", "N/A")
            marker = "**" if sig else ""
            lines.append(
                f"- {pair_key} ({var}): t={t_stat:.3f}, p={p_val:.4f}, "
                f"{marker}{'significant' if sig else 'not significant'}{marker}, "
                f"better={better}"
            )

    # Physics check
    lines.append("")
    lines.append("## Physics Checks")
    lines.append("")
    lines.append("### Cold Air Pool Reproduction")
    lines.append("")
    lines.append(f"- Valley stations (TPI < -20m): {len(valley_stations)}")
    lines.append(f"- Ridge stations (TPI > 50m): {len(ridge_stations)}")

    if cap_result is not None:
        lines.append(f"- **Passed:** {cap_result.passed}")
        if cap_result.not_applicable:
            lines.append(f"- Not applicable: {cap_result.details.get('reason', 'N/A')}")
        else:
            lines.append(f"- Mean valley temperature bias: {cap_result.mean_value:.2f} K")
            lines.append(f"- Max valley temperature bias: {cap_result.max_value:.2f} K")
            lines.append(f"- Threshold: {cap_result.threshold} K")
            lines.append(f"- Inversions detected: {cap_result.details.get('n_inversions', 0)}")
    else:
        lines.append("- Insufficient valley/ridge station pairs for analysis")

    # Interpretation
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("This is a **proper holdout evaluation** using WY2024 data")
    lines.append("that was not used during any phase of model training.")
    lines.append("The AnomalyNormalizer removes per-station per-month climatology")
    lines.append("before z-scoring, preventing seasonal cycle conflation.")
    lines.append("")
    lines.append("Key questions answered:")
    lines.append("1. Does the terrain downscaling head generalize to unseen time? (temporal holdout)")
    lines.append("2. Does terrain correction beat all baselines? (skill scores)")
    lines.append("3. How does performance vary by season? (DJF/MAM/JJA/SON)")
    lines.append("4. How does performance vary by elevation? (treeline bands)")
    lines.append("5. Does the model respect cold air pooling physics? (CAP check)")
    lines.append("")

    return "\n".join(lines)


def run_cold_air_pool_check(station_obs, predictions, tpi_values, stations_df):
    """Run cold air pool physics check using TPI to classify valley/ridge."""
    tpi_threshold_valley = -20.0
    tpi_threshold_ridge = 50.0

    valley_stations = [
        sid for sid, tpi in tpi_values.items()
        if tpi < tpi_threshold_valley and sid in station_obs and sid in predictions
    ]
    ridge_stations = [
        sid for sid, tpi in tpi_values.items()
        if tpi > tpi_threshold_ridge and sid in station_obs and sid in predictions
    ]

    if not valley_stations or not ridge_stations:
        return None, valley_stations, ridge_stations

    valley_obs_temps = []
    valley_pred_temps = []
    ridge_obs_temps = []

    for sid in valley_stations:
        obs_df = station_obs[sid]
        if "temperature" not in obs_df.columns or "temperature" not in predictions[sid]:
            continue
        valley_obs_temps.append(obs_df["temperature"].values.astype(np.float64))
        valley_pred_temps.append(predictions[sid]["temperature"].astype(np.float64))

    for sid in ridge_stations:
        obs_df = station_obs[sid]
        if "temperature" not in obs_df.columns:
            continue
        ridge_obs_temps.append(obs_df["temperature"].values.astype(np.float64))

    if not valley_obs_temps or not ridge_obs_temps:
        return None, valley_stations, ridge_stations

    min_len = min(
        min(len(a) for a in valley_obs_temps),
        min(len(a) for a in ridge_obs_temps),
        min(len(a) for a in valley_pred_temps),
    )

    obs_valley = np.mean([a[:min_len] for a in valley_obs_temps], axis=0)
    pred_valley = np.mean([a[:min_len] for a in valley_pred_temps], axis=0)
    obs_ridge = np.mean([a[:min_len] for a in ridge_obs_temps], axis=0)

    result = check_cold_air_pool(obs_valley, obs_ridge, pred_valley, threshold=3.0)
    return result, valley_stations, ridge_stations


def main():
    t_start = time.time()
    device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info("Device: %s", device_name)

    # 1. Load model
    logger.info("Loading Phase 3 model...")
    head, mapper, normalizer = load_model(PHASE3_CKPT)
    head.to(device_name)

    # 2. Load SNOTEL station metadata
    stations_df = pd.read_csv(STATIONS_PATH)
    stations_df["elevation"] = stations_df["elevation_ft"] * 0.3048
    logger.info("Loaded %d SNOTEL stations", len(stations_df))

    # 3. Extract DEM patches
    logger.info("Extracting DEM patches...")
    patches = extract_dem_patches(stations_df, DEM_PATH, PATCH_SIZE)
    logger.info("Extracted %d valid patches (all stations)", len(patches))

    # 4. Compute terrain features
    logger.info("Computing terrain features...")
    terrain_tensors = compute_all_terrain(patches, TERRAIN_CACHE)
    terrain_tensors = {sid: t.to(device_name) for sid, t in terrain_tensors.items()}

    # 5. Load HRRR 12Z data (all 366 days)
    logger.info("Loading HRRR data (all 366 days)...")
    hrrr_tensors, hrrr_lat, hrrr_lon = load_hrrr_12z(HRRR_DIR)
    logger.info("Loaded %d HRRR timesteps", len(hrrr_tensors))
    hrrr_tensors_device = {k: v.to(device_name) for k, v in hrrr_tensors.items()}

    # 6. Load WY2024 SNOTEL observations (holdout)
    logger.info("Loading WY2024 SNOTEL observations (holdout)...")
    station_obs = load_snotel_wy2024(SNOTEL_DIR, list(patches.keys()))
    logger.info("WY2024 observations for %d stations", len(station_obs))

    if not station_obs:
        logger.error("No station observations found.")
        sys.exit(1)

    # 7. Load WY2020-2023 training observations (for baseline fitting)
    logger.info("Loading WY2020-2023 training observations for baselines...")
    train_obs = load_snotel_training(SNOTEL_DIR, list(patches.keys()))
    logger.info("Training observations for %d stations", len(train_obs))

    # 8. Run model predictions
    logger.info("Running model predictions...")
    t_pred = time.time()
    model_preds = run_model_predictions(
        head, mapper, normalizer, terrain_tensors, hrrr_tensors_device,
        hrrr_lat, hrrr_lon, stations_df, station_obs,
    )
    logger.info("Model predictions done in %.1fs for %d stations",
                time.time() - t_pred, len(model_preds))

    # 9. Run all baselines
    hrrr_cpu = {k: v.cpu() for k, v in hrrr_tensors_device.items()}

    logger.info("Running baseline: raw HRRR...")
    hrrr_preds = run_hrrr_baseline(
        hrrr_cpu, hrrr_lat, hrrr_lon, stations_df, station_obs,
    )

    logger.info("Running baseline: lapse-rate adjusted HRRR...")
    lapse_preds = run_lapse_rate_baseline(
        hrrr_cpu, hrrr_lat, hrrr_lon, stations_df, station_obs, patches,
    )

    logger.info("Running baseline: climatology (WY2020-2023)...")
    clim_preds = run_climatology_baseline(station_obs, train_obs)

    logger.info("Running baseline: persistence...")
    persist_preds = run_persistence_baseline(station_obs)

    logger.info("Running baseline: bias-corrected HRRR...")
    bias_preds = run_bias_corrected_baseline(
        hrrr_cpu, hrrr_lat, hrrr_lon, stations_df, station_obs, train_obs,
    )

    # 10. Evaluate all models
    logger.info("Computing metrics...")
    model_results = evaluate_predictions(station_obs, model_preds, "terrain_downscaling")
    hrrr_results = evaluate_predictions(station_obs, hrrr_preds, "hrrr_raw")
    lapse_results = evaluate_predictions(station_obs, lapse_preds, "hrrr_lapse_rate")
    clim_results = evaluate_predictions(station_obs, clim_preds, "climatology")
    persist_results = evaluate_predictions(station_obs, persist_preds, "persistence")
    bias_results = evaluate_predictions(station_obs, bias_preds, "hrrr_bias_corrected")

    all_results = [model_results, hrrr_results, lapse_results,
                   clim_results, persist_results, bias_results]

    logger.info("Model: %d stations", len(model_results.per_station))
    for r in all_results[1:]:
        logger.info("Baseline %s: %d stations", r.name, len(r.per_station))

    # 11. Generate comparison report
    comparison = generate_comparison_report(all_results)

    # 12. Compute skill scores vs each baseline
    skill_scores = {}
    model_agg = comparison.aggregate_metrics.get("terrain_downscaling", {})
    for baseline_r in all_results[1:]:
        b_agg = comparison.aggregate_metrics.get(baseline_r.name, {})
        for var in ["temperature", "precipitation"]:
            m_rmse = model_agg.get(var, {}).get("rmse", float("nan"))
            b_rmse = b_agg.get(var, {}).get("rmse", float("nan"))
            key = f"{var}_vs_{baseline_r.name}"
            skill_scores[key] = compute_skill_score(m_rmse, b_rmse)

    # 13. Per-season and per-elevation metrics
    logger.info("Computing per-season metrics...")
    model_season = compute_per_season_metrics(station_obs, model_preds, stations_df)
    baseline_seasons = {}
    for r in all_results[1:]:
        preds_for_r = {
            "hrrr_raw": hrrr_preds, "hrrr_lapse_rate": lapse_preds,
            "climatology": clim_preds, "persistence": persist_preds,
            "hrrr_bias_corrected": bias_preds,
        }.get(r.name, {})
        baseline_seasons[r.name] = compute_per_season_metrics(
            station_obs, preds_for_r, stations_df
        )

    logger.info("Computing per-elevation metrics...")
    model_elev = compute_per_elevation_metrics(station_obs, model_preds, stations_df)
    baseline_elevs = {}
    for r in all_results[1:]:
        preds_for_r = {
            "hrrr_raw": hrrr_preds, "hrrr_lapse_rate": lapse_preds,
            "climatology": clim_preds, "persistence": persist_preds,
            "hrrr_bias_corrected": bias_preds,
        }.get(r.name, {})
        baseline_elevs[r.name] = compute_per_elevation_metrics(
            station_obs, preds_for_r, stations_df
        )

    # 14. Cold air pool physics check
    logger.info("Running physics checks...")
    tpi_values = compute_station_tpi(terrain_tensors)
    cap_result, valley_stations, ridge_stations = run_cold_air_pool_check(
        station_obs, model_preds, tpi_values, stations_df,
    )

    # 15. Generate and save report
    elapsed = time.time() - t_start
    report = generate_comprehensive_report(
        model_results, all_results[1:], comparison,
        model_season, baseline_seasons, model_elev, baseline_elevs,
        skill_scores, cap_result, valley_stations, ridge_stations,
        len(model_results.per_station), elapsed,
        stations_df, station_obs, model_preds,
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "validation_phase3_holdout_wy2024.md"
    report_path.write_text(report)
    logger.info("Report saved to %s", report_path)

    # 16. Print summary
    print()
    print("=" * 60)
    print("Phase 3 Validation: WY2024 Holdout (AnomalyNormalizer)")
    print("=" * 60)

    if "temperature" in model_agg:
        m_t = model_agg["temperature"]
        print("\nTemperature (aggregate):")
        print(f"  Model RMSE:    {m_t.get('rmse', 'N/A'):.2f} K")
        for r in all_results[1:]:
            b_agg = comparison.aggregate_metrics.get(r.name, {})
            b_t = b_agg.get("temperature", {})
            print(f"  {r.name} RMSE: {b_t.get('rmse', float('nan')):.2f} K")

    if "precipitation" in model_agg:
        m_p = model_agg["precipitation"]
        print("\nPrecipitation (aggregate):")
        print(f"  Model RMSE:    {m_p.get('rmse', 'N/A'):.6f} kg/m^2/s")

    print("\nSkill scores (temperature):")
    for bn in [r.name for r in all_results[1:]]:
        key = f"temperature_vs_{bn}"
        ss = skill_scores.get(key, float("nan"))
        print(f"  vs {bn}: {ss:+.4f}")

    print("\nPer-season temperature RMSE:")
    for season in ("DJF", "MAM", "JJA", "SON"):
        sm = model_season.get(season, {})
        print(f"  {season}: {sm.get('rmse', float('nan')):.2f} K")

    if cap_result is not None:
        print(f"\nCold Air Pool: {'PASS' if cap_result.passed else 'FAIL'}")
        if not cap_result.not_applicable:
            print(f"  Mean bias: {cap_result.mean_value:.2f} K")
    else:
        print("\nCold Air Pool: N/A (insufficient station pairs)")

    print(f"\nStations: {len(model_results.per_station)}")
    print(f"Runtime: {elapsed:.1f}s")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
