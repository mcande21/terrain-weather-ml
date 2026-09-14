"""Phase 3 LOSO cross-validation: evaluate Alps->Rockies terrain downscaling.

Loads the Phase 3 checkpoint and runs leave-one-station-out evaluation
against SNOTEL observations for January 2024. Compares model predictions
(HRRR + terrain downscaling) vs raw HRRR baseline. Runs cold air pool
physics check on valley vs ridge stations.

Note: this evaluates on the training period (Jan 2024). True generalization
requires a holdout period, but this validates the pipeline and model behavior.
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

from terrain_weather_ml.evaluation.metrics import (
    compute_continuous_metrics,
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
OBS_PATH = PROJECT_ROOT / "data" / "snotel" / "observations_wy2024.csv"
TERRAIN_CACHE = PROJECT_ROOT / "data" / "cache" / "terrain_patches_colorado"
REPORT_DIR = PROJECT_ROOT / "reports"

# --- Config ---
PATCH_SIZE = 64
MAX_EVAL_STATIONS = 15
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


def load_model(checkpoint_path):
    """Load Phase 3 model head and quantile mapper from checkpoint."""
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

    return head, mapper


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


def load_snotel_january(obs_path, stations_path, station_ids):
    """Load SNOTEL January 2024 observations: temp (K) and precip rate (kg/m^2/s)."""
    obs = pd.read_csv(obs_path)
    obs["date"] = pd.to_datetime(obs["date"])

    jan_mask = (obs["date"].dt.month == 1) & (obs["date"].dt.year == 2024)
    obs_jan = obs[jan_mask].copy()

    records = {}
    for sid in station_ids:
        stn_obs = obs_jan[obs_jan["station_id"] == sid].sort_values("date")
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


def run_model_predictions(head, mapper, terrain_tensors, hrrr_tensors,
                          hrrr_lat, hrrr_lon, stations_df, station_obs):
    """Run model predictions for each station on each day.

    Returns dict of station_id -> {"temperature": array, "precipitation": array}
    matching the daily observations.
    """
    stn_meta = stations_df.set_index("station_id")
    predictions = {}

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
            date_str = pd.Timestamp(date_val).strftime("%Y%m%d")
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
            temp_preds.append(out[0, IDX_TEMPERATURE, h_mid, w_mid].item())
            prec_preds.append(max(0.0, out[0, IDX_PRECIPITATION, h_mid, w_mid].item()))

        stn_preds = {}
        if "temperature" in obs_df.columns:
            stn_preds["temperature"] = np.array(temp_preds)
        if "precipitation" in obs_df.columns:
            stn_preds["precipitation"] = np.array(prec_preds)

        predictions[sid] = stn_preds

    return predictions


def run_hrrr_baseline(hrrr_tensors, hrrr_lat, hrrr_lon,
                      stations_df, station_obs):
    """Extract raw HRRR value at each station location as baseline.

    HRRR t2m is in Kelvin. HRRR prate is in kg/m^2/s.
    """
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


def compute_station_tpi(terrain_tensors):
    """Extract mean TPI from terrain tensor for each station."""
    tpi_idx = 15
    tpi_values = {}
    for sid, tensor in terrain_tensors.items():
        tpi_values[sid] = tensor[tpi_idx].mean().item()
    return tpi_values


def evaluate_predictions(station_obs, predictions, model_name="model"):
    """Compute per-station metrics and return ModelResults."""
    per_station = {}

    for sid, obs_df in station_obs.items():
        if sid not in predictions:
            continue

        stn_preds = predictions[sid]
        stn_metrics = {}

        if "temperature" in obs_df.columns and "temperature" in stn_preds:
            obs_arr = obs_df["temperature"].values.astype(np.float64)
            pred_arr = stn_preds["temperature"].astype(np.float64)
            m = compute_continuous_metrics(obs_arr, pred_arr)
            stn_metrics["temperature"] = {
                "rmse": m.rmse, "mae": m.mae, "bias": m.bias,
                "r_squared": m.r_squared,
            }

        if "precipitation" in obs_df.columns and "precipitation" in stn_preds:
            obs_arr = obs_df["precipitation"].values.astype(np.float64)
            pred_arr = stn_preds["precipitation"].astype(np.float64)
            m = compute_continuous_metrics(obs_arr, pred_arr)
            stn_metrics["precipitation"] = {
                "rmse": m.rmse, "mae": m.mae, "bias": m.bias,
                "r_squared": m.r_squared,
            }

        if stn_metrics:
            per_station[sid] = stn_metrics

    return ModelResults(name=model_name, per_station=per_station)


def run_cold_air_pool_check(station_obs, predictions, tpi_values, stations_df):
    """Run cold air pool physics check using TPI to classify valley/ridge."""
    tpi_threshold_valley = -20.0
    tpi_threshold_ridge = 50.0

    valley_stations = [sid for sid, tpi in tpi_values.items()
                       if tpi < tpi_threshold_valley and sid in station_obs and sid in predictions]
    ridge_stations = [sid for sid, tpi in tpi_values.items()
                      if tpi > tpi_threshold_ridge and sid in station_obs and sid in predictions]

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


def generate_report(model_results, baseline_results, comparison,
                    cap_result, valley_stations, ridge_stations,
                    n_stations, elapsed):
    """Generate markdown validation report."""
    lines = []
    lines.append("# Phase 3 Validation Report: LOSO Cross-Validation (January 2024)")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("- **Evaluation period:** January 2024 (training set)")
    lines.append(f"- **Stations evaluated:** {n_stations}")
    lines.append(f"- **Runtime:** {elapsed:.1f}s")
    lines.append("- **Variables:** temperature (K), precipitation (kg/m^2/s)")
    lines.append("- **Baseline:** Raw HRRR nearest grid cell (3km)")
    lines.append("")

    # Aggregate comparison
    lines.append("## Model vs HRRR Baseline (Aggregate)")
    lines.append("")
    lines.append("| Variable | Metric | Model | HRRR Baseline | Improvement |")
    lines.append("|----------|--------|-------|---------------|-------------|")

    model_agg = comparison.aggregate_metrics.get("terrain_downscaling", {})
    baseline_agg = comparison.aggregate_metrics.get("hrrr_raw", {})

    for var in ["temperature", "precipitation"]:
        if var not in model_agg or var not in baseline_agg:
            continue
        for metric in ["rmse", "mae", "bias", "r_squared"]:
            m_val = model_agg[var].get(metric, float("nan"))
            b_val = baseline_agg[var].get(metric, float("nan"))
            if np.isnan(m_val) or np.isnan(b_val):
                continue
            if metric == "r_squared":
                imp = f"{m_val - b_val:+.4f}"
            else:
                pct = (b_val - m_val) / abs(b_val) * 100 if abs(b_val) > 1e-10 else 0
                imp = f"{pct:+.1f}%"
            lines.append(f"| {var} | {metric} | {m_val:.4f} | {b_val:.4f} | {imp} |")

    # Per-station temperature metrics
    lines.append("")
    lines.append("## Per-Station Temperature Metrics")
    lines.append("")
    lines.append("| Station | Model RMSE | HRRR RMSE | Model Bias | HRRR Bias | Model R^2 |")
    lines.append("|---------|-----------|-----------|------------|-----------|-----------|")

    for sid in sorted(model_results.per_station.keys()):
        m_t = model_results.per_station[sid].get("temperature", {})
        b_t = baseline_results.per_station[sid].get("temperature", {})
        if not m_t:
            continue
        m_rmse = m_t.get("rmse", float("nan"))
        b_rmse = b_t.get("rmse", float("nan"))
        m_bias = m_t.get("bias", float("nan"))
        b_bias = b_t.get("bias", float("nan"))
        m_r2 = m_t.get("r_squared", float("nan"))
        lines.append(
            f"| {sid} | {m_rmse:.2f} | {b_rmse:.2f} | "
            f"{m_bias:+.2f} | {b_bias:+.2f} | {m_r2:.4f} |"
        )

    # Significance test
    lines.append("")
    lines.append("## Statistical Significance")
    lines.append("")
    for pair_key, pair_results in comparison.significance.items():
        for var, sig_data in pair_results.items():
            lines.append(f"- **{pair_key} ({var}):** "
                         f"t={sig_data.get('t_statistic', 'N/A'):.3f}, "
                         f"p={sig_data.get('p_value', 'N/A'):.4f}, "
                         f"significant={sig_data.get('significant', 'N/A')}, "
                         f"better={sig_data.get('better_model', 'N/A')}")

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
    lines.append("This is a **training-set evaluation** (January 2024 was used for Phase 3")
    lines.append("adaptation). Results validate the pipeline works and the model produces")
    lines.append("physically reasonable predictions. True generalization requires evaluation")
    lines.append("on a holdout period (e.g., February-March 2024).")
    lines.append("")
    lines.append("Key questions answered:")
    lines.append("1. Does the terrain downscaling head produce valid predictions? (pipeline check)")
    lines.append("2. Does terrain correction improve over raw HRRR? (value-add check)")
    lines.append("3. Does the model respect cold air pooling physics? (physics check)")
    lines.append("")

    return "\n".join(lines)


def main():
    t_start = time.time()
    device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info("Device: %s", device_name)

    # 1. Load model
    logger.info("Loading Phase 3 model...")
    head, mapper = load_model(PHASE3_CKPT)
    head.to(device_name)

    # 2. Load SNOTEL station metadata
    stations_df = pd.read_csv(STATIONS_PATH)
    logger.info("Loaded %d SNOTEL stations", len(stations_df))

    # 3. Extract DEM patches
    logger.info("Extracting DEM patches...")
    patches = extract_dem_patches(stations_df, DEM_PATH, PATCH_SIZE)
    logger.info("Extracted %d valid patches", len(patches))

    # 4. Subset for speed
    selected_ids = sorted(patches.keys())[:MAX_EVAL_STATIONS]
    patches = {k: patches[k] for k in selected_ids}
    logger.info("Selected %d stations for evaluation", len(patches))

    # 5. Compute terrain features
    logger.info("Computing terrain features...")
    terrain_tensors = compute_all_terrain(patches, TERRAIN_CACHE)

    # Move terrain to device
    terrain_tensors = {sid: t.to(device_name) for sid, t in terrain_tensors.items()}

    # 6. Load HRRR 12Z data
    logger.info("Loading HRRR data...")
    hrrr_tensors, hrrr_lat, hrrr_lon = load_hrrr_12z(HRRR_DIR)
    logger.info("Loaded %d HRRR timesteps", len(hrrr_tensors))

    # Move HRRR to device
    hrrr_tensors = {k: v.to(device_name) for k, v in hrrr_tensors.items()}

    # 7. Load SNOTEL observations
    logger.info("Loading SNOTEL observations...")
    station_obs = load_snotel_january(OBS_PATH, STATIONS_PATH, list(patches.keys()))
    logger.info("Observations loaded for %d stations", len(station_obs))

    if not station_obs:
        logger.error("No station observations found.")
        sys.exit(1)

    # 8. Run model predictions
    logger.info("Running model predictions...")
    t_pred = time.time()
    model_preds = run_model_predictions(
        head, mapper, terrain_tensors, hrrr_tensors,
        hrrr_lat, hrrr_lon, stations_df, station_obs,
    )
    logger.info("Model predictions done in %.1fs for %d stations",
                time.time() - t_pred, len(model_preds))

    # 9. Run HRRR baseline
    logger.info("Running HRRR baseline...")
    # Baseline uses CPU tensors for raw extraction
    hrrr_cpu = {k: v.cpu() for k, v in hrrr_tensors.items()}
    baseline_preds = run_hrrr_baseline(
        hrrr_cpu, hrrr_lat, hrrr_lon, stations_df, station_obs,
    )
    logger.info("Baseline predictions for %d stations", len(baseline_preds))

    # 10. Evaluate model and baseline
    logger.info("Computing metrics...")
    model_results = evaluate_predictions(station_obs, model_preds, "terrain_downscaling")
    baseline_results = evaluate_predictions(station_obs, baseline_preds, "hrrr_raw")

    logger.info("Model: %d stations evaluated", len(model_results.per_station))
    logger.info("Baseline: %d stations evaluated", len(baseline_results.per_station))

    # 11. Generate comparison report
    comparison = generate_comparison_report([model_results, baseline_results])

    # 12. Cold air pool physics check
    logger.info("Running physics checks...")
    tpi_values = compute_station_tpi(terrain_tensors)
    cap_result, valley_stations, ridge_stations = run_cold_air_pool_check(
        station_obs, model_preds, tpi_values, stations_df,
    )

    # 13. Generate and save report
    elapsed = time.time() - t_start
    report = generate_report(
        model_results, baseline_results, comparison,
        cap_result, valley_stations, ridge_stations,
        len(model_results.per_station), elapsed,
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "validation_phase3_jan2024.md"
    report_path.write_text(report)
    logger.info("Report saved to %s", report_path)

    # Print summary to stdout
    print()
    print("=" * 60)
    print("Phase 3 Validation Summary")
    print("=" * 60)

    model_agg = comparison.aggregate_metrics.get("terrain_downscaling", {})
    baseline_agg = comparison.aggregate_metrics.get("hrrr_raw", {})

    if "temperature" in model_agg:
        m_t = model_agg["temperature"]
        b_t = baseline_agg.get("temperature", {})
        print("\nTemperature:")
        print(f"  Model RMSE:    {m_t.get('rmse', 'N/A'):.2f} K")
        print(f"  HRRR RMSE:     {b_t.get('rmse', 'N/A'):.2f} K")
        print(f"  Model Bias:    {m_t.get('bias', 'N/A'):+.2f} K")
        print(f"  HRRR Bias:     {b_t.get('bias', 'N/A'):+.2f} K")
        print(f"  Model R^2:     {m_t.get('r_squared', 'N/A'):.4f}")
        print(f"  HRRR R^2:      {b_t.get('r_squared', 'N/A'):.4f}")

    if "precipitation" in model_agg:
        m_p = model_agg["precipitation"]
        b_p = baseline_agg.get("precipitation", {})
        print("\nPrecipitation:")
        print(f"  Model RMSE:    {m_p.get('rmse', 'N/A'):.6f} kg/m^2/s")
        print(f"  HRRR RMSE:     {b_p.get('rmse', 'N/A'):.6f} kg/m^2/s")

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
