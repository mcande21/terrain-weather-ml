## Context

See proposal.md for motivation (cross-domain terrain-weather transfer learning, Alps to Rockies).

Current state: greenfield project scaffolded with dependencies (`torch`, `xarray`, `rasterio`, `duckdb`, `httpx`, `huggingface-hub`, `safetensors`) but no implementation code. Three open-weight models form the foundation: StormCast (3km weather backbone), DEVINE (30m wind downscaling), and granite-wxc-downscaling (baseline comparison). Dense Alpine station networks (PeakWeather 302 stations, Swiss IMIS 180 stations) provide pre-training data; sparse Colorado SNOTEL (118 stations) is the fine-tuning target.

Key constraints shaping the design:
- **Compute budget:** Apple Silicon (MPS) for development, GPU cloud for training. LoRA keeps fine-tuning feasible on consumer hardware for iteration.
- **Data asymmetry:** Alpine data is dense and abundant; Colorado data is sparse. The entire architecture exists to bridge this gap via transfer learning.
- **Safety-critical downstream:** colorado-avalanche-ml consumes predictions for avalanche forecasting. Tail-event accuracy matters more than average skill.
- **Resolution mismatch:** HRRR input is 3km, DEM terrain is 30m, target output is sub-km (~250m). The downscaling head must bridge 12x resolution.

## Goals / Non-Goals

**Goals:**
- Dual-branch architecture separating static terrain features from dynamic weather, enabling terrain computation to be amortized across timesteps
- StormCast backbone integration with LoRA adapters that preserve the pre-trained weather representation while enabling Colorado specialization
- DEVINE-extended U-Net that downscales multi-variable weather (not just wind) from 3km to sub-km with mass-conservation enforcement
- 3-phase training pipeline with domain alignment between Alpine and Colorado distributions
- Inference API contract compatible with colorado-avalanche-ml Phase 2/3

**Non-Goals:**
- Training StormCast from scratch or modifying its architecture (frozen backbone, LoRA only)
- Real-time operational forecasting (batch/on-demand inference, not streaming)
- Global coverage (Colorado mountains are the fine-tuning target; generalization to other ranges is a future concern)
- Ensemble or probabilistic output (deterministic predictions first, uncertainty quantification deferred)
- Processing the full 21TB Reynolds Creek archive (staged subset for validation only)

## Decisions

### 1. Dual-branch input architecture: static terrain + dynamic weather

Terrain features (elevation, slope, aspect, curvature, SVF, Sx, roughness) are computed once from the DEM and reused across all timesteps. Weather features (temperature, wind, precipitation, pressure, BLH) change per timestep. These enter as separate branches and are concatenated before the U-Net decoder.

**Why over single-branch:** Computing terrain features is expensive (Sx wind exposure requires directional ray-tracing). Separating the branches means terrain computation is amortized — computed once per geographic region, stored as tensors, and loaded alongside dynamic weather inputs. A single-branch approach would either recompute terrain every timestep (wasteful) or require the model to learn terrain invariance from data (harder, more data-hungry).

**Alternative considered:** Late fusion (separate encoders merging at bottleneck). Rejected because terrain features need to condition every spatial scale of the U-Net — a mid-level concat before the decoder lets skip connections carry terrain information to all decoder layers.

### 2. StormCast surface variable selection

StormCast produces 99 output variables across multiple pressure levels. We consume only surface-relevant fields:
- 2m temperature (T2M)
- 10m wind u-component (U10)
- 10m wind v-component (V10)
- Precipitation rate (PRATE)
- Surface pressure (SP)
- Boundary layer height (BLH)

**Why this subset:** These six variables are what the terrain downscaling head needs to produce sub-km surface weather. Upper-atmosphere variables don't directly inform surface micro-weather in complex terrain — the terrain itself creates the micro-scale patterns (valley cold pools, ridge acceleration, orographic precipitation) from surface-level inputs. BLH is the one non-surface variable included because it governs whether terrain effects penetrate the boundary layer or are damped.

**Alternative considered:** Including 850hPa and 700hPa winds/temperature for better representation of above-ridge flow. Deferred to a future iteration — adds complexity to the input tensor and the Alpine pre-training data may not have the vertical structure to validate it.

### 3. DEVINE extension for multi-variable output

Start from DEVINE's pre-trained wind downscaling weights (U, V components). Extend the output head with parallel channels for:
- Temperature (terrain-adjusted via lapse rate and cold-air pooling)
- Precipitation (orographic enhancement/shadow)

**Training strategy:** Freeze the wind output channels during initial Alpine fine-tuning, allowing the new temperature and precipitation channels to train without disturbing DEVINE's wind skill. After temperature/precipitation channels converge, unfreeze wind channels for end-to-end joint optimization.

**Why extend DEVINE over training from scratch:** DEVINE already encodes the terrain→wind relationship from 30m Swiss data. Its convolutional structure is proven for this exact task geometry. Extending it preserves months of pre-training compute and the learned terrain-wind physics.

**Alternative considered:** Separate models per variable (one for wind, one for temperature, one for precipitation). Rejected because variables are physically coupled — orographic precipitation depends on wind speed and direction, valley temperature depends on wind-driven mixing. Joint prediction enforces physical consistency.

### 4. LoRA adapter placement and configuration

LoRA adapters on StormCast's attention layers only, rank r=8, following the WeatherPEFT pattern for regional specialization.

**Why attention layers only:** Attention layers capture long-range spatial dependencies — exactly what changes between Alpine and Colorado terrain geometries. Convolutional layers encode local feature extraction patterns that transfer well across mountain regions. Adapting only attention keeps the parameter count low (~0.5% of backbone) while targeting the layers most sensitive to domain shift.

**Why r=8:** Empirical sweet spot from WeatherPEFT (Nguyen et al.). r=4 under-adapts for cross-continental domain shift; r=16 risks overfitting to 118 SNOTEL stations. r=8 with 118 stations gives ~15x more data points per parameter than r=16.

**Alternative considered:** Full fine-tuning of final layers. Rejected due to catastrophic forgetting risk and compute cost — LoRA preserves the backbone's weather prior.

### 5. Loss function: MSE + mass-conservation penalty

```
L = L_mse(y_pred, y_true) + lambda_div * L_div(u_pred, v_pred) + lambda_oro * L_oro(p_pred, u_pred, v_pred, terrain)
```

- **L_mse:** Per-variable MSE reconstruction loss, with per-variable weighting to balance wind/temperature/precipitation scales
- **L_div:** Divergence penalty on predicted wind field: `||div(u, v)||^2` computed via finite differences on the output grid. Enforces mass conservation as a soft constraint during training.
- **L_oro (optional):** Orographic precipitation consistency — predicted precipitation should increase on windward slopes and decrease on leeward slopes relative to the terrain gradient and wind direction. Formulated as a correlation penalty.

**Why soft constraint over hard projection:** Hard divergence-free projection (Helmholtz decomposition) can be applied at inference time, but as a training loss the soft constraint lets the model learn to produce near-divergence-free fields naturally. The gradient signal from the penalty shapes the learned representation. At inference, a post-hoc Helmholtz projection enforces exact mass conservation.

**Alternative considered:** Physics-informed neural operator (PINO) approach with PDE residual losses. Too complex for v1 — the divergence penalty captures the most important physical constraint without requiring a full PDE solver in the training loop.

### 6. Domain adaptation via quantile mapping

Quantile-mapping alignment between Alpine and Colorado weather distributions, applied at the Alpine→Colorado transition (phase 2→phase 3 of training). Per-variable, per-season mapping using the RainShift protocol:

1. Compute empirical CDFs for each variable from Alpine stations and Colorado SNOTEL
2. Map Alpine training targets to Colorado-equivalent distributions
3. Apply mapped distributions during the LoRA fine-tuning phase so the backbone adapts to Colorado-scale values

**Why quantile mapping over learned domain adaptation:** The distribution shift between Alps and Rockies is well-characterized (different absolute elevations, continental vs. maritime climate influence, different precipitation regimes). Quantile mapping handles this with zero additional parameters. Learned approaches (domain-adversarial training, MMD losses) add training complexity and can fail silently when the shift is simple and well-understood.

**Alternative considered:** CycleGAN-style domain translation on the weather fields. Overkill for this application — the shift is in marginal distributions, not in spatial structure.

### 7. Inference API design

FastAPI endpoint serving sub-km weather predictions:

**Primary endpoint:** `POST /predict`
```
Input:  { lat, lon, datetime, radius_km? }
Output: { predictions: [{ lat, lon, elevation, temperature, wind_speed, wind_direction, precipitation_rate }], metadata: { model_version, hrrr_cycle, resolution_m } }
```

**colorado-avalanche-ml integration contract:** `POST /terrain-weather`
```
Input:  { points: [{ lat, lon, elevation, aspect }], datetime, horizon_hours? }
Output: { forecasts: [{ lat, lon, hourly: [{ datetime, temperature, wind_speed, wind_direction, precipitation_rate, surface_pressure }] }] }
```

The integration endpoint adds temporal dimension (hourly forecasts over a horizon) and accepts pre-defined terrain points rather than gridded output, matching what the avalanche instability model needs.

**Model loading:** Lazy initialization on first request, safetensors format. StormCast backbone + LoRA weights + terrain head loaded as separate components, composed at inference time.

## Risks / Trade-offs

**[StormCast model size may exceed Apple Silicon memory]** StormCast is a diffusion model trained on full HRRR resolution — its memory footprint is unknown until weights are profiled. → Mitigation: Profile immediately in task 1. If too large for MPS, use float16/bfloat16 quantization or extract only the surface-variable encoder layers. Worst case: run backbone inference on cloud GPU, cache embeddings locally.

**[DEVINE weights may not load cleanly]** DEVINE was trained with a specific PyTorch version and custom layers. Weight compatibility is not guaranteed. → Mitigation: Load and validate weights in task 1 before building the architecture around them. If incompatible, re-implement the architecture from the paper and train from scratch on Alpine data (adds ~2 weeks).

**[118 SNOTEL stations may be insufficient for LoRA fine-tuning]** The Colorado fine-tuning dataset is sparse (118 stations, ~3 years of hourly data). Overfitting risk is real. → Mitigation: Aggressive regularization (LoRA rank r=8, dropout 0.1, early stopping on held-out stations). Leave-one-station-out cross-validation will quantify this risk before deployment. Fall back to r=4 if overfitting is observed.

**[Alpine→Colorado transfer may fail for precipitation]** Wind and temperature terrain effects transfer across mountain ranges (physics is similar). Precipitation is more climate-dependent — Alpine convective patterns differ from Colorado's monsoonal moisture. → Mitigation: Precipitation channel gets the weakest transfer prior (no frozen DEVINE weights). Quantile mapping addresses distribution shift. If precipitation skill is poor, fall back to raw HRRR precipitation and downscale only wind and temperature.

**[Mass-conservation constraint may conflict with station observations]** Station observations have measurement errors and represent point measurements, not grid-cell averages. Enforcing divergence-free wind on a grid trained against point observations may create tension. → Mitigation: Weight the divergence loss low initially (lambda_div=0.01), increase after reconstruction loss converges. At inference, the hard Helmholtz projection is optional — evaluate both constrained and unconstrained predictions against held-out stations.

**[Safety-critical tail events]** Avalanche-triggering weather involves extreme wind loading, rapid temperature changes, and intense precipitation — exactly the distribution tails where ML models perform worst. → Mitigation: Evaluation framework includes explicit tail-event metrics (99th percentile wind errors, cold-pool detection rate, heavy precipitation bias). If tail performance is unacceptable, apply tail-aware loss weighting (emphasize high-magnitude events during training).

## Module Interfaces

### terrain/feature-encoder ↔ terrain/downscaling-head

```python
# Terrain tensor: precomputed, static per geographic region
terrain_features: Tensor  # shape (B, C_terrain, H_fine, W_fine)
# C_terrain = 7: elevation, slope, aspect_sin, aspect_cos, curvature, svf, sx
# H_fine, W_fine = sub-km output grid (e.g., 256x256 at 250m)
```

The feature encoder normalizes DEM-derived features to zero-mean unit-variance using statistics from the training regions. Aspect is encoded as (sin, cos) to avoid the 0/360 discontinuity.

### backbone/stormcast-adapter ↔ terrain/downscaling-head

```python
# Weather tensor: per-timestep, from StormCast inference
weather_features: Tensor  # shape (B, C_weather, H_coarse, W_coarse)
# C_weather = 6: T2M, U10, V10, PRATE, SP, BLH
# H_coarse, W_coarse = 3km HRRR grid (subset covering region of interest)
```

The adapter extracts the 6 surface variables from StormCast's full output, applies LoRA-adapted inference, and bilinearly interpolates to a coarse grid aligned with the terrain grid's geographic extent.

### training/transfer-pipeline ↔ data pipelines

```python
@dataclass
class TerrainWeatherSample:
    terrain: Tensor       # (C_terrain, H, W) — static terrain features
    weather_in: Tensor    # (C_weather, H_coarse, W_coarse) — coarse input
    weather_target: Tensor  # (C_out, H_fine, W_fine) — fine-resolution target
    mask: Tensor          # (H_fine, W_fine) — valid observation locations
    metadata: dict        # station_id, datetime, phase, source_dataset
```

All data pipelines (alpine, cfd, reynolds-creek) produce this common format. The `mask` tensor handles sparse station observations vs. gridded targets (CFD/Reynolds Creek have full grids, station data has point observations on a grid).

### inference/micro-weather-api ↔ colorado-avalanche-ml

```python
@dataclass
class TerrainWeatherForecast:
    lat: float
    lon: float
    hourly: list[HourlyForecast]

@dataclass
class HourlyForecast:
    datetime: str          # ISO 8601
    temperature: float     # Celsius
    wind_speed: float      # m/s
    wind_direction: float  # degrees from north
    precipitation_rate: float  # mm/hr
    surface_pressure: float    # hPa
```

## Open Questions

- **Reynolds Creek data volume:** The full 21TB archive at hourly/10m resolution may be impractical to download and process. Need to determine the minimum subset (temporal range, spatial extent) required for meaningful validation. This can be decided after initial model training — it's a validation dataset, not a training dependency.
- **StormCast inference mode:** StormCast is a diffusion model that can run multi-step denoising for higher quality or single-step for speed. The quality/speed tradeoff for our use case (input conditioning, not standalone forecasting) needs empirical testing. Deferrable — single-step is the default, multi-step is an optimization.
- **Orographic precipitation loss weighting:** The L_oro term's contribution to training stability and final skill is uncertain. It may help or hurt depending on the Alpine→Colorado precipitation transfer quality. Can be tuned empirically during training without changing the architecture.
