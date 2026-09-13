## Tier 1 — Foundation (no model dependencies, parallelizable)

## 1. Terrain Feature Encoder

Spec: `terrain/feature-encoder`. No upstream dependencies. Produces the static terrain tensor consumed by the downscaling head and all data pipelines (station terrain linkage).

- [ ] 1.1 DEM ingestion module — accept GeoTIFF DEM rasters (SRTM 30m, Copernicus 30m, USGS 1/3 arc-second), reproject to target CRS, resample to target resolution via rasterio. Verify: test loads a real SRTM tile, reprojects from EPSG:4326 to UTM, and outputs an elevation grid at the requested resolution with CRS metadata preserved.

- [ ] 1.2 Core terrain feature computation — elevation passthrough, slope via Horn's method (degrees), aspect as (sin, cos) pair to avoid 0/360 discontinuity (flat pixels → aspect = 0,0), plan curvature and profile curvature (1/m). Verify: test against a synthetic DEM patch with known geometry (plane, cone, saddle) producing expected slope/aspect/curvature values within tolerance.

- [ ] 1.3 Sky-view factor (SVF) computation — 36-azimuth-sector ray-tracing, configurable max search radius (default 5km). Verify: test flat surface returns SVF ≈ 1.0, synthetic hemispherical valley returns SVF < 0.5, runtime scales linearly with search radius.

- [ ] 1.4 Sx wind exposure computation — 8 cardinal directions (N, NE, E, SE, S, SW, W, NW), positive = sheltered, negative = exposed, configurable search radius (default 3km). Verify: test ridgeline pixel has negative Sx (exposed), valley-bottom pixel has positive Sx (sheltered), values are in degrees.

- [ ] 1.5 TPI and surface roughness — TPI at 500m radius (elevation minus mean within radius, meters), NLCD land-cover to aerodynamic roughness z0 lookup table, configurable default roughness when NLCD unavailable (log warning on fallback). Verify: test TPI on synthetic peak/valley DEM, test NLCD mapping produces valid z0 values, test missing-NLCD fallback triggers warning and uses default.

- [ ] 1.6 Tensor assembly and caching — assemble all features into a single PyTorch tensor of shape (C, H, W) where C=17 (elevation, slope, aspect_sin, aspect_cos, plan_curv, profile_curv, SVF, 8×Sx, TPI, roughness), support CPU/CUDA/MPS devices, cache computed tensors to disk keyed by (DEM path, target resolution, CRS), return cached tensor on repeat requests without recomputation. Verify: test full pipeline on a small DEM produces shape (17, H, W), second call returns identical tensor from cache without recomputation (assert no DEM read on cache hit).

- [ ] 1.7 Normalization — z-score normalization using per-channel mean and std computed from training regions, save/load normalization parameters alongside model checkpoints. Verify: round-trip test — compute stats, save, load from checkpoint, apply to same data, verify output matches original normalized values exactly.

## 2. Alpine Data Pipeline

Spec: `data/alpine-pipeline`. No upstream dependencies (terrain linkage deferred to integration). Produces the Phase 2 training dataset.

- [ ] 2.1 PeakWeather HuggingFace download — download and parse MeteoSwiss/PeakWeather dataset for 302 Swiss stations at 10-min resolution, extract T2m, wind speed, wind direction, precipitation, surface pressure, relative humidity, global radiation. Support incremental updates (download only missing date ranges). Verify: test with mocked HuggingFace responses returns expected station count (302), variable set, and date range; test incremental mode skips already-downloaded ranges.

- [ ] 2.2 Swiss IMIS ingestion — SLF open API client for 180 high-alpine stations at 30-min resolution (wind speed, wind direction, air temperature, snow surface temperature, snow depth, relative humidity), EnviDat bulk file fallback for historical data outside API temporal window. Verify: test API client with mocked responses returns expected station count (180) and variables; test EnviDat fallback triggers when API returns 404 for historical range.

- [ ] 2.3 Station metadata schema — unified metadata dataclass across both networks: station_id (string), network (`peakweather` | `imis`), latitude/longitude (WGS84), elevation (m), terrain feature reference (coordinates for downstream terrain encoder linkage). Verify: test both PeakWeather and IMIS produce records with all required metadata fields, no field is None.

- [ ] 2.4 Temporal alignment — aggregate PeakWeather 10-min to configurable target resolution (default 30-min): mean for temperature/pressure/humidity, vector-mean for wind, sum for precipitation. Mark intervals with fewer than 2-of-3 expected 10-min records as NaN. Verify: test aggregation rules on synthetic 10-min data, test incomplete-interval NaN marking, test aligned timestamps match IMIS 30-min boundaries exactly.

- [ ] 2.5 Quality control and unit normalization — normalize to common units: temperature → Kelvin, wind speed → m/s, wind direction → degrees (meteorological, 0=N), precipitation → mm/interval, pressure → hPa, humidity → fraction (0-1). Reject physically impossible values (negative wind speed, T < 180K or > 340K, wind direction outside 0-360). Verify: test unit conversions on known values, test QC rejection on each impossible-value category, test cross-network variable consistency in a mixed batch.

- [ ] 2.6 Storage format with random access — store aligned dataset as per-station time series supporting O(log N) random access by station ID and time range. Include timestamp, all weather variables, station metadata reference, per-variable quality flag. Verify: test random access retrieval for a specific station and time range returns correct slice; benchmark confirms sub-linear access time (not full-scan).

## 3. CFD Data Pipeline

Spec: `data/cfd-pipeline`. No upstream dependencies (uses terrain encoder internally for feature computation). Produces the Phase 1 pre-training dataset.

- [ ] 3.1 wind-cfd-trial HuggingFace download and indexing — download and index ~10,000 RANS simulation cases, each containing a DEM patch with wind boundary conditions and the solved 3D wind field. Verify: test with mocked HuggingFace responses returns indexed case metadata with expected count; test index supports lookup by case ID.

- [ ] 3.2 Input/output tensor formatting — structure each case as input tuple (DEM patch tensor `(1, H, W)`, terrain features `(C_terrain, H, W)`, boundary conditions `(3,)` for speed/direction/height) and target wind field `(3, H, W)` for u, v, w components. Verify: test tensor shapes match spec for a loaded case, spatial dimensions are consistent between input and output.

- [ ] 3.3 WindNinja synthetic generation — optional integration: accept DEM patch + wind boundary conditions, run WindNinja solver, return result in standard tensor format. Skip gracefully when WindNinja binary is not installed (log warning, operate on wind-cfd-trial only). Verify: test WindNinja-not-found path logs warning and pipeline continues; if WindNinja is available, test a synthetic case produces valid tensor shapes.

- [ ] 3.4 Domain randomization — random wind direction rotation, wind speed scaling (0.5-30 m/s), DEM patch random cropping (minimum 64×64), Gaussian elevation noise (configurable sigma, default 5m). Target wind field must be transformed consistently with augmentations. Verify: test each augmentation independently on a known case, verify target wind field rotates with direction augmentation, test crop produces valid minimum size, test augmentations are independently randomized per sample in a batch.

- [ ] 3.5 Mass conservation labeling — compute divergence field from target wind field, flag samples exceeding configurable max absolute divergence threshold (default 0.01 s⁻¹) for exclusion. Verify: test divergence computation on a known divergence-free field returns ~0, test a constructed non-conservative field is flagged, test threshold-based exclusion filters correctly.

## Tier 2 — Model (depends on Tier 1 data pipelines)

## 4. StormCast Backbone Adapter

Spec: `backbone/stormcast-adapter`. Depends on: Tier 1 complete (needs terrain features for interface validation). Produces the dynamic weather input for the downscaling head.

- [ ] 4.1 StormCast checkpoint loading — load nvidia/stormcast-v1-era5-hrrr from HuggingFace or local path, set all parameters to requires_grad=False, report parameter count and memory footprint. Handle checkpoint-not-found with clear error message. Verify: test successful load reports non-zero parameter count, all parameters have requires_grad=False; test missing checkpoint raises descriptive error.

- [ ] 4.2 Surface variable extraction — extract 6 surface variables (T2m, U10, V10, precipitation rate, surface pressure, BLH) from StormCast's 99-variable output by index. Validate variable indices against model metadata at load time. Verify: test extraction from a mock 99-channel tensor produces shape (6, H, W); test variable index validation catches metadata mismatch.

- [ ] 4.3 LoRA adapter injection — inject LoRA adapters into StormCast attention layers (query, key, value projections) with configurable rank (default r=8), alpha (default 16), dropout (default 0.0). Only LoRA parameters trainable. Verify: test trainable parameter count is < 1% of total; test save/load round-trip of LoRA-only checkpoint reproduces identical adapted model output.

- [ ] 4.4 HRRR input conditioning — parse HRRR GRIB2 files, extract required input variables, reproject to StormCast grid, normalize to expected input schema. Verify: test GRIB2 processing on a real HRRR file (or fixture) produces a valid input tensor matching StormCast's expected shape and value ranges.

- [ ] 4.5 Interface contract validation — validate at initialization that adapter output shape (6, H_coarse, W_coarse) matches downscaling head expected dynamic input channels. Raise error on mismatch. Verify: test matching shapes pass validation; test mismatched channel count raises descriptive error.

## 5. Terrain Downscaling Head

Spec: `terrain/downscaling-head`. Depends on: terrain/feature-encoder (tensor format), backbone/stormcast-adapter (weather tensor format). Core model architecture.

- [ ] 5.1 U-Net encoder-decoder architecture — encoder with 4+ downsampling stages, decoder with matching upsampling stages, skip connections concatenating encoder features at each resolution level. Verify: test forward pass with input (C, H, W) produces output with same (H, W) spatial dimensions; test skip connections are active (ablation: removing one changes output).

- [ ] 5.2 Dual-branch input processing — upsample coarse weather tensor (C_weather, H_coarse, W_coarse) to match fine terrain tensor (C_terrain, H_fine, W_fine) via bilinear interpolation, concatenate along channel dimension before U-Net encoder. Verify: test with 3km weather (6 channels) and 100m terrain (17 channels) produces concatenated input of 23 channels at fine resolution; test terrain resolution mismatch with output raises error.

- [ ] 5.3 Multi-variable output head — 4 output channels: u-wind, v-wind, temperature, precipitation. Provide u/v to speed/direction conversion utility at the output interface. Verify: test output shape is (4, H_fine, W_fine); test speed/direction conversion on known (u, v) vectors produces correct values.

- [ ] 5.4 Divergence-free projection layer — compute 2D divergence of (u, v) wind field via finite differences, solve Poisson equation for pressure correction, subtract gradient to produce divergence-free field. Apply only to wind channels; temperature and precipitation pass through unchanged. Verify: test output wind field has max absolute divergence < 1e-5 s⁻¹; test temperature and precipitation channels are bitwise identical before and after projection.

- [ ] 5.5 DEVINE weight initialization — load wind-related encoder/decoder weights from louisletoumelin/wind_downscaling_cnn checkpoint. Temperature and precipitation channels randomly initialized. Support freezing DEVINE-loaded wind parameters (requires_grad=False) during warmup. Verify: test DEVINE weights load without error; test freeze flag disables gradients on wind channels while temp/precip remain trainable; test with no DEVINE path uses random init for all channels.

- [ ] 5.6 Loss function — per-variable weighted MSE (configurable weights, default equal) + mean squared divergence penalty on pre-projection wind field (configurable lambda_div, default 0.1) + optional orographic precipitation penalty (configurable lambda_oro, default 0.0). Verify: test loss computation includes MSE, divergence, and orographic terms with correct weighting; test lambda_div=0 and lambda_oro=0 produces MSE-only loss; test lambda_oro>0 adds orographic correlation penalty; test gradient flows through all active terms.

## 6. Transfer Training Pipeline

Spec: `training/transfer-pipeline`. Depends on: all Tier 1 data pipelines (cfd-pipeline for Phase 1, alpine-pipeline for Phase 2), backbone/stormcast-adapter, terrain/downscaling-head. Orchestrates the 3-phase training.

- [ ] 6.1 Phase 1 CFD pre-training — training loop consuming cfd-pipeline data, MSE + mass-conservation loss (lambda=0.1), full downscaling head trainable, no StormCast involvement. Save Phase 1 checkpoint with metadata (phase=1, epochs, validation loss). Verify: test one epoch completes without error on synthetic CFD data; test checkpoint contains only downscaling head parameters (no StormCast weights); test checkpoint metadata is complete.

- [ ] 6.2 Phase 2 Alpine fine-tuning — load Phase 1 downscaling head checkpoint, inject LoRA into StormCast, initialize wind channels from DEVINE, support configurable warmup period with wind channels frozen. Train on alpine-pipeline data paired with nearest HRRR grid cell. Verify: test Phase 2 initialization from Phase 1 checkpoint succeeds; test DEVINE weights are loaded into wind channels; test warmup freezing disables wind channel gradients; test station-HRRR pairing produces valid training samples.

- [ ] 6.3 Phase 3 Colorado LoRA adaptation — load Phase 2 checkpoint, freeze downscaling head, train LoRA adapters only on SNOTEL data with quantile-mapped coarse input. Save checkpoint containing Phase 2 head weights, Phase 3 LoRA weights, and quantile mapping parameters. Verify: test Phase 3 initialization from Phase 2 checkpoint succeeds; test downscaling head is frozen (no gradients); test only LoRA parameters update during training step; test checkpoint includes quantile mapping parameters.

- [ ] 6.4 Quantile mapping module — compute empirical CDFs per variable per season from Alpine and Colorado station data, map Alpine distributions to Colorado-equivalent via CDF matching (RainShift protocol). Apply mapping to coarse input during Phase 3 training. Verify: test mapping transforms a sample from Alpine distribution to match Colorado distribution moments; test per-season stratification produces different mappings for winter vs. summer; test round-trip map/inverse-map recovers original values within tolerance.

- [ ] 6.5 Phase-gated checkpointing — enforce phase ordering (Phase 2 requires Phase 1 checkpoint, Phase 3 requires Phase 2 checkpoint). Each checkpoint includes: phase number, total epochs, best validation loss, training data hash, parent checkpoint hash. Verify: test launching Phase 3 without Phase 2 checkpoint raises descriptive error; test launching Phase 2 without Phase 1 checkpoint raises descriptive error; test checkpoint metadata fields are all populated and parent hash matches the input checkpoint.

## Tier 3 — Evaluation + Deployment (depends on Tier 2 model)

## 7. Cross-Validation Framework

Spec: `evaluation/cross-validation`. Depends on: trained model from transfer-pipeline (Phase 3 checkpoint). Parallelizable with tasks 8 and 9 at the implementation level (evaluation code can be written before the model is trained).

- [ ] 7.1 LOSO framework — leave-one-station-out cross-validation over 118 Colorado SNOTEL stations, per-station and aggregate statistics (mean, median, std across stations). Support single-station evaluation mode. Verify: test with mock model and 5 synthetic stations produces 5 folds with per-station metrics and correct aggregate statistics; test single-station mode evaluates only the specified fold.

- [ ] 7.2 Evaluation metrics — per-variable computation: RMSE, MAE, bias, R² for wind speed/temperature/precipitation; circular RMSE and circular bias for wind direction; precipitation categorical metrics (frequency bias, POD, FAR, CSI) at configurable thresholds (default 0.1, 1.0, 5.0 mm/hr). Handle missing observations by excluding those timesteps. Verify: test each metric on synthetic data with known expected values; test missing-observation exclusion produces correct results.

- [ ] 7.3 Physics consistency checks — (a) wind mass conservation: mean and max absolute divergence, pass/fail at 0.01 s⁻¹ threshold; (b) cold air pool reproduction: valley-ridgeline temperature bias during nocturnal inversions (TPI < -50m), pass/fail at 2K threshold; (c) orographic precipitation: elevation-precipitation gradient vs. PRISM 30-year normals, pass/fail at 20% agreement. Verify: test each check with synthetic data designed to pass and fail, verify correct pass/fail determination.

- [ ] 7.4 Baseline model runners — implement evaluation wrappers for: (a) raw HRRR nearest-grid-cell (no downscaling), (b) WindNinja solver with HRRR boundary conditions, (c) granite-wxc-downscaling. All baselines produce metrics in the same format as the terrain-weather model. Verify: test each baseline runner produces a metrics dict with all required keys matching the terrain-weather model output format.

- [ ] 7.5 Comparison report and tail-event evaluation — comparison table across all models with per-variable aggregate metrics, paired t-test for statistical significance (p < 0.05). Separate tail-event metrics: wind > 90th percentile, temperature < 10th percentile, precipitation > 95th percentile. Verify: test report generation with mock metrics from 3 models produces formatted table with significance flags; test tail-event filtering selects correct timestep subsets.

## 8. Reynolds Creek Data

Spec: `data/reynolds-creek`. Depends on: no model dependency (validation data), but logically Tier 3 because it validates the trained model. Parallelizable with tasks 7 and 9.

- [ ] 8.1 Variable and temporal subsetting — download only 6 variables (air temperature 2m, wind speed 10m, wind direction 10m, precipitation, relative humidity, incoming shortwave radiation) from the Reynolds Creek gridded product. Support configurable temporal window (default: 5 most recent complete water years, Oct 1 – Sep 30). Full spatial domain (239 km², no spatial subset). Verify: test variable filter rejects files containing only excluded variables; test temporal bounds produce correct date range; test spatial extent covers full watershed boundary.

- [ ] 8.2 Staged download with checkpointing — resumable download with checkpoint files, file hash verification on restart. Interruption and restart does not re-download completed files. Verify: test simulated interruption and restart resumes from last completed file; test hash mismatch triggers re-download of corrupted file.

- [ ] 8.3 NetCDF format conversion — convert to per-variable daily NetCDF files at 10m grid resolution, organized as `{variable}/{year}/{variable}_{YYYYMMDD}.nc` with hourly time steps. Verify: test converted files have correct directory structure, NetCDF dimensions (time=24, lat, lon matching 10m grid), and variable metadata.

## 9. Micro-Weather Inference API

Spec: `inference/micro-weather-api`. Depends on: trained model (Phase 3 checkpoint), terrain/feature-encoder, backbone/stormcast-adapter. Parallelizable with tasks 7 and 8 at the code level.

- [ ] 9.1 Prediction endpoint — `POST /predict` accepting (latitude, longitude, datetime). Returns wind_speed, wind_direction, temperature, precipitation_rate, elevation, aspect, model_version, source_hrrr_cycle. Return HTTP 422 for coordinates outside model domain (with valid bounds in message) and for datetime > 18 hours in future. Verify: test valid request returns all response fields; test out-of-domain coordinates return 422 with bounds; test future datetime returns 422 with horizon limit.

- [ ] 9.2 Batch prediction endpoint — `POST /predict/batch` accepting up to 1000 (lat, lon, datetime) tuples. Responses in same order as input. Optimize shared StormCast forward pass for same-time queries. Return HTTP 422 for batch size > 1000. Verify: test 100-point batch returns 100 ordered predictions; test batch size 1001 returns 422; test same-datetime points share a single forward pass (mock StormCast call count = 1).

- [ ] 9.3 Model loading and health — lazy initialization on first request using safetensors format. Load StormCast backbone + LoRA weights + terrain head as separate components, compose at inference time. `GET /health` returns 200 when ready, 503 when loading/failed. `GET /metadata` returns model version, checkpoint hash, domain bounds, forecast horizon, output variable list with units. Verify: test first request triggers model load; test /health returns 503 before load, 200 after; test /metadata returns all required fields.

- [ ] 9.4 HRRR input refresh — automatic fetch of latest HRRR analysis cycle from AWS S3 at configurable interval (default hourly). Log active HRRR cycle, include in prediction responses. On fetch failure, continue serving from most recent successful cycle with warning log. Verify: test refresh picks up new HRRR cycle; test fetch failure logs warning and predictions continue with stale cycle; test HRRR cycle identifier appears in prediction response.

- [ ] 9.5 colorado-avalanche-ml integration contract — `POST /terrain-weather` accepting points (lat, lon, elevation, aspect) and datetime with optional horizon_hours. Returns hourly forecasts per point containing temperature, wind_speed, wind_direction, precipitation_rate, surface_pressure. Response includes elevation and aspect so downstream gets a complete feature vector. Verify: test request with 5 points and 24-hour horizon returns 5 forecast arrays each with 24 hourly entries containing all required fields.
