## Why

Mountain weather prediction at terrain scale (sub-km) is critical for avalanche forecasting, wildfire weather, hydroelectric planning, and renewable energy siting. Current weather models (HRRR at 3km, ERA5 at 31km) cannot resolve micro-weather in valleys, on ridgelines, and at specific terrain aspects that drives real-world outcomes. Dense mountain weather station networks exist (Swiss IMIS, MeteoSwiss) but only in a few regions — most mountain terrain worldwide has sparse or no observations. No one has attempted Alps-to-Rockies terrain-weather transfer learning, despite successful cross-domain precedents in adjacent fields (Swiss-to-Pyrenees avalanche transfer at 60-70% zero-shot, Sierra Nevada-to-Colorado SWE transfer at +20% R2).

## What Changes

- Add a terrain feature encoder that derives sub-km predictors from DEM data (elevation, slope, aspect, curvature, sky view factor, wind exposure Sx, surface roughness)
- Integrate NVIDIA StormCast (open-weight, 3km HRRR-trained) as the weather backbone with LoRA regional adapters
- Build a DEVINE-inspired U-Net terrain downscaling head that maps 3km weather fields to sub-km resolution, conditioned on terrain features
- Implement mass-conservation (divergence-free projection) as a hard physics constraint on wind output
- Create a multi-phase pre-training pipeline: synthetic CFD data → dense Alpine station networks → Reynolds Creek gridded product
- Fine-tune on Colorado SNOTEL stations via LoRA + quantile-mapping domain alignment
- Build an evaluation framework: leave-one-station-out cross-validation, physics consistency checks, baseline comparisons
- Deploy as a micro-weather inference API that feeds colorado-avalanche-ml Phase 2/3

## Capabilities

### New Capabilities

- `terrain/feature-encoder`: DEM-derived terrain feature extraction (elevation, slope, aspect, curvature, SVF, Sx, roughness) at arbitrary resolution from raster inputs
- `terrain/downscaling-head`: DEVINE-style U-Net that downscales 3km weather grids to sub-km, conditioned on terrain features, with mass-conservation constraint
- `backbone/stormcast-adapter`: StormCast integration with LoRA adapters for regional fine-tuning, including checkpoint loading and inference
- `data/alpine-pipeline`: Data ingestion for PeakWeather (HuggingFace), Swiss IMIS (REST API), and SAFRAN reanalysis — normalization, QC, and station metadata alignment
- `data/cfd-pipeline`: Synthetic CFD training data pipeline from wind-cfd-trial (HuggingFace) and WindNinja solver runs
- `data/reynolds-creek`: Reynolds Creek gridded 10m product ingestion and preprocessing (21TB hourly, 40-year record)
- `training/transfer-pipeline`: Multi-phase pre-training orchestration (CFD → Alpine → Reynolds Creek) with phase-gated checkpointing and domain alignment via quantile mapping
- `evaluation/cross-validation`: Leave-one-station-out evaluation framework with physics consistency checks (mass conservation, cold air pools, orographic precipitation ratios) and baseline comparison (raw HRRR, WindNinja, granite-wxc-downscaling)
- `inference/micro-weather-api`: REST API serving sub-km weather predictions at arbitrary terrain coordinates, designed as input layer for colorado-avalanche-ml

### Modified Capabilities

(none — greenfield project, no existing specs)

## Impact

- **Code:** New Python package under `src/` with modules for terrain, backbone, data, training, evaluation, and inference
- **Data dependencies:** PeakWeather (HuggingFace, ~50GB), Swiss IMIS (REST API, ongoing), wind-cfd-trial (HuggingFace, ~10GB), Reynolds Creek (21TB, staged download), SNOTEL Colorado (existing in colorado-avalanche-ml)
- **Model dependencies:** StormCast weights (nvidia/stormcast-v1-era5-hrrr), DEVINE weights (louisletoumelin/wind_downscaling_cnn), granite-wxc-downscaling (ibm-granite, baseline only)
- **Compute:** GPU training required (LoRA keeps it manageable vs. full fine-tuning), StormCast model size needs profiling
- **Downstream:** colorado-avalanche-ml Phase 2/3 consumes the micro-weather API output, replacing raw HRRR inputs with terrain-resolved predictions
- **Safety:** Avalanche and wildfire applications are safety-critical — evaluation must cover dangerous tail events, not just average skill metrics
