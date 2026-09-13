# terrain-weather-ml

Cross-domain terrain-weather physics model. Pre-trained on dense Alpine station networks and synthetic CFD data, fine-tuned for mountain terrain micro-weather prediction.

## Build & Run

```bash
pip install -e ".[dev]"
pytest tests/
ruff check src/
```

## Architecture

StormCast backbone (NVIDIA, 3km HRRR) + DEVINE-style terrain downscaling head + LoRA adaptation.

- **Backbone:** Frozen StormCast with LoRA adapters for regional specialization
- **Terrain Head:** U-Net downscaling from 3km to sub-km using DEM terrain features
- **Physics Constraint:** Mass-conservation (divergence-free) on wind output
- **Pre-training:** Synthetic CFD data + PeakWeather (302 Swiss stations) + Swiss IMIS (180 stations)
- **Fine-tuning:** Colorado SNOTEL (118 stations) via few-shot domain adaptation

## Key Data Sources

| Source | Purpose | Access |
|--------|---------|--------|
| PeakWeather (MeteoSwiss) | Pre-training: 302 Swiss stations, 10-min | HuggingFace |
| Swiss IMIS (SLF) | Pre-training: 180 high-alpine stations | Open API + EnviDat |
| HRRR (NOAA) | Coarse input conditioning | AWS S3 |
| Reynolds Creek | Validation: gridded 10m product | ESSD open |
| wind-cfd-trial | Synthetic CFD pre-training | HuggingFace |
| SNOTEL (Colorado) | Fine-tuning target | AWDB REST API |

## Pre-Trained Models

| Model | Source | License |
|-------|--------|---------|
| StormCast | nvidia/stormcast-v1-era5-hrrr | NVIDIA Open Model |
| DEVINE | louisletoumelin/wind_downscaling_cnn | Open (Zenodo) |
| FuXi-CFD | daisy1212/FuXi-CFD-model | HuggingFace |
| granite-wxc-downscaling | ibm-granite/granite-geospatial-wxc-downscaling | Apache 2.0 |

## Downstream Applications

- Avalanche terrain micro-weather (colorado-avalanche-ml Phase 2/3)
- Wildfire weather prediction
- Hydroelectric inflow forecasting
- Agricultural frost prediction
- Renewable energy siting

## Conventions

- Python 3.11+, ruff for linting (line-length: 100)
- No trailing whitespace
- PyTorch with MPS auto-detection for Apple Silicon
