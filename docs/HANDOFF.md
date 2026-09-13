# terrain-weather-ml Handoff

## What This Project Is

A cross-domain terrain-weather physics model for predicting weather at sub-km
resolution in complex mountain terrain. Pre-trained on dense Alpine networks
and synthetic CFD data, fine-tuned for Colorado.

## Current State

- Project scaffolded with OpenSpec lifecycle
- Proposal written, specs/design/tasks still need to be driven
- No implementation code yet — this is day 1

## To Continue

1. Run `/reload` to load context
2. Run `openspec status --change terrain-weather-foundation --json` to see where planning stands
3. Drive remaining OpenSpec artifacts: `specs -> design -> tasks`
4. Then implement via `opsx-apply`

## Research That Produced This Project

This project was born from the colorado-avalanche-ml Phase 2 research. A 7-agent
research workflow identified that:

- Cross-domain terrain-weather transfer is PROMISING (not yet done Alps to Rockies)
- NVIDIA StormCast (open weights, 3km HRRR-trained) is the ideal backbone
- DEVINE wind downscaling (open weights, 30m mountain terrain) is the terrain head template
- PeakWeather (302 Swiss stations, HuggingFace) is the best pre-training dataset
- Mass-conservation constraints improve out-of-distribution generalization

Full research synthesis is in the qm memory store under project `terrain-weather-ml`.

## Related Project

colorado-avalanche-ml (~/work/personal/code/colorado-avalanche-ml) is the downstream
consumer. Its Phase 2/3 (spatial downscaling to terrain-resolved instability predictions)
will use this model's output as input.

## Key HuggingFace Resources

- nvidia/stormcast-v1-era5-hrrr (backbone)
- MeteoSwiss/PeakWeather (pre-training data)
- souravsud/wind-cfd-trial (synthetic CFD data)
- ibm-granite/granite-geospatial-wxc-downscaling (baseline comparison)
- daisy1212/FuXi-CFD-model (wind field baseline)

## Action Plan (8 steps)

1. Evaluate existing pre-trained models on Colorado SNOTEL
2. Build DEM terrain feature encoder
3. Download PeakWeather + Swiss IMIS
4. Generate synthetic WindNinja data
5. Train DEVINE terrain head
6. Integrate StormCast backbone + LoRA
7. Validate (leave-one-station-out + physics checks)
8. Deploy as micro-weather API
