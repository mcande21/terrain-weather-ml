## Why

Validation showed the current terrain downscaling head (7.7K RMSE) does not beat bias-corrected HRRR (2.56K). The root cause: channel concatenation treats terrain as "just another input," diluting its signal across 23 input channels. FiLM (Feature-wise Linear Modulation) lets terrain actively shape how the network processes weather at every decoder layer via learned affine transforms. Three independent projects converge on this pattern for weather downscaling: Karpos (CNRM/Meteo-France), TA-ViT, and MambaDS.

## What Changes

- **New `terrain/film-conditioning` module** — FiLM generator that maps a 17-channel terrain tensor to per-decoder-block (gamma, beta) modulation parameters. Terrain features are computed once (static) and cached across timesteps.
- **BREAKING: Modified `terrain/downscaling-head`** — U-Net input drops from 23 channels (terrain+weather concat) to 6 channels (weather only). Terrain influence enters via FiLM modulation at each decoder stage instead of input concatenation. All existing checkpoints are incompatible.
- Training pipeline is architecturally unchanged but requires full retrain with the new head (Phase 2 + Phase 3).

## Capabilities

### New Capabilities

- `terrain/film-conditioning`: FiLM generator that produces per-decoder-block affine modulation parameters (gamma, beta) from static terrain features, enabling terrain to condition weather processing at multiple spatial scales.

### Modified Capabilities

- `terrain/downscaling-head`: Input architecture changes from dual-branch concatenation (23ch) to weather-only input (6ch) with FiLM terrain conditioning at each decoder stage. U-Net `in_channels` changes from `c_terrain + c_weather` to `c_weather` only.

## Impact

- **Code:** `src/terrain_weather_ml/terrain/downscaling/head.py` (forward pass, constructor), `unet.py` (decoder blocks accept FiLM params), new `film.py` module.
- **Weights:** All existing downscaling head checkpoints are invalidated. DEVINE weight loading (`devine.py`) needs update — encoder init changes due to reduced input channels; wind channel output init remains compatible.
- **Training:** Phase 2 and Phase 3 must be retrained from scratch.
- **Tests:** Existing head/U-Net tests need update for new interface. New tests for FiLM generator.
- **Config:** Phase training configs need `c_weather=6` and FiLM generator parameters.
