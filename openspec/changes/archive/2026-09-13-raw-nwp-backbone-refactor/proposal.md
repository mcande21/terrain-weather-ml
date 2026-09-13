## Why

The backbone architecture assumed StormCast would provide 6 surface weather variables (T2M, U10, V10, PRATE, SP, BLH) at 3km resolution. Real-weight validation revealed StormCast outputs only 4 surface variables (T2M, U10, V10, MSLP) — precipitation rate, surface pressure, and boundary layer height are not in its 99-channel output. Additionally, StormCast is US-only (trained on HRRR), making it unusable during Phase 2 Alpine pre-training — the LoRA adapters get zero transfer learning from the Swiss station networks that justify the project's cross-domain approach. The terrain downscaling head and terrain encoder are where the physics innovation lives; the backbone should be a thin data adapter, not a 200M-parameter neural inference layer.

## What Changes

- **BREAKING**: Replace `StormCastAdapter` as the default backbone with a raw NWP passthrough adapter that extracts surface variables directly from HRRR (Colorado) or ERA5 (Alps/global) without neural inference
- Add ERA5 reanalysis input conditioning for Phase 2 Alpine pre-training, enabling the transfer learning path that StormCast's US-only coverage blocked
- Update Phase 2 training to use ERA5 as coarse weather input instead of StormCast, making Alpine→Colorado transfer functional
- Preserve StormCast adapter as an optional experimental backbone (not default) with corrected real-weight compatibility (.mdlus loading, 4-var extraction, updated LoRA targets)
- C_WEATHER remains 6 (T2M, U10, V10, PRATE, SP, BLH) — all natively available in both HRRR and ERA5

## Capabilities

### New Capabilities

- `backbone/nwp-passthrough`: Thin NWP variable extractor that reads raw HRRR or ERA5 surface fields. Drop-in replacement for StormCastAdapter with identical interface contract (C_WEATHER=6). No neural inference.
- `backbone/era5-conditioner`: ERA5 reanalysis input conditioning for Alpine pre-training. CDS API access, variable mapping (ECMWF short names to project schema), grid extraction for the Alpine and Colorado domains.

### Modified Capabilities

- `backbone/stormcast-adapter`: Demote from default to experimental. Fix real-weight compatibility (.mdlus format, correct surface indices [2,0,1] for 4 vars, LoRA targets to `affine` layers). Mark as optional A/B test path.
- `training/transfer-pipeline`: Phase 2 trainer uses ERA5 via the new conditioner instead of StormCast. Phase 3 uses HRRR via the passthrough adapter. Phase 1 unchanged (c_weather=0).
- `inference/micro-weather-api`: Default to NWP passthrough adapter. Single data source (HRRR) at inference — no ERA5 latency dependency.

## Impact

- **backbone/**: New `nwp_passthrough.py`, `era5_conditioner.py`. Modified `stormcast.py` (experimental flag), `interface.py` (adapter protocol), `__init__.py` (exports).
- **training/**: Modified `phases.py` (Phase2 uses ERA5, Phase3 uses HRRR passthrough), `datasets.py` (ERA5 dataset class).
- **inference/**: Modified `service.py` (default adapter selection).
- **Dependencies**: Add `cdsapi` for ERA5 access. No new ML framework dependencies.
- **Interface contract preserved**: C_WEATHER=6, downscaling head input=23 channels. No changes to terrain encoder, downscaling head architecture, or evaluation framework.
