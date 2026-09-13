## 1. NWP Passthrough Adapter

- [ ] 1.1 Create `backbone/nwp_passthrough.py` with `NWPPassthroughAdapter` class implementing the `WeatherAdapter` protocol. Accepts raw HRRR or ERA5 data dict, extracts 6 surface variables by name mapping, returns `(batch, 6, H, W)` tensor. Include HRRR and ERA5 variable name maps with unit normalization (ERA5 tp accumulated → rate conversion). **Verify:** Tests with mock HRRR and ERA5 data dicts produce correct 6-channel tensors.
- [ ] 1.2 Create `backbone/adapter_protocol.py` with `WeatherAdapter` Protocol defining the `extract()` method signature. Both `NWPPassthroughAdapter` and `StormCastAdapter` SHALL satisfy this protocol. Update `backbone/__init__.py` exports. **Verify:** `isinstance` check passes for both adapters against the protocol.
- [ ] 1.3 Update `backbone/interface.py` to validate against `WeatherAdapter` protocol instead of hardcoded StormCast assumptions. `validate_interface_contract()` accepts any adapter satisfying the protocol. **Verify:** Contract validation passes with NWP passthrough adapter, C_WEATHER=6, C_TERRAIN=17.
- [ ] 1.4 Wire NWP passthrough as default in `backbone/__init__.py`. Import order: `NWPPassthroughAdapter` as the primary export, `StormCastAdapter` available but not default. **Verify:** `from terrain_weather_ml.backbone import NWPPassthroughAdapter` works.

## 2. ERA5 Conditioner

- [ ] 2.1 Create `backbone/era5_conditioner.py` with `ERA5Conditioner` class. Handles CDS API data download for configurable domains (Alpine: lat 45-48, lon 5-11; Colorado: lat 37-41, lon -109 to -105). Supports both legacy and beta CDS API. **Verify:** Tests with mocked CDS responses produce correctly shaped output.
- [ ] 2.2 Implement variable extraction and mapping: ERA5 short names (2t, 10u, 10v, tp, sp, blh) → project schema (T2M, U10, V10, PRATE, SP, BLH). Handle ERA5 accumulated precipitation → instantaneous rate conversion by differencing consecutive timesteps. **Verify:** Unit tests confirm mapping and precipitation conversion match HRRR-equivalent output.
- [ ] 2.3 Implement temporal alignment: match ERA5 hourly timestamps to PeakWeather (10-min) and IMIS (30-min) observations. Select nearest ERA5 timestep within configurable tolerance (default 30 min). **Verify:** Tests confirm correct pairing and out-of-tolerance rejection.
- [ ] 2.4 Implement resumable download with checkpoint tracking (same pattern as Reynolds Creek downloader). Skip already-downloaded timesteps on re-invocation. **Verify:** Interrupted download resumes correctly.

## 3. Training Pipeline Updates

- [ ] 3.1 Update `training/phases.py` Phase2Trainer to accept ERA5 data via the NWP passthrough adapter instead of StormCast. Remove LoRA injection from Phase 2 (no neural backbone to adapt). Keep DEVINE weight initialization and warmup freeze. **Verify:** Phase 2 trainer initializes and runs a forward pass with ERA5 input tensor.
- [ ] 3.2 Update `training/phases.py` Phase3Trainer to use HRRR via NWP passthrough adapter. Replace StormCast adapter reference with passthrough adapter. Keep quantile mapping and LoRA-only training as options (LoRA only when experimental StormCast backbone is active). **Verify:** Phase 3 trainer runs with HRRR passthrough, quantile mapping applied.
- [ ] 3.3 Create `training/datasets.py` ERA5 dataset class (`ERA5StationDataset`) that pairs ERA5 grid cells with Alpine station observations. Analogous to the existing HRRR-station pairing but for ERA5 format. **Verify:** Dataset produces (weather_tensor, terrain_tensor, target_observation) tuples with correct shapes.
- [ ] 3.4 Update Phase 2 → Phase 3 weight transfer to handle the backbone switch (ERA5 passthrough → HRRR passthrough). The head weights transfer directly; no backbone weights carry over. **Verify:** Phase 3 loads Phase 2 head checkpoint without errors.

## 4. StormCast Experimental Adapter

- [ ] 4.1 Refactor `backbone/stormcast.py` to implement `WeatherAdapter` protocol. Mark as experimental with clear documentation. Fix `.mdlus` loading (repack zip, strip `model/` prefix). **Verify:** Real StormCast weights load from `data/models/stormcast/StormCastUNet.0.0.mdlus`.
- [ ] 4.2 Correct surface variable indices to [2, 0, 1, 3] (T2M, U10, V10, MSLP — 4 channels). Add HRRR passthrough for remaining 2 channels (PRATE, BLH) to reach C_WEATHER=6. **Verify:** Tests confirm 4 extracted + 2 passthrough = 6 channels.
- [ ] 4.3 Update LoRA targets from `["attn_q", "attn_k", "attn_v"]` to `["affine"]` (55 nn.Linear layers in real StormCast). **Verify:** LoRA injection targets affine layers, skips fused QKV Conv2d.
- [ ] 4.4 Add configuration flag to select StormCast experimental backbone: `backbone: stormcast` in config. Default remains NWP passthrough. **Verify:** Config switch toggles between adapters.

## 5. Inference Service Updates

- [ ] 5.1 Update `inference/service.py` to default to `NWPPassthroughAdapter`. Accept backbone selection via `InferenceConfig`. **Verify:** Service starts with default config and serves predictions from raw HRRR.
- [ ] 5.2 Add StormCast experimental option in `InferenceConfig(backbone="stormcast")`. Service loads StormCast adapter when configured. **Verify:** Config switch activates StormCast adapter with LoRA weights.
- [ ] 5.3 Verify single data source at inference: default mode reads only HRRR, no ERA5 dependency. HRRR refresh cycle continues as-is. **Verify:** Health endpoint reports HRRR as sole data source.

## 6. Test Updates

- [ ] 6.1 Update `tests/backbone/test_surface_extraction.py` — rewrite for NWP passthrough extraction (6 vars from HRRR/ERA5 dicts) and experimental StormCast (4+2 passthrough).
- [ ] 6.2 Update `tests/backbone/test_interface.py` — validate protocol-based contract with both adapters.
- [ ] 6.3 Update `tests/backbone/test_lora.py` — correct LoRA target assertions for `affine` layers.
- [ ] 6.4 Update `tests/training/test_phase2.py` — Phase 2 uses ERA5 passthrough, no LoRA injection.
- [ ] 6.5 Update `tests/training/test_phase3.py` — Phase 3 uses HRRR passthrough, optional LoRA.
- [ ] 6.6 Create `tests/backbone/test_nwp_passthrough.py` — NWP passthrough adapter tests (HRRR + ERA5 sources).
- [ ] 6.7 Create `tests/backbone/test_era5_conditioner.py` — ERA5 conditioner tests with mocked CDS API.
- [ ] 6.8 Run full test suite — 404+ tests passing, zero regressions.
