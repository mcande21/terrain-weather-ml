## 1. FiLM Conditioning Module

- [ ] 1.1 Create `src/terrain_weather_ml/terrain/downscaling/film.py` with `FiLMGenerator` class: CNN that maps terrain tensor (B, 17, H, W) to per-decoder-block (gamma, beta) pairs via 3 conv layers + adaptive avg pool + per-block linear heads. Gamma initialized to 1.0, beta to 0.0. Verify: unit test confirms output shapes match expected `[(B, 8f, 1, 1), ...]` for base_features=64 and identity initialization values.

- [ ] 1.2 Create `FiLMBlock` wrapper in `film.py`: applies `gamma * GroupNorm(x) + beta` given pre-computed (gamma, beta). Verify: unit test confirms identity-initialized FiLMBlock produces same output as plain GroupNorm, and non-identity params change the output.

- [ ] 1.3 Add FiLM module tests in `tests/terrain/downscaling/test_film.py`: test FiLMGenerator output shapes for various base_features, test identity init, test parameter count is <10% of UNet, test FiLMBlock modulation correctness. Verify: `pytest tests/terrain/downscaling/test_film.py` passes.

## 2. U-Net Decoder Modification

- [ ] 2.1 Modify `DecoderBlock` in `unet.py` to accept optional `(gamma, beta)` FiLM parameters. When provided, apply FiLM modulation after the first GroupNorm in ConvBlock. When not provided, behave as before (backward compatible). Verify: existing U-Net tests still pass with no FiLM params; new test confirms FiLM params change decoder block output.

- [ ] 2.2 Update `UNet.forward()` and `UNet._decode()` in `unet.py` to accept an optional `film_params` list of (gamma, beta) tuples and pass each to the corresponding decoder block. Update `forward_ablated` similarly. Verify: existing `test_unet.py` tests pass unchanged; new test with film_params confirms conditioning flows through.

- [ ] 2.3 Update `ConvBlock` in `unet.py` to accept optional FiLM modulation — when provided, apply after the first GroupNorm (index 1 in the Sequential) and before the first ReLU. Verify: unit test confirms `gamma * GroupNorm(conv(x)) + beta` placement.

## 3. Downscaling Head Integration

- [ ] 3.1 Modify `TerrainDownscalingHead.__init__` in `head.py`: instantiate `FiLMGenerator`, change U-Net `in_channels` from `c_terrain + c_weather` to `c_weather` only. Remove terrain concatenation. Verify: `head.unet.init_conv` input channels equals `c_weather` (6).

- [ ] 3.2 Modify `TerrainDownscalingHead.forward()` in `head.py`: compute FiLM params from terrain tensor via FiLMGenerator, pass only upsampled weather to U-Net encoder, pass FiLM params to decoder. Verify: forward pass produces correct output shape (B, 4, H_fine, W_fine) with both terrain and weather inputs.

- [ ] 3.3 Update `tests/terrain/downscaling/test_head.py`: adjust all existing tests for new interface (weather-only U-Net input, FiLM conditioning). Test that terrain tensor shape mismatch raises error. Test output shape preserved. Verify: `pytest tests/terrain/downscaling/test_head.py` passes.

## 4. DEVINE Weight Loading Update

- [ ] 4.1 Update `load_devine_weights` in `devine.py` to handle the init_conv shape mismatch (23ch DEVINE → 6ch new head) by skipping init_conv weights when shapes don't match. Decoder weights load where shapes are compatible. FiLM generator parameters are excluded from DEVINE loading. Verify: unit test loads a mock DEVINE checkpoint into the new head without error, confirms init_conv was skipped and decoder weights were loaded.

- [ ] 4.2 Update `create_mock_devine_checkpoint` in `devine.py` to work with both old (23ch) and new (6ch) head configurations. Verify: `pytest tests/terrain/downscaling/test_devine.py` passes with updated mock.

## 5. Integration Verification

- [ ] 5.1 Run full test suite: `pytest tests/terrain/downscaling/` — all tests pass with the new FiLM architecture. Verify: zero failures.

- [ ] 5.2 Run linter: `ruff check src/terrain_weather_ml/terrain/downscaling/` — no lint errors in modified or new files. Verify: clean output.

- [ ] 5.3 Verify loss function compatibility: `DownscalingLoss` in `loss.py` works unchanged with the new head (it receives the same pred/target/terrain tensor shapes). Verify: existing loss tests pass without modification.

## 6. Training Pipeline Config Update

- [ ] 6.1 Update Phase 2 and Phase 3 training configs/scripts to use `c_weather=6` and include FiLM generator parameters. Verify: config loads without error and head instantiation with config values produces expected architecture.

- [ ] 6.2 Retrain Phase 2 with FiLM-conditioned head and run Phase 3 fine-tuning. Verify: training loss decreases and validation metrics are computed.

- [ ] 6.3 Run validation against baselines (bias-corrected HRRR, previous concat head). Verify: validation report generated with comparison metrics.
