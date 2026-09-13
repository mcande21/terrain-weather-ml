## Context

See proposal.md for motivation. The current backbone (`StormCastAdapter`) was designed around assumptions invalidated by real-weight testing: StormCast outputs only 4 of 6 assumed surface variables, and its US-only training data breaks the Alpine transfer learning path. The existing interface contract (C_WEATHER=6, downscaling head input=23 channels) is correct in intent but wrong in implementation.

Existing code state:
- `backbone/stormcast.py`: 200+ lines, C_WEATHER=6 (wrong indices), LoRA injection (wrong targets)
- `backbone/hrrr.py`: HRRR conditioning — mostly correct, needs variable name flexibility
- `backbone/interface.py`: Contract validation — correct interface, reusable
- `training/phases.py`: 3-phase trainers — Phase 2 assumes StormCast, needs ERA5 path
- `inference/service.py`: Hardcoded to StormCastAdapter

## Goals / Non-Goals

**Goals:**
- Raw NWP passthrough as default backbone — HRRR for Colorado, ERA5 for Alps
- ERA5 conditioner enabling functional Alpine pre-training (Phase 2)
- C_WEATHER=6 interface preserved — no downstream changes to head, eval, or API schemas
- StormCast preserved as experimental option with corrected real-weight loading

**Non-Goals:**
- AIFS integration (rejected by adversarial review — 28km adds no terrain-scale signal)
- Hybrid multi-backbone fusion (deferred — measure single-backbone value first)
- Deriving additional variables from StormCast hybrid levels (future experiment)
- SNOTEL data pipeline changes (not affected by backbone refactor)

## Decisions

### D1: NWP passthrough as default backbone

**Choice:** Replace neural inference with raw NWP field extraction.

**Rationale:** The downscaling head learns terrain-weather corrections from the coarse-to-fine mapping. Whether the coarse input comes from a neural model or raw NWP doesn't change the head's learning signal — the station observations are the same either way. Raw NWP eliminates 200M parameters of inference overhead, provides all 6 surface variables natively, and works in both Alpine (ERA5) and Colorado (HRRR) domains.

**Alternatives considered:**
- StormCast as default (current): Broken — missing 3 variables, can't run on Alpine data
- Option C dual backbone (StormCast + AIFS): Over-engineered — 400M params for marginal signal at wrong resolution
- Option A (StormCast 4 vars only): Loses precipitation, critical for avalanche modeling

### D2: ERA5 for Phase 2, HRRR for Phase 3

**Choice:** Use the NWP source native to each training domain.

**Rationale:** ERA5 has global coverage including the Alps (31km). HRRR covers CONUS at 3km. The downscaling head doesn't need the coarse source to be identical between phases — the quantile mapping (RainShift) bridges the distribution gap. This makes the transfer learning path functional: Phase 2 trains on ERA5+terrain→stations, Phase 3 adapts to HRRR+terrain→stations.

**Alternatives considered:**
- ERA5 for both phases: Loses HRRR's 3km resolution advantage in Colorado
- HRRR for both phases: Impossible — no Alpine coverage
- Separate backbones per phase: Unnecessary complexity when raw NWP suffices

### D3: StormCast as experimental, not deleted

**Choice:** Preserve StormCast adapter with corrected real-weight support, behind an experimental flag.

**Rationale:** The adversarial review questioned whether neural backbone adds value over raw NWP. The correct response is to measure, not assume. Keeping StormCast as an experiment enables A/B comparison: run the same training with raw HRRR vs StormCast-processed HRRR and compare downstream metrics. If StormCast adds measurable value, promote it. If not, delete it with evidence.

**Corrections needed for real weights:**
- `.mdlus` format: repack zip to strip `model/` prefix
- Surface indices: [2, 0, 1, 3] (T2M, U10, V10, MSLP) — only 4 vars
- LoRA targets: `affine` layers (55 nn.Linear), not QKV (1 fused Conv2d)
- Supplement 2 missing vars (PRATE, BLH) from HRRR passthrough for C_WEATHER=6

### D4: Adapter protocol (duck typing)

**Choice:** Define an adapter protocol that both NWPPassthrough and StormCastAdapter implement.

**Rationale:** The downscaling head, training pipeline, and inference service should be agnostic to which adapter is active. A Python Protocol (structural subtyping) avoids inheritance coupling while enforcing the interface contract.

```python
class WeatherAdapter(Protocol):
    def extract(self, nwp_data: dict[str, torch.Tensor]) -> torch.Tensor:
        """Returns (batch, C_WEATHER, H, W) weather tensor."""
        ...
```

### D5: No LoRA in Phase 2

**Choice:** Phase 2 trains the downscaling head only — no LoRA adapters.

**Rationale:** LoRA adapters are StormCast-specific. When Phase 2 uses ERA5 passthrough, there's no neural backbone to adapt. The head learns terrain-weather mapping directly from ERA5 coarse fields + terrain features → station observations. If the StormCast experimental path is enabled, LoRA activates only in Phase 3 (Colorado).

## Risks / Trade-offs

**[Raw NWP may be worse than neural backbone for some variables]** → The experiment (D3) measures this. The terrain head's physics-informed architecture (divergence-free projection, terrain features) should compensate for any lost error correction from the backbone. Mitigation: A/B test before committing.

**[ERA5 resolution (31km) is coarser than HRRR (3km) for Phase 2]** → The terrain head upsamples coarse weather to terrain resolution. ERA5's large-scale fields (pressure, temperature, precipitation) are what Alpine pre-training needs — the terrain features provide the sub-grid physics. Swiss stations are dense enough (302 + 180) to supervise the head's learning despite coarse input.

**[ERA5 latency for real-time inference]** → ERA5 is used only for training (Phase 2). Inference uses HRRR (hourly updates). No operational latency issue.

**[Quantile mapping Alpine→Colorado harder with different NWP sources]** → The RainShift protocol maps per-variable CDFs. ERA5 and HRRR have different biases, but quantile mapping is designed for exactly this — mapping between different source distributions. The mapping calibration uses overlapping periods where both ERA5 and HRRR are available for the Colorado domain.

## Open Questions

- **ERA5 vs ERA5-Land:** ERA5-Land has 9km resolution (vs 31km ERA5) with better surface fields, but limited variable set. Worth evaluating as Phase 2 source if the 31km resolution proves too coarse for Alpine terrain.
- **StormCast regression vs diffusion:** The real checkpoint has both `StormCastUNet` (regression, 78M) and `EDMPrecond` (diffusion, 121M). The experimental adapter should use the regression model (deterministic, single-step) for feature extraction. Diffusion adds stochasticity inappropriate for conditioning.
