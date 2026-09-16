## Context

The current `TerrainDownscalingHead` concatenates 17 terrain channels with 6 weather channels into a 23-channel input to the U-Net. This treats terrain as an undifferentiated extra signal at the input layer only. See proposal.md for why this underperforms.

The U-Net uses `ConvBlock` (Conv-GroupNorm-ReLU-Conv-GroupNorm-ReLU), `EncoderBlock` (MaxPool-ConvBlock), and `DecoderBlock` (ConvTranspose-CatSkip-ConvBlock) with 4 stages and base_features=64. Feature progression: 64 → 128 → 256 → 512 → 1024 (bottleneck) → 512 → 256 → 128 → 64.

## Goals / Non-Goals

**Goals:**
- Terrain conditions weather processing at every decoder resolution via FiLM
- Identity initialization so the untrained FiLM model degrades gracefully to unconditioned U-Net
- Maintain compatibility with divergence-free projection and loss function (unchanged)
- Keep DEVINE weight init working for compatible layers

**Non-Goals:**
- Encoder-side FiLM conditioning (decoder-only for this change)
- Attention-based terrain conditioning (future work, more complex)
- Automatic DEVINE weight adaptation for mismatched input channels (skip incompatible layers)

## Decisions

### D1: FiLM over concatenation

**Choice:** Replace input concatenation with FiLM conditioning at each decoder stage.

**Why:** Concatenation gives terrain one chance to influence processing (at the input). FiLM gives terrain influence at 4 spatial scales (each decoder block). The multiplicative interaction (`gamma * x`) lets terrain selectively amplify or suppress weather features, which concatenation cannot express. Karpos, TA-ViT, and MambaDS all converge on this pattern independently.

**Alternatives considered:**
- Cross-attention (terrain queries attend to weather keys): more expressive but O(n^2) in spatial dims, too expensive at fine resolution
- Hypernetwork (terrain generates all conv weights): too many parameters, unstable training
- Keep concatenation with deeper encoder: doesn't address the fundamental single-scale limitation

### D2: Lightweight FiLM generator — small CNN

**Choice:** 3-4 conv layer CNN that globally pools terrain features and projects to per-block (gamma, beta) via linear layers.

Architecture:
```
terrain (B, 17, H, W)
  → Conv2d(17, 64, 3, padding=1) → GroupNorm → ReLU
  → Conv2d(64, 128, 3, stride=2, padding=1) → GroupNorm → ReLU
  → Conv2d(128, 256, 3, stride=2, padding=1) → GroupNorm → ReLU
  → AdaptiveAvgPool2d(1) → flatten
  → per-block Linear(256, 2 * C_block) → split → (gamma, beta)
```

**Why:** Terrain is a static, smooth signal (DEM derivatives). It doesn't need a heavy encoder — the spatial structure is captured by the convolutions, and the global pool aggregates a region-level terrain "fingerprint." One linear head per decoder block lets each scale learn its own terrain sensitivity.

**Alternatives considered:**
- Full U-Net encoder for terrain: redundant complexity, the main U-Net already handles spatial features
- Single shared linear head: forces all decoder blocks to use the same terrain representation, losing scale-specific information
- Spatial FiLM (gamma/beta per pixel, not globally pooled): much more parameters, and terrain is spatially smooth enough that global stats suffice for modulation

### D3: FiLM placement — after GroupNorm, before ReLU

**Choice:** Apply `FiLM(GroupNorm(conv(x))) = gamma * GroupNorm(conv(x)) + beta` within `ConvBlock`, after the first GroupNorm and before the first ReLU.

**Why:** This is the standard placement from the original FiLM paper (Perez et al., 2018). GroupNorm normalizes the activations to zero mean/unit variance, then FiLM re-scales and re-shifts them conditioned on terrain. Applying before normalization would be partially undone by the norm.

**Alternatives considered:**
- After both conv-norm-relu pairs: delays terrain influence, and the second conv has already mixed features without terrain context
- After ReLU: the ReLU clips negative values before FiLM can modulate them, losing information

### D4: DEVINE weight init compatibility

**Choice:** Wind channels in the decoder still init from DEVINE. The initial convolution layer (encoder input) is skipped during DEVINE loading because input channels change from 23 to 6. FiLM generator trains from scratch with identity init (gamma=1, beta=0).

**Why:** DEVINE's decoder weights encode spatial upsampling patterns for wind that are independent of the input channel count. The encoder's first layer is shape-incompatible (23→6 channels), so we skip it — the weather-only input is a fundamentally different representation. Identity init for FiLM means the initial model behaves as an unconditioned U-Net, preserving a sensible starting point.

**Alternatives considered:**
- Slice DEVINE's init_conv weights to first 6 channels: DEVINE's first 17 channels are terrain, not weather — channel semantics don't align
- Skip DEVINE entirely and train from scratch: wastes the spatial upsampling priors in the decoder

### D5: Reference implementation

**Choice:** Karpos (github.com/maurinl26/karpos-downscaling, Apache 2.0) as primary reference for FiLM-in-U-Net weather downscaling.

**Why:** Production-quality implementation, same domain (weather downscaling), same architecture pattern (FiLM-conditioned U-Net decoder). Apache 2.0 license permits reference.

## Risks / Trade-offs

- **[Risk] All existing checkpoints invalidated** → Expected and accepted. Phase 2/3 must retrain. No rollback path for saved models.
- **[Risk] FiLM generator may be too simple** → Global pooling loses spatial terrain variation within a tile. Mitigation: if validation shows spatial bias, upgrade to spatial FiLM (per-pixel gamma/beta) in a follow-up change.
- **[Risk] DEVINE encoder init lost** → The encoder's init_conv no longer matches DEVINE shapes. Mitigation: decoder weights (the spatial upsampling patterns) are the more valuable part; encoder trains from scratch on weather-only input, which is a cleaner signal.
- **[Trade-off] Parameter count increases slightly** → FiLM generator adds ~3-5% parameters. Acceptable for the multi-scale conditioning benefit.

## Open Questions

- Should the FiLM generator share any layers across decoder blocks or use fully independent linear heads? Current design uses independent heads. Can be revisited after initial training metrics.
