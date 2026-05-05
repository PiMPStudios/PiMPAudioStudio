# Changelog

All notable changes to AnvilApp are documented here.

---

## [Unreleased] — Phase 2: Core Inference Stack

### Added

- **MusicPipeline** coordinator actor — top-level orchestrator that runs diffusion then VAE decode, returning `AudioResult` with stereo audio at 44.1 kHz
- **DiffusionActor** — full generation loop with seeded noise initialization, CFG batch doubling, APG guidance, and `AsyncStream<DiffusionProgress>` for progress reporting
- **DiffusionSampler** — ODE and SDE stepping methods, APG (Adaptive Projected Guidance) with momentum and norm thresholding, seeded noise generation
- **DiTDecoder** (`DiTModel.swift`) — 24-layer Diffusion Transformer decoder with patch embedding (Conv1d stride 2), de-patchify (ConvTransposed1d), adaptive layer norm output, and sliding window mask caching
- **DiTActor** — thread-safe actor wrapping `DiTDecoder` with `step()` and `loadWeights()` methods
- **DiT components** (`DiTComponents.swift`):
  - `RotaryEmbedding` — precomputed cos/sin tables, base frequency 1,000,000, max 32,768 tokens
  - `DiTAttention` — Grouped Query Attention (16 query / 8 KV heads), QK-RMSNorm, optional sliding window mask, cross-attention with KV cache support
  - `SwiGLUMLP` — `down(silu(gate(x)) * up(x))`, intermediate size 6144
  - `TimestepEmbedding` — sinusoidal → linear → SiLU → linear → 6×D projection for AdaLN
  - `DiTLayer` — AdaLN-modulated self-attention + cross-attention + MLP
  - `createSlidingWindowMask` — additive -1e9 mask for bidirectional window of 128 tokens
- **VAEActor** — actor wrapping `AutoEncoderOobleck` with fp16/fp32 precision switching and tiled decode delegation
- **VAE model** (`VAEModel.swift`) — `OobleckEncoder`, `OobleckDecoder`, `AutoEncoderOobleck` with `encodeSample` (diagonal Gaussian), `encodeMean`, and `decode`
- **VAETiledDecode** — overlap-discard chunked decode for latent sequences >2048 frames (chunk=2048, overlap=64, stride=1920); prevents OOM on songs >85s
- **VAE components** (`VAEComponents.swift`) — `Snake1d` activation, `OobleckResidualUnit` (dilated conv1d + skip, dilations 1/3/9), `OobleckEncoderBlock`, `OobleckDecoderBlock`
- **WeightConverter** actor — converts PyTorch `.bin` state dicts to MLX-Swift weight lists for on-device use (pure Swift, no Python subprocess)
- **VAENameMap** — `weight_norm` fusion (g+v → fused), Conv1d/ConvTransposed1d axis transposition, Snake1d parameter squeeze
- **TensorNameMap** — DiT key remapping (`proj_in.1.*` → `proj_in.*`), rotary embedding key filtering
- **CrossAttentionCache** — layer-indexed (key, value) dictionary cache; populated on first diffusion step, reused for all subsequent steps
- **TimestepSchedule** — three scheduling modes: lookup table (shift 1.0/2.0/3.0), linspace with shift transform, custom timestep snapping
- **Shared inference types** — `Precision`, `InferMethod`, `DiffusionConfig`, `DiffusionResult`, `AudioResult`, `TimeCosts`, `PipelineState`, `GenerationProgress`, `ConversionProgress`
- mlx-swift dependency added to `AnvilCore/Package.swift` (MLX, MLXNN, MLXRandom, MLXFast)
- Design spec: `docs/superpowers/specs/2026-05-05-phase2-core-inference-stack.md`
- Implementation plan: `docs/superpowers/plans/2026-05-05-phase2-core-inference.md`
- Audio editor roadmap entries (DSP — Phase 5; AI inpaint/repaint — Phase 6)

---

## [Unreleased] — Phase 1: Foundation

### Added

- **AnvilApp Xcode project** — SwiftUI shell targeting macOS 14+, App Store sandbox entitlements, Hardened Runtime for notarization
- **Three-panel main shell** — sidebar / canvas / inspector layout
- **First-launch download UI** — model weight download flow with progress display
- **AppState** — observable state container with injectable `ModelRegistry` for testing
- **ModelRegistry** — status tracking and download orchestration for model weights
- **DownloadClient** — HuggingFace URL building, async download with progress, SHA256 verification
- **ModelSpec** — model weight metadata type (name, URL, expected SHA256, size)
- **AnvilCore Swift package** — initialized at `AnvilCore/`, separate from the Xcode app target
- Design spec: `docs/superpowers/specs/` (initial Anvil Swift app design)
- Implementation plan: `docs/superpowers/plans/2026-05-05-phase1-foundation.md` (implicit)

---

## Road Map

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | ✅ Complete | Foundation — Xcode project, SwiftUI shell, ModelRegistry, DownloadClient |
| 2 | ✅ Complete | Core Inference Stack — VAE, DiT, Diffusion Sampler, MusicPipeline, WeightConverter |
| 3 | Planned | Text Encoders — T5 and CLAP conditioning |
| 4 | Planned | UI Integration — IntentRouter, LyricWriter, full generation flow |
| 5 | Planned | DSP Audio Editor — trim, fade, normalize, EQ, reverb, time stretch, pitch shift |
| 6 | Planned | AI Audio Editor — inpaint/repaint via VAE encode → mask latents → denoise |
| 7 | Planned | LoRA Fine-Tuning — on-device model personalization |
