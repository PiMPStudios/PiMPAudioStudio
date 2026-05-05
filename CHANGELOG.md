# Changelog

All notable changes to AnvilApp are documented here.

---

## [Unreleased] — Phase 3: Conditioning Pipeline

### Added

- **Qwen3Model** (`Qwen3Model.swift`) — 28-layer bidirectional text encoder, 1024-dim hidden, 151,669-token vocabulary; `callAsFunction(inputIds:)` returns `[B, T, 1024]`, `embedTokens(_:)` returns embedding-only lookup
- **Qwen3Layer / Qwen3Attention** (`Qwen3Layer.swift`) — pre-norm transformer layer with GQA (16 query / 8 KV heads, head_dim=128), QK-RMSNorm, RoPE via shared `RotaryEmbedding`, no causal mask
- **LyricEncoder** (`LyricEncoder.swift`) — 8-layer encoder with `Linear(1024→2048)` input projection; takes Qwen3 token embeddings, returns `[B, T_lyric, 2048]`
- **TimbreEncoder** (`TimbreEncoder.swift`) — 4-layer encoder with `Linear(64→2048)` input projection, followed by 2-layer `AttentionPooler` with 5 learnable query tokens; takes VAE latents `[B, 750, 64]`, returns timbre tokens `[B, 5, 2048]`; silence fallback (`zeros([1, 750, 64])`) for text-only generation
- **ConditionEncoder** (`ConditionEncoder.swift`) — orchestrates sub-encoders: projects text hidden states `[B, T, 1024→2048]`, concatenates text + lyric + timbre tokens along sequence axis; returns `encoderStates [B, T_combined, 2048]` and combined attention mask
- **ConditioningActor** (`ConditioningActor.swift`) — top-level Swift actor: tokenizes text and lyrics, runs Qwen3 forward pass, runs LyricEncoder, VAE-encodes reference audio (pad/truncate to 750 frames), runs TimbreEncoder, assembles `ConditioningResult`; produces `contextLatents [B, T, 128]` = zeros for text-only or `cat([latents, chunkMask], axis:-1)` for reference audio
- **AnvilTokenizer** (`AnvilTokenizer.swift`) — thin wrapper around swift-transformers `AutoTokenizer`; `encodeText(_:maxLength:)` and `encodeLyrics(_:maxLength:)` return `(ids: MLXArray, mask: MLXArray)` shaped `[1, maxLength]` as Int32; `forTesting()` factory with minimal BPE vocab for unit tests
- **Qwen3NameMap** (`Weights/Qwen3NameMap.swift`) — maps Qwen3-Embedding-0.6B HuggingFace safetensors keys to MLX-Swift parameter names; filters `lm_head.*`
- **ConditionEncoderNameMap** (`Weights/ConditionEncoderNameMap.swift`) — maps ACE-Step DiT checkpoint keys (`condition_encoder.{text_projector,lyric_encoder,timbre_encoder,timbre_attention_pooler}.*`) to Swift module paths; handles `special_token` → `attentionPooler.queryTokens`
- **WeightConverter** additions — `convertQwen3Weights(from:progress:)` and `convertConditionEncoderWeights(from:progress:)` methods
- **ModelComponent** additions — `.qwen3` and `.conditionEncoder` cases
- **ConditioningResult** type — `encoderStates`, `encoderMask`, `contextLatents` (`@unchecked Sendable`)
- **MusicPipeline** convenience method — `generate(text:lyrics:referenceAudio:duration:config:conditioning:)` runs conditioning then diffusion in one call
- swift-transformers (HuggingFace, v0.1.12+) added as dependency to `AnvilCore/Package.swift`
- Design spec: `docs/superpowers/specs/2026-05-05-phase3-conditioning-pipeline.md`
- Implementation plan: `docs/superpowers/plans/2026-05-05-phase3-conditioning-pipeline.md`

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
| 3 | ✅ Complete | Conditioning Pipeline — Qwen3 text encoder, LyricEncoder, TimbreEncoder, ConditionEncoder, ConditioningActor |
| 4 | Planned | UI Integration — IntentRouter, LyricWriter, full generation flow |
| 5 | Planned | DSP Audio Editor — trim, fade, normalize, EQ, reverb, time stretch, pitch shift |
| 6 | Planned | AI Audio Editor — inpaint/repaint via VAE encode → mask latents → denoise |
| 7 | Planned | LoRA Fine-Tuning — on-device model personalization |
