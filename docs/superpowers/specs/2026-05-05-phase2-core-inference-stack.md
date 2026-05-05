# Phase 2: Core Inference Stack — Design Spec

## Overview

Port the ACE-Step MLX Python inference layer (~1500 lines) to native MLX-Swift
for Apple Silicon. This phase delivers the complete audio generation pipeline
(VAE + DiT + Diffusion Sampler) as testable actors within the existing
`AnvilCore` Swift package.

**Scope boundary:** Phase 2 does NOT include the text encoder (T5/CLAP), UI
integration, or sound FX pipeline. The `MusicPipeline` accepts pre-computed
conditioning tensors so the full stack can be validated end-to-end against the
Python reference.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Weight format | On-device PyTorch `.bin` → safetensors conversion at first launch | No separate weight repo to maintain; ~30s one-time cost |
| Precision | fp32 + fp16, user-selectable | Apple Silicon unified memory makes fp16 a free 2× throughput boost |
| Tiled VAE decode | Included from day one | Required for full-length songs (>85s) |
| ODE/SDE | Both methods, ODE default | SDE is ~10 extra lines; gives users variation control |
| KV cache | Included from day one | ~20-30% speedup; simple dict-based implementation |
| CFG + APG | Full implementation from day one | Audio quality without guidance is noticeably worse |
| Weight conversion | Pure Swift | App Store sandbox prohibits shelling out to Python |
| Architecture | Actor-per-component + MusicPipeline coordinator | Thread-safe, independently testable, clean public API |
| Build approach | Bottom-up | Each component tested in isolation before integration |

## Package Structure

```text
AnvilCore/Sources/AnvilCore/
├── Inference/
│   ├── Pipeline/
│   │   └── MusicPipeline.swift
│   ├── VAE/
│   │   ├── VAEModel.swift
│   │   ├── VAEComponents.swift
│   │   ├── VAEActor.swift
│   │   └── VAETiledDecode.swift
│   ├── DiT/
│   │   ├── DiTModel.swift
│   │   ├── DiTComponents.swift
│   │   ├── DiTActor.swift
│   │   └── CrossAttentionCache.swift
│   ├── Diffusion/
│   │   ├── DiffusionSampler.swift
│   │   ├── TimestepSchedule.swift
│   │   └── DiffusionActor.swift
│   └── Weights/
│       ├── WeightConverter.swift
│       ├── TensorNameMap.swift
│       └── VAENameMap.swift
```

## Component 1: Weight Converter

Runs once at first launch. Reads PyTorch `.bin` state dicts, renames tensors,
transposes axes, and writes `.safetensors` to the app's model cache.

### Key Operations

1. **Tensor renaming** — Python keys (e.g.,
   `decoder.layers.0.self_attn.q_proj.weight`) map to the Swift module
   hierarchy.
2. **Axis transposition** — PyTorch Conv1d weights are `[out_ch, in_ch, kernel]`
   (NCK); MLX-Swift expects `[out_ch, kernel, in_ch]` (NKC).
3. **Shape validation** — After renaming, each tensor's shape is validated
   against the expected model architecture. Mismatches throw immediately rather
   than producing a silently broken model.

### API

```swift
public actor WeightConverter {
    public func convert(
        pytorchBinURL: URL,
        outputDirectory: URL,
        modelType: ModelComponent
    ) async throws -> URL

    public var progress: AsyncStream<ConversionProgress> { get }
}

public enum ModelComponent {
    case vae
    case dit
}

public struct ConversionProgress: Sendable {
    public let current: Int
    public let total: Int
    public let currentTensorName: String
}
```

## Component 2: VAE Model

Oobleck autoencoder (same architecture as Stable Audio). Converts between stereo
audio waveforms and compact 64-dimensional latent representations.

### Architecture

```text
Audio [B, L, 2] (stereo, 44.1kHz)
  → Encoder: Conv1d(k=7) → 5 EncoderBlocks (strides [2,4,4,8,8]) → Conv1d(k=3)
  → Latent [B, L/2048, 64]
  → Decoder: Conv1d(k=7) → 5 DecoderBlocks (strides [8,8,4,4,2]) → Conv1d(k=7)
  → Audio [B, L, 2]
```

- **Hop length:** 2 × 4 × 4 × 8 × 8 = 2048 samples per latent frame
- **Latent dim:** 64 channels
- **Audio:** Stereo (2 channels), 44.1kHz
- **Convention:** Channels-last (NLC) throughout

### Sub-components

- **Snake1d** — `x + (1/exp(β)) × sin²(exp(α) × x)` — learnable per-channel
  nonlinearity
- **OobleckResidualUnit** — Dilated Conv1d(k=7) + Conv1d(k=1) with Snake
  activations and skip connection. Dilations: 1, 3, 9.
- **EncoderBlock** — 3 residual units → Snake → strided Conv1d downsampling
- **DecoderBlock** — Snake → strided ConvTranspose1d upsampling → 3 residual
  units

### Tiled Decode

Activates automatically when latent frames exceed 2048:

- **Chunk size:** 2048 latent frames
- **Overlap:** 64 frames on each side (discarded after decode)
- **Stride:** 2048 - 128 = 1920 frames per chunk
- **Purpose:** Prevents OOM on long audio; a 4-minute song ≈ 5160 latent frames
  → 3 tiles

### API

```swift
public actor VAEActor {
    public func decode(
        latents: MLXArray,
        precision: Precision
    ) async -> MLXArray

    public func encodeSample(
        audio: MLXArray,
        precision: Precision
    ) async -> MLXArray
}

public enum Precision: String, Sendable, CaseIterable {
    case float32
    case float16
}
```

## Component 3: DiT Model

Diffusion Transformer decoder — takes noisy latents + text conditioning and
predicts the velocity field for denoising.

### Architecture

```text
Noisy latents [B, T, 64] + Context [B, T, 128] → concat → [B, T, 192]
  → Patch embedding: Conv1d(k=2, stride=2) → [B, T/2, 2048]
  → 24 Transformer layers (alternating sliding-window / full attention)
  → Output norm + ConvTranspose1d(k=2, stride=2) → [B, T, 64]
```

### Sub-components

- **RotaryEmbedding** — Precomputed cos/sin tables. Base frequency = 1,000,000.
  Max sequence length = 32,768 (~11 minutes of audio).
- **Attention** — Grouped Query Attention: 16 query heads, 8 KV heads,
  head_dim = 128. QK-RMSNorm for training stability. Uses
  `MLXFast.scaledDotProductAttention`.
- **Sliding window** — Odd-indexed layers use bidirectional window of 128
  tokens (additive mask, -1e9 outside window). Even layers attend to full
  sequence.
- **Cross-attention** — Attends to encoder hidden states (text conditioning).
  KV projected once on first step, cached for subsequent steps.
- **AdaLN** — Adaptive Layer Norm modulated by timestep embeddings. Per-layer
  `scale_shift_table` produces 6 modulation values (shift/scale/gate for
  self-attn and MLP).
- **SwiGLU MLP** — `down_proj(silu(gate_proj(x)) × up_proj(x))`,
  intermediate_size = 6144.
- **Timestep embedding** — Sinusoidal → Linear → SiLU → Linear → project to
  6×D for AdaLN. Dual embeddings (t and t−r) summed together.

### Cross-Attention KV Cache

Simple dictionary mapping `layer_idx → (key, value)`. Populated on the first
diffusion step and reused for all subsequent steps. Invalidated when switching
from cover to non-cover conditioning mid-generation.

```swift
public final class CrossAttentionCache: @unchecked Sendable {
    public func update(key: MLXArray, value: MLXArray, layerIdx: Int)
    public func get(layerIdx: Int) -> (key: MLXArray, value: MLXArray)?
    public func isUpdated(layerIdx: Int) -> Bool
    public func reset()
}
```

### API

```swift
public actor DiTActor {
    public func step(
        noisyLatents: MLXArray,
        timestep: Float,
        timestepR: Float,
        encoderStates: MLXArray,
        contextLatents: MLXArray,
        cache: CrossAttentionCache?,
        useCache: Bool
    ) async -> (output: MLXArray, cache: CrossAttentionCache?)
}
```

## Component 4: Diffusion Sampler

Orchestrates the iterative denoising loop — starts with pure noise and
progressively refines it into clean latents using the DiT.

### Timestep Scheduling

Three modes (checked in priority order):

1. **Custom timesteps** — User-provided list, snapped to nearest valid value
2. **Linspace** — When `steps` is specified: `1.0 - i/steps` for `i` in
   `0..<steps`, with shift transformation `shift×t / (1 + (shift-1)×t)`
3. **Lookup table** — Default 8-step schedules per shift value (1.0, 2.0, 3.0)

### Stepping Methods

- **ODE (default)** — Deterministic Euler: `x_{t-1} = x_t - v_t × dt`
- **SDE** — Stochastic: predict clean sample `x_0 = x_t - v_t × t`, then
  re-noise at next level: `x_{t-1} = t_{next} × noise + (1 - t_{next}) × x_0`

### CFG + APG

When `guidanceScale > 1.0`:

1. Double the batch (conditioned + null-conditioned)
2. Run DiT on both halves
3. APG decomposition: split guidance vector into parallel and orthogonal
   components relative to the conditioned prediction
4. Apply guidance only to orthogonal: `pred_cond + (scale - 1) × orthogonal`
5. Norm thresholding (default 2.5) prevents guidance explosion
6. Momentum tracking across steps for temporal stability

### API

```swift
public actor DiffusionActor {
    /// Initialized with a reference to the DiTActor used for each denoising step.
    public init(dit: DiTActor)

    public func generate(
        encoderStates: MLXArray,
        contextLatents: MLXArray,
        shape: (batch: Int, frames: Int, channels: Int),
        config: DiffusionConfig
    ) async -> DiffusionResult

    public var progress: AsyncStream<DiffusionProgress> { get }
}

public struct DiffusionConfig: Sendable {
    public var seed: Int?
    public var method: InferMethod
    public var shift: Float
    public var steps: Int?
    public var guidanceScale: Float
    public var cfgIntervalStart: Float
    public var cfgIntervalEnd: Float
    public var audioCoverStrength: Float
    public var precision: Precision
}

public enum InferMethod: String, Sendable {
    case ode
    case sde
}

public struct DiffusionResult: Sendable {
    public let latents: MLXArray
    public let timeCosts: TimeCosts
}

public struct DiffusionProgress: Sendable {
    public let currentStep: Int
    public let totalSteps: Int
    public let elapsedTime: TimeInterval
}

public struct TimeCosts: Sendable {
    public let diffusionTime: TimeInterval
    public let perStepTime: TimeInterval
    public let totalTime: TimeInterval
}
```

## Component 5: MusicPipeline (Coordinator)

Top-level actor that orchestrates the full generation flow.

### Generation Flow

```text
MusicPipeline.generate(encoderStates, contextLatents, duration, config)
  1. DiffusionActor.generate(...) → iterates DiTActor.step() for N steps
     → denoised latents [B, T, 64]
  2. VAEActor.decode(latents, precision) → tiled if needed
     → audio [B, samples, 2]
  3. Return AudioResult
```

### API

```swift
public actor MusicPipeline {
    public func loadModel(
        from directory: URL,
        precision: Precision
    ) async throws

    public func generate(
        encoderStates: MLXArray,
        contextLatents: MLXArray,
        duration: TimeInterval,
        config: DiffusionConfig
    ) async throws -> AudioResult

    public var state: PipelineState { get }
    public var progress: AsyncStream<GenerationProgress> { get }
}

public struct AudioResult: Sendable {
    public let samples: MLXArray
    public let sampleRate: Int        // 44100
    public let duration: TimeInterval
    public let timeCosts: TimeCosts
}

public enum PipelineState: Sendable {
    case unloaded
    case loading
    case ready
    case generating(step: Int, totalSteps: Int)
    case error(String)
}

public enum GenerationProgress: Sendable {
    case loadingWeights(Float)
    case diffusion(DiffusionProgress)
    case decoding
    case complete
}
```

## Testing Strategy

Each component is validated against the Python reference using deterministic
inputs:

1. **Weight Converter** — Convert a known checkpoint, verify tensor names,
   shapes, and sample values match expected output.
2. **VAE** — Feed known latent tensors (exported from Python), compare decoded
   audio within floating-point tolerance (atol=1e-4 for fp32, atol=1e-2 for
   fp16).
3. **DiT** — Feed known noisy latents + conditioning + timestep, compare single
   step output against Python reference.
4. **Diffusion Sampler** — Full loop with fixed seed, compare final latents
   against Python reference output.
5. **MusicPipeline** — End-to-end: same seed + conditioning → same audio output
   as Python.

Reference test fixtures (numpy `.npz` files) will be exported from the Python
implementation and stored in `AnvilCore/Tests/Fixtures/`.

## Dependencies

- **mlx-swift** (ml-explore/mlx-swift) — Core ML framework
- **swift-argument-parser** (existing in AnvilCore) — Not needed for this phase
- **CryptoKit** (system) — SHA256 for weight verification (existing)

## Non-Goals (Future Phases)

- Text encoder (T5/CLAP) — Phase 3
- Sound FX pipeline (Stable Audio) — Phase 3
- IntentRouter / LyricWriter — Phase 4
- UI integration — Phase 4
- DSP audio editor (trim, fade, normalize, EQ, reverb, time stretch, pitch shift) — Phase 5
- AI audio editor (inpaint/repaint via VAE encode → mask latents → denoise) — Phase 6
- LoRA fine-tuning — Phase 7
