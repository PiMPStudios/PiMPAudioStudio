# Phase 5: Sound FX Pipeline — Design Spec

## Overview

Add Stable Audio Open 1.0 as a second inference pipeline for sound effects and
ambient generation. The existing IntentRouter classifies prompts and routes to
the appropriate pipeline: music → ACE-Step (existing), soundFX/ambient → Stable
Audio Open (new). Users can override the automatic routing.

**Scope boundary:** Phase 5 does NOT include LoRA fine-tuning, audio editing,
A/B comparison, loop playback, or prompt presets. It delivers a fully functional
second pipeline for non-music audio generation.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Target model | Stable Audio Open 1.0 | Proven, open weights, compatible DiT+VAE architecture |
| Routing | Automatic with manual override | IntentRouter already classifies; power users can force pipeline |
| Download timing | Deferred on first SFX/ambient use | Keeps first launch fast; music is primary use case |
| Architecture sharing | Parallel pipeline, shared primitives | Independent types avoid fragile branching; shared scheduling/attention |
| Conditioning | CLAP text encoder + timing scalars | Native Stable Audio Open architecture |
| UI feedback | Subtle pipeline indicator in Inspector | Minimal clutter, override available for power users |
| Max duration | 47 seconds | Stable Audio Open's trained limit |

## Component 1: CLAPEncoder (AnvilCore)

CLAP (Contrastive Language-Audio Pretraining) text encoder. Converts raw text
prompts into embedding vectors for the SFX DiT model.

### Location

`AnvilCore/Sources/AnvilCore/Inference/StableAudio/CLAPEncoder.swift`

### API

```swift
public actor CLAPEncoder {
    private var weights: [String: MLXArray]?
    private var tokenizer: CLAPTokenizer?
    private let modelPath: URL

    public init(modelPath: URL)

    public func loadModel() async throws
    public var isLoaded: Bool { get }

    public func encode(text: String) async throws -> MLXArray
}
```

### Behavior

- `encode` tokenizes input text using CLAP's vocabulary, runs the text
  transformer, and returns a fixed-size embedding vector (512-dim).
- Model loading is explicit via `loadModel()`. Deferred until first generation.
- Uses CLAP's own tokenizer (separate from Qwen3).
- Weights stored as safetensors in the Stable Audio model directory.

## Component 2: CLAPTokenizer (AnvilCore)

Tokenizer for the CLAP text encoder. Separate from the Qwen3/AnvilTokenizer
used by the music pipeline.

### Location

`AnvilCore/Sources/AnvilCore/Inference/StableAudio/CLAPTokenizer.swift`

### API

```swift
public struct CLAPTokenizer: Sendable {
    public init(vocabularyURL: URL, configURL: URL) throws

    public func encode(text: String) -> [Int]
    public var maxLength: Int { get }
}
```

### Behavior

- Loads vocabulary and config from JSON files in the model directory.
- Encodes text to token IDs with padding/truncation to `maxLength`.
- CLAP uses a standard WordPiece or BPE tokenizer (determined by the model's
  tokenizer_config.json).

## Component 3: TimingEncoder (AnvilCore)

Converts timing parameters into sinusoidal positional embeddings for the DiT.

### Location

`AnvilCore/Sources/AnvilCore/Inference/StableAudio/TimingEncoder.swift`

### API

```swift
public struct TimingEncoder: Sendable {
    public let embeddingDim: Int

    public init(embeddingDim: Int)

    public func encode(startSeconds: Float, durationSeconds: Float) -> MLXArray
}
```

### Behavior

- Pure computation — no model weights, no loading step.
- Produces sinusoidal embeddings for the two scalar values (start time, total
  duration).
- Output shape: `[1, embeddingDim]` — concatenated with CLAP embedding before
  feeding into the DiT's cross-attention.
- `embeddingDim` matches the DiT's conditioning dimension (from model config).

## Component 4: SFXDiTModel + SFXDiTActor (AnvilCore)

The Stable Audio Open DiT (Diffusion Transformer) model. Separate type from the
ACE-Step DiT to avoid fragile branching.

### Location

- `AnvilCore/Sources/AnvilCore/Inference/StableAudio/SFXDiTModel.swift`
- `AnvilCore/Sources/AnvilCore/Inference/StableAudio/SFXDiTActor.swift`

### API

```swift
public struct SFXDiTModel: Sendable {
    public init(config: SFXDiTConfig)

    public func forward(
        latents: MLXArray,
        timestep: MLXArray,
        conditioning: MLXArray
    ) -> MLXArray
}

public struct SFXDiTConfig: Codable, Sendable {
    public let hiddenSize: Int
    public let numLayers: Int
    public let numHeads: Int
    public let latentDim: Int
}

public actor SFXDiTActor {
    public init(config: SFXDiTConfig)

    public func loadWeights(_ weights: [String: MLXArray]) throws
    public var isLoaded: Bool { get }

    public func predict(
        latents: MLXArray,
        timestep: MLXArray,
        conditioning: MLXArray
    ) async -> MLXArray
}
```

### Behavior

- Standard DiT architecture: transformer blocks with cross-attention to
  conditioning, adaptive layer norm conditioned on timestep.
- Config loaded from `config.json` in model directory.
- Shares attention layer primitives (scaled dot-product attention, RoPE) with
  ACE-Step DiT but is its own type — different layer count, hidden dims, head
  count.

## Component 5: SFXVAEModel + SFXVAEActor (AnvilCore)

Stable Audio Open's VAE decoder. Converts DiT latents back to audio waveform.

### Location

- `AnvilCore/Sources/AnvilCore/Inference/StableAudio/SFXVAEModel.swift`
- `AnvilCore/Sources/AnvilCore/Inference/StableAudio/SFXVAEActor.swift`

### API

```swift
public struct SFXVAEModel: Sendable {
    public init(config: SFXVAEConfig)

    public func decode(latents: MLXArray) -> MLXArray
}

public struct SFXVAEConfig: Codable, Sendable {
    public let latentDim: Int
    public let outputChannels: Int
    public let sampleRate: Int
}

public actor SFXVAEActor {
    public init(config: SFXVAEConfig)

    public func loadWeights(_ weights: [String: MLXArray]) throws
    public var isLoaded: Bool { get }

    public func decode(latents: MLXArray) async -> MLXArray
}
```

### Behavior

- Decodes latent representations into raw audio samples.
- Output: mono or stereo waveform at 44.1kHz (Stable Audio Open's native rate).
- Separate from ACE-Step's VAE — different latent dimensions and architecture.

## Component 6: SFXDiffusionActor (AnvilCore)

Runs the diffusion sampling loop for Stable Audio Open.

### Location

`AnvilCore/Sources/AnvilCore/Inference/StableAudio/SFXDiffusionActor.swift`

### API

```swift
public actor SFXDiffusionActor {
    private let dit: SFXDiTActor

    public init(dit: SFXDiTActor)

    public func sample(
        conditioning: MLXArray,
        config: SFXDiffusionConfig
    ) async throws -> MLXArray

    public var progress: AsyncStream<PipelineProgress> { get }
}

public struct SFXDiffusionConfig: Sendable {
    public let steps: Int
    public let guidanceScale: Float
    public let seed: UInt64?
    public let latentShape: [Int]
}
```

### Behavior

- Shares `TimestepSchedule` logic from existing `DiffusionSampler` (same
  mathematical concept, potentially different schedule parameters).
- Runs the denoising loop: noise → iterative DiT predictions → clean latents.
- Reports progress via `AsyncStream<PipelineProgress>` (reuses existing enum).
- Supports cooperative cancellation via `Task.checkCancellation()` between steps.
- CFG (classifier-free guidance) applied when `guidanceScale > 1.0`.

## Component 7: SoundFXPipeline (AnvilCore)

Top-level orchestrator actor for the Stable Audio pipeline. Parallel to
`MusicPipeline`.

### Location

`AnvilCore/Sources/AnvilCore/Inference/StableAudio/SoundFXPipeline.swift`

### API

```swift
public actor SoundFXPipeline {
    private let clap: CLAPEncoder
    private let timingEncoder: TimingEncoder
    private let diffusion: SFXDiffusionActor
    private let vae: SFXVAEActor

    public init(
        clap: CLAPEncoder,
        diffusion: SFXDiffusionActor,
        vae: SFXVAEActor,
        timingEncoder: TimingEncoder
    )

    public func generate(
        text: String,
        duration: TimeInterval,
        config: SFXDiffusionConfig
    ) async throws -> AudioResult

    public var progress: AsyncStream<PipelineProgress> { get }
}
```

### Behavior

- Full generation flow: CLAP encode → timing encode → combine conditioning →
  diffusion sampling → VAE decode → `AudioResult`.
- `AudioResult` is the existing type (samples + sampleRate) used by
  `MusicPipeline`.
- Duration clamped to 0.5–47 seconds.
- Progress stream reports conditioning, diffusion steps, and decoding phases.
- Start time defaults to 0 (no offset generation).

## Component 8: SFXModelManager (AnvilApp)

Manages download and activation of Stable Audio Open weights. Follows the same
pattern as `LLMModelManager`.

### Location

`AnvilApp/Services/SFXModelManager.swift`

### API

```swift
@MainActor
final class SFXModelManager: ObservableObject {
    @Published var downloadState: SFXDownloadState = .notDownloaded
    @Published var pipeline: SoundFXPipeline?

    func downloadIfNeeded() async throws
    func deleteWeights()
    var isReady: Bool { get }
    var modelPath: URL { get }
}

enum SFXDownloadState: Equatable, Sendable {
    case notDownloaded
    case downloading(progress: Float)
    case downloaded
    case error(String)
}
```

### Behavior

- On init, checks if weights exist at
  `~/Library/Application Support/Anvil/models/stable-audio-open-1.0/`.
- `downloadIfNeeded()` — downloads CLAP + DiT + VAE weights from HuggingFace if
  not present. Updates `downloadState` progressively.
- After download completes, creates and stores `SoundFXPipeline` instance.
- `pipeline` is nil until weights are downloaded and loaded.
- Uses `URLSession.download(from:)` for large files (same fix applied to
  LLMModelManager in Phase 4b).

### Weight Files

Downloaded from HuggingFace `stabilityai/stable-audio-open-1.0`:

- `model.safetensors` (or sharded) — DiT weights
- `vae.safetensors` — VAE decoder weights
- `clap_model.safetensors` — CLAP text encoder weights
- `config.json` — model configuration
- `tokenizer.json` — CLAP tokenizer vocabulary

### Storage

`~/Library/Application Support/Anvil/models/stable-audio-open-1.0/`

## Component 9: GenerationService Routing (AnvilApp)

The existing `GenerationService` gains pipeline routing based on the resolved
generation mode.

### Changes to GenerationService

```swift
@MainActor
final class GenerationService: ObservableObject {
    // Existing
    private let musicPipeline: MusicPipeline
    private let conditioning: ConditioningActor

    // New
    private let sfxModelManager: SFXModelManager

    // Modified generate method
    func generate(request: GenerationRequest)
}
```

### GenerationRequest Changes

```swift
struct GenerationRequest {
    // Existing fields
    let text: String
    let lyrics: String?
    let duration: TimeInterval
    let seed: UInt64?
    let method: InferMethod
    let guidanceScale: Float
    let precision: Precision

    // New
    let resolvedMode: GenerationMode
    let pipelineOverride: PipelineChoice?
}

enum PipelineChoice: String, CaseIterable, Sendable {
    case auto
    case aceStep
    case stableAudio
}
```

### Routing Logic

```text
1. Determine pipeline:
   - If pipelineOverride == .aceStep → MusicPipeline
   - If pipelineOverride == .stableAudio → SoundFXPipeline
   - If pipelineOverride == .auto (default):
     - resolvedMode == .music → MusicPipeline
     - resolvedMode == .soundFX or .ambient → SoundFXPipeline

2. If SoundFXPipeline selected:
   - Check sfxModelManager.isReady
   - If not → state = .downloadingModel, await sfxModelManager.downloadIfNeeded()
   - Clamp duration to 47s max
   - Generate via SoundFXPipeline

3. If MusicPipeline selected:
   - Existing flow unchanged
```

### Progress Reporting

Both pipelines emit `PipelineProgress`. The existing progress handling in
`GenerationService` (step counter, ETA) works unchanged regardless of which
pipeline is active.

## Component 10: UI Changes (AnvilApp)

### GenerateView Modifications

- **Duration slider**: When resolved pipeline is Stable Audio, max value capped
  at 47 seconds. If user had set >47s and switches to SFX mode, value clamps
  down with a brief animation.
- **Download trigger**: If user taps Generate and Stable Audio weights aren't
  downloaded, progress section shows download progress before generation begins.
  Same `GenerationState` enum, new case: `.downloadingModel(progress: Float)`.
- **Lyrics section**: Already hidden for soundFX/ambient modes (no change).

### MainShellView Inspector Modifications

The placeholder Inspector gains a real pipeline row:

```swift
Section("Pipeline") {
    LabeledContent("Active") {
        Text(resolvedPipelineName)
            .foregroundStyle(.secondary)
    }
    Picker("Override", selection: $pipelineOverride) {
        Text("Auto").tag(PipelineChoice.auto)
        Text("ACE-Step (Music)").tag(PipelineChoice.aceStep)
        Text("Stable Audio (SFX)").tag(PipelineChoice.stableAudio)
    }
    .pickerStyle(.menu)
}
```

### Mode Picker Badge

When Stable Audio is not downloaded, the Sound FX and Ambient mode buttons show
a small download arrow badge (SF Symbol `arrow.down.circle`). This disappears
after download completes.

## Component 11: Weight Name Maps (AnvilCore)

New weight conversion maps for Stable Audio's tensor naming conventions.

### Location

- `AnvilCore/Sources/AnvilCore/Inference/Weights/CLAPNameMap.swift`
- `AnvilCore/Sources/AnvilCore/Inference/Weights/SFXDiTNameMap.swift`
- `AnvilCore/Sources/AnvilCore/Inference/Weights/SFXVAENameMap.swift`

### Behavior

Each map converts from HuggingFace/PyTorch tensor names to the MLX-Swift
property paths used by the corresponding model structs. Same pattern as existing
`TensorNameMap`, `VAENameMap`, etc.

## Package Structure

```text
AnvilCore/Sources/AnvilCore/
├── Inference/
│   ├── StableAudio/
│   │   ├── CLAPEncoder.swift               (new)
│   │   ├── CLAPTokenizer.swift             (new)
│   │   ├── TimingEncoder.swift             (new)
│   │   ├── SFXDiTModel.swift              (new)
│   │   ├── SFXDiTActor.swift             (new)
│   │   ├── SFXVAEModel.swift             (new)
│   │   ├── SFXVAEActor.swift             (new)
│   │   ├── SFXDiffusionActor.swift        (new)
│   │   └── SoundFXPipeline.swift          (new)
│   ├── Weights/
│   │   ├── CLAPNameMap.swift              (new)
│   │   ├── SFXDiTNameMap.swift            (new)
│   │   └── SFXVAENameMap.swift            (new)
│   └── Types.swift                         (modified — add SFXDiffusionConfig, SFXDiTConfig, SFXVAEConfig)
├── ModelRegistry/
│   └── ModelSpec.swift                     (modified — add Stable Audio spec)

AnvilApp/
├── Services/
│   ├── GenerationService.swift             (modified — routing layer)
│   └── SFXModelManager.swift              (new)
├── UI/
│   ├── MainShellView.swift                 (modified — Inspector pipeline row)
│   └── Components/
│       └── GenerateView.swift              (modified — duration cap, download badge)
├── Models/
│   └── PipelineChoice.swift               (new)
```

## Testing Strategy

1. **CLAPTokenizer** — unit test: known text → expected token IDs, padding,
   truncation at maxLength.
2. **CLAPEncoder** — test with synthetic weights: verify output shape matches
   expected embedding dim.
3. **TimingEncoder** — pure function test: known inputs → expected embedding
   shape, verify sinusoidal pattern properties.
4. **SFXDiTModel** — forward pass with synthetic weights: verify output shape
   matches input latent shape.
5. **SFXVAEModel** — decode with synthetic weights: verify output is audio-shaped
   (samples × channels).
6. **SFXDiffusionActor** — test full sampling loop with synthetic DiT: verify
   progress reports, cancellation, output shape.
7. **SoundFXPipeline** — integration test: full pipeline with random weights,
   verify WAV-writable AudioResult output.
8. **SFXModelManager** — test state machine (notDownloaded → downloading →
   downloaded), path resolution.
9. **GenerationService routing** — test mode → pipeline dispatch (music → ACE,
   soundFX → SA, ambient → SA, override → forced choice).
10. **Weight name maps** — test known PyTorch tensor names map to expected MLX
    paths.

## Dependencies

- **MLX, MLXNN, MLXRandom** (existing — for model inference)
- **Security.framework** (existing — Keychain for API keys, unchanged)
- No new external packages

## Non-Goals (Later Phases)

- Audio editing (trim, fade, effects)
- A/B generation comparison
- Loop playback mode
- LoRA fine-tuning for either pipeline
- Prompt presets / templates library
- Sound FX model fine-tuning or alternative checkpoints
- Real-time preview / streaming audio output during generation
