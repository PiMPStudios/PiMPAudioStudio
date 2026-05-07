# Cover-Song Phase 2A — Foundation Modules Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the four foundation modules (`ResidualFSQ`, `AudioTokenizer`, `AudioTokenDetokenizer`, `AceStepLM`) plus the supporting weight infrastructure (sharded safetensors loader, three new name maps, `mapDiTKey` extension), so we can load every checkpoint asset cover-song generation needs. This phase produces no user-visible UI change — it's a pure infrastructure layer that Phase 2B will plug into the conditioning path.

**Architecture:** Each module is a self-contained `Module` subclass under `AnvilCore/Sources/AnvilCore/Inference/`. Weight conversion goes through a per-component name map under `AnvilCore/Sources/AnvilCore/Inference/Weights/`, mirroring the existing `Qwen3NameMap` / `ConditionEncoderNameMap` pattern. `mapDiTKey` learns to route the DiT bundle's `tokenizer.*` and `detokenizer.*` subtrees out — they currently get dropped by the `decoder.` guard. Sharded safetensors get a single new method on `WeightConverter` that reads `model.safetensors.index.json` and merges per-shard arrays.

**Tech Stack:** Swift, MLX-Swift, MLXNN. Existing modules to reuse: `Qwen3Layer` (used by Lyric/Timbre encoders) for the LM, `RMSNorm` and `RotaryEmbedding` from `DiTComponents.swift`. Verify via Xcode build (no SPM tests because Metal shaders don't compile in `swift test`):

```bash
xcodebuild -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | grep -E "error:|BUILD"
```

Pre-existing warnings (MLXArray Sendable, AppState Swift 6) are unrelated and expected.

---

## Reference: actual safetensors key shapes (verified from disk)

These are the keys we must produce or accept after each name map runs. Memorize the shapes — they encode the architecture.

### DiT bundle's `tokenizer.*` subtree (32 keys)

```
tokenizer.audio_acoustic_proj.weight          [2048, 64]
tokenizer.audio_acoustic_proj.bias            [2048]
tokenizer.attention_pooler.embed_tokens.weight [2048, 2048]   ← Linear, not Embedding
tokenizer.attention_pooler.embed_tokens.bias  [2048]
tokenizer.attention_pooler.special_token      [1, 1, 2048]    ← cls-style single query
tokenizer.attention_pooler.layers.0..1.{...}                  ← 2 self-attn transformer layers
  .self_attn.{q,k,v,o}_proj.weight  ([2048,2048] for q/o, [1024,2048] for k/v)
  .self_attn.{q,k}_norm.weight  [128]
  .input_layernorm.weight       [2048]
  .post_attention_layernorm.weight [2048]
  .mlp.{gate,up}_proj.weight    [6144, 2048]
  .mlp.down_proj.weight         [2048, 6144]
tokenizer.attention_pooler.norm.weight        [2048]
tokenizer.quantizer.project_in.weight         [6, 2048]       ← FSQ projection 2048→6
tokenizer.quantizer.project_in.bias           [6]
tokenizer.quantizer.project_out.weight        [2048, 6]       ← back to 2048
tokenizer.quantizer.project_out.bias          [2048]
```

### DiT bundle's `detokenizer.*` subtree (28 keys)

```
detokenizer.embed_tokens.weight               [2048, 2048]    ← Linear
detokenizer.embed_tokens.bias                 [2048]
detokenizer.special_tokens                    [1, 5, 2048]    ← 5 expansion tokens (note the `s`)
detokenizer.layers.0..1.{...}                                  ← 2 self-attn transformer layers (same layout as tokenizer's pooler)
detokenizer.norm.weight                       [2048]
detokenizer.proj_out.weight                   [64, 2048]      ← back to acoustic dim
detokenizer.proj_out.bias                     [64]
```

### Qwen3-style LM checkpoint (`acestep-5Hz-lm-0.6B/model.safetensors` — 308 + 2 = 310 keys)

```
embed_tokens.weight                           [vocab, hidden]
norm.weight                                   [hidden]
layers.0..N.{...}                                            ← N = num_hidden_layers from config
  .input_layernorm.weight                     [hidden]
  .post_attention_layernorm.weight            [hidden]
  .self_attn.{q,k,v,o}_proj.weight
  .self_attn.{q,k}_norm.weight                [head_dim]
  .mlp.{gate,up}_proj.weight                  [intermediate, hidden]
  .mlp.down_proj.weight                       [hidden, intermediate]
```

The 4B variant has the same structure but sharded across `model-00001-of-00002.safetensors` + `model-00002-of-00002.safetensors`, indexed by `model.safetensors.index.json`.

---

## File structure (created in this plan)

| File | Responsibility |
|---|---|
| `AnvilCore/Sources/AnvilCore/Inference/AudioTokenizer/ResidualFSQ.swift` | Finite Scalar Quantization with `project_in` / `project_out` linear bottleneck |
| `AnvilCore/Sources/AnvilCore/Inference/AudioTokenizer/AudioTokenizerPooler.swift` | The tokenizer's internal `AttentionPooler`: cls-token + 2 layers + norm |
| `AnvilCore/Sources/AnvilCore/Inference/AudioTokenizer/AudioTokenizer.swift` | Top-level: `audio_acoustic_proj` + pooler + quantizer; `forward` returns `(quantized, indices)` |
| `AnvilCore/Sources/AnvilCore/Inference/AudioTokenizer/AudioTokenDetokenizer.swift` | `embed_tokens` (Linear) + special_tokens expansion + 2 layers + norm + `proj_out` |
| `AnvilCore/Sources/AnvilCore/Inference/LM/AceStepLMConfig.swift` | Decodes the LM variant's `config.json` into Swift constants |
| `AnvilCore/Sources/AnvilCore/Inference/LM/AceStepLM.swift` | Qwen-architecture decoder using existing `Qwen3Layer`; returns final hidden states |
| `AnvilCore/Sources/AnvilCore/Inference/Weights/AudioTokenizerNameMap.swift` | `tokenizer.*` → Swift parameter paths |
| `AnvilCore/Sources/AnvilCore/Inference/Weights/AudioTokenDetokenizerNameMap.swift` | `detokenizer.*` → Swift parameter paths |
| `AnvilCore/Sources/AnvilCore/Inference/Weights/AceStepLMNameMap.swift` | LM safetensors keys → Swift parameter paths |
| `AnvilCore/Sources/AnvilCore/Inference/Weights/WeightConverter.swift` (modify) | New sharded loader; new `convertAudioTokenizerWeights` / `convertAudioTokenDetokenizerWeights` / `convertAceStepLMWeights` methods |
| `AnvilCore/Sources/AnvilCore/Inference/Weights/TensorNameMap.swift` (modify) | `mapDiTKey` extension: keep `tokenizer.*` and `detokenizer.*` keys via dedicated maps |

`AceStepEncoderLayer` (used by both the tokenizer's pooler and the detokenizer) is structurally identical to our existing `Qwen3Layer`. We reuse `Qwen3Layer` and don't introduce a new type — the tokenizer/detokenizer constructors just instantiate `Qwen3Layer` instances directly.

---

## Task 1: ResidualFSQ — Finite Scalar Quantization with linear bottleneck

**Files:**
- Create: `AnvilCore/Sources/AnvilCore/Inference/AudioTokenizer/ResidualFSQ.swift`

The upstream module wraps `vector_quantize_pytorch.ResidualFSQ`. For ACE-Step v1.5 turbo `num_quantizers=1` (`fsq_input_num_quantizers: 1`), so "residual" reduces to a single FSQ pass — but we keep the name for parity with upstream. The math: project hidden_size → 6 dims, bound each scalar to its level via `tanh`/`floor`, look up index, project 6 → hidden_size.

- [ ] **Step 1: Create the file with a stub Module that compiles**

```swift
// AnvilCore/Sources/AnvilCore/Inference/AudioTokenizer/ResidualFSQ.swift
import Foundation
import MLX
import MLXNN

/// Finite Scalar Quantization with a Linear bottleneck on each side.
///
/// Mirrors upstream's `vector_quantize_pytorch.ResidualFSQ` configured with
/// `num_quantizers=1`. The quantizer projects `hidden_size`-dim continuous
/// features down to `dim` (= len(levels)) dims, rounds each scalar to the
/// nearest of its `levels[i]` evenly-spaced bins, then projects back up.
///
/// The "residual" prefix is structurally a no-op when `num_quantizers=1`
/// but kept for naming parity with upstream and the safetensors keys
/// (`tokenizer.quantizer.project_in.*`, `tokenizer.quantizer.project_out.*`).
public class ResidualFSQ: Module {
    /// Per-axis quantization granularity. For ACE-Step v1.5 this is
    /// `[8, 8, 8, 5, 5, 5]` from `config.fsq_input_levels`. Total codebook
    /// size = product = 64000.
    public let levels: [Int]
    /// `project_in: Linear(hidden_size, len(levels))` — typically Linear(2048, 6).
    public var projectIn: Linear
    /// `project_out: Linear(len(levels), hidden_size)` — Linear(6, 2048).
    public var projectOut: Linear

    public init(hiddenSize: Int, levels: [Int]) {
        self.levels = levels
        self.projectIn = Linear(hiddenSize, levels.count, bias: true)
        self.projectOut = Linear(levels.count, hiddenSize, bias: true)
    }

    /// Run FSQ on continuous input.
    /// - Parameter hidden: `[B, T, hidden_size]`.
    /// - Returns: `(quantized, indices)` where:
    ///   - `quantized`: `[B, T, hidden_size]` — same shape as input, run through the bottleneck.
    ///   - `indices`:   `[B, T, 1]` — single integer per token (residual depth = 1).
    public func callAsFunction(_ hidden: MLXArray) -> (quantized: MLXArray, indices: MLXArray) {
        let z = projectIn(hidden)                       // [B, T, dim]
        let (zq, idx) = quantize(z)                     // [B, T, dim], [B, T]
        let out = projectOut(zq)                        // [B, T, hidden_size]
        return (out, idx.expandedDimensions(axis: -1))  // [B, T, 1]
    }

    /// Snap each scalar to the nearest of `levels[i]` evenly-spaced bins in [-1, 1],
    /// and pack the per-axis bin indices into a single integer per token.
    /// Mirrors upstream `lucidrains/vector-quantize-pytorch` FSQ.quantize logic.
    public func quantize(_ z: MLXArray) -> (quantized: MLXArray, indices: MLXArray) {
        let bsz = z.shape[0]
        let seqL = z.shape[1]

        // bound input via tanh and round to nearest bin per axis
        let zBounded = MLX.tanh(z)                                        // [B, T, dim] in (-1, 1)
        let levelArr = MLXArray(levels.map { Float($0) })                 // [dim]
        // half-step grid: bin i in 0..<level corresponds to value
        //   ((2*i + 1) / level) - 1   ∈ (-1, 1)
        // To pick: i = round((zBounded + 1) * level / 2 - 0.5),
        // clamped to [0, level-1].
        let halfLevels = levelArr / 2.0
        let scaled = (zBounded + 1.0) * halfLevels - 0.5                  // [B, T, dim]
        let rounded = MLX.round(scaled)
        let clamped = MLX.clip(rounded, min: MLXArray.zeros(like: rounded), max: levelArr - 1.0)
        let idxPerAxis = clamped.asType(.int32)                            // [B, T, dim]

        // recover quantized continuous values from bin indices
        let zq = (idxPerAxis.asType(.float32) + 0.5) * 2.0 / levelArr - 1.0 // [B, T, dim]

        // pack per-axis indices into a single integer:
        // idx = sum_i (idxPerAxis[i] * prod(levels[i+1..]))
        var multiplier = 1
        var packed = MLXArray.zeros([bsz, seqL], dtype: .int32)
        for i in (0..<levels.count).reversed() {
            let axis = idxPerAxis[0..., 0..., i] * multiplier
            packed = packed + axis
            multiplier *= levels[i]
        }
        return (zq, packed)
    }

    /// Inverse of `quantize` for the per-token packed integer index path.
    /// Used during cover generation when `audio_codes` are supplied directly:
    /// `lm_hints_5Hz = quantizer.get_output_from_indices(audio_codes)`.
    public func getOutputFromIndices(_ packed: MLXArray) -> MLXArray {
        let bsz = packed.shape[0]
        let seqL = packed.shape[1]
        let dim = levels.count

        var idxPerAxis = MLXArray.zeros([bsz, seqL, dim], dtype: .int32)
        var remaining = packed
        for i in (0..<dim).reversed() {
            let level = levels[i]
            let axis = remaining % level
            remaining = remaining / level
            // place `axis` at slot i along axis 2
            idxPerAxis[0..., 0..., i] = axis
        }

        let levelArr = MLXArray(levels.map { Float($0) })
        let zq = (idxPerAxis.asType(.float32) + 0.5) * 2.0 / levelArr - 1.0  // [B, T, dim]
        return projectOut(zq)                                                 // [B, T, hidden_size]
    }
}
```

- [ ] **Step 2: Build to verify the file compiles**

Run: `xcodebuild -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | grep -E "error:|BUILD" | tail -5`
Expected: `** BUILD SUCCEEDED **`

- [ ] **Step 3: Commit**

```bash
cd ~/Developer/GitRepos/AnvilApp
git add AnvilCore/Sources/AnvilCore/Inference/AudioTokenizer/ResidualFSQ.swift
git commit -m "feat(audio-tokenizer): ResidualFSQ quantizer with project_in/project_out

First component of cover-song generation foundation. Mirrors upstream's
vector_quantize_pytorch.ResidualFSQ with num_quantizers=1: hidden_size →
dim=len(levels) bottleneck via Linear, per-axis FSQ rounding, then
back to hidden_size.

quantize() packs the per-axis bin indices into a single integer per
token. getOutputFromIndices() reverses the pack and runs project_out
to recover the continuous representation — used when audio_codes are
supplied directly to skip the encoder pass.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 2: AudioTokenizerPooler — cls-token attention pooling

**Files:**
- Create: `AnvilCore/Sources/AnvilCore/Inference/AudioTokenizer/AudioTokenizerPooler.swift`

This is the tokenizer's internal `AttentionPooler`. It takes patches of pool_window_size audio tokens, prepends a learnable cls token, runs 2 self-attention transformer layers, and reads out the cls position. The output is one token per pool window — i.e., 5 Hz from a 25 Hz input.

The pooler reuses our existing `Qwen3Layer` for its self-attention layers (architecturally identical: SwiGLU MLP, RMSNorm, GQA with q_norm/k_norm). The "embed_tokens" name on the safetensors side is misleading — it's actually a Linear projection (input shape `[2048, 2048]` plus bias).

- [ ] **Step 1: Create the file**

```swift
// AnvilCore/Sources/AnvilCore/Inference/AudioTokenizer/AudioTokenizerPooler.swift
import Foundation
import MLX
import MLXNN

/// The cls-token attention pooler that lives inside AudioTokenizer.
///
/// Input layout: `[B, num_windows, pool_window_size, hidden_size]` where each
/// `pool_window_size` slice is one window's worth of audio patches at the
/// VAE-latent rate. The pooler:
///   1. Reshape so each window is processed independently:
///        `[B * num_windows, pool_window_size, hidden_size]`
///   2. Prepend a learnable special_token (cls):
///        `[B * num_windows, 1 + pool_window_size, hidden_size]`
///   3. Run 2 self-attention transformer layers
///   4. Read out position 0 (the cls slot) and reshape back to
///        `[B, num_windows, hidden_size]`
///
/// The result is one pooled vector per window — the "5 Hz" representation
/// fed to the FSQ quantizer.
public class AudioTokenizerPooler: Module {
    /// `tokenizer.attention_pooler.embed_tokens` in the safetensors. Despite the
    /// name, this is a Linear (weight shape [hidden_size, hidden_size]) that
    /// projects each input patch before the cls is prepended.
    public var embedTokens: Linear
    /// `tokenizer.attention_pooler.special_token`, shape [1, 1, hidden_size].
    /// The cls query that gets prepended to each window's patches.
    public var specialToken: MLXArray
    /// Two self-attention transformer layers from `tokenizer.attention_pooler.layers.0..1`.
    public let layers: [Qwen3Layer]
    /// Final RMSNorm after the layers.
    public let norm: RMSNorm
    /// Rotary positional embedding shared across the layers.
    public let rotaryEmbedding: RotaryEmbedding

    public let hiddenSize: Int

    public init(
        hiddenSize: Int,
        intermediateSize: Int,
        numAttentionHeads: Int,
        numKeyValueHeads: Int,
        headDim: Int,
        numLayers: Int,
        rmsNormEps: Float,
        ropeTheta: Float,
        maxPositionEmbeddings: Int
    ) {
        self.hiddenSize = hiddenSize
        self.embedTokens = Linear(hiddenSize, hiddenSize, bias: true)
        self.specialToken = MLXArray.zeros([1, 1, hiddenSize])
        self.layers = (0..<numLayers).map { _ in
            Qwen3Layer(
                hiddenSize: hiddenSize,
                intermediateSize: intermediateSize,
                numAttentionHeads: numAttentionHeads,
                numKeyValueHeads: numKeyValueHeads,
                headDim: headDim,
                rmsNormEps: rmsNormEps
            )
        }
        self.norm = RMSNorm(dimensions: hiddenSize, eps: rmsNormEps)
        self.rotaryEmbedding = RotaryEmbedding(
            headDim: headDim,
            maxLen: maxPositionEmbeddings,
            base: ropeTheta
        )
    }

    /// Pool `[B, num_windows, pool_window_size, hidden_size]` →
    /// `[B, num_windows, hidden_size]` via cls-token self-attention.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let bsz = x.shape[0]
        let nWindows = x.shape[1]
        let poolWin = x.shape[2]
        let hSize = x.shape[3]

        // Linear-project each patch
        let projected = embedTokens(x.reshaped([bsz * nWindows, poolWin, hSize]))

        // Prepend cls (broadcast special_token across the [B*num_windows] axis)
        let cls = MLX.broadcast(specialToken, to: [bsz * nWindows, 1, hSize])
        var h = MLX.concatenated([cls, projected], axis: 1)               // [B*N, 1+poolWin, H]

        let seqLen = h.shape[1]
        let positionCosSin = rotaryEmbedding.forward(seqLen: seqLen)
        for layer in layers {
            h = layer(
                hiddenStates: h,
                positionCosSin: positionCosSin,
                attentionMask: nil
            )
        }
        h = norm(h)

        // Read out cls position (index 0), reshape back to [B, num_windows, H]
        let cls_out = h[0..., 0..<1, 0...]                                 // [B*N, 1, H]
        return cls_out.reshaped([bsz, nWindows, hSize])
    }
}
```

- [ ] **Step 2: Build**

Run: `xcodebuild -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | grep -E "error:|BUILD" | tail -5`
Expected: `** BUILD SUCCEEDED **`

- [ ] **Step 3: Commit**

```bash
git add AnvilCore/Sources/AnvilCore/Inference/AudioTokenizer/AudioTokenizerPooler.swift
git commit -m "feat(audio-tokenizer): cls-token AttentionPooler used inside AudioTokenizer

Pools windows of pool_window_size audio patches into a single token by
prepending a learnable cls token and running self-attention. Output is
one pooled vector per window — the 5 Hz representation fed to FSQ.

Reuses our existing Qwen3Layer for the self-attention transformer
blocks (same SwiGLU+RMSNorm+GQA architecture as the lyric/timbre
encoders, identical safetensors layout).

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 3: AudioTokenizer — top-level acoustic→codes module

**Files:**
- Create: `AnvilCore/Sources/AnvilCore/Inference/AudioTokenizer/AudioTokenizer.swift`

The full tokenizer: `audio_acoustic_proj` linear lifts 64-d acoustic features to 2048-d hidden, then the pooler reduces the time axis by `pool_window_size`, then `ResidualFSQ` quantizes.

- [ ] **Step 1: Create the file**

```swift
// AnvilCore/Sources/AnvilCore/Inference/AudioTokenizer/AudioTokenizer.swift
import Foundation
import MLX
import MLXNN

/// Audio→token encoder. Converts a stream of VAE acoustic latents at 25 Hz
/// into 5 Hz quantized tokens via:
///
///     audio_acoustic_proj  : [B, T, 64]    → [B, T, 2048]
///     reshape windows      : [B, T, 2048]  → [B, T/pool, pool, 2048]
///     attention_pooler     : [B, T/pool, pool, 2048] → [B, T/pool, 2048]
///     residual FSQ         : [B, T/pool, 2048] → quantized [B, T/pool, 2048] + indices [B, T/pool, 1]
///
/// Mirrors upstream `AceStepAudioTokenizer` in
/// `acestep/models/turbo/modeling_acestep_v15_turbo.py:1178`.
public class AudioTokenizer: Module {
    /// `tokenizer.audio_acoustic_proj` — Linear(audio_acoustic_hidden_dim, hidden_size).
    public var audioAcousticProj: Linear
    public let attentionPooler: AudioTokenizerPooler
    public let quantizer: ResidualFSQ
    public let poolWindowSize: Int

    public init(
        audioAcousticHiddenDim: Int = 64,
        hiddenSize: Int = 2048,
        intermediateSize: Int = 6144,
        numAttentionHeads: Int = 16,
        numKeyValueHeads: Int = 8,
        headDim: Int = 128,
        numPoolerLayers: Int = 2,
        poolWindowSize: Int = 5,
        fsqLevels: [Int] = [8, 8, 8, 5, 5, 5],
        rmsNormEps: Float = 1e-6,
        ropeTheta: Float = 1_000_000.0,
        maxPositionEmbeddings: Int = 32768
    ) {
        self.audioAcousticProj = Linear(audioAcousticHiddenDim, hiddenSize, bias: true)
        self.attentionPooler = AudioTokenizerPooler(
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize,
            numAttentionHeads: numAttentionHeads,
            numKeyValueHeads: numKeyValueHeads,
            headDim: headDim,
            numLayers: numPoolerLayers,
            rmsNormEps: rmsNormEps,
            ropeTheta: ropeTheta,
            maxPositionEmbeddings: maxPositionEmbeddings
        )
        self.quantizer = ResidualFSQ(hiddenSize: hiddenSize, levels: fsqLevels)
        self.poolWindowSize = poolWindowSize
    }

    /// Tokenize a continuous acoustic stream.
    /// - Parameter audioLatents: `[B, T, audio_acoustic_hidden_dim]` where T must
    ///   be a multiple of `poolWindowSize`. The caller is responsible for
    ///   padding to that boundary (upstream pads with `silence_latent`).
    /// - Returns: `(quantized, indices)`:
    ///   - `quantized`: `[B, T/pool, hidden_size]` — continuous post-FSQ representation
    ///   - `indices`:   `[B, T/pool, 1]` — packed FSQ codes per window
    public func callAsFunction(_ audioLatents: MLXArray) -> (quantized: MLXArray, indices: MLXArray) {
        let bsz = audioLatents.shape[0]
        let tFrames = audioLatents.shape[1]
        let hSize = audioAcousticProj.weight.shape[0]
        precondition(
            tFrames % poolWindowSize == 0,
            "AudioTokenizer.forward: T (\(tFrames)) must be divisible by pool_window_size (\(poolWindowSize))"
        )

        let projected = audioAcousticProj(audioLatents)                               // [B, T, hSize]
        let nWindows = tFrames / poolWindowSize
        let windowed = projected.reshaped([bsz, nWindows, poolWindowSize, hSize])     // [B, T/pool, pool, H]
        let pooled = attentionPooler(windowed)                                         // [B, T/pool, H]
        return quantizer(pooled)
    }
}
```

- [ ] **Step 2: Build**

Run: `xcodebuild -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | grep -E "error:|BUILD" | tail -5`
Expected: `** BUILD SUCCEEDED **`

- [ ] **Step 3: Commit**

```bash
git add AnvilCore/Sources/AnvilCore/Inference/AudioTokenizer/AudioTokenizer.swift
git commit -m "feat(audio-tokenizer): AudioTokenizer top-level module

Wires audio_acoustic_proj + AudioTokenizerPooler + ResidualFSQ into a
single module that converts [B, T, 64] VAE latents at 25 Hz into
[B, T/pool, 2048] quantized representations + packed indices.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 4: AudioTokenDetokenizer — codes→acoustic upsampler

**Files:**
- Create: `AnvilCore/Sources/AnvilCore/Inference/AudioTokenizer/AudioTokenDetokenizer.swift`

The 5 Hz → 25 Hz upsampler. Takes the LM's hidden states (one per pool window), prepends a Linear projection, expands via 5 learnable position tokens (one per intra-window slot), runs 2 self-attn layers per expanded sequence, and projects back to acoustic dim.

- [ ] **Step 1: Create the file**

```swift
// AnvilCore/Sources/AnvilCore/Inference/AudioTokenizer/AudioTokenDetokenizer.swift
import Foundation
import MLX
import MLXNN

/// Codes→acoustic upsampler. Mirrors upstream `AudioTokenDetokenizer` in
/// `acestep/models/turbo/modeling_acestep_v15_turbo.py:859`.
///
/// Input is a 5 Hz stream (one token per pool window), output is the 25 Hz
/// acoustic-feature stream:
///
///     embed_tokens          : [B, T_5Hz, hidden] → [B, T_5Hz, hidden]
///     unsqueeze + repeat    : → [B, T_5Hz, pool, hidden]
///     + special_tokens      : add learnable per-slot position tokens
///                              → [B, T_5Hz, pool, hidden]
///     reshape (b t p h)→(bt p h): for window-local self-attention
///     2 transformer layers   : self-attention within each window
///     norm                  : RMSNorm
///     proj_out              : [..., hidden] → [..., audio_acoustic_hidden_dim]
///     reshape back          : → [B, T_5Hz * pool, audio_acoustic_hidden_dim]
public class AudioTokenDetokenizer: Module {
    /// `detokenizer.embed_tokens` — Linear(hidden_size, hidden_size).
    public var embedTokens: Linear
    /// `detokenizer.special_tokens`, shape `[1, pool_window_size, hidden_size]`.
    public var specialTokens: MLXArray
    public let layers: [Qwen3Layer]
    public let norm: RMSNorm
    public let rotaryEmbedding: RotaryEmbedding
    /// `detokenizer.proj_out` — Linear(hidden_size, audio_acoustic_hidden_dim).
    public var projOut: Linear

    public let poolWindowSize: Int
    public let hiddenSize: Int

    public init(
        hiddenSize: Int = 2048,
        intermediateSize: Int = 6144,
        numAttentionHeads: Int = 16,
        numKeyValueHeads: Int = 8,
        headDim: Int = 128,
        numLayers: Int = 2,
        poolWindowSize: Int = 5,
        audioAcousticHiddenDim: Int = 64,
        rmsNormEps: Float = 1e-6,
        ropeTheta: Float = 1_000_000.0,
        maxPositionEmbeddings: Int = 32768
    ) {
        self.hiddenSize = hiddenSize
        self.poolWindowSize = poolWindowSize
        self.embedTokens = Linear(hiddenSize, hiddenSize, bias: true)
        self.specialTokens = MLXArray.zeros([1, poolWindowSize, hiddenSize])
        self.layers = (0..<numLayers).map { _ in
            Qwen3Layer(
                hiddenSize: hiddenSize,
                intermediateSize: intermediateSize,
                numAttentionHeads: numAttentionHeads,
                numKeyValueHeads: numKeyValueHeads,
                headDim: headDim,
                rmsNormEps: rmsNormEps
            )
        }
        self.norm = RMSNorm(dimensions: hiddenSize, eps: rmsNormEps)
        self.rotaryEmbedding = RotaryEmbedding(
            headDim: headDim,
            maxLen: maxPositionEmbeddings,
            base: ropeTheta
        )
        self.projOut = Linear(hiddenSize, audioAcousticHiddenDim, bias: true)
    }

    /// Detokenize 5 Hz hidden states into 25 Hz acoustic features.
    /// - Parameter lmHints5Hz: `[B, T_5Hz, hidden_size]`.
    /// - Returns: `[B, T_5Hz * pool_window_size, audio_acoustic_hidden_dim]`.
    public func callAsFunction(_ lmHints5Hz: MLXArray) -> MLXArray {
        let bsz = lmHints5Hz.shape[0]
        let t5Hz = lmHints5Hz.shape[1]
        let hSize = self.hiddenSize
        let pool = self.poolWindowSize

        // Linear-project the LM hidden states
        var x = embedTokens(lmHints5Hz)                                                // [B, T, H]

        // Expand each token into `pool` patches and add learnable special_tokens
        x = x.expandedDimensions(axis: 2)                                              // [B, T, 1, H]
        x = MLX.broadcast(x, to: [bsz, t5Hz, pool, hSize])                             // [B, T, pool, H]
        let special = MLX.broadcast(specialTokens, to: [bsz, t5Hz, pool, hSize])
        x = x + special

        // Reshape so each window's `pool` patches are processed independently.
        // (b t p h) → (b*t p h)
        var h = x.reshaped([bsz * t5Hz, pool, hSize])

        let seqLen = h.shape[1]
        let positionCosSin = rotaryEmbedding.forward(seqLen: seqLen)
        for layer in layers {
            h = layer(
                hiddenStates: h,
                positionCosSin: positionCosSin,
                attentionMask: nil
            )
        }
        h = norm(h)
        let acoustic = projOut(h)                                                       // [B*T, pool, 64]

        // Reshape back to a flat 25 Hz stream
        let outDim = acoustic.shape[2]
        return acoustic.reshaped([bsz, t5Hz * pool, outDim])
    }
}
```

- [ ] **Step 2: Build**

Run: `xcodebuild -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | grep -E "error:|BUILD" | tail -5`
Expected: `** BUILD SUCCEEDED **`

- [ ] **Step 3: Commit**

```bash
git add AnvilCore/Sources/AnvilCore/Inference/AudioTokenizer/AudioTokenDetokenizer.swift
git commit -m "feat(audio-tokenizer): AudioTokenDetokenizer 5Hz → 25Hz upsampler

Mirrors upstream AudioTokenDetokenizer. Takes LM hidden states (one per
pool window), expands via 5 learnable special_tokens, runs 2 layers of
self-attention within each window, and projects to acoustic dim.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 5: AceStepLMConfig — JSON config decoder

**Files:**
- Create: `AnvilCore/Sources/AnvilCore/Inference/LM/AceStepLMConfig.swift`

The 0.6B and 4B variants have different hidden sizes / layer counts. We can't hardcode constants — we read from each variant's `config.json` after download. Also handles the `tie_word_embeddings` and rope-base fields.

- [ ] **Step 1: Create the file**

```swift
// AnvilCore/Sources/AnvilCore/Inference/LM/AceStepLMConfig.swift
import Foundation

/// Decoded form of an ACE-Step LM variant's `config.json`.
///
/// Both the 0.6B and 4B variants ship a Qwen-architecture LM. The model size
/// differs but the schema is identical, so we drive layer construction from
/// these values rather than hardcoding constants per variant.
public struct AceStepLMConfig: Sendable, Codable {
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let intermediateSize: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let rmsNormEps: Float
    public let vocabSize: Int
    public let maxPositionEmbeddings: Int
    public let ropeTheta: Float

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabSize = "vocab_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
    }

    /// Loads from `config.json` on disk. The path comes from the
    /// `ace-step-lm-*` model spec's `localDirectory(base:)`.
    public static func load(from configURL: URL) throws -> AceStepLMConfig {
        let data = try Data(contentsOf: configURL)
        // Some checkpoints omit head_dim and rope_theta; fall back if needed.
        let raw = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
        let hiddenSize = (raw["hidden_size"] as? Int) ?? 1024
        let numLayers = (raw["num_hidden_layers"] as? Int) ?? 28
        let intermediate = (raw["intermediate_size"] as? Int) ?? 3072
        let nHeads = (raw["num_attention_heads"] as? Int) ?? 16
        let nKV = (raw["num_key_value_heads"] as? Int) ?? 8
        let hDim = (raw["head_dim"] as? Int) ?? (hiddenSize / nHeads)
        let eps = (raw["rms_norm_eps"] as? Double).map(Float.init) ?? 1e-6
        let vocab = (raw["vocab_size"] as? Int) ?? 64003
        let maxPos = (raw["max_position_embeddings"] as? Int) ?? 32768
        let rope = (raw["rope_theta"] as? Double).map(Float.init) ?? 1_000_000.0

        return AceStepLMConfig(
            hiddenSize: hiddenSize,
            numHiddenLayers: numLayers,
            intermediateSize: intermediate,
            numAttentionHeads: nHeads,
            numKeyValueHeads: nKV,
            headDim: hDim,
            rmsNormEps: eps,
            vocabSize: vocab,
            maxPositionEmbeddings: maxPos,
            ropeTheta: rope
        )
    }
}
```

- [ ] **Step 2: Build**

Run: `xcodebuild -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | grep -E "error:|BUILD" | tail -5`
Expected: `** BUILD SUCCEEDED **`

- [ ] **Step 3: Commit**

```bash
git add AnvilCore/Sources/AnvilCore/Inference/LM/AceStepLMConfig.swift
git commit -m "feat(lm): AceStepLMConfig decodes Qwen-style config.json

Reads hidden_size / num_hidden_layers / etc from each LM variant's
config.json on disk. Both 0.6B and 4B ship the same schema with
different sizes — the constructor is data-driven rather than hardcoded
per variant.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 6: AceStepLM — the Qwen-architecture audio LM

**Files:**
- Create: `AnvilCore/Sources/AnvilCore/Inference/LM/AceStepLM.swift`

Decoder-only transformer. Reuses our existing `Qwen3Layer`. No `lm_head` — we want hidden states (which feed the detokenizer), not next-token logits.

- [ ] **Step 1: Create the file**

```swift
// AnvilCore/Sources/AnvilCore/Inference/LM/AceStepLM.swift
import Foundation
import MLX
import MLXNN

/// 5 Hz Audio→Token Language Model used to produce LM hints from FSQ-coded
/// audio tokens. Architecture is Qwen-style decoder-only (we reuse our
/// existing Qwen3Layer). We expose final hidden states rather than next-token
/// logits — those hidden states feed AudioTokenDetokenizer to upsample to
/// 25 Hz acoustic features.
public class AceStepLM: Module {
    public var embedTokens: Embedding
    public let layers: [Qwen3Layer]
    public let norm: RMSNorm
    public let rotaryEmbedding: RotaryEmbedding

    public let hiddenSize: Int

    public init(config: AceStepLMConfig) {
        self.hiddenSize = config.hiddenSize
        self.embedTokens = Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.hiddenSize
        )
        self.layers = (0..<config.numHiddenLayers).map { _ in
            Qwen3Layer(
                hiddenSize: config.hiddenSize,
                intermediateSize: config.intermediateSize,
                numAttentionHeads: config.numAttentionHeads,
                numKeyValueHeads: config.numKeyValueHeads,
                headDim: config.headDim,
                rmsNormEps: config.rmsNormEps
            )
        }
        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.rotaryEmbedding = RotaryEmbedding(
            headDim: config.headDim,
            maxLen: config.maxPositionEmbeddings,
            base: config.ropeTheta
        )
    }

    /// Forward through the LM and return final hidden states.
    /// - Parameter audioCodes: `[B, T_5Hz]` integer FSQ indices.
    /// - Parameter attentionMask: `[B, T_5Hz]` 1.0 for valid, 0.0 for PAD.
    /// - Returns: `[B, T_5Hz, hidden_size]` final hidden states (post-norm).
    public func callAsFunction(audioCodes: MLXArray, attentionMask: MLXArray?) -> MLXArray {
        var h = embedTokens(audioCodes)                                  // [B, T, H]
        let seqLen = h.shape[1]
        let positionCosSin = rotaryEmbedding.forward(seqLen: seqLen)

        // Convert padding mask to additive bias for attention.
        let additiveMask: MLXArray? = attentionMask.map { mask in
            let inverted = (1.0 - mask) * Float(-1e9)
            return inverted
        }

        for layer in layers {
            h = layer(
                hiddenStates: h,
                positionCosSin: positionCosSin,
                attentionMask: additiveMask
            )
        }
        return norm(h)
    }
}
```

- [ ] **Step 2: Build**

Run: `xcodebuild -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | grep -E "error:|BUILD" | tail -5`
Expected: `** BUILD SUCCEEDED **`

- [ ] **Step 3: Commit**

```bash
git add AnvilCore/Sources/AnvilCore/Inference/LM/AceStepLM.swift
git commit -m "feat(lm): AceStepLM Qwen-architecture audio language model

Decoder-only transformer reusing Qwen3Layer. Constructed from an
AceStepLMConfig (loaded per-variant from config.json on disk). No
lm_head: we want final hidden states for the detokenizer, not
next-token logits.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 7: WeightConverter — sharded safetensors loading

**Files:**
- Modify: `AnvilCore/Sources/AnvilCore/Inference/Weights/WeightConverter.swift`

Currently `loadSafetensors(from:)` handles a single file. The 4B LM variant ships sharded across `model-00001-of-00002.safetensors` + `model-00002-of-00002.safetensors`, indexed by `model.safetensors.index.json`. Add a method that consumes the index, loads each shard, and merges into one `[(String, MLXArray)]`.

- [ ] **Step 1: Add the new method to WeightConverter**

Find the existing `loadSafetensors(from:)` method (around the top of the file, public actor method). Add this new method right after it inside the same actor:

```swift
/// Load a sharded safetensors checkpoint described by `model.safetensors.index.json`.
///
/// The index file lives alongside the shards (e.g.
/// `<dir>/model.safetensors.index.json`,
/// `<dir>/model-00001-of-00002.safetensors`, ...). We read the
/// `weight_map` field which is `{tensor_name: shard_filename}`, group keys
/// by shard, load each shard once via `MLX.loadArrays`, then concatenate
/// the per-shard `(name, array)` pairs into a single sorted list.
///
/// Falls back to single-file loading if the supplied URL is not an index
/// (returns the same shape as `loadSafetensors(from:)`).
public func loadShardedSafetensors(indexURL: URL) throws -> [(String, MLXArray)] {
    struct SafetensorsIndex: Decodable {
        let weight_map: [String: String]
    }

    let data = try Data(contentsOf: indexURL)
    let index = try JSONDecoder().decode(SafetensorsIndex.self, from: data)
    let dir = indexURL.deletingLastPathComponent()

    // Group keys by shard so we only load each shard file once.
    var byShard: [String: [String]] = [:]
    for (key, shard) in index.weight_map {
        byShard[shard, default: []].append(key)
    }

    var merged: [String: MLXArray] = [:]
    for (shard, keys) in byShard {
        let shardURL = dir.appendingPathComponent(shard)
        let dict = try MLX.loadArrays(url: shardURL)
        for key in keys {
            guard let arr = dict[key] else {
                throw NSError(
                    domain: "WeightConverter",
                    code: 1,
                    userInfo: [NSLocalizedDescriptionKey: "Sharded checkpoint claims \(key) is in \(shard) but the file does not contain it"]
                )
            }
            merged[key] = arr
        }
    }
    return merged.sorted { $0.key < $1.key }.map { ($0.key, $0.value) }
}
```

- [ ] **Step 2: Build**

Run: `xcodebuild -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | grep -E "error:|BUILD" | tail -5`
Expected: `** BUILD SUCCEEDED **`

- [ ] **Step 3: Commit**

```bash
git add AnvilCore/Sources/AnvilCore/Inference/Weights/WeightConverter.swift
git commit -m "feat(weights): WeightConverter.loadShardedSafetensors

Reads model.safetensors.index.json's weight_map, groups keys by shard
filename, loads each shard via MLX.loadArrays, and merges into a single
sorted (name, array) list. Needed for the 4B LM variant which ships
across model-00001-of-00002 + model-00002-of-00002.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 8: AudioTokenizerNameMap + convertAudioTokenizerWeights

**Files:**
- Create: `AnvilCore/Sources/AnvilCore/Inference/Weights/AudioTokenizerNameMap.swift`
- Modify: `AnvilCore/Sources/AnvilCore/Inference/Weights/WeightConverter.swift`

Maps the DiT bundle's `tokenizer.*` keys to the Swift parameter paths. The mapping rules:
- Strip the `tokenizer.` prefix.
- Convert each path component snake_case → camelCase, preserving numeric layer indices.
- `embed_tokens` → `embedTokens`, `attention_pooler` → `attentionPooler`, `audio_acoustic_proj` → `audioAcousticProj`, `special_token` → `specialToken`, `project_in` / `project_out` → `projectIn` / `projectOut`. (All handled by the generic snake→camel pass; no special cases needed.)
- The pooler's transformer-layer sub-keys (`self_attn.q_proj`, `mlp.down_proj`, `input_layernorm`, etc.) need the same renames as our other Qwen-style mappers — reuse the helper from `Qwen3NameMap` if needed, or just inline.

- [ ] **Step 1: Create the name-map file**

```swift
// AnvilCore/Sources/AnvilCore/Inference/Weights/AudioTokenizerNameMap.swift
import Foundation

public enum AudioTokenizerNameMap {
    /// Maps `tokenizer.*` keys from the DiT bundle to Swift parameter paths
    /// on the AudioTokenizer module. Returns nil for keys outside the
    /// `tokenizer.` subtree.
    public static func mapKey(_ key: String) -> String? {
        guard key.hasPrefix("tokenizer.") else { return nil }
        let stripped = String(key.dropFirst("tokenizer.".count))

        // Per-component snake→camel, preserving numeric layer indices and the
        // raw `special_token` -> `specialToken` rename which is just snake→camel.
        let components = stripped.split(separator: ".", omittingEmptySubsequences: false).map { component -> String in
            let s = String(component)
            if Int(s) != nil { return s }
            return snakeToCamel(s)
        }
        var mapped = components.joined(separator: ".")

        // Fix Qwen-style sub-paths inside attention_pooler.layers.N.* — these
        // need the same self_attn → attention rename used by Qwen3NameMap, so
        // they line up with our existing Qwen3Layer property names.
        mapped = mapped
            .replacingOccurrences(of: ".selfAttn.qProj", with: ".attention.qProj")
            .replacingOccurrences(of: ".selfAttn.kProj", with: ".attention.kProj")
            .replacingOccurrences(of: ".selfAttn.vProj", with: ".attention.vProj")
            .replacingOccurrences(of: ".selfAttn.oProj", with: ".attention.oProj")
            .replacingOccurrences(of: ".selfAttn.qNorm", with: ".attention.qNorm")
            .replacingOccurrences(of: ".selfAttn.kNorm", with: ".attention.kNorm")
            .replacingOccurrences(of: ".inputLayernorm", with: ".inputNorm")
            .replacingOccurrences(of: ".postAttentionLayernorm", with: ".postNorm")

        return mapped
    }

    private static func snakeToCamel(_ s: String) -> String {
        let parts = s.split(separator: "_")
        guard let first = parts.first else { return s }
        var result = String(first)
        for part in parts.dropFirst() {
            result += part.prefix(1).uppercased() + part.dropFirst()
        }
        return result
    }
}
```

- [ ] **Step 2: Add the converter method on `WeightConverter`**

Find where `convertConditionEncoderWeights` is defined in `WeightConverter.swift` and add this right below it:

```swift
public func convertAudioTokenizerWeights(
    from rawWeights: [(String, MLXArray)],
    progress: ((ConversionProgress) -> Void)? = nil
) -> [(String, MLXArray)] {
    var result: [(String, MLXArray)] = []
    let total = rawWeights.count

    for (index, (key, value)) in rawWeights.enumerated() {
        progress?(ConversionProgress(current: index + 1, total: total, currentTensorName: key))
        guard let mappedKey = AudioTokenizerNameMap.mapKey(key) else { continue }
        result.append((mappedKey, value))
    }

    return result
}
```

- [ ] **Step 3: Build**

Run: `xcodebuild -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | grep -E "error:|BUILD" | tail -5`
Expected: `** BUILD SUCCEEDED **`

- [ ] **Step 4: Commit**

```bash
git add AnvilCore/Sources/AnvilCore/Inference/Weights/AudioTokenizerNameMap.swift \
        AnvilCore/Sources/AnvilCore/Inference/Weights/WeightConverter.swift
git commit -m "feat(weights): AudioTokenizerNameMap + convertAudioTokenizerWeights

Strips the tokenizer. prefix and converts snake_case → camelCase per
path component, plus the Qwen-style self_attn → attention and
input_layernorm → inputNorm / post_attention_layernorm → postNorm
renames so the pooler's transformer layers line up with our Qwen3Layer
property paths.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 9: AudioTokenDetokenizerNameMap + convertAudioTokenDetokenizerWeights

**Files:**
- Create: `AnvilCore/Sources/AnvilCore/Inference/Weights/AudioTokenDetokenizerNameMap.swift`
- Modify: `AnvilCore/Sources/AnvilCore/Inference/Weights/WeightConverter.swift`

Same pattern as the tokenizer map but for the `detokenizer.` subtree. The detokenizer has `special_tokens` (plural — 5 tokens), `embed_tokens` (Linear), `proj_out`, plus 2 self-attn layers.

- [ ] **Step 1: Create the name-map file**

```swift
// AnvilCore/Sources/AnvilCore/Inference/Weights/AudioTokenDetokenizerNameMap.swift
import Foundation

public enum AudioTokenDetokenizerNameMap {
    /// Maps `detokenizer.*` keys from the DiT bundle to Swift parameter
    /// paths on the AudioTokenDetokenizer module. Returns nil for keys
    /// outside the `detokenizer.` subtree.
    public static func mapKey(_ key: String) -> String? {
        guard key.hasPrefix("detokenizer.") else { return nil }
        let stripped = String(key.dropFirst("detokenizer.".count))

        let components = stripped.split(separator: ".", omittingEmptySubsequences: false).map { component -> String in
            let s = String(component)
            if Int(s) != nil { return s }
            return snakeToCamel(s)
        }
        var mapped = components.joined(separator: ".")

        // Same Qwen3Layer-internal renames as AudioTokenizerNameMap.
        mapped = mapped
            .replacingOccurrences(of: ".selfAttn.qProj", with: ".attention.qProj")
            .replacingOccurrences(of: ".selfAttn.kProj", with: ".attention.kProj")
            .replacingOccurrences(of: ".selfAttn.vProj", with: ".attention.vProj")
            .replacingOccurrences(of: ".selfAttn.oProj", with: ".attention.oProj")
            .replacingOccurrences(of: ".selfAttn.qNorm", with: ".attention.qNorm")
            .replacingOccurrences(of: ".selfAttn.kNorm", with: ".attention.kNorm")
            .replacingOccurrences(of: ".inputLayernorm", with: ".inputNorm")
            .replacingOccurrences(of: ".postAttentionLayernorm", with: ".postNorm")

        return mapped
    }

    private static func snakeToCamel(_ s: String) -> String {
        let parts = s.split(separator: "_")
        guard let first = parts.first else { return s }
        var result = String(first)
        for part in parts.dropFirst() {
            result += part.prefix(1).uppercased() + part.dropFirst()
        }
        return result
    }
}
```

- [ ] **Step 2: Add the converter on `WeightConverter`**

Add right after `convertAudioTokenizerWeights`:

```swift
public func convertAudioTokenDetokenizerWeights(
    from rawWeights: [(String, MLXArray)],
    progress: ((ConversionProgress) -> Void)? = nil
) -> [(String, MLXArray)] {
    var result: [(String, MLXArray)] = []
    let total = rawWeights.count

    for (index, (key, value)) in rawWeights.enumerated() {
        progress?(ConversionProgress(current: index + 1, total: total, currentTensorName: key))
        guard let mappedKey = AudioTokenDetokenizerNameMap.mapKey(key) else { continue }
        result.append((mappedKey, value))
    }

    return result
}
```

- [ ] **Step 3: Build**

Run: `xcodebuild -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | grep -E "error:|BUILD" | tail -5`
Expected: `** BUILD SUCCEEDED **`

- [ ] **Step 4: Commit**

```bash
git add AnvilCore/Sources/AnvilCore/Inference/Weights/AudioTokenDetokenizerNameMap.swift \
        AnvilCore/Sources/AnvilCore/Inference/Weights/WeightConverter.swift
git commit -m "feat(weights): AudioTokenDetokenizerNameMap + converter

Mirrors AudioTokenizerNameMap for the detokenizer.* subtree. Same
prefix strip + snake→camel + Qwen3Layer-internal renames.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 10: AceStepLMNameMap + convertAceStepLMWeights

**Files:**
- Create: `AnvilCore/Sources/AnvilCore/Inference/Weights/AceStepLMNameMap.swift`
- Modify: `AnvilCore/Sources/AnvilCore/Inference/Weights/WeightConverter.swift`

The LM safetensors keys are nearly identical to Qwen3-Embedding-0.6B (`embed_tokens`, `layers.N.*`, `norm`). Different from `Qwen3NameMap` only in that `embed_tokens` maps to `embedTokens` (Embedding) here, while in our text-encoder path we mapped it to `embeddingTable`. These are purely property-name choices — see the `AceStepLM` definition in Task 6 for the property name we used.

- [ ] **Step 1: Create the name-map file**

```swift
// AnvilCore/Sources/AnvilCore/Inference/Weights/AceStepLMNameMap.swift
import Foundation

public enum AceStepLMNameMap {
    /// Maps a key from an ACE-Step LM variant's safetensors to the parameter
    /// path expected by our Swift `AceStepLM` module. Both 0.6B (single file)
    /// and 4B (sharded) ship the same Qwen-style key naming.
    ///
    /// No `lm_head` filter — these checkpoints were saved without one
    /// (verified for 0.6B; 4B follows the same convention).
    public static func mapKey(_ key: String) -> String? {
        var mapped = key

        // Strip the optional model. prefix some Qwen checkpoints carry.
        if mapped.hasPrefix("model.") {
            mapped = String(mapped.dropFirst("model.".count))
        }

        // Filter lm_head if it ever appears.
        if mapped.hasPrefix("lm_head.") { return nil }

        // embed_tokens → embedTokens (note: AceStepLM uses Embedding, not
        // Linear, so the parameter path is .embedTokens.weight directly).
        mapped = mapped.replacingOccurrences(of: "embed_tokens.", with: "embedTokens.")

        // Standard Qwen-style sub-renames.
        mapped = mapped
            .replacingOccurrences(of: ".self_attn.q_proj", with: ".attention.qProj")
            .replacingOccurrences(of: ".self_attn.k_proj", with: ".attention.kProj")
            .replacingOccurrences(of: ".self_attn.v_proj", with: ".attention.vProj")
            .replacingOccurrences(of: ".self_attn.o_proj", with: ".attention.oProj")
            .replacingOccurrences(of: ".self_attn.q_norm", with: ".attention.qNorm")
            .replacingOccurrences(of: ".self_attn.k_norm", with: ".attention.kNorm")
            .replacingOccurrences(of: ".mlp.gate_proj", with: ".mlp.gateProj")
            .replacingOccurrences(of: ".mlp.up_proj", with: ".mlp.upProj")
            .replacingOccurrences(of: ".mlp.down_proj", with: ".mlp.downProj")
            .replacingOccurrences(of: ".input_layernorm", with: ".inputNorm")
            .replacingOccurrences(of: ".post_attention_layernorm", with: ".postNorm")

        return mapped
    }
}
```

- [ ] **Step 2: Add the converter on `WeightConverter`**

```swift
public func convertAceStepLMWeights(
    from rawWeights: [(String, MLXArray)],
    progress: ((ConversionProgress) -> Void)? = nil
) -> [(String, MLXArray)] {
    var result: [(String, MLXArray)] = []
    let total = rawWeights.count

    for (index, (key, value)) in rawWeights.enumerated() {
        progress?(ConversionProgress(current: index + 1, total: total, currentTensorName: key))
        guard let mappedKey = AceStepLMNameMap.mapKey(key) else { continue }
        result.append((mappedKey, value))
    }

    return result
}
```

- [ ] **Step 3: Build**

Run: `xcodebuild -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | grep -E "error:|BUILD" | tail -5`
Expected: `** BUILD SUCCEEDED **`

- [ ] **Step 4: Commit**

```bash
git add AnvilCore/Sources/AnvilCore/Inference/Weights/AceStepLMNameMap.swift \
        AnvilCore/Sources/AnvilCore/Inference/Weights/WeightConverter.swift
git commit -m "feat(weights): AceStepLMNameMap + converter

Maps Qwen-style LM safetensors keys to AceStepLM property paths. Mirror
of Qwen3NameMap with embed_tokens → embedTokens (Embedding, not
embeddingTable like the text encoder uses) and standard self_attn /
mlp / layernorm renames.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 11: mapDiTKey extension — keep tokenizer.* and detokenizer.* keys

**Files:**
- Modify: `AnvilCore/Sources/AnvilCore/Inference/Weights/TensorNameMap.swift`

The current `mapDiTKey` returns nil for everything outside `decoder.` (with one exception for the top-level `null_condition_emb`). The DiT bundle's `tokenizer.*` and `detokenizer.*` keys get filtered out — fine when nobody loaded them, but Phase 2A needs them to flow through to their dedicated converters.

The cleanest move: the DiT loader in `ACEStepGenerationService.loadDiTWeights` continues using `mapDiTKey` (which still drops tokenizer/detokenizer/encoder), and a new `CoverSongCoordinator` (Phase 2B) calls `WeightConverter.convertAudioTokenizerWeights` / `convertAudioTokenDetokenizerWeights` on the same `rawWeights`. Those converters don't need `mapDiTKey` to do anything special — they have their own prefix-aware maps.

So `mapDiTKey` actually doesn't need to change for Phase 2A. The audio-tokenizer / detokenizer converters work on raw safetensors keys directly (they expect `tokenizer.*` / `detokenizer.*` as input). This task is essentially a no-op verification.

- [ ] **Step 1: Verify `mapDiTKey` already filters `tokenizer.*` / `detokenizer.*` to nil and that Phase 2B can load them via the new converters**

Read the file and confirm the expected behavior:

```bash
cd ~/Developer/GitRepos/AnvilApp
grep -n 'guard.*hasPrefix("decoder' AnvilCore/Sources/AnvilCore/Inference/Weights/TensorNameMap.swift
```

Expected output: a single match showing the prefix guard. If found, the existing behavior is correct: `tokenizer.foo` and `detokenizer.foo` already return nil from `mapDiTKey`, so the existing DiT loader silently drops them. The new audio-tokenizer / detokenizer converters work on the same `rawWeights` independently and pick out only the keys they care about. No change needed here.

- [ ] **Step 2: Add a comment to `TensorNameMap.swift` explaining the contract**

Open `AnvilCore/Sources/AnvilCore/Inference/Weights/TensorNameMap.swift` and find the doc comment above `mapDiTKey`. Append this paragraph at the end of the doc comment (right before `public static func mapDiTKey`):

```swift
    /// Note: returns nil for the bundle's `tokenizer.*` and `detokenizer.*`
    /// subtrees. Those weights are loaded separately by
    /// `WeightConverter.convertAudioTokenizerWeights` /
    /// `convertAudioTokenDetokenizerWeights` (cover-song generation,
    /// Phase 2A+) which work on the raw safetensors `(key, array)` list and
    /// have their own prefix-aware maps. This means callers can safely
    /// load the same `rawWeights` through both `convertDiTWeights` (DiT
    /// decoder) and `convertAudioTokenizerWeights` (audio tokenizer)
    /// without keys colliding — each converter ignores keys outside its
    /// subtree.
```

- [ ] **Step 3: Build**

Run: `xcodebuild -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | grep -E "error:|BUILD" | tail -5`
Expected: `** BUILD SUCCEEDED **`

- [ ] **Step 4: Commit**

```bash
git add AnvilCore/Sources/AnvilCore/Inference/Weights/TensorNameMap.swift
git commit -m "docs(weights): document mapDiTKey contract for audio tokenizer subtrees

Clarify that mapDiTKey returns nil for tokenizer.* / detokenizer.*
keys — those flow through dedicated AudioTokenizerNameMap /
AudioTokenDetokenizerNameMap converters that work on the raw
safetensors list independently of mapDiTKey.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 12: Smoke test — load all four module weight sets without crashing

**Files:**
- No new files. Pure verification: run a one-off test from the existing test infrastructure or Swift REPL.

This task verifies the full Phase 2A stack actually loads weights end-to-end against the user's local model files. We don't have integration tests in CI (Metal shaders don't compile in SPM), so this is a manual smoke test the engineer runs from a macOS app context.

- [ ] **Step 1: Add a `testCoverSongFoundation()` helper to `ACEStepGenerationService` (temporary, will be removed when Phase 2B lands)**

Open `AnvilApp/Services/ACEStepGenerationService.swift` and add this method anywhere inside the class:

```swift
#if DEBUG
    /// Phase 2A smoke test: load AudioTokenizer, AudioTokenDetokenizer, and
    /// AceStepLM weights, log per-module match counts. Intended to be called
    /// once from a debug menu or via lldb after a generation run.
    /// Will be removed when Phase 2B introduces CoverSongCoordinator.
    func testCoverSongFoundation() async {
        guard let ditSpec = await registry.activeSpec(for: .aceStepDiT) else {
            logger.warning("[cover-smoke] no active DiT spec")
            return
        }
        let baseDir = await registry.baseDirectory
        let ditWeightsPath = ditSpec.localPath(for: "model.safetensors", base: baseDir)

        do {
            let raw = try await weightConverter.loadSafetensors(from: ditWeightsPath)

            // Audio tokenizer
            let tokWeights = await weightConverter.convertAudioTokenizerWeights(from: raw)
            let tokenizer = AudioTokenizer()
            let tokModelKeys = Set(tokenizer.parameters().flattened().map { $0.0 })
            let tokSuppliedKeys = Set(tokWeights.map { $0.0 })
            logger.info("[cover-smoke] tokenizer supplied=\(tokSuppliedKeys.count) model=\(tokModelKeys.count) matched=\(tokSuppliedKeys.intersection(tokModelKeys).count)")
            tokenizer.update(parameters: ModuleParameters.unflattened(tokWeights))
            eval(tokenizer.parameters())

            // Audio detokenizer
            let detokWeights = await weightConverter.convertAudioTokenDetokenizerWeights(from: raw)
            let detok = AudioTokenDetokenizer()
            let detokModelKeys = Set(detok.parameters().flattened().map { $0.0 })
            let detokSuppliedKeys = Set(detokWeights.map { $0.0 })
            logger.info("[cover-smoke] detokenizer supplied=\(detokSuppliedKeys.count) model=\(detokModelKeys.count) matched=\(detokSuppliedKeys.intersection(detokModelKeys).count)")
            detok.update(parameters: ModuleParameters.unflattened(detokWeights))
            eval(detok.parameters())

            // LM (try 0.6B first, fall back to 4B if only that one is downloaded)
            let lmIDs = ["ace-step-lm-0.6b", "ace-step-lm-4b"]
            var lmLoaded = false
            for lmID in lmIDs {
                guard let lmSpec = await registry.allSpecs.first(where: { $0.id == lmID }) else { continue }
                let lmDir = lmSpec.localDirectory(base: baseDir)
                let configURL = lmDir.appending(path: "config.json")
                guard FileManager.default.fileExists(atPath: configURL.path(percentEncoded: false)) else { continue }
                let config = try AceStepLMConfig.load(from: configURL)

                let weightsURL = lmDir.appending(path: "model.safetensors")
                let indexURL = lmDir.appending(path: "model.safetensors.index.json")
                let lmRaw: [(String, MLXArray)]
                if FileManager.default.fileExists(atPath: indexURL.path(percentEncoded: false)) {
                    lmRaw = try await weightConverter.loadShardedSafetensors(indexURL: indexURL)
                } else if FileManager.default.fileExists(atPath: weightsURL.path(percentEncoded: false)) {
                    lmRaw = try await weightConverter.loadSafetensors(from: weightsURL)
                } else {
                    continue
                }
                let lmWeights = await weightConverter.convertAceStepLMWeights(from: lmRaw)
                let lm = AceStepLM(config: config)
                let lmModelKeys = Set(lm.parameters().flattened().map { $0.0 })
                let lmSuppliedKeys = Set(lmWeights.map { $0.0 })
                logger.info("[cover-smoke] LM \(lmID) supplied=\(lmSuppliedKeys.count) model=\(lmModelKeys.count) matched=\(lmSuppliedKeys.intersection(lmModelKeys).count)")
                lm.update(parameters: ModuleParameters.unflattened(lmWeights))
                eval(lm.parameters())
                lmLoaded = true
                break
            }
            if !lmLoaded {
                logger.warning("[cover-smoke] no LM variant available; download one from Settings → Models → ACE-Step LM")
            }
        } catch {
            logger.error("[cover-smoke] failed: \(error.localizedDescription)")
        }
    }
#endif
```

The required Swift imports for this method (`AnvilCore` types like `AudioTokenizer`, `AudioTokenDetokenizer`, `AceStepLM`, `AceStepLMConfig`) are already covered by the existing `import AnvilCore` at the top of `ACEStepGenerationService.swift`.

- [ ] **Step 2: Build**

Run: `xcodebuild -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | grep -E "error:|BUILD" | tail -5`
Expected: `** BUILD SUCCEEDED **`

If the build fails because the new types aren't `public`, return to Tasks 3, 4, 6 and ensure the `AudioTokenizer`, `AudioTokenDetokenizer`, `AceStepLM`, and `AceStepLMConfig` types and their `init`s are declared `public`. Re-run the build.

- [ ] **Step 3: Trigger the smoke test from the running app via lldb**

After running the app once (so a DiT model is active), pause execution in Xcode and run from the lldb prompt:

```
po await ACEStepGenerationService.testCoverSongFoundation()
```

Or simpler — temporarily call `await self.testCoverSongFoundation()` once at the top of `loadDiTWeights()` body (right after `guard let spec`), launch, generate once, then revert.

Expected log output (numbers are approximate; what matters is matched ≈ supplied):

```
[cover-smoke] tokenizer supplied=32 model=33 matched=32
[cover-smoke] detokenizer supplied=28 model=29 matched=28
[cover-smoke] LM ace-step-lm-0.6b supplied=310 model=311 matched=310
```

The 1-key gap on the model side in each row corresponds to a `rotaryEmbedding.invFreq` runtime buffer that doesn't get supplied — same pattern as DiT/Qwen3.

- [ ] **Step 4: Commit the smoke test (with note that it'll be replaced by Phase 2B's coordinator)**

```bash
git add AnvilApp/Services/ACEStepGenerationService.swift
git commit -m "test(cover-song): testCoverSongFoundation smoke test for Phase 2A

DEBUG-only method on ACEStepGenerationService that loads the audio
tokenizer + detokenizer + LM weights and logs per-module match counts.
Intended for one-off verification that the Phase 2A foundation can
actually load the user's downloaded models. Will be replaced by
CoverSongCoordinator's loadWeights() in Phase 2B.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 13: CHANGELOG entry

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add a new section at the top of CHANGELOG.md**

Open `CHANGELOG.md`. Find the line right under the top heading (`# Changelog`) and add a new section before any existing `[Unreleased]` block:

```markdown
## [Unreleased] — Cover-Song Generation Phase 2A (foundation)

### Added

- **`ResidualFSQ`** (`AnvilCore/Inference/AudioTokenizer/`) — Finite Scalar Quantization with `project_in` / `project_out` linear bottleneck. Mirrors upstream's `vector_quantize_pytorch.ResidualFSQ` configured with `num_quantizers=1`. Provides `quantize` (continuous → indices) and `getOutputFromIndices` (indices → continuous) for both encode and skip-encode paths.
- **`AudioTokenizerPooler`** — cls-token AttentionPooler used inside `AudioTokenizer`. Reuses `Qwen3Layer` for its 2 self-attention transformer blocks. Pools `pool_window_size` audio patches into a single token via prepended cls + self-attention.
- **`AudioTokenizer`** — top-level acoustic→codes module. Composes `audio_acoustic_proj` Linear + `AudioTokenizerPooler` + `ResidualFSQ`. Converts 25 Hz `[B, T, 64]` VAE latents into 5 Hz `[B, T/pool, 2048]` quantized representations + packed indices.
- **`AudioTokenDetokenizer`** — codes→acoustic upsampler. Linear projection + 5 learnable special_tokens for window expansion + 2 self-attention layers per window + `proj_out` back to acoustic dim.
- **`AceStepLMConfig`** — JSON decoder for each LM variant's `config.json` (hidden_size / num_hidden_layers / etc). Drives variant-aware module construction; no hardcoded constants per variant.
- **`AceStepLM`** — Qwen-architecture decoder-only transformer reusing `Qwen3Layer`. Returns final hidden states (no `lm_head`). Used in cover-song path to convert audio codes → 5 Hz hidden states for the detokenizer.
- **`WeightConverter.loadShardedSafetensors(indexURL:)`** — reads `model.safetensors.index.json`'s `weight_map`, groups keys by shard, loads each shard once, and merges into a single sorted `(name, array)` list. Required for the 4B LM variant which ships across 2 safetensors files.
- **Three new weight name maps**: `AudioTokenizerNameMap`, `AudioTokenDetokenizerNameMap`, `AceStepLMNameMap`. Each strips its own subtree prefix and applies the standard snake_case → camelCase conversion plus the Qwen-style sub-renames (`self_attn` → `attention`, `input_layernorm` → `inputNorm`, etc.).
- **Three new `WeightConverter` methods** — `convertAudioTokenizerWeights`, `convertAudioTokenDetokenizerWeights`, `convertAceStepLMWeights`. All actor-isolated, called via `await`.
- **DEBUG-only smoke test** `ACEStepGenerationService.testCoverSongFoundation()` — verifies all three module weight sets load against the user's local files and logs match counts. Will be removed when Phase 2B's `CoverSongCoordinator` lands.

### Documentation

- `mapDiTKey` doc comment now explicitly notes that `tokenizer.*` and `detokenizer.*` keys are intentionally returned as nil; their loading happens through dedicated converters that work on the raw safetensors list independently.

Phase 2B (CoverSongCoordinator + ConditioningActor hook), 2C (UI), and 2D (polish) are deferred to separate plans.
```

- [ ] **Step 2: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs: changelog for cover-song Phase 2A foundation

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Self-Review (post-write check)

**Spec coverage:** 12 tasks cover Phase 2A's spec section 6.1 ("Foundation") items 1-9 cleanly:
- spec item 1 (silence_latent.pt) — deferred to Phase 2B; the foundation modules don't actually consume silence_latent (it's a runtime input, not a parameter), so no Phase 2A work needed
- spec item 2 (mapDiTKey extension) — Task 11 (clarified as a no-op + doc)
- spec item 3 (FSQQuantizer) — Task 1
- spec item 4 (AudioTokenizer) — Tasks 2 + 3
- spec item 5 (AudioTokenDetokenizer) — Task 4
- spec item 6 (AceStepLMConfig) — Task 5
- spec item 7 (AceStepLM) — Task 6
- spec item 8 (sharded safetensors loading) — Task 7
- spec item 9 (AceStepLMNameMap) — Task 10
- spec item 10 (unit tests) — replaced with the smoke test in Task 12 since SPM tests don't run

**Placeholders:** none — every code step has full code, every command has expected output, every file path is exact.

**Type consistency check:**
- `AudioTokenizer` declares `audioAcousticProj: Linear`, `attentionPooler: AudioTokenizerPooler`, `quantizer: ResidualFSQ`. Used by `convertAudioTokenizerWeights` via `AudioTokenizerNameMap`. Map produces `audioAcousticProj.*` / `attentionPooler.*` / `quantizer.*` paths. Match. ✅
- `AudioTokenizerPooler` declares `embedTokens`, `specialToken`, `layers: [Qwen3Layer]`, `norm`, `rotaryEmbedding`. Map produces `attentionPooler.embedTokens.*` / `attentionPooler.specialToken` / `attentionPooler.layers.N.*` / `attentionPooler.norm.*`. Match. ✅
- `AudioTokenDetokenizer` declares `embedTokens`, `specialTokens` (plural — matches safetensors `detokenizer.special_tokens` plural), `layers`, `norm`, `rotaryEmbedding`, `projOut`. Map produces matching paths. ✅
- `AceStepLM` declares `embedTokens` (Embedding), `layers`, `norm`, `rotaryEmbedding`. Map's `embed_tokens.` → `embedTokens.` matches. ✅
- `Qwen3Layer` property paths (referenced by all three name-map sub-renames) — verified the existing module has `attention.qProj/kProj/vProj/oProj/qNorm/kNorm`, `inputNorm`, `postNorm`, `mlp.gateProj/upProj/downProj`. All three new maps target those names consistently. ✅

**Numeric layer index handling:** AudioTokenizerNameMap and AudioTokenDetokenizerNameMap use `if Int(s) != nil { return s }` to preserve `0`, `1` etc. AceStepLMNameMap uses substring `replacingOccurrences` and never touches numeric components. All three handle layer indices correctly. ✅

---

## Plan complete

Plan saved to `docs/superpowers/plans/2026-05-07-cover-song-phase-2a.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. Best for plans of this length (12 tasks) since context stays clean per task.

**2. Inline Execution** — Execute tasks in this session using `superpowers:executing-plans`, batch execution with checkpoints for review.

Which approach?
