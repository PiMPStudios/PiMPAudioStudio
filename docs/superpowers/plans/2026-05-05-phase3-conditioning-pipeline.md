# Phase 3: Conditioning Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the text/lyric/timbre conditioning pipeline in native MLX-Swift that transforms raw text prompts, lyrics, and optional reference audio into the `encoderStates` and `contextLatents` tensors the Phase 2 `MusicPipeline` already accepts.

**Architecture:** Bottom-up build: add swift-tokenizers dependency, then implement Qwen3 transformer components (reusing Phase 2 RotaryEmbedding/RMSNorm/SwiGLUMLP), LyricEncoder, TimbreEncoder with attention pooler, ConditionEncoder orchestrator, and finally the top-level ConditioningActor. Each encoder is a standalone `Module` class tested in isolation.

**Tech Stack:** Swift 5.10, MLX-Swift (MLX, MLXNN, MLXFast), swift-tokenizers (HuggingFace BPE), Swift Testing framework

---

## File Structure

```text
AnvilCore/Sources/AnvilCore/Inference/
├── Conditioning/
│   ├���─ Qwen3Model.swift          — Full Qwen3 text encoder (28-layer transformer)
│   ├── Qwen3Layer.swift          — Single encoder layer (self-attn + SwiGLU MLP)
│   ├── LyricEncoder.swift        — 8-layer lyric encoder with input projection
│   ├── TimbreEncoder.swift       — 4-layer timbre encoder + attention pooler
│   ��── ConditionEncoder.swift    — Orchestrates sub-encoders, concatenates outputs
│   ├── ConditioningActor.swift   — Top-level actor: tokenize → encode → result
│   └── AnvilTokenizer.swift      — Wrapper around swift-tokenizers
├── Weights/
│   ├── Qwen3NameMap.swift        — Weight key remapping for Qwen3
│   └── ConditionEncoderNameMap.swift — Weight key remapping for condition encoder
├── Types.swift                   — (existing, add ConditioningResult)

AnvilCore/Tests/AnvilCoreTests/Inference/
├── Qwen3ModelTests.swift
├── LyricEncoderTests.swift
├── TimbreEncoderTests.swift
├── ConditionEncoderTests.swift
├── ConditioningActorTests.swift
└── AnvilTokenizerTests.swift
```

---

### Task 1: Add swift-tokenizers Dependency and ConditioningResult Type

**Files:**
- Modify: `AnvilCore/Package.swift`
- Modify: `AnvilCore/Sources/AnvilCore/Inference/Types.swift`

- [ ] **Step 1: Add swift-tokenizers dependency to Package.swift**

```swift
// In dependencies array, add:
.package(url: "https://github.com/huggingface/swift-transformers", from: "0.1.12"),

// In AnvilCore target dependencies, add:
.product(name: "Tokenizers", package: "swift-transformers"),
```

The full Package.swift should become:

```swift
// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "AnvilCore",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "AnvilCore", targets: ["AnvilCore"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.21.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "0.1.12"),
    ],
    targets: [
        .target(
            name: "AnvilCore",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
        .testTarget(name: "AnvilCoreTests", dependencies: ["AnvilCore"]),
    ]
)
```

- [ ] **Step 2: Add ConditioningResult to Types.swift**

Add at the end of `AnvilCore/Sources/AnvilCore/Inference/Types.swift`:

```swift
public struct ConditioningResult: @unchecked Sendable {
    public let encoderStates: MLXArray
    public let encoderMask: MLXArray
    public let contextLatents: MLXArray

    public init(encoderStates: MLXArray, encoderMask: MLXArray, contextLatents: MLXArray) {
        self.encoderStates = encoderStates
        self.encoderMask = encoderMask
        self.contextLatents = contextLatents
    }
}
```

- [ ] **Step 3: Verify build**

Run: `cd AnvilCore && swift build 2>&1 | tail -5`
Expected: Build succeeds (swift-transformers resolves and compiles)

- [ ] **Step 4: Commit**

```bash
git add AnvilCore/Package.swift AnvilCore/Sources/AnvilCore/Inference/Types.swift
git commit -m "feat: add swift-transformers dependency and ConditioningResult type"
```

---

### Task 2: Qwen3Layer (Encoder Layer)

**Files:**
- Create: `AnvilCore/Sources/AnvilCore/Inference/Conditioning/Qwen3Layer.swift`
- Create: `AnvilCore/Tests/AnvilCoreTests/Inference/Qwen3ModelTests.swift`

The Qwen3Layer is a standard pre-norm transformer encoder layer: RMSNorm → self-attention (GQA + RoPE, bidirectional) → residual → RMSNorm → SwiGLU MLP → residual. It reuses `RotaryEmbedding`, `RMSNorm`, and `SwiGLUMLP` from `DiTComponents.swift`. The attention module is separate from `DiTAttention` because it lacks cross-attention, adaptive norm, QK-norm, and sliding window — it's simpler.

- [ ] **Step 1: Write the failing test**

Create `AnvilCore/Tests/AnvilCoreTests/Inference/Qwen3ModelTests.swift`:

```swift
import Testing
import MLX
import MLXNN
@testable import AnvilCore

@Suite("Qwen3 Components")
struct Qwen3ModelTests {
    @Test("Qwen3Attention produces correct output shape")
    func qwen3AttentionShape() {
        let attn = Qwen3Attention(
            hiddenSize: 1024,
            numAttentionHeads: 16,
            numKeyValueHeads: 8,
            headDim: 128,
            rmsNormEps: 1e-6
        )
        let input = MLXArray.zeros([1, 10, 1024])
        let rope = RotaryEmbedding(headDim: 128, maxLen: 32768, base: 1_000_000.0)
        let (cos, sin) = rope.forward(seqLen: 10)
        let output = attn(hiddenStates: input, positionCosSin: (cos, sin), attentionMask: nil)
        #expect(output.shape == [1, 10, 1024])
    }

    @Test("Qwen3Layer produces correct output shape")
    func qwen3LayerShape() {
        let layer = Qwen3Layer(
            hiddenSize: 1024,
            intermediateSize: 3072,
            numAttentionHeads: 16,
            numKeyValueHeads: 8,
            headDim: 128,
            rmsNormEps: 1e-6
        )
        let input = MLXArray.zeros([1, 10, 1024])
        let rope = RotaryEmbedding(headDim: 128, maxLen: 32768, base: 1_000_000.0)
        let (cos, sin) = rope.forward(seqLen: 10)
        let output = layer(hiddenStates: input, positionCosSin: (cos, sin), attentionMask: nil)
        #expect(output.shape == [1, 10, 1024])
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd AnvilCore && swift build 2>&1 | grep "error"`
Expected: Errors about `Qwen3Attention` and `Qwen3Layer` not found

- [ ] **Step 3: Implement Qwen3Layer**

Create `AnvilCore/Sources/AnvilCore/Inference/Conditioning/Qwen3Layer.swift`:

```swift
import Foundation
import MLX
import MLXNN
import MLXFast

public class Qwen3Attention: Module {
    let hiddenSize: Int
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let nRep: Int
    let scale: Float

    let qProj: Linear
    let kProj: Linear
    let vProj: Linear
    let oProj: Linear
    let qNorm: RMSNorm
    let kNorm: RMSNorm

    public init(
        hiddenSize: Int,
        numAttentionHeads: Int,
        numKeyValueHeads: Int,
        headDim: Int,
        rmsNormEps: Float
    ) {
        self.hiddenSize = hiddenSize
        self.numHeads = numAttentionHeads
        self.numKVHeads = numKeyValueHeads
        self.headDim = headDim
        self.nRep = numAttentionHeads / numKeyValueHeads
        self.scale = pow(Float(headDim), -0.5)

        self.qProj = Linear(hiddenSize, numAttentionHeads * headDim, bias: false)
        self.kProj = Linear(hiddenSize, numKeyValueHeads * headDim, bias: false)
        self.vProj = Linear(hiddenSize, numKeyValueHeads * headDim, bias: false)
        self.oProj = Linear(numAttentionHeads * headDim, hiddenSize, bias: false)
        self.qNorm = RMSNorm(dimensions: headDim, eps: rmsNormEps)
        self.kNorm = RMSNorm(dimensions: headDim, eps: rmsNormEps)
    }

    private func repeatKV(_ x: MLXArray, nRep: Int) -> MLXArray {
        if nRep == 1 { return x }
        let (b, nKV, seqL, d) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        let expanded = x.expandedDimensions(axis: 2)
        let broadcasted = broadcast(expanded, to: [b, nKV, nRep, seqL, d])
        return broadcasted.reshaped([b, nKV * nRep, seqL, d])
    }

    public func callAsFunction(
        hiddenStates: MLXArray,
        positionCosSin: (cos: MLXArray, sin: MLXArray),
        attentionMask: MLXArray?
    ) -> MLXArray {
        let b = hiddenStates.shape[0]
        let seqL = hiddenStates.shape[1]

        var q = qProj(hiddenStates)
        q = qNorm(q.reshaped([b, seqL, numHeads, headDim]))
        q = q.transposed(0, 2, 1, 3)

        var k = kProj(hiddenStates)
        k = kNorm(k.reshaped([b, seqL, numKVHeads, headDim]))
        k = k.transposed(0, 2, 1, 3)

        let v = vProj(hiddenStates).reshaped([b, seqL, numKVHeads, headDim])
            .transposed(0, 2, 1, 3)

        (q, k) = applyRotaryPosEmb(q: q, k: k, cos: positionCosSin.cos, sin: positionCosSin.sin)

        let kExpanded = repeatKV(k, nRep: nRep)
        let vExpanded = repeatKV(v, nRep: nRep)

        let attnOut = MLXFast.scaledDotProductAttention(
            queries: q, keys: kExpanded, values: vExpanded, scale: scale, mask: attentionMask
        )

        let merged = attnOut.transposed(0, 2, 1, 3).reshaped([b, seqL, -1])
        return oProj(merged)
    }
}

public class Qwen3Layer: Module {
    let inputNorm: RMSNorm
    let attention: Qwen3Attention
    let postNorm: RMSNorm
    let mlp: SwiGLUMLP

    public init(
        hiddenSize: Int,
        intermediateSize: Int,
        numAttentionHeads: Int,
        numKeyValueHeads: Int,
        headDim: Int,
        rmsNormEps: Float
    ) {
        self.inputNorm = RMSNorm(dimensions: hiddenSize, eps: rmsNormEps)
        self.attention = Qwen3Attention(
            hiddenSize: hiddenSize,
            numAttentionHeads: numAttentionHeads,
            numKeyValueHeads: numKeyValueHeads,
            headDim: headDim,
            rmsNormEps: rmsNormEps
        )
        self.postNorm = RMSNorm(dimensions: hiddenSize, eps: rmsNormEps)
        self.mlp = SwiGLUMLP(hiddenSize: hiddenSize, intermediateSize: intermediateSize)
    }

    public func callAsFunction(
        hiddenStates: MLXArray,
        positionCosSin: (cos: MLXArray, sin: MLXArray),
        attentionMask: MLXArray?
    ) -> MLXArray {
        var h = hiddenStates
        let normed = inputNorm(h)
        let attnOut = attention(hiddenStates: normed, positionCosSin: positionCosSin, attentionMask: attentionMask)
        h = h + attnOut

        let postNormed = postNorm(h)
        let ffOut = mlp(postNormed)
        h = h + ffOut

        return h
    }
}
```

- [ ] **Step 4: Verify build succeeds**

Run: `cd AnvilCore && swift build 2>&1 | tail -5`
Expected: Build succeeds

- [ ] **Step 5: Commit**

```bash
git add AnvilCore/Sources/AnvilCore/Inference/Conditioning/Qwen3Layer.swift \
        AnvilCore/Tests/AnvilCoreTests/Inference/Qwen3ModelTests.swift
git commit -m "feat: add Qwen3Layer and Qwen3Attention for conditioning encoder"
```

---

### Task 3: Qwen3Model (Full Text Encoder)

**Files:**
- Create: `AnvilCore/Sources/AnvilCore/Inference/Conditioning/Qwen3Model.swift`
- Modify: `AnvilCore/Tests/AnvilCoreTests/Inference/Qwen3ModelTests.swift`

The Qwen3Model is a 28-layer transformer used as a text encoder. It has an embedding table (151,669 tokens → 1024 dims), 28 × Qwen3Layer, and a final RMSNorm. No causal mask — bidirectional attention. Only `last_hidden_state` is used; no LM head.

- [ ] **Step 1: Add tests to Qwen3ModelTests.swift**

Append to the existing test file:

```swift
    @Test("Qwen3Model forward produces [B, T, 1024]")
    func qwen3ModelForwardShape() {
        let model = Qwen3Model()
        let inputIds = MLXArray(converting: [1, 2, 3, 4, 5], type: .int32).reshaped([1, 5])
        let output = model(inputIds: inputIds)
        #expect(output.shape == [1, 5, 1024])
    }

    @Test("Qwen3Model embedTokens returns embeddings only")
    func qwen3EmbedTokensShape() {
        let model = Qwen3Model()
        let inputIds = MLXArray(converting: [1, 2, 3], type: .int32).reshaped([1, 3])
        let embeddings = model.embedTokens(inputIds)
        #expect(embeddings.shape == [1, 3, 1024])
    }
```

- [ ] **Step 2: Run build to verify it fails**

Run: `cd AnvilCore && swift build 2>&1 | grep "error"`
Expected: Error about `Qwen3Model` not found

- [ ] **Step 3: Implement Qwen3Model**

Create `AnvilCore/Sources/AnvilCore/Inference/Conditioning/Qwen3Model.swift`:

```swift
import Foundation
import MLX
import MLXNN

public class Qwen3Model: Module {
    let embeddingTable: Embedding
    let layers: [Qwen3Layer]
    let norm: RMSNorm
    let rotaryEmbedding: RotaryEmbedding

    static let vocabSize = 151_669
    static let hiddenSize = 1024
    static let intermediateSize = 3072
    static let numLayers = 28
    static let numAttentionHeads = 16
    static let numKeyValueHeads = 8
    static let headDim = 128
    static let rmsNormEps: Float = 1e-6

    public init() {
        self.embeddingTable = Embedding(
            embeddingCount: Self.vocabSize,
            dimensions: Self.hiddenSize
        )
        self.layers = (0..<Self.numLayers).map { _ in
            Qwen3Layer(
                hiddenSize: Self.hiddenSize,
                intermediateSize: Self.intermediateSize,
                numAttentionHeads: Self.numAttentionHeads,
                numKeyValueHeads: Self.numKeyValueHeads,
                headDim: Self.headDim,
                rmsNormEps: Self.rmsNormEps
            )
        }
        self.norm = RMSNorm(dimensions: Self.hiddenSize, eps: Self.rmsNormEps)
        self.rotaryEmbedding = RotaryEmbedding(
            headDim: Self.headDim,
            maxLen: 32768,
            base: 1_000_000.0
        )
    }

    public func callAsFunction(inputIds: MLXArray) -> MLXArray {
        var hiddenStates = embeddingTable(inputIds)
        let seqLen = hiddenStates.shape[1]
        let positionCosSin = rotaryEmbedding.forward(seqLen: seqLen)

        for layer in layers {
            hiddenStates = layer(
                hiddenStates: hiddenStates,
                positionCosSin: positionCosSin,
                attentionMask: nil
            )
        }

        return norm(hiddenStates)
    }

    public func embedTokens(_ inputIds: MLXArray) -> MLXArray {
        embeddingTable(inputIds)
    }
}
```

- [ ] **Step 4: Verify build succeeds**

Run: `cd AnvilCore && swift build 2>&1 | tail -5`
Expected: Build succeeds

- [ ] **Step 5: Commit**

```bash
git add AnvilCore/Sources/AnvilCore/Inference/Conditioning/Qwen3Model.swift \
        AnvilCore/Tests/AnvilCoreTests/Inference/Qwen3ModelTests.swift
git commit -m "feat: add Qwen3Model text encoder (28-layer, 1024-dim)"
```

---

### Task 4: LyricEncoder

**Files:**
- Create: `AnvilCore/Sources/AnvilCore/Inference/Conditioning/LyricEncoder.swift`
- Create: `AnvilCore/Tests/AnvilCoreTests/Inference/LyricEncoderTests.swift`

The LyricEncoder receives raw embeddings from Qwen3's embedding table (dim=1024), projects them to 2048 via a Linear layer, then processes through 8 transformer layers (with GQA 16/8, head_dim=128, intermediate=6144) and a final RMSNorm. Uses bidirectional attention with RoPE. Supports an optional attention mask for padded lyrics.

- [ ] **Step 1: Write the failing test**

Create `AnvilCore/Tests/AnvilCoreTests/Inference/LyricEncoderTests.swift`:

```swift
import Testing
import MLX
import MLXNN
@testable import AnvilCore

@Suite("Lyric Encoder")
struct LyricEncoderTests {
    @Test("LyricEncoder produces [B, T, 2048] from [B, T, 1024] input")
    func lyricEncoderOutputShape() {
        let encoder = LyricEncoder()
        let embeddings = MLXArray.zeros([1, 100, 1024])
        let output = encoder(embeddings: embeddings, attentionMask: nil)
        #expect(output.shape == [1, 100, 2048])
    }

    @Test("LyricEncoder handles short sequences")
    func lyricEncoderShortSeq() {
        let encoder = LyricEncoder()
        let embeddings = MLXArray.zeros([1, 5, 1024])
        let output = encoder(embeddings: embeddings, attentionMask: nil)
        #expect(output.shape == [1, 5, 2048])
    }

    @Test("LyricEncoder respects max sequence length 2048")
    func lyricEncoderMaxLength() {
        let encoder = LyricEncoder()
        let embeddings = MLXArray.zeros([1, 2048, 1024])
        let output = encoder(embeddings: embeddings, attentionMask: nil)
        #expect(output.shape == [1, 2048, 2048])
    }
}
```

- [ ] **Step 2: Run build to verify it fails**

Run: `cd AnvilCore && swift build 2>&1 | grep "error"`
Expected: Error about `LyricEncoder` not found

- [ ] **Step 3: Implement LyricEncoder**

Create `AnvilCore/Sources/AnvilCore/Inference/Conditioning/LyricEncoder.swift`:

```swift
import Foundation
import MLX
import MLXNN

public class LyricEncoder: Module {
    let inputProjection: Linear
    let layers: [Qwen3Layer]
    let norm: RMSNorm
    let rotaryEmbedding: RotaryEmbedding

    static let inputDim = 1024
    static let hiddenSize = 2048
    static let intermediateSize = 6144
    static let numLayers = 8
    static let numAttentionHeads = 16
    static let numKeyValueHeads = 8
    static let headDim = 128
    static let rmsNormEps: Float = 1e-6

    public init() {
        self.inputProjection = Linear(Self.inputDim, Self.hiddenSize, bias: false)
        self.layers = (0..<Self.numLayers).map { _ in
            Qwen3Layer(
                hiddenSize: Self.hiddenSize,
                intermediateSize: Self.intermediateSize,
                numAttentionHeads: Self.numAttentionHeads,
                numKeyValueHeads: Self.numKeyValueHeads,
                headDim: Self.headDim,
                rmsNormEps: Self.rmsNormEps
            )
        }
        self.norm = RMSNorm(dimensions: Self.hiddenSize, eps: Self.rmsNormEps)
        self.rotaryEmbedding = RotaryEmbedding(
            headDim: Self.headDim,
            maxLen: 32768,
            base: 1_000_000.0
        )
    }

    public func callAsFunction(
        embeddings: MLXArray,
        attentionMask: MLXArray?
    ) -> MLXArray {
        var hiddenStates = inputProjection(embeddings)
        let seqLen = hiddenStates.shape[1]
        let positionCosSin = rotaryEmbedding.forward(seqLen: seqLen)

        for layer in layers {
            hiddenStates = layer(
                hiddenStates: hiddenStates,
                positionCosSin: positionCosSin,
                attentionMask: attentionMask
            )
        }

        return norm(hiddenStates)
    }
}
```

- [ ] **Step 4: Verify build succeeds**

Run: `cd AnvilCore && swift build 2>&1 | tail -5`
Expected: Build succeeds

- [ ] **Step 5: Commit**

```bash
git add AnvilCore/Sources/AnvilCore/Inference/Conditioning/LyricEncoder.swift \
        AnvilCore/Tests/AnvilCoreTests/Inference/LyricEncoderTests.swift
git commit -m "feat: add LyricEncoder (8-layer, 1024→2048 projection)"
```

---

### Task 5: TimbreEncoder with Attention Pooler

**Files:**
- Create: `AnvilCore/Sources/AnvilCore/Inference/Conditioning/TimbreEncoder.swift`
- Create: `AnvilCore/Tests/AnvilCoreTests/Inference/TimbreEncoderTests.swift`

The TimbreEncoder takes VAE-encoded reference audio latents `[B, 750, 64]`, projects them to 2048, prepends a learnable special token, processes through 4 self-attention layers, then extracts the special token (position 0) as the timbre embedding. The AttentionPooler then takes these timbre embeddings and uses 2 layers with a learnable query token to produce 5 timbre tokens via a reshape from the `pool_window_size=5` grouping.

**Important implementation detail from reference:** The TimbreEncoder prepends a special token to the sequence, processes through layers, then extracts position 0 as the per-sample timbre embedding. The AttentionPooler is separate — it takes the VAE latents grouped into patches of `pool_window_size=5`, prepends its own special token per patch, processes through 2 layers, and extracts position 0 per patch.

However, per our spec (simplified for batch-size-1), the TimbreEncoder:
1. Projects `[B, 750, 64]` → `[B, 750, 2048]`
2. Processes through 4 self-attention layers
3. Passes to AttentionPooler (2 layers, 5 learnable query tokens that cross-attend to the 750-token sequence)
4. Outputs `[B, 5, 2048]`

- [ ] **Step 1: Write the failing test**

Create `AnvilCore/Tests/AnvilCoreTests/Inference/TimbreEncoderTests.swift`:

```swift
import Testing
import MLX
import MLXNN
@testable import AnvilCore

@Suite("Timbre Encoder")
struct TimbreEncoderTests {
    @Test("TimbreEncoder produces [B, 5, 2048] from [B, 750, 64]")
    func timbreEncoderOutputShape() {
        let encoder = TimbreEncoder()
        let latents = MLXArray.zeros([1, 750, 64])
        let output = encoder(latents: latents, attentionMask: nil)
        #expect(output.shape == [1, 5, 2048])
    }

    @Test("TimbreEncoder silence fallback (zeros input)")
    func timbreEncoderSilenceFallback() {
        let encoder = TimbreEncoder()
        let silence = MLXArray.zeros([1, 750, 64])
        let output = encoder(latents: silence, attentionMask: nil)
        #expect(output.shape == [1, 5, 2048])
    }

    @Test("AttentionPooler compresses sequence to query count")
    func attentionPoolerShape() {
        let pooler = AttentionPooler(
            hiddenSize: 2048,
            numQueries: 5,
            numLayers: 2,
            numAttentionHeads: 16,
            numKeyValueHeads: 8,
            headDim: 128,
            intermediateSize: 6144,
            rmsNormEps: 1e-6
        )
        let encoded = MLXArray.zeros([1, 750, 2048])
        let output = pooler(encodedSequence: encoded)
        #expect(output.shape == [1, 5, 2048])
    }
}
```

- [ ] **Step 2: Run build to verify it fails**

Run: `cd AnvilCore && swift build 2>&1 | grep "error"`
Expected: Errors about `TimbreEncoder` and `AttentionPooler` not found

- [ ] **Step 3: Implement TimbreEncoder and AttentionPooler**

Create `AnvilCore/Sources/AnvilCore/Inference/Conditioning/TimbreEncoder.swift`:

```swift
import Foundation
import MLX
import MLXNN
import MLXFast

public class AttentionPooler: Module {
    let queryTokens: MLXArray
    let layers: [Qwen3Layer]
    let norm: RMSNorm
    let rotaryEmbedding: RotaryEmbedding
    let numQueries: Int

    public init(
        hiddenSize: Int,
        numQueries: Int,
        numLayers: Int,
        numAttentionHeads: Int,
        numKeyValueHeads: Int,
        headDim: Int,
        intermediateSize: Int,
        rmsNormEps: Float
    ) {
        self.numQueries = numQueries
        self.queryTokens = MLXRandom.normal([1, numQueries, hiddenSize]) * 0.02
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
            maxLen: 32768,
            base: 1_000_000.0
        )
    }

    public func callAsFunction(encodedSequence: MLXArray) -> MLXArray {
        let b = encodedSequence.shape[0]
        let queries = broadcast(queryTokens, to: [b, numQueries, encodedSequence.shape[2]])
        var hiddenStates = concatenated([queries, encodedSequence], axis: 1)

        let seqLen = hiddenStates.shape[1]
        let positionCosSin = rotaryEmbedding.forward(seqLen: seqLen)

        for layer in layers {
            hiddenStates = layer(
                hiddenStates: hiddenStates,
                positionCosSin: positionCosSin,
                attentionMask: nil
            )
        }

        hiddenStates = norm(hiddenStates)
        return hiddenStates[0..<b, 0..<numQueries]
    }
}

public class TimbreEncoder: Module {
    let inputProjection: Linear
    let layers: [Qwen3Layer]
    let norm: RMSNorm
    let rotaryEmbedding: RotaryEmbedding
    let attentionPooler: AttentionPooler

    static let inputDim = 64
    static let hiddenSize = 2048
    static let intermediateSize = 6144
    static let numEncoderLayers = 4
    static let numPoolerLayers = 2
    static let numQueries = 5
    static let numAttentionHeads = 16
    static let numKeyValueHeads = 8
    static let headDim = 128
    static let rmsNormEps: Float = 1e-6

    public init() {
        self.inputProjection = Linear(Self.inputDim, Self.hiddenSize, bias: false)
        self.layers = (0..<Self.numEncoderLayers).map { _ in
            Qwen3Layer(
                hiddenSize: Self.hiddenSize,
                intermediateSize: Self.intermediateSize,
                numAttentionHeads: Self.numAttentionHeads,
                numKeyValueHeads: Self.numKeyValueHeads,
                headDim: Self.headDim,
                rmsNormEps: Self.rmsNormEps
            )
        }
        self.norm = RMSNorm(dimensions: Self.hiddenSize, eps: Self.rmsNormEps)
        self.rotaryEmbedding = RotaryEmbedding(
            headDim: Self.headDim,
            maxLen: 32768,
            base: 1_000_000.0
        )
        self.attentionPooler = AttentionPooler(
            hiddenSize: Self.hiddenSize,
            numQueries: Self.numQueries,
            numLayers: Self.numPoolerLayers,
            numAttentionHeads: Self.numAttentionHeads,
            numKeyValueHeads: Self.numKeyValueHeads,
            headDim: Self.headDim,
            intermediateSize: Self.intermediateSize,
            rmsNormEps: Self.rmsNormEps
        )
    }

    public func callAsFunction(
        latents: MLXArray,
        attentionMask: MLXArray?
    ) -> MLXArray {
        var hiddenStates = inputProjection(latents)
        let seqLen = hiddenStates.shape[1]
        let positionCosSin = rotaryEmbedding.forward(seqLen: seqLen)

        for layer in layers {
            hiddenStates = layer(
                hiddenStates: hiddenStates,
                positionCosSin: positionCosSin,
                attentionMask: attentionMask
            )
        }

        hiddenStates = norm(hiddenStates)
        return attentionPooler(encodedSequence: hiddenStates)
    }
}
```

- [ ] **Step 4: Verify build succeeds**

Run: `cd AnvilCore && swift build 2>&1 | tail -5`
Expected: Build succeeds

Note: `MLXRandom` needs to be imported. Add `import MLXRandom` at the top of the file.

- [ ] **Step 5: Commit**

```bash
git add AnvilCore/Sources/AnvilCore/Inference/Conditioning/TimbreEncoder.swift \
        AnvilCore/Tests/AnvilCoreTests/Inference/TimbreEncoderTests.swift
git commit -m "feat: add TimbreEncoder with AttentionPooler (4+2 layers, 5 query tokens)"
```

---

### Task 6: ConditionEncoder

**Files:**
- Create: `AnvilCore/Sources/AnvilCore/Inference/Conditioning/ConditionEncoder.swift`
- Create: `AnvilCore/Tests/AnvilCoreTests/Inference/ConditionEncoderTests.swift`

The ConditionEncoder orchestrates the sub-encoders. It:
1. Projects text hidden states from 1024 → 2048 via a linear layer
2. Takes pre-computed lyric hidden states (already 2048 from LyricEncoder)
3. Takes pre-computed timbre tokens (already 2048 from TimbreEncoder)
4. Concatenates all three along the sequence dimension: `[B, T_text + T_lyric + 5, 2048]`
5. Produces a combined attention mask

- [ ] **Step 1: Write the failing test**

Create `AnvilCore/Tests/AnvilCoreTests/Inference/ConditionEncoderTests.swift`:

```swift
import Testing
import MLX
import MLXNN
@testable import AnvilCore

@Suite("Condition Encoder")
struct ConditionEncoderTests {
    @Test("ConditionEncoder concatenates all inputs correctly")
    func conditionEncoderOutputShape() {
        let encoder = ConditionEncoder()
        let textHidden = MLXArray.zeros([1, 50, 1024])
        let lyricHidden = MLXArray.zeros([1, 100, 2048])
        let timbreTokens = MLXArray.zeros([1, 5, 2048])
        let textMask = MLXArray.ones([1, 50])
        let lyricMask = MLXArray.ones([1, 100])

        let (states, mask) = encoder(
            textHiddenStates: textHidden,
            textMask: textMask,
            lyricHiddenStates: lyricHidden,
            lyricMask: lyricMask,
            timbreTokens: timbreTokens
        )
        #expect(states.shape == [1, 155, 2048])
        #expect(mask.shape == [1, 155])
    }

    @Test("ConditionEncoder handles max sequence (256 + 2048 + 5)")
    func conditionEncoderMaxSequence() {
        let encoder = ConditionEncoder()
        let textHidden = MLXArray.zeros([1, 256, 1024])
        let lyricHidden = MLXArray.zeros([1, 2048, 2048])
        let timbreTokens = MLXArray.zeros([1, 5, 2048])
        let textMask = MLXArray.ones([1, 256])
        let lyricMask = MLXArray.ones([1, 2048])

        let (states, mask) = encoder(
            textHiddenStates: textHidden,
            textMask: textMask,
            lyricHiddenStates: lyricHidden,
            lyricMask: lyricMask,
            timbreTokens: timbreTokens
        )
        #expect(states.shape == [1, 2309, 2048])
        #expect(mask.shape == [1, 2309])
    }
}
```

- [ ] **Step 2: Run build to verify it fails**

Run: `cd AnvilCore && swift build 2>&1 | grep "error"`
Expected: Error about `ConditionEncoder` not found

- [ ] **Step 3: Implement ConditionEncoder**

Create `AnvilCore/Sources/AnvilCore/Inference/Conditioning/ConditionEncoder.swift`:

```swift
import Foundation
import MLX
import MLXNN

public class ConditionEncoder: Module {
    let textProjection: Linear

    static let textInputDim = 1024
    static let hiddenSize = 2048

    public init() {
        self.textProjection = Linear(Self.textInputDim, Self.hiddenSize, bias: false)
    }

    public func callAsFunction(
        textHiddenStates: MLXArray,
        textMask: MLXArray?,
        lyricHiddenStates: MLXArray,
        lyricMask: MLXArray?,
        timbreTokens: MLXArray
    ) -> (encoderStates: MLXArray, mask: MLXArray) {
        let textProjected = textProjection(textHiddenStates)
        let b = textProjected.shape[0]
        let tText = textProjected.shape[1]
        let tLyric = lyricHiddenStates.shape[1]
        let tTimbre = timbreTokens.shape[1]
        let tCombined = tText + tLyric + tTimbre

        let encoderStates = concatenated(
            [textProjected, lyricHiddenStates, timbreTokens],
            axis: 1
        )

        let effectiveTextMask = textMask ?? MLXArray.ones([b, tText])
        let effectiveLyricMask = lyricMask ?? MLXArray.ones([b, tLyric])
        let timbreMask = MLXArray.ones([b, tTimbre])

        let combinedMask = concatenated(
            [effectiveTextMask, effectiveLyricMask, timbreMask],
            axis: 1
        )

        return (encoderStates, combinedMask)
    }
}
```

- [ ] **Step 4: Verify build succeeds**

Run: `cd AnvilCore && swift build 2>&1 | tail -5`
Expected: Build succeeds

- [ ] **Step 5: Commit**

```bash
git add AnvilCore/Sources/AnvilCore/Inference/Conditioning/ConditionEncoder.swift \
        AnvilCore/Tests/AnvilCoreTests/Inference/ConditionEncoderTests.swift
git commit -m "feat: add ConditionEncoder (text projection + sequence concatenation)"
```

---

### Task 7: AnvilTokenizer

**Files:**
- Create: `AnvilCore/Sources/AnvilCore/Inference/Conditioning/AnvilTokenizer.swift`
- Create: `AnvilCore/Tests/AnvilCoreTests/Inference/AnvilTokenizerTests.swift`

Thin wrapper around swift-tokenizers that loads a Qwen3 `tokenizer.json` and provides typed encoding for text (max 256 tokens) and lyrics (max 2048 tokens). Returns `(ids: MLXArray, mask: MLXArray)` tuples where ids are int32 token IDs and mask is a float32 attention mask (1 for real tokens, 0 for padding).

- [ ] **Step 1: Write the failing test**

Create `AnvilCore/Tests/AnvilCoreTests/Inference/AnvilTokenizerTests.swift`:

```swift
import Testing
import MLX
@testable import AnvilCore

@Suite("Anvil Tokenizer")
struct AnvilTokenizerTests {
    @Test("AnvilTokenizer encodeText returns padded ids and mask")
    func encodeTextShape() throws {
        let tokenizer = try AnvilTokenizer.forTesting()
        let (ids, mask) = tokenizer.encodeText("Hello world", maxLength: 256)
        #expect(ids.shape[0] == 1)
        #expect(ids.shape[1] == 256)
        #expect(mask.shape == ids.shape)
    }

    @Test("AnvilTokenizer encodeLyrics respects max length")
    func encodeLyricsShape() throws {
        let tokenizer = try AnvilTokenizer.forTesting()
        let (ids, mask) = tokenizer.encodeLyrics("La la la", maxLength: 2048)
        #expect(ids.shape[0] == 1)
        #expect(ids.shape[1] == 2048)
        #expect(mask.shape == ids.shape)
    }

    @Test("AnvilTokenizer mask has 1s for real tokens and 0s for padding")
    func maskValues() throws {
        let tokenizer = try AnvilTokenizer.forTesting()
        let (_, mask) = tokenizer.encodeText("Hi", maxLength: 10)
        let maskArray = mask.asArray(Int32.self)
        let realTokenCount = maskArray.filter { $0 == 1 }.count
        #expect(realTokenCount > 0)
        #expect(realTokenCount < 10)
    }
}
```

- [ ] **Step 2: Run build to verify it fails**

Run: `cd AnvilCore && swift build 2>&1 | grep "error"`
Expected: Error about `AnvilTokenizer` not found

- [ ] **Step 3: Implement AnvilTokenizer**

Create `AnvilCore/Sources/AnvilCore/Inference/Conditioning/AnvilTokenizer.swift`:

```swift
import Foundation
import MLX
import Tokenizers

public struct AnvilTokenizer: Sendable {
    private let tokenizer: any Tokenizer

    public init(configURL: URL) throws {
        self.tokenizer = try AutoTokenizer.from(tokenizerConfig: configURL)
    }

    public func encodeText(_ text: String, maxLength: Int = 256) -> (ids: MLXArray, mask: MLXArray) {
        encode(text, maxLength: maxLength)
    }

    public func encodeLyrics(_ lyrics: String, maxLength: Int = 2048) -> (ids: MLXArray, mask: MLXArray) {
        encode(lyrics, maxLength: maxLength)
    }

    private func encode(_ text: String, maxLength: Int) -> (ids: MLXArray, mask: MLXArray) {
        var tokenIds = tokenizer.encode(text: text)

        if tokenIds.count > maxLength {
            tokenIds = Array(tokenIds.prefix(maxLength))
        }

        let realLength = tokenIds.count
        let paddedIds = tokenIds + Array(repeating: 0, count: maxLength - realLength)
        let maskValues = Array(repeating: Int32(1), count: realLength)
            + Array(repeating: Int32(0), count: maxLength - realLength)

        let ids = MLXArray(converting: paddedIds.map { Int32($0) }).reshaped([1, maxLength])
        let mask = MLXArray(converting: maskValues).reshaped([1, maxLength])

        return (ids, mask)
    }
}

extension AnvilTokenizer {
    /// Creates a tokenizer for testing using a minimal config.
    /// In production, use init(configURL:) with the real Qwen3 tokenizer.json.
    public static func forTesting() throws -> AnvilTokenizer {
        let config: [String: Any] = [
            "version": "1.0",
            "model": [
                "type": "BPE",
                "vocab": ["H": 0, "e": 1, "l": 2, "o": 3, " ": 4, "w": 5, "r": 6, "d": 7,
                          "i": 8, "L": 9, "a": 10, "He": 11, "ll": 12, "lo": 13,
                          "wo": 14, "rl": 15, "Hello": 16, "world": 17],
                "merges": ["H e", "l l", "l o", "w o", "r l", "He ll", "Hel lo", "wo rl", "Hello", "world"]
            ]
        ]
        let data = try JSONSerialization.data(withJSONObject: config)
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("test_tokenizer.json")
        try data.write(to: tempURL)
        return try AnvilTokenizer(configURL: tempURL)
    }
}
```

- [ ] **Step 4: Verify build succeeds**

Run: `cd AnvilCore && swift build 2>&1 | tail -5`
Expected: Build succeeds

**Note:** The `Tokenizers` module from `swift-transformers` provides `AutoTokenizer.from(tokenizerConfig:)`. The exact API may require `AutoTokenizer.from(pretrained:)` or a different initializer — the implementer should check the swift-transformers documentation and adapt accordingly. The key contract is: load a BPE tokenizer from a JSON config and call `encode(text:)` to get token IDs.

- [ ] **Step 5: Commit**

```bash
git add AnvilCore/Sources/AnvilCore/Inference/Conditioning/AnvilTokenizer.swift \
        AnvilCore/Tests/AnvilCoreTests/Inference/AnvilTokenizerTests.swift
git commit -m "feat: add AnvilTokenizer wrapper around swift-transformers"
```

---

### Task 8: Weight Name Maps

**Files:**
- Create: `AnvilCore/Sources/AnvilCore/Inference/Weights/Qwen3NameMap.swift`
- Create: `AnvilCore/Sources/AnvilCore/Inference/Weights/ConditionEncoderNameMap.swift`

These provide key remapping from HuggingFace checkpoint keys to our Swift model property paths.

- [ ] **Step 1: Implement Qwen3NameMap**

Create `AnvilCore/Sources/AnvilCore/Inference/Weights/Qwen3NameMap.swift`:

```swift
import Foundation

public enum Qwen3NameMap {
    public static func mapKey(_ key: String) -> String? {
        var mapped = key

        // Filter out keys we don't need (lm_head, etc.)
        if key.hasPrefix("lm_head.") { return nil }

        // model.embed_tokens.weight → embeddingTable.weight
        mapped = mapped.replacingOccurrences(of: "model.embed_tokens", with: "embeddingTable")

        // model.norm.weight → norm.weight
        mapped = mapped.replacingOccurrences(of: "model.norm", with: "norm")

        // model.layers.N.self_attn.q_proj → layers.N.attention.qProj
        mapped = mapped.replacingOccurrences(of: "model.layers.", with: "layers.")
        mapped = mapped.replacingOccurrences(of: ".self_attn.q_proj", with: ".attention.qProj")
        mapped = mapped.replacingOccurrences(of: ".self_attn.k_proj", with: ".attention.kProj")
        mapped = mapped.replacingOccurrences(of: ".self_attn.v_proj", with: ".attention.vProj")
        mapped = mapped.replacingOccurrences(of: ".self_attn.o_proj", with: ".attention.oProj")
        mapped = mapped.replacingOccurrences(of: ".self_attn.q_norm", with: ".attention.qNorm")
        mapped = mapped.replacingOccurrences(of: ".self_attn.k_norm", with: ".attention.kNorm")

        // model.layers.N.mlp.gate_proj → layers.N.mlp.gateProj
        mapped = mapped.replacingOccurrences(of: ".mlp.gate_proj", with: ".mlp.gateProj")
        mapped = mapped.replacingOccurrences(of: ".mlp.up_proj", with: ".mlp.upProj")
        mapped = mapped.replacingOccurrences(of: ".mlp.down_proj", with: ".mlp.downProj")

        // model.layers.N.input_layernorm → layers.N.inputNorm
        mapped = mapped.replacingOccurrences(of: ".input_layernorm", with: ".inputNorm")
        mapped = mapped.replacingOccurrences(of: ".post_attention_layernorm", with: ".postNorm")

        return mapped
    }
}
```

- [ ] **Step 2: Implement ConditionEncoderNameMap**

Create `AnvilCore/Sources/AnvilCore/Inference/Weights/ConditionEncoderNameMap.swift`:

```swift
import Foundation

public enum ConditionEncoderNameMap {
    public static func mapKey(_ key: String) -> String? {
        var mapped = key

        // condition_encoder.text_projector.weight → textProjection.weight
        if key.hasPrefix("condition_encoder.text_projector") {
            mapped = mapped.replacingOccurrences(
                of: "condition_encoder.text_projector",
                with: "conditionEncoder.textProjection"
            )
            return mapped
        }

        // condition_encoder.lyric_encoder.* → lyricEncoder.*
        if key.hasPrefix("condition_encoder.lyric_encoder") {
            mapped = mapped.replacingOccurrences(
                of: "condition_encoder.lyric_encoder.",
                with: "lyricEncoder."
            )
            return mapEncoderKey(mapped, prefix: "lyricEncoder")
        }

        // condition_encoder.timbre_encoder.* → timbreEncoder.*
        if key.hasPrefix("condition_encoder.timbre_encoder") {
            mapped = mapped.replacingOccurrences(
                of: "condition_encoder.timbre_encoder.",
                with: "timbreEncoder."
            )
            return mapEncoderKey(mapped, prefix: "timbreEncoder")
        }

        return nil
    }

    private static func mapEncoderKey(_ key: String, prefix: String) -> String {
        var mapped = key

        // embed_tokens → inputProjection
        mapped = mapped.replacingOccurrences(of: ".embed_tokens", with: ".inputProjection")

        // layers.N.self_attn.{q,k,v,o}_proj → layers.N.attention.{q,k,v,o}Proj
        mapped = mapped.replacingOccurrences(of: ".self_attn.q_proj", with: ".attention.qProj")
        mapped = mapped.replacingOccurrences(of: ".self_attn.k_proj", with: ".attention.kProj")
        mapped = mapped.replacingOccurrences(of: ".self_attn.v_proj", with: ".attention.vProj")
        mapped = mapped.replacingOccurrences(of: ".self_attn.o_proj", with: ".attention.oProj")
        mapped = mapped.replacingOccurrences(of: ".self_attn.q_norm", with: ".attention.qNorm")
        mapped = mapped.replacingOccurrences(of: ".self_attn.k_norm", with: ".attention.kNorm")

        // mlp projections
        mapped = mapped.replacingOccurrences(of: ".mlp.gate_proj", with: ".mlp.gateProj")
        mapped = mapped.replacingOccurrences(of: ".mlp.up_proj", with: ".mlp.upProj")
        mapped = mapped.replacingOccurrences(of: ".mlp.down_proj", with: ".mlp.downProj")

        // norms
        mapped = mapped.replacingOccurrences(of: ".input_layernorm", with: ".inputNorm")
        mapped = mapped.replacingOccurrences(of: ".post_attention_layernorm", with: ".postNorm")

        // special_token → attentionPooler.queryTokens (for timbre encoder pooler)
        mapped = mapped.replacingOccurrences(of: ".special_token", with: ".attentionPooler.queryTokens")

        return mapped
    }
}
```

- [ ] **Step 3: Verify build succeeds**

Run: `cd AnvilCore && swift build 2>&1 | tail -5`
Expected: Build succeeds

- [ ] **Step 4: Commit**

```bash
git add AnvilCore/Sources/AnvilCore/Inference/Weights/Qwen3NameMap.swift \
        AnvilCore/Sources/AnvilCore/Inference/Weights/ConditionEncoderNameMap.swift
git commit -m "feat: add weight name maps for Qwen3 and ConditionEncoder"
```

---

### Task 9: ConditioningActor

**Files:**
- Create: `AnvilCore/Sources/AnvilCore/Inference/Conditioning/ConditioningActor.swift`
- Create: `AnvilCore/Tests/AnvilCoreTests/Inference/ConditioningActorTests.swift`

The top-level actor that owns the full conditioning pipeline:
1. Tokenize text and lyrics via AnvilTokenizer
2. Run Qwen3 forward pass on text tokens → text_hidden_states `[B, T_text, 1024]`
3. Get lyric embeddings from Qwen3 embedding table → lyric_embeddings `[B, T_lyric, 1024]`
4. Run LyricEncoder on lyric embeddings → lyric_hidden_states `[B, T_lyric, 2048]`
5. Pad/truncate reference latents to 750 frames (or use zeros for silence fallback)
6. Run TimbreEncoder → timbre_tokens `[B, 5, 2048]`
7. Run ConditionEncoder (project text + concatenate) → encoderStates `[B, T_combined, 2048]`
8. Prepare contextLatents `[B, T_frames, 128]` = `cat([src_latents(64), chunk_mask(64)])`

- [ ] **Step 1: Write the failing test**

Create `AnvilCore/Tests/AnvilCoreTests/Inference/ConditioningActorTests.swift`:

```swift
import Testing
import MLX
@testable import AnvilCore

@Suite("Conditioning Actor")
struct ConditioningActorTests {
    @Test("ConditioningActor encode produces correct shapes (text-only)")
    func encodeTextOnly() async throws {
        let actor = try ConditioningActor.forTesting()
        let result = await actor.encode(
            text: "A dreamy synthwave track",
            lyrics: "Floating through the neon lights",
            referenceAudio: nil,
            duration: 30.0,
            precision: .float32
        )
        #expect(result.encoderStates.shape[0] == 1)
        #expect(result.encoderStates.shape[2] == 2048)
        #expect(result.encoderMask.shape[0] == 1)
        #expect(result.encoderMask.shape[1] == result.encoderStates.shape[1])
        #expect(result.contextLatents.shape[0] == 1)
        #expect(result.contextLatents.shape[2] == 128)
    }

    @Test("ConditioningActor encode with reference audio")
    func encodeWithReference() async throws {
        let actor = try ConditioningActor.forTesting()
        let refAudio = MLXArray.zeros([1, 800, 64])
        let result = await actor.encode(
            text: "A pop song",
            lyrics: "Hey hey hey",
            referenceAudio: refAudio,
            duration: 10.0,
            precision: .float32
        )
        #expect(result.encoderStates.shape[0] == 1)
        #expect(result.encoderStates.shape[2] == 2048)
        #expect(result.contextLatents.shape[2] == 128)
    }

    @Test("ConditioningActor contextLatents uses zeros for text-only")
    func contextLatentsZerosPath() async throws {
        let actor = try ConditioningActor.forTesting()
        let result = await actor.encode(
            text: "Test",
            lyrics: "",
            referenceAudio: nil,
            duration: 5.0,
            precision: .float32
        )
        let sum = MLX.sum(MLX.abs(result.contextLatents))
        eval(sum)
        #expect(sum.item(Float.self) == 0.0)
    }
}
```

- [ ] **Step 2: Run build to verify it fails**

Run: `cd AnvilCore && swift build 2>&1 | grep "error"`
Expected: Error about `ConditioningActor` not found

- [ ] **Step 3: Implement ConditioningActor**

Create `AnvilCore/Sources/AnvilCore/Inference/Conditioning/ConditioningActor.swift`:

```swift
import Foundation
import MLX
import MLXNN

public actor ConditioningActor {
    private let qwen3: Qwen3Model
    private let lyricEncoder: LyricEncoder
    private let timbreEncoder: TimbreEncoder
    private let conditionEncoder: ConditionEncoder
    private let tokenizer: AnvilTokenizer

    static let timbreFixFrame = 750
    static let sampleRate = 44100
    static let hopLength = 512
    static let latentFrameRate: Float = 25.0

    public init(
        qwen3: Qwen3Model,
        lyricEncoder: LyricEncoder,
        timbreEncoder: TimbreEncoder,
        conditionEncoder: ConditionEncoder,
        tokenizer: AnvilTokenizer
    ) {
        self.qwen3 = qwen3
        self.lyricEncoder = lyricEncoder
        self.timbreEncoder = timbreEncoder
        self.conditionEncoder = conditionEncoder
        self.tokenizer = tokenizer
    }

    public func encode(
        text: String,
        lyrics: String,
        referenceAudio: MLXArray?,
        duration: TimeInterval,
        precision: Precision
    ) -> ConditioningResult {
        let (textIds, textMask) = tokenizer.encodeText(text, maxLength: 256)
        let (lyricIds, lyricMask) = tokenizer.encodeLyrics(lyrics, maxLength: 2048)

        let textHidden = qwen3(inputIds: textIds)
        let lyricEmbeddings = qwen3.embedTokens(lyricIds)
        let lyricHidden = lyricEncoder(embeddings: lyricEmbeddings, attentionMask: lyricMask)

        let referenceLatents = prepareReferenceLatents(referenceAudio)
        let timbreTokens = timbreEncoder(latents: referenceLatents, attentionMask: nil)

        let (encoderStates, encoderMask) = conditionEncoder(
            textHiddenStates: textHidden,
            textMask: textMask,
            lyricHiddenStates: lyricHidden,
            lyricMask: lyricMask,
            timbreTokens: timbreTokens
        )

        let tFrames = Int(Float(duration) * Self.latentFrameRate)
        let contextLatents = MLXArray.zeros([1, tFrames, 128])

        if precision == .float16 {
            return ConditioningResult(
                encoderStates: encoderStates.asType(.float16),
                encoderMask: encoderMask,
                contextLatents: contextLatents.asType(.float16)
            )
        }

        return ConditioningResult(
            encoderStates: encoderStates,
            encoderMask: encoderMask,
            contextLatents: contextLatents
        )
    }

    private func prepareReferenceLatents(_ referenceAudio: MLXArray?) -> MLXArray {
        guard let ref = referenceAudio else {
            return MLXArray.zeros([1, Self.timbreFixFrame, 64])
        }

        let currentFrames = ref.shape[1]
        if currentFrames == Self.timbreFixFrame {
            return ref
        } else if currentFrames > Self.timbreFixFrame {
            return ref[0..<ref.shape[0], 0..<Self.timbreFixFrame]
        } else {
            let padding = MLXArray.zeros([ref.shape[0], Self.timbreFixFrame - currentFrames, 64])
            return concatenated([ref, padding], axis: 1)
        }
    }

    public func loadWeights(
        qwen3Weights: [(String, MLXArray)],
        conditionWeights: [(String, MLXArray)]
    ) {
        qwen3.update(parameters: ModuleParameters.unflattened(qwen3Weights))
        eval(qwen3.parameters())

        let lyricWeights = conditionWeights.compactMap { (key, value) -> (String, MLXArray)? in
            guard key.hasPrefix("lyricEncoder.") else { return nil }
            return (String(key.dropFirst("lyricEncoder.".count)), value)
        }
        lyricEncoder.update(parameters: ModuleParameters.unflattened(lyricWeights))
        eval(lyricEncoder.parameters())

        let timbreWeights = conditionWeights.compactMap { (key, value) -> (String, MLXArray)? in
            guard key.hasPrefix("timbreEncoder.") else { return nil }
            return (String(key.dropFirst("timbreEncoder.".count)), value)
        }
        timbreEncoder.update(parameters: ModuleParameters.unflattened(timbreWeights))
        eval(timbreEncoder.parameters())

        let condEncWeights = conditionWeights.compactMap { (key, value) -> (String, MLXArray)? in
            guard key.hasPrefix("conditionEncoder.") else { return nil }
            return (String(key.dropFirst("conditionEncoder.".count)), value)
        }
        conditionEncoder.update(parameters: ModuleParameters.unflattened(condEncWeights))
        eval(conditionEncoder.parameters())
    }
}

extension ConditioningActor {
    public static func forTesting() throws -> ConditioningActor {
        let tokenizer = try AnvilTokenizer.forTesting()
        return ConditioningActor(
            qwen3: Qwen3Model(),
            lyricEncoder: LyricEncoder(),
            timbreEncoder: TimbreEncoder(),
            conditionEncoder: ConditionEncoder(),
            tokenizer: tokenizer
        )
    }
}
```

- [ ] **Step 4: Verify build succeeds**

Run: `cd AnvilCore && swift build 2>&1 | tail -5`
Expected: Build succeeds

- [ ] **Step 5: Commit**

```bash
git add AnvilCore/Sources/AnvilCore/Inference/Conditioning/ConditioningActor.swift \
        AnvilCore/Tests/AnvilCoreTests/Inference/ConditioningActorTests.swift
git commit -m "feat: add ConditioningActor (tokenize → encode → ConditioningResult)"
```

---

### Task 10: Add Conditioning Weight Conversion to WeightConverter

**Files:**
- Modify: `AnvilCore/Sources/AnvilCore/Inference/Weights/WeightConverter.swift`
- Modify: `AnvilCore/Sources/AnvilCore/Inference/Types.swift` (add `.qwen3` and `.conditionEncoder` to `ModelComponent`)

- [ ] **Step 1: Add new ModelComponent cases**

In `AnvilCore/Sources/AnvilCore/Inference/Types.swift`, update `ModelComponent`:

```swift
public enum ModelComponent: Sendable {
    case vae
    case dit
    case qwen3
    case conditionEncoder
}
```

- [ ] **Step 2: Add conversion methods to WeightConverter**

Add to `AnvilCore/Sources/AnvilCore/Inference/Weights/WeightConverter.swift`:

```swift
    public func convertQwen3Weights(
        from rawWeights: [(String, MLXArray)],
        progress: ((ConversionProgress) -> Void)? = nil
    ) -> [(String, MLXArray)] {
        var result: [(String, MLXArray)] = []
        let total = rawWeights.count

        for (index, (key, value)) in rawWeights.enumerated() {
            progress?(ConversionProgress(current: index + 1, total: total, currentTensorName: key))
            guard let mappedKey = Qwen3NameMap.mapKey(key) else { continue }
            result.append((mappedKey, value))
        }

        return result
    }

    public func convertConditionEncoderWeights(
        from rawWeights: [(String, MLXArray)],
        progress: ((ConversionProgress) -> Void)? = nil
    ) -> [(String, MLXArray)] {
        var result: [(String, MLXArray)] = []
        let total = rawWeights.count

        for (index, (key, value)) in rawWeights.enumerated() {
            progress?(ConversionProgress(current: index + 1, total: total, currentTensorName: key))
            guard let mappedKey = ConditionEncoderNameMap.mapKey(key) else { continue }
            result.append((mappedKey, value))
        }

        return result
    }
```

- [ ] **Step 3: Verify build succeeds**

Run: `cd AnvilCore && swift build 2>&1 | tail -5`
Expected: Build succeeds

- [ ] **Step 4: Commit**

```bash
git add AnvilCore/Sources/AnvilCore/Inference/Types.swift \
        AnvilCore/Sources/AnvilCore/Inference/Weights/WeightConverter.swift
git commit -m "feat: add Qwen3 and ConditionEncoder weight conversion to WeightConverter"
```

---

### Task 11: Final Integration — Wire ConditioningActor into MusicPipeline Interface

**Files:**
- Modify: `AnvilCore/Sources/AnvilCore/Inference/Pipeline/MusicPipeline.swift`

The `MusicPipeline` already accepts `encoderStates` and `contextLatents`. This task adds an optional convenience method that takes raw text/lyrics/reference and uses the `ConditioningActor` internally, so callers can choose either the low-level or high-level API.

- [ ] **Step 1: Add convenience generate method to MusicPipeline**

Add to `AnvilCore/Sources/AnvilCore/Inference/Pipeline/MusicPipeline.swift`:

```swift
    public func generate(
        text: String,
        lyrics: String,
        referenceAudio: MLXArray?,
        duration: TimeInterval,
        config: DiffusionConfig,
        conditioning: ConditioningActor
    ) async throws -> AudioResult {
        let cond = await conditioning.encode(
            text: text,
            lyrics: lyrics,
            referenceAudio: referenceAudio,
            duration: duration,
            precision: config.precision
        )
        return try await generate(
            encoderStates: cond.encoderStates,
            contextLatents: cond.contextLatents,
            duration: duration,
            config: config
        )
    }
```

- [ ] **Step 2: Verify build succeeds**

Run: `cd AnvilCore && swift build 2>&1 | tail -5`
Expected: Build succeeds

- [ ] **Step 3: Commit**

```bash
git add AnvilCore/Sources/AnvilCore/Inference/Pipeline/MusicPipeline.swift
git commit -m "feat: add convenience generate method with ConditioningActor to MusicPipeline"
```

---

## Summary

| Task | Component | Key Output |
|------|-----------|------------|
| 1 | swift-transformers dep + ConditioningResult | Package builds with new dependency |
| 2 | Qwen3Layer + Qwen3Attention | Reusable encoder layer building block |
| 3 | Qwen3Model | Full 28-layer text encoder |
| 4 | LyricEncoder | 8-layer lyric processor (1024→2048) |
| 5 | TimbreEncoder + AttentionPooler | Audio → 5 timbre tokens |
| 6 | ConditionEncoder | Concat + project all conditioning |
| 7 | AnvilTokenizer | BPE tokenization wrapper |
| 8 | Weight name maps | Qwen3 + ConditionEncoder key remapping |
| 9 | ConditioningActor | Top-level orchestrator actor |
| 10 | Weight conversion methods | End-to-end weight loading |
| 11 | Pipeline integration | Convenience API on MusicPipeline |
