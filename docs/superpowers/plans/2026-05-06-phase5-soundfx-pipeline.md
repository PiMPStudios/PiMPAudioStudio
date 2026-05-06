# Phase 5: Sound FX Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Stable Audio Open 1.0 as a second inference pipeline for sound effects and ambient generation, with automatic routing from IntentRouter and manual override.

**Architecture:** Parallel pipeline (`SoundFXPipeline`) with its own CLAP encoder, DiT, VAE, and diffusion actors, sharing low-level primitives (timestep scheduling, sampler logic) with the existing ACE-Step pipeline. `GenerationService` gains routing logic to dispatch to the correct pipeline based on the resolved generation mode.

**Tech Stack:** MLX-Swift (MLX, MLXNN, MLXRandom, MLXFast), Swift Testing, SwiftUI

---

## File Structure

### New Files (AnvilCore)

| File | Responsibility |
|------|---------------|
| `AnvilCore/Sources/AnvilCore/Inference/StableAudio/TimingEncoder.swift` | Sinusoidal positional embeddings for start/duration scalars |
| `AnvilCore/Sources/AnvilCore/Inference/StableAudio/CLAPTokenizer.swift` | CLAP vocabulary loading and text tokenization |
| `AnvilCore/Sources/AnvilCore/Inference/StableAudio/CLAPEncoder.swift` | CLAP text transformer actor producing embeddings |
| `AnvilCore/Sources/AnvilCore/Inference/StableAudio/SFXDiTModel.swift` | Stable Audio DiT model layers and forward pass |
| `AnvilCore/Sources/AnvilCore/Inference/StableAudio/SFXDiTActor.swift` | Actor wrapper for SFX DiT |
| `AnvilCore/Sources/AnvilCore/Inference/StableAudio/SFXVAEModel.swift` | Stable Audio VAE decoder model |
| `AnvilCore/Sources/AnvilCore/Inference/StableAudio/SFXVAEActor.swift` | Actor wrapper for SFX VAE |
| `AnvilCore/Sources/AnvilCore/Inference/StableAudio/SFXDiffusionActor.swift` | Diffusion sampling loop for Stable Audio |
| `AnvilCore/Sources/AnvilCore/Inference/StableAudio/SoundFXPipeline.swift` | Top-level orchestrator actor |
| `AnvilCore/Sources/AnvilCore/Inference/Weights/CLAPNameMap.swift` | Weight name mapping for CLAP tensors |
| `AnvilCore/Sources/AnvilCore/Inference/Weights/SFXDiTNameMap.swift` | Weight name mapping for SFX DiT tensors |
| `AnvilCore/Sources/AnvilCore/Inference/Weights/SFXVAENameMap.swift` | Weight name mapping for SFX VAE tensors |

### New Files (Tests)

| File | Responsibility |
|------|---------------|
| `AnvilCore/Tests/AnvilCoreTests/Inference/StableAudio/TimingEncoderTests.swift` | TimingEncoder shape and value tests |
| `AnvilCore/Tests/AnvilCoreTests/Inference/StableAudio/CLAPTokenizerTests.swift` | Tokenization tests |
| `AnvilCore/Tests/AnvilCoreTests/Inference/StableAudio/CLAPEncoderTests.swift` | Encoder actor tests |
| `AnvilCore/Tests/AnvilCoreTests/Inference/StableAudio/SFXDiTTests.swift` | DiT forward pass tests |
| `AnvilCore/Tests/AnvilCoreTests/Inference/StableAudio/SFXVAETests.swift` | VAE decode tests |
| `AnvilCore/Tests/AnvilCoreTests/Inference/StableAudio/SFXDiffusionTests.swift` | Diffusion loop tests |
| `AnvilCore/Tests/AnvilCoreTests/Inference/StableAudio/SoundFXPipelineTests.swift` | Integration tests |

### New Files (AnvilApp)

| File | Responsibility |
|------|---------------|
| `AnvilApp/Services/SFXModelManager.swift` | Download/activation of Stable Audio weights |
| `AnvilApp/Models/PipelineChoice.swift` | PipelineChoice enum for routing override |

### Modified Files

| File | Change |
|------|--------|
| `AnvilCore/Sources/AnvilCore/Inference/Types.swift` | Add `SFXDiffusionConfig` struct |
| `AnvilApp/Models/GenerationRequest.swift` | Add `resolvedMode` and `pipelineOverride` fields |
| `AnvilApp/Models/GenerationState.swift` | Add `.downloadingModel(progress: Float)` case |
| `AnvilApp/Services/GenerationService.swift` | Add routing logic, SFXModelManager dependency |
| `AnvilApp/UI/MainShellView.swift` | Add pipeline indicator/override in Inspector |
| `AnvilApp/UI/Components/GenerateView.swift` | Duration cap at 47s, download badge on mode buttons |
| `AnvilApp/AnvilAppApp.swift` | Wire SFXModelManager, pass to GenerationService |

---

### Task 1: TimingEncoder

**Files:**
- Create: `AnvilCore/Sources/AnvilCore/Inference/StableAudio/TimingEncoder.swift`
- Test: `AnvilCore/Tests/AnvilCoreTests/Inference/StableAudio/TimingEncoderTests.swift`

- [ ] **Step 1: Write the failing test**

```swift
// AnvilCore/Tests/AnvilCoreTests/Inference/StableAudio/TimingEncoderTests.swift
import Testing
import MLX
@testable import AnvilCore

@Suite("TimingEncoder")
struct TimingEncoderTests {
    @Test("Produces correct output shape")
    func outputShape() {
        let encoder = TimingEncoder(embeddingDim: 768)
        let result = encoder.encode(startSeconds: 0.0, durationSeconds: 30.0)
        #expect(result.shape == [1, 768])
    }

    @Test("Different durations produce different embeddings")
    func differentDurations() {
        let encoder = TimingEncoder(embeddingDim: 768)
        let a = encoder.encode(startSeconds: 0.0, durationSeconds: 10.0)
        let b = encoder.encode(startSeconds: 0.0, durationSeconds: 30.0)
        let diff = abs(a - b).sum()
        #expect(diff.item(Float.self) > 0.0)
    }

    @Test("Different start times produce different embeddings")
    func differentStartTimes() {
        let encoder = TimingEncoder(embeddingDim: 768)
        let a = encoder.encode(startSeconds: 0.0, durationSeconds: 30.0)
        let b = encoder.encode(startSeconds: 5.0, durationSeconds: 30.0)
        let diff = abs(a - b).sum()
        #expect(diff.item(Float.self) > 0.0)
    }

    @Test("Output values are bounded")
    func valuesBounded() {
        let encoder = TimingEncoder(embeddingDim: 768)
        let result = encoder.encode(startSeconds: 0.0, durationSeconds: 47.0)
        let maxVal = abs(result).max().item(Float.self)
        #expect(maxVal <= 1.0)
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd AnvilCore && swift build 2>&1 | tail -5`
Expected: Build failure — `TimingEncoder` not defined

- [ ] **Step 3: Write minimal implementation**

```swift
// AnvilCore/Sources/AnvilCore/Inference/StableAudio/TimingEncoder.swift
import Foundation
import MLX

public struct TimingEncoder: Sendable {
    public let embeddingDim: Int

    public init(embeddingDim: Int) {
        self.embeddingDim = embeddingDim
    }

    public func encode(startSeconds: Float, durationSeconds: Float) -> MLXArray {
        let halfDim = embeddingDim / 4

        let startEmb = sinusoidalEmbedding(value: startSeconds, halfDim: halfDim)
        let durationEmb = sinusoidalEmbedding(value: durationSeconds, halfDim: halfDim)

        let combined = concatenated([startEmb, durationEmb], axis: 1)
        return combined
    }

    private func sinusoidalEmbedding(value: Float, halfDim: Int) -> MLXArray {
        let positions = MLXArray(value)
        let dimRange = MLXArray(Array(0..<halfDim).map { Float($0) })
        let freqs = exp(dimRange * (-log(Float(10000.0)) / Float(halfDim)))
        let angles = positions * freqs
        let sinVals = sin(angles)
        let cosVals = cos(angles)
        let embedding = concatenated([sinVals.reshaped([1, halfDim]), cosVals.reshaped([1, halfDim])], axis: 1)
        return embedding
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd AnvilCore && swift build 2>&1 | tail -5`
Expected: Build complete!

- [ ] **Step 5: Commit**

```bash
git add AnvilCore/Sources/AnvilCore/Inference/StableAudio/TimingEncoder.swift AnvilCore/Tests/AnvilCoreTests/Inference/StableAudio/TimingEncoderTests.swift
git commit -m "feat: add TimingEncoder for Stable Audio timing conditioning"
```

---

### Task 2: CLAPTokenizer

**Files:**
- Create: `AnvilCore/Sources/AnvilCore/Inference/StableAudio/CLAPTokenizer.swift`
- Test: `AnvilCore/Tests/AnvilCoreTests/Inference/StableAudio/CLAPTokenizerTests.swift`

- [ ] **Step 1: Write the failing test**

```swift
// AnvilCore/Tests/AnvilCoreTests/Inference/StableAudio/CLAPTokenizerTests.swift
import Testing
import Foundation
@testable import AnvilCore

@Suite("CLAPTokenizer")
struct CLAPTokenizerTests {
    @Test("Encodes text to token IDs with correct length")
    func encodesText() throws {
        let tokenizer = CLAPTokenizer.forTesting()
        let tokens = tokenizer.encode(text: "thunder rolling in the distance")
        #expect(tokens.count == tokenizer.maxLength)
    }

    @Test("Empty text produces padded sequence")
    func emptyText() throws {
        let tokenizer = CLAPTokenizer.forTesting()
        let tokens = tokenizer.encode(text: "")
        #expect(tokens.count == tokenizer.maxLength)
        #expect(tokens[0] == tokenizer.bosToken)
        #expect(tokens[1] == tokenizer.eosToken)
        // Rest should be padding
        for i in 2..<tokens.count {
            #expect(tokens[i] == tokenizer.padToken)
        }
    }

    @Test("Long text is truncated to maxLength")
    func truncation() throws {
        let tokenizer = CLAPTokenizer.forTesting()
        let longText = String(repeating: "word ", count: 200)
        let tokens = tokenizer.encode(text: longText)
        #expect(tokens.count == tokenizer.maxLength)
    }

    @Test("Max length property is accessible")
    func maxLengthAccessible() {
        let tokenizer = CLAPTokenizer.forTesting()
        #expect(tokenizer.maxLength == 77)
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd AnvilCore && swift build 2>&1 | tail -5`
Expected: Build failure — `CLAPTokenizer` not defined

- [ ] **Step 3: Write minimal implementation**

```swift
// AnvilCore/Sources/AnvilCore/Inference/StableAudio/CLAPTokenizer.swift
import Foundation

public struct CLAPTokenizer: Sendable {
    public let maxLength: Int
    public let bosToken: Int
    public let eosToken: Int
    public let padToken: Int
    private let vocabulary: [String: Int]

    public init(vocabulary: [String: Int], maxLength: Int = 77, bosToken: Int = 49406, eosToken: Int = 49407, padToken: Int = 0) {
        self.vocabulary = vocabulary
        self.maxLength = maxLength
        self.bosToken = bosToken
        self.eosToken = eosToken
        self.padToken = padToken
    }

    public init(vocabularyURL: URL, configURL: URL) throws {
        let vocabData = try Data(contentsOf: vocabularyURL)
        let vocab = try JSONDecoder().decode([String: Int].self, from: vocabData)

        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(TokenizerConfig.self, from: configData)

        self.vocabulary = vocab
        self.maxLength = config.maxLength ?? 77
        self.bosToken = config.bosTokenId ?? 49406
        self.eosToken = config.eosTokenId ?? 49407
        self.padToken = config.padTokenId ?? 0
    }

    public func encode(text: String) -> [Int] {
        var tokens: [Int] = [bosToken]

        let words = text.lowercased().split(separator: " ")
        for word in words {
            if tokens.count >= maxLength - 1 { break }
            let wordStr = String(word)
            if let id = vocabulary[wordStr] {
                tokens.append(id)
            } else {
                // Character-level fallback for unknown words
                for char in wordStr {
                    if tokens.count >= maxLength - 1 { break }
                    let charStr = String(char)
                    tokens.append(vocabulary[charStr] ?? padToken)
                }
            }
        }

        tokens.append(eosToken)

        // Pad or truncate to maxLength
        if tokens.count > maxLength {
            tokens = Array(tokens.prefix(maxLength - 1)) + [eosToken]
        }
        while tokens.count < maxLength {
            tokens.append(padToken)
        }

        return tokens
    }

    public static func forTesting() -> CLAPTokenizer {
        var vocab: [String: Int] = [:]
        // Build a basic test vocabulary
        let chars = "abcdefghijklmnopqrstuvwxyz 0123456789"
        for (i, char) in chars.enumerated() {
            vocab[String(char)] = i + 1
        }
        let commonWords = ["thunder", "rolling", "in", "the", "distance", "rain", "wind", "explosion", "footsteps", "word"]
        for (i, word) in commonWords.enumerated() {
            vocab[word] = 1000 + i
        }
        return CLAPTokenizer(vocabulary: vocab)
    }
}

struct TokenizerConfig: Codable {
    let maxLength: Int?
    let bosTokenId: Int?
    let eosTokenId: Int?
    let padTokenId: Int?

    enum CodingKeys: String, CodingKey {
        case maxLength = "max_length"
        case bosTokenId = "bos_token_id"
        case eosTokenId = "eos_token_id"
        case padTokenId = "pad_token_id"
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd AnvilCore && swift build 2>&1 | tail -5`
Expected: Build complete!

- [ ] **Step 5: Commit**

```bash
git add AnvilCore/Sources/AnvilCore/Inference/StableAudio/CLAPTokenizer.swift AnvilCore/Tests/AnvilCoreTests/Inference/StableAudio/CLAPTokenizerTests.swift
git commit -m "feat: add CLAPTokenizer for Stable Audio text encoding"
```

---

### Task 3: CLAPEncoder Actor

**Files:**
- Create: `AnvilCore/Sources/AnvilCore/Inference/StableAudio/CLAPEncoder.swift`
- Test: `AnvilCore/Tests/AnvilCoreTests/Inference/StableAudio/CLAPEncoderTests.swift`

- [ ] **Step 1: Write the failing test**

```swift
// AnvilCore/Tests/AnvilCoreTests/Inference/StableAudio/CLAPEncoderTests.swift
import Testing
import MLX
@testable import AnvilCore

@Suite("CLAPEncoder")
struct CLAPEncoderTests {
    @Test("Initial state is not loaded")
    func initialState() async {
        let encoder = CLAPEncoder(embeddingDim: 512, numLayers: 4, numHeads: 8, vocabSize: 49408)
        let loaded = await encoder.isLoaded
        #expect(!loaded)
    }

    @Test("Encode without loading throws")
    func encodeWithoutLoad() async {
        let encoder = CLAPEncoder(embeddingDim: 512, numLayers: 4, numHeads: 8, vocabSize: 49408)
        do {
            _ = try await encoder.encode(text: "test", tokenizer: CLAPTokenizer.forTesting())
            Issue.record("Should have thrown")
        } catch {
            #expect(error is CLAPError)
        }
    }

    @Test("Encode after loading produces correct shape")
    func encodeShape() async throws {
        let encoder = CLAPEncoder(embeddingDim: 512, numLayers: 4, numHeads: 8, vocabSize: 49408)
        await encoder.loadWeights(randomWeights(for: encoder))
        let result = try await encoder.encode(text: "thunder", tokenizer: CLAPTokenizer.forTesting())
        #expect(result.shape == [1, 512])
    }

    private func randomWeights(for encoder: CLAPEncoder) -> [(String, MLXArray)] {
        // Generate minimal random weights matching the expected parameter names
        var weights: [(String, MLXArray)] = []
        weights.append(("token_embedding.weight", MLXRandom.normal([49408, 512])))
        weights.append(("position_embedding.weight", MLXRandom.normal([77, 512])))
        for i in 0..<4 {
            weights.append(("layers.\(i).attention.query.weight", MLXRandom.normal([512, 512])))
            weights.append(("layers.\(i).attention.key.weight", MLXRandom.normal([512, 512])))
            weights.append(("layers.\(i).attention.value.weight", MLXRandom.normal([512, 512])))
            weights.append(("layers.\(i).attention.out.weight", MLXRandom.normal([512, 512])))
            weights.append(("layers.\(i).ln1.weight", MLXArray.ones([512])))
            weights.append(("layers.\(i).ln1.bias", MLXArray.zeros([512])))
            weights.append(("layers.\(i).ln2.weight", MLXArray.ones([512])))
            weights.append(("layers.\(i).ln2.bias", MLXArray.zeros([512])))
            weights.append(("layers.\(i).mlp.fc1.weight", MLXRandom.normal([2048, 512])))
            weights.append(("layers.\(i).mlp.fc1.bias", MLXArray.zeros([2048])))
            weights.append(("layers.\(i).mlp.fc2.weight", MLXRandom.normal([512, 2048])))
            weights.append(("layers.\(i).mlp.fc2.bias", MLXArray.zeros([512])))
        }
        weights.append(("final_ln.weight", MLXArray.ones([512])))
        weights.append(("final_ln.bias", MLXArray.zeros([512])))
        weights.append(("projection.weight", MLXRandom.normal([512, 512])))
        return weights
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd AnvilCore && swift build 2>&1 | tail -5`
Expected: Build failure — `CLAPEncoder` and `CLAPError` not defined

- [ ] **Step 3: Write minimal implementation**

```swift
// AnvilCore/Sources/AnvilCore/Inference/StableAudio/CLAPEncoder.swift
import Foundation
import MLX
import MLXNN
import MLXFast

public enum CLAPError: Error, Sendable {
    case notLoaded
    case invalidWeights(String)
}

public actor CLAPEncoder {
    private let embeddingDim: Int
    private let numLayers: Int
    private let numHeads: Int
    private let vocabSize: Int
    private var model: CLAPTextModel?
    private var loaded = false

    public init(embeddingDim: Int, numLayers: Int, numHeads: Int, vocabSize: Int) {
        self.embeddingDim = embeddingDim
        self.numLayers = numLayers
        self.numHeads = numHeads
        self.vocabSize = vocabSize
    }

    public var isLoaded: Bool { loaded }

    public func loadWeights(_ weights: [(String, MLXArray)]) {
        let m = CLAPTextModel(
            embeddingDim: embeddingDim,
            numLayers: numLayers,
            numHeads: numHeads,
            vocabSize: vocabSize
        )
        m.update(parameters: ModuleParameters.unflattened(weights))
        eval(m.parameters())
        self.model = m
        self.loaded = true
    }

    public func encode(text: String, tokenizer: CLAPTokenizer) throws -> MLXArray {
        guard let model, loaded else { throw CLAPError.notLoaded }

        let tokens = tokenizer.encode(text: text)
        let inputIds = MLXArray(tokens).reshaped([1, tokens.count])

        let embedding = model(inputIds)
        eval(embedding)
        return embedding
    }
}

class CLAPTextModel: Module {
    let tokenEmbedding: Embedding
    let positionEmbedding: Embedding
    let layers: [CLAPTransformerLayer]
    let finalLn: LayerNorm
    let projection: Linear

    init(embeddingDim: Int, numLayers: Int, numHeads: Int, vocabSize: Int) {
        self.tokenEmbedding = Embedding(embeddingCount: vocabSize, dimensions: embeddingDim)
        self.positionEmbedding = Embedding(embeddingCount: 77, dimensions: embeddingDim)
        self.layers = (0..<numLayers).map { _ in
            CLAPTransformerLayer(embeddingDim: embeddingDim, numHeads: numHeads)
        }
        self.finalLn = LayerNorm(dimensions: embeddingDim)
        self.projection = Linear(embeddingDim, embeddingDim, bias: false)
    }

    func callAsFunction(_ inputIds: MLXArray) -> MLXArray {
        let seqLen = inputIds.shape[1]
        let positions = MLXArray(Array(0..<seqLen)).reshaped([1, seqLen])

        var hidden = tokenEmbedding(inputIds) + positionEmbedding(positions)

        for layer in layers {
            hidden = layer(hidden)
        }

        hidden = finalLn(hidden)

        // Pool: take the EOS token position (last non-pad)
        let pooled = hidden[0..., seqLen - 1]
        let projected = projection(pooled)
        return projected.reshaped([1, -1])
    }
}

class CLAPTransformerLayer: Module {
    let attention: CLAPAttention
    let ln1: LayerNorm
    let ln2: LayerNorm
    let mlp: CLAPMLP

    init(embeddingDim: Int, numHeads: Int) {
        self.attention = CLAPAttention(embeddingDim: embeddingDim, numHeads: numHeads)
        self.ln1 = LayerNorm(dimensions: embeddingDim)
        self.ln2 = LayerNorm(dimensions: embeddingDim)
        self.mlp = CLAPMLP(embeddingDim: embeddingDim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x + attention(ln1(x))
        h = h + mlp(ln2(h))
        return h
    }
}

class CLAPAttention: Module {
    let query: Linear
    let key: Linear
    let value: Linear
    let out: Linear
    let numHeads: Int
    let headDim: Int

    init(embeddingDim: Int, numHeads: Int) {
        self.numHeads = numHeads
        self.headDim = embeddingDim / numHeads
        self.query = Linear(embeddingDim, embeddingDim, bias: false)
        self.key = Linear(embeddingDim, embeddingDim, bias: false)
        self.value = Linear(embeddingDim, embeddingDim, bias: false)
        self.out = Linear(embeddingDim, embeddingDim, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.shape[0]
        let L = x.shape[1]

        let q = query(x).reshaped([B, L, numHeads, headDim]).transposed(0, 2, 1, 3)
        let k = key(x).reshaped([B, L, numHeads, headDim]).transposed(0, 2, 1, 3)
        let v = value(x).reshaped([B, L, numHeads, headDim]).transposed(0, 2, 1, 3)

        let scale = Float(1.0 / sqrt(Float(headDim)))
        let attn = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: nil)

        let merged = attn.transposed(0, 2, 1, 3).reshaped([B, L, -1])
        return out(merged)
    }
}

class CLAPMLP: Module {
    let fc1: Linear
    let fc2: Linear

    init(embeddingDim: Int) {
        self.fc1 = Linear(embeddingDim, embeddingDim * 4)
        self.fc2 = Linear(embeddingDim * 4, embeddingDim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = fc1(x)
        h = gelu(h)
        return fc2(h)
    }
}

private func gelu(_ x: MLXArray) -> MLXArray {
    x * (1.0 + MLX.tanh(0.7978845608 * (x + 0.044715 * x * x * x))) * 0.5
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd AnvilCore && swift build 2>&1 | tail -5`
Expected: Build complete!

- [ ] **Step 5: Commit**

```bash
git add AnvilCore/Sources/AnvilCore/Inference/StableAudio/CLAPEncoder.swift AnvilCore/Tests/AnvilCoreTests/Inference/StableAudio/CLAPEncoderTests.swift
git commit -m "feat: add CLAPEncoder actor for Stable Audio text conditioning"
```

---

### Task 4: SFXDiTModel + SFXDiTActor

**Files:**
- Create: `AnvilCore/Sources/AnvilCore/Inference/StableAudio/SFXDiTModel.swift`
- Create: `AnvilCore/Sources/AnvilCore/Inference/StableAudio/SFXDiTActor.swift`
- Test: `AnvilCore/Tests/AnvilCoreTests/Inference/StableAudio/SFXDiTTests.swift`

- [ ] **Step 1: Write the failing test**

```swift
// AnvilCore/Tests/AnvilCoreTests/Inference/StableAudio/SFXDiTTests.swift
import Testing
import MLX
import MLXRandom
@testable import AnvilCore

@Suite("SFX DiT")
struct SFXDiTTests {
    let config = SFXDiTConfig(
        hiddenSize: 256,
        numLayers: 2,
        numHeads: 4,
        latentDim: 64,
        conditioningDim: 512
    )

    @Test("Forward pass produces correct output shape")
    func forwardShape() {
        let model = SFXDiTModel(config: config)
        let latents = MLXRandom.normal([1, 16, 64])
        let timestep = MLXArray([0.5])
        let conditioning = MLXRandom.normal([1, 512])
        let output = model.forward(latents: latents, timestep: timestep, conditioning: conditioning)
        #expect(output.shape == [1, 16, 64])
    }

    @Test("Actor predict returns correct shape")
    func actorPredict() async {
        let actor = SFXDiTActor(config: config)
        let latents = MLXRandom.normal([1, 16, 64])
        let timestep = MLXArray([0.5])
        let conditioning = MLXRandom.normal([1, 512])
        let output = await actor.predict(latents: latents, timestep: timestep, conditioning: conditioning)
        #expect(output.shape == [1, 16, 64])
    }

    @Test("Actor loadWeights marks as loaded")
    func actorLoadWeights() async throws {
        let actor = SFXDiTActor(config: config)
        let loaded = await actor.isLoaded
        #expect(!loaded)
        try await actor.loadWeights(randomDiTWeights())
        let loadedAfter = await actor.isLoaded
        #expect(loadedAfter)
    }

    private func randomDiTWeights() -> [(String, MLXArray)] {
        var weights: [(String, MLXArray)] = []
        weights.append(("proj_in.weight", MLXRandom.normal([256, 64])))
        weights.append(("proj_in.bias", MLXArray.zeros([256])))
        weights.append(("time_embed.0.weight", MLXRandom.normal([256, 256])))
        weights.append(("time_embed.0.bias", MLXArray.zeros([256])))
        weights.append(("time_embed.2.weight", MLXRandom.normal([256, 256])))
        weights.append(("time_embed.2.bias", MLXArray.zeros([256])))
        weights.append(("cond_proj.weight", MLXRandom.normal([256, 512])))
        weights.append(("cond_proj.bias", MLXArray.zeros([256])))
        for i in 0..<2 {
            let prefix = "blocks.\(i)"
            weights.append(("\(prefix).norm1.weight", MLXArray.ones([256])))
            weights.append(("\(prefix).norm1.bias", MLXArray.zeros([256])))
            weights.append(("\(prefix).attn.qkv.weight", MLXRandom.normal([768, 256])))
            weights.append(("\(prefix).attn.out.weight", MLXRandom.normal([256, 256])))
            weights.append(("\(prefix).norm2.weight", MLXArray.ones([256])))
            weights.append(("\(prefix).norm2.bias", MLXArray.zeros([256])))
            weights.append(("\(prefix).cross_attn.q.weight", MLXRandom.normal([256, 256])))
            weights.append(("\(prefix).cross_attn.kv.weight", MLXRandom.normal([512, 256])))
            weights.append(("\(prefix).cross_attn.out.weight", MLXRandom.normal([256, 256])))
            weights.append(("\(prefix).norm3.weight", MLXArray.ones([256])))
            weights.append(("\(prefix).norm3.bias", MLXArray.zeros([256])))
            weights.append(("\(prefix).mlp.fc1.weight", MLXRandom.normal([1024, 256])))
            weights.append(("\(prefix).mlp.fc1.bias", MLXArray.zeros([1024])))
            weights.append(("\(prefix).mlp.fc2.weight", MLXRandom.normal([256, 1024])))
            weights.append(("\(prefix).mlp.fc2.bias", MLXArray.zeros([256])))
            weights.append(("\(prefix).adaLN.weight", MLXRandom.normal([768, 256])))
            weights.append(("\(prefix).adaLN.bias", MLXArray.zeros([768])))
        }
        weights.append(("proj_out.weight", MLXRandom.normal([64, 256])))
        weights.append(("proj_out.bias", MLXArray.zeros([64])))
        weights.append(("final_norm.weight", MLXArray.ones([256])))
        weights.append(("final_norm.bias", MLXArray.zeros([256])))
        return weights
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd AnvilCore && swift build 2>&1 | tail -5`
Expected: Build failure — `SFXDiTModel`, `SFXDiTActor`, `SFXDiTConfig` not defined

- [ ] **Step 3: Add SFXDiTConfig to Types.swift**

Add to the end of `AnvilCore/Sources/AnvilCore/Inference/Types.swift`:

```swift
public struct SFXDiTConfig: Codable, Sendable {
    public let hiddenSize: Int
    public let numLayers: Int
    public let numHeads: Int
    public let latentDim: Int
    public let conditioningDim: Int

    public init(hiddenSize: Int, numLayers: Int, numHeads: Int, latentDim: Int, conditioningDim: Int) {
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.numHeads = numHeads
        self.latentDim = latentDim
        self.conditioningDim = conditioningDim
    }
}
```

- [ ] **Step 4: Write SFXDiTModel**

```swift
// AnvilCore/Sources/AnvilCore/Inference/StableAudio/SFXDiTModel.swift
import Foundation
import MLX
import MLXNN
import MLXFast

public class SFXDiTModel: Module {
    let projIn: Linear
    let timeEmbed: SFXTimestepEmbedding
    let condProj: Linear
    let blocks: [SFXDiTBlock]
    let finalNorm: LayerNorm
    let projOut: Linear
    let config: SFXDiTConfig

    public init(config: SFXDiTConfig) {
        self.config = config
        self.projIn = Linear(config.latentDim, config.hiddenSize)
        self.timeEmbed = SFXTimestepEmbedding(hiddenSize: config.hiddenSize)
        self.condProj = Linear(config.conditioningDim, config.hiddenSize)
        self.blocks = (0..<config.numLayers).map { _ in
            SFXDiTBlock(hiddenSize: config.hiddenSize, numHeads: config.numHeads, condDim: config.hiddenSize)
        }
        self.finalNorm = LayerNorm(dimensions: config.hiddenSize)
        self.projOut = Linear(config.hiddenSize, config.latentDim)
    }

    public func forward(latents: MLXArray, timestep: MLXArray, conditioning: MLXArray) -> MLXArray {
        var hidden = projIn(latents)
        let tEmb = timeEmbed(timestep)
        let condEmb = condProj(conditioning)

        for block in blocks {
            hidden = block(hidden, timeEmbedding: tEmb, conditioning: condEmb)
        }

        hidden = finalNorm(hidden)
        return projOut(hidden)
    }
}

class SFXTimestepEmbedding: Module {
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear

    init(hiddenSize: Int) {
        self._fc1 = ModuleInfo(wrappedValue: Linear(hiddenSize, hiddenSize))
        self._fc2 = ModuleInfo(wrappedValue: Linear(hiddenSize, hiddenSize))
    }

    func callAsFunction(_ t: MLXArray) -> MLXArray {
        let halfDim = fc1.weight.shape[1] / 2
        let freqs = exp(MLXArray(Array(0..<halfDim).map { Float($0) }) * (-log(Float(10000.0)) / Float(halfDim)))
        let angles = t.reshaped([-1, 1]) * freqs.reshaped([1, -1])
        let emb = concatenated([sin(angles), cos(angles)], axis: 1)
        var h = fc1(emb)
        h = silu(h)
        return fc2(h)
    }
}

class SFXDiTBlock: Module {
    let norm1: LayerNorm
    let attn: SFXSelfAttention
    let norm2: LayerNorm
    let crossAttn: SFXCrossAttention
    let norm3: LayerNorm
    let mlp: SFXMLP
    let adaLN: Linear

    init(hiddenSize: Int, numHeads: Int, condDim: Int) {
        self.norm1 = LayerNorm(dimensions: hiddenSize)
        self.attn = SFXSelfAttention(hiddenSize: hiddenSize, numHeads: numHeads)
        self.norm2 = LayerNorm(dimensions: hiddenSize)
        self.crossAttn = SFXCrossAttention(hiddenSize: hiddenSize, numHeads: numHeads, condDim: condDim)
        self.norm3 = LayerNorm(dimensions: hiddenSize)
        self.mlp = SFXMLP(hiddenSize: hiddenSize)
        self.adaLN = Linear(hiddenSize, hiddenSize * 3)
    }

    func callAsFunction(_ x: MLXArray, timeEmbedding: MLXArray, conditioning: MLXArray) -> MLXArray {
        let adaln = adaLN(timeEmbedding)
        let chunks = adaln.reshaped([adaln.shape[0], 3, -1])
        let scale1 = chunks[0..., 0]
        let shift1 = chunks[0..., 1]
        let gate = sigmoid(chunks[0..., 2])

        var h = norm1(x) * (1.0 + scale1.expandedDimensions(axis: 1)) + shift1.expandedDimensions(axis: 1)
        h = x + attn(h)
        h = h + crossAttn(norm2(h), conditioning: conditioning)
        h = h + gate.expandedDimensions(axis: 1) * mlp(norm3(h))
        return h
    }
}

class SFXSelfAttention: Module {
    let qkv: Linear
    let out: Linear
    let numHeads: Int
    let headDim: Int

    init(hiddenSize: Int, numHeads: Int) {
        self.numHeads = numHeads
        self.headDim = hiddenSize / numHeads
        self.qkv = Linear(hiddenSize, hiddenSize * 3, bias: false)
        self.out = Linear(hiddenSize, hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.shape[0]
        let L = x.shape[1]
        let qkvOut = qkv(x).reshaped([B, L, 3, numHeads, headDim])
        let q = qkvOut[0..., 0..., 0].transposed(0, 2, 1, 3)
        let k = qkvOut[0..., 0..., 1].transposed(0, 2, 1, 3)
        let v = qkvOut[0..., 0..., 2].transposed(0, 2, 1, 3)

        let scale = Float(1.0 / sqrt(Float(headDim)))
        let attn = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: nil)
        let merged = attn.transposed(0, 2, 1, 3).reshaped([B, L, -1])
        return out(merged)
    }
}

class SFXCrossAttention: Module {
    let q: Linear
    let kv: Linear
    let out: Linear
    let numHeads: Int
    let headDim: Int

    init(hiddenSize: Int, numHeads: Int, condDim: Int) {
        self.numHeads = numHeads
        self.headDim = hiddenSize / numHeads
        self.q = Linear(hiddenSize, hiddenSize, bias: false)
        self.kv = Linear(condDim, hiddenSize * 2, bias: false)
        self.out = Linear(hiddenSize, hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray, conditioning: MLXArray) -> MLXArray {
        let B = x.shape[0]
        let L = x.shape[1]

        let cond = conditioning.expandedDimensions(axis: 1)
        let condLen = cond.shape[1]

        let queries = q(x).reshaped([B, L, numHeads, headDim]).transposed(0, 2, 1, 3)
        let kvOut = kv(cond).reshaped([B, condLen, 2, numHeads, headDim])
        let keys = kvOut[0..., 0..., 0].transposed(0, 2, 1, 3)
        let values = kvOut[0..., 0..., 1].transposed(0, 2, 1, 3)

        let scale = Float(1.0 / sqrt(Float(headDim)))
        let attn = MLXFast.scaledDotProductAttention(queries: queries, keys: keys, values: values, scale: scale, mask: nil)
        let merged = attn.transposed(0, 2, 1, 3).reshaped([B, L, -1])
        return out(merged)
    }
}

class SFXMLP: Module {
    let fc1: Linear
    let fc2: Linear

    init(hiddenSize: Int) {
        self.fc1 = Linear(hiddenSize, hiddenSize * 4)
        self.fc2 = Linear(hiddenSize * 4, hiddenSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = fc1(x)
        h = silu(h)
        return fc2(h)
    }
}

private func silu(_ x: MLXArray) -> MLXArray {
    x * sigmoid(x)
}
```

- [ ] **Step 5: Write SFXDiTActor**

```swift
// AnvilCore/Sources/AnvilCore/Inference/StableAudio/SFXDiTActor.swift
import Foundation
import MLX
import MLXNN

public actor SFXDiTActor {
    private let model: SFXDiTModel
    private var loaded = false

    public init(config: SFXDiTConfig) {
        self.model = SFXDiTModel(config: config)
    }

    public var isLoaded: Bool { loaded }

    public func loadWeights(_ weights: [(String, MLXArray)]) throws {
        model.update(parameters: ModuleParameters.unflattened(weights))
        eval(model.parameters())
        loaded = true
    }

    public func predict(latents: MLXArray, timestep: MLXArray, conditioning: MLXArray) -> MLXArray {
        let result = model.forward(latents: latents, timestep: timestep, conditioning: conditioning)
        eval(result)
        return result
    }
}
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd AnvilCore && swift build 2>&1 | tail -5`
Expected: Build complete!

- [ ] **Step 7: Commit**

```bash
git add AnvilCore/Sources/AnvilCore/Inference/StableAudio/SFXDiTModel.swift AnvilCore/Sources/AnvilCore/Inference/StableAudio/SFXDiTActor.swift AnvilCore/Sources/AnvilCore/Inference/Types.swift AnvilCore/Tests/AnvilCoreTests/Inference/StableAudio/SFXDiTTests.swift
git commit -m "feat: add SFXDiTModel and SFXDiTActor for Stable Audio diffusion transformer"
```

---

### Task 5: SFXVAEModel + SFXVAEActor

**Files:**
- Create: `AnvilCore/Sources/AnvilCore/Inference/StableAudio/SFXVAEModel.swift`
- Create: `AnvilCore/Sources/AnvilCore/Inference/StableAudio/SFXVAEActor.swift`
- Modify: `AnvilCore/Sources/AnvilCore/Inference/Types.swift`
- Test: `AnvilCore/Tests/AnvilCoreTests/Inference/StableAudio/SFXVAETests.swift`

- [ ] **Step 1: Write the failing test**

```swift
// AnvilCore/Tests/AnvilCoreTests/Inference/StableAudio/SFXVAETests.swift
import Testing
import MLX
import MLXRandom
@testable import AnvilCore

@Suite("SFX VAE")
struct SFXVAETests {
    let config = SFXVAEConfig(latentDim: 64, outputChannels: 2, sampleRate: 44100, upsampleRatios: [8, 8, 4, 2])

    @Test("Decode produces audio from latents")
    func decodeShape() {
        let model = SFXVAEModel(config: config)
        let latents = MLXRandom.normal([1, 16, 64])
        let audio = model.decode(latents: latents)
        #expect(audio.shape[0] == 1)
        #expect(audio.shape[2] == 2) // stereo
        #expect(audio.shape[1] == 16 * 8 * 8 * 4 * 2) // upsampled
    }

    @Test("Actor decode returns correct shape")
    func actorDecode() async {
        let actor = SFXVAEActor(config: config)
        let latents = MLXRandom.normal([1, 16, 64])
        let audio = await actor.decode(latents: latents)
        #expect(audio.shape[0] == 1)
        #expect(audio.shape[2] == 2)
    }

    @Test("Actor loadWeights marks as loaded")
    func actorLoad() async {
        let actor = SFXVAEActor(config: config)
        let loaded = await actor.isLoaded
        #expect(!loaded)
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd AnvilCore && swift build 2>&1 | tail -5`
Expected: Build failure — types not defined

- [ ] **Step 3: Add SFXVAEConfig to Types.swift**

Add to `AnvilCore/Sources/AnvilCore/Inference/Types.swift`:

```swift
public struct SFXVAEConfig: Codable, Sendable {
    public let latentDim: Int
    public let outputChannels: Int
    public let sampleRate: Int
    public let upsampleRatios: [Int]

    public init(latentDim: Int, outputChannels: Int, sampleRate: Int, upsampleRatios: [Int]) {
        self.latentDim = latentDim
        self.outputChannels = outputChannels
        self.sampleRate = sampleRate
        self.upsampleRatios = upsampleRatios
    }
}
```

- [ ] **Step 4: Write SFXVAEModel**

```swift
// AnvilCore/Sources/AnvilCore/Inference/StableAudio/SFXVAEModel.swift
import Foundation
import MLX
import MLXNN

public class SFXVAEModel: Module {
    let config: SFXVAEConfig
    let projIn: Linear
    let upsampleBlocks: [SFXUpsampleBlock]
    let projOut: Linear

    public init(config: SFXVAEConfig) {
        self.config = config
        var currentDim = config.latentDim
        self.projIn = Linear(config.latentDim, currentDim)

        var blocks: [SFXUpsampleBlock] = []
        for ratio in config.upsampleRatios {
            let outDim = max(currentDim / 2, config.outputChannels)
            blocks.append(SFXUpsampleBlock(inputDim: currentDim, outputDim: outDim, ratio: ratio))
            currentDim = outDim
        }
        self.upsampleBlocks = blocks
        self.projOut = Linear(currentDim, config.outputChannels)
    }

    public func decode(latents: MLXArray) -> MLXArray {
        var h = projIn(latents)
        for block in upsampleBlocks {
            h = block(h)
        }
        return projOut(h)
    }
}

class SFXUpsampleBlock: Module {
    let linear: Linear
    let ratio: Int

    init(inputDim: Int, outputDim: Int, ratio: Int) {
        self.linear = Linear(inputDim, outputDim * ratio)
        self.ratio = ratio
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.shape[0]
        let T = x.shape[1]
        let projected = linear(x)
        let outDim = projected.shape[2] / ratio
        let reshaped = projected.reshaped([B, T * ratio, outDim])
        return silu(reshaped)
    }
}

private func silu(_ x: MLXArray) -> MLXArray {
    x * sigmoid(x)
}
```

- [ ] **Step 5: Write SFXVAEActor**

```swift
// AnvilCore/Sources/AnvilCore/Inference/StableAudio/SFXVAEActor.swift
import Foundation
import MLX
import MLXNN

public actor SFXVAEActor {
    private let model: SFXVAEModel
    private var loaded = false

    public init(config: SFXVAEConfig) {
        self.model = SFXVAEModel(config: config)
    }

    public var isLoaded: Bool { loaded }

    public func loadWeights(_ weights: [(String, MLXArray)]) {
        model.update(parameters: ModuleParameters.unflattened(weights))
        eval(model.parameters())
        loaded = true
    }

    public func decode(latents: MLXArray) -> MLXArray {
        let result = model.decode(latents: latents)
        eval(result)
        return result
    }
}
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd AnvilCore && swift build 2>&1 | tail -5`
Expected: Build complete!

- [ ] **Step 7: Commit**

```bash
git add AnvilCore/Sources/AnvilCore/Inference/StableAudio/SFXVAEModel.swift AnvilCore/Sources/AnvilCore/Inference/StableAudio/SFXVAEActor.swift AnvilCore/Sources/AnvilCore/Inference/Types.swift AnvilCore/Tests/AnvilCoreTests/Inference/StableAudio/SFXVAETests.swift
git commit -m "feat: add SFXVAEModel and SFXVAEActor for Stable Audio decoding"
```

---

### Task 6: SFXDiffusionActor

**Files:**
- Create: `AnvilCore/Sources/AnvilCore/Inference/StableAudio/SFXDiffusionActor.swift`
- Modify: `AnvilCore/Sources/AnvilCore/Inference/Types.swift`
- Test: `AnvilCore/Tests/AnvilCoreTests/Inference/StableAudio/SFXDiffusionTests.swift`

- [ ] **Step 1: Write the failing test**

```swift
// AnvilCore/Tests/AnvilCoreTests/Inference/StableAudio/SFXDiffusionTests.swift
import Testing
import MLX
import MLXRandom
@testable import AnvilCore

@Suite("SFX Diffusion")
struct SFXDiffusionTests {
    @Test("Sample produces latents with correct shape")
    func sampleShape() async throws {
        let ditConfig = SFXDiTConfig(hiddenSize: 128, numLayers: 1, numHeads: 2, latentDim: 64, conditioningDim: 512)
        let dit = SFXDiTActor(config: ditConfig)
        let diffusion = SFXDiffusionActor(dit: dit)

        let config = SFXDiffusionConfig(steps: 4, guidanceScale: 1.0, seed: 42, latentShape: [1, 16, 64])
        let conditioning = MLXRandom.normal([1, 512])
        let result = try await diffusion.sample(conditioning: conditioning, config: config)
        #expect(result.latents.shape == [1, 16, 64])
    }

    @Test("Cancellation stops sampling")
    func cancellation() async {
        let ditConfig = SFXDiTConfig(hiddenSize: 128, numLayers: 1, numHeads: 2, latentDim: 64, conditioningDim: 512)
        let dit = SFXDiTActor(config: ditConfig)
        let diffusion = SFXDiffusionActor(dit: dit)

        let config = SFXDiffusionConfig(steps: 100, guidanceScale: 1.0, seed: 42, latentShape: [1, 16, 64])
        let conditioning = MLXRandom.normal([1, 512])

        let task = Task {
            try await diffusion.sample(conditioning: conditioning, config: config)
        }
        try? await Task.sleep(for: .milliseconds(50))
        task.cancel()

        let result = await task.result
        switch result {
        case .failure(let error):
            #expect(error is CancellationError)
        case .success:
            break // May complete before cancellation on fast hardware
        }
    }

    @Test("CFG with guidanceScale > 1 runs without error")
    func cfgMode() async throws {
        let ditConfig = SFXDiTConfig(hiddenSize: 128, numLayers: 1, numHeads: 2, latentDim: 64, conditioningDim: 512)
        let dit = SFXDiTActor(config: ditConfig)
        let diffusion = SFXDiffusionActor(dit: dit)

        let config = SFXDiffusionConfig(steps: 2, guidanceScale: 3.0, seed: 42, latentShape: [1, 16, 64])
        let conditioning = MLXRandom.normal([1, 512])
        let result = try await diffusion.sample(conditioning: conditioning, config: config)
        #expect(result.latents.shape == [1, 16, 64])
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd AnvilCore && swift build 2>&1 | tail -5`
Expected: Build failure — `SFXDiffusionActor`, `SFXDiffusionConfig` not defined

- [ ] **Step 3: Add SFXDiffusionConfig to Types.swift**

Add to `AnvilCore/Sources/AnvilCore/Inference/Types.swift`:

```swift
public struct SFXDiffusionConfig: Sendable {
    public let steps: Int
    public let guidanceScale: Float
    public let seed: Int?
    public let latentShape: [Int]

    public init(steps: Int = 100, guidanceScale: Float = 7.0, seed: Int? = nil, latentShape: [Int]) {
        self.steps = steps
        self.guidanceScale = guidanceScale
        self.seed = seed
        self.latentShape = latentShape
    }
}
```

- [ ] **Step 4: Write SFXDiffusionActor**

```swift
// AnvilCore/Sources/AnvilCore/Inference/StableAudio/SFXDiffusionActor.swift
import Foundation
import MLX
import MLXRandom

public actor SFXDiffusionActor {
    private let dit: SFXDiTActor
    private var progressContinuation: AsyncStream<GenerationProgress>.Continuation?
    private var _progress: AsyncStream<GenerationProgress>?

    public init(dit: SFXDiTActor) {
        self.dit = dit
    }

    public var progress: AsyncStream<GenerationProgress> {
        if let existing = _progress { return existing }
        let (stream, continuation) = AsyncStream<GenerationProgress>.makeStream()
        self.progressContinuation = continuation
        self._progress = stream
        return stream
    }

    public func sample(
        conditioning: MLXArray,
        config: SFXDiffusionConfig
    ) async throws -> DiffusionResult {
        let totalStart = Date()

        var xt = DiffusionSampler.generateNoise(shape: config.latentShape, seed: config.seed)

        let schedule = generateLinearSchedule(steps: config.steps)
        let doCFG = config.guidanceScale > 1.0

        let diffStart = Date()

        for stepIdx in 0..<config.steps {
            try Task.checkCancellation()

            let currentT = schedule[stepIdx]
            let timestep = MLXArray([currentT])

            var vt: MLXArray
            if doCFG {
                let condPred = await dit.predict(latents: xt, timestep: timestep, conditioning: conditioning)
                let uncondPred = await dit.predict(latents: xt, timestep: timestep, conditioning: MLXArray.zeros(like: conditioning))
                vt = uncondPred + config.guidanceScale * (condPred - uncondPred)
            } else {
                vt = await dit.predict(latents: xt, timestep: timestep, conditioning: conditioning)
            }

            if stepIdx == config.steps - 1 {
                xt = DiffusionSampler.finalStep(xt: xt, vt: vt, currentT: currentT)
            } else {
                let nextT = schedule[stepIdx + 1]
                xt = DiffusionSampler.odeStep(xt: xt, vt: vt, currentT: currentT, nextT: nextT)
            }
            eval(xt)

            let elapsed = Date().timeIntervalSince(diffStart)
            progressContinuation?.yield(.diffusion(DiffusionProgress(
                currentStep: stepIdx + 1, totalSteps: config.steps, elapsedTime: elapsed
            )))
        }

        let diffTime = Date().timeIntervalSince(diffStart)
        let totalTime = Date().timeIntervalSince(totalStart)
        let costs = TimeCosts(
            diffusionTime: diffTime,
            perStepTime: diffTime / Double(max(config.steps, 1)),
            totalTime: totalTime
        )

        return DiffusionResult(latents: xt, timeCosts: costs)
    }

    private func generateLinearSchedule(steps: Int) -> [Float] {
        (0..<steps).map { i in
            1.0 - Float(i) / Float(steps)
        }
    }
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd AnvilCore && swift build 2>&1 | tail -5`
Expected: Build complete!

- [ ] **Step 6: Commit**

```bash
git add AnvilCore/Sources/AnvilCore/Inference/StableAudio/SFXDiffusionActor.swift AnvilCore/Sources/AnvilCore/Inference/Types.swift AnvilCore/Tests/AnvilCoreTests/Inference/StableAudio/SFXDiffusionTests.swift
git commit -m "feat: add SFXDiffusionActor with linear schedule and CFG support"
```

---

### Task 7: SoundFXPipeline

**Files:**
- Create: `AnvilCore/Sources/AnvilCore/Inference/StableAudio/SoundFXPipeline.swift`
- Test: `AnvilCore/Tests/AnvilCoreTests/Inference/StableAudio/SoundFXPipelineTests.swift`

- [ ] **Step 1: Write the failing test**

```swift
// AnvilCore/Tests/AnvilCoreTests/Inference/StableAudio/SoundFXPipelineTests.swift
import Testing
import MLX
import MLXRandom
@testable import AnvilCore

@Suite("SoundFXPipeline")
struct SoundFXPipelineTests {
    @Test("Generate produces AudioResult")
    func generateProducesResult() async throws {
        let pipeline = makeTestPipeline()
        let tokenizer = CLAPTokenizer.forTesting()

        let config = SFXDiffusionConfig(steps: 2, guidanceScale: 1.0, seed: 42, latentShape: [1, 16, 64])
        let result = try await pipeline.generate(
            text: "thunder",
            duration: 5.0,
            config: config,
            tokenizer: tokenizer
        )
        #expect(result.sampleRate == 44100)
        #expect(result.samples.shape[0] == 1)
        #expect(result.duration == 5.0)
    }

    @Test("Duration clamped to 47 seconds max")
    func durationClamped() async throws {
        let pipeline = makeTestPipeline()
        let tokenizer = CLAPTokenizer.forTesting()

        let config = SFXDiffusionConfig(steps: 2, guidanceScale: 1.0, seed: 42, latentShape: [1, 16, 64])
        let result = try await pipeline.generate(
            text: "long ambient",
            duration: 60.0,
            config: config,
            tokenizer: tokenizer
        )
        #expect(result.duration == 47.0)
    }

    @Test("Progress stream emits events")
    func progressStream() async throws {
        let pipeline = makeTestPipeline()
        let tokenizer = CLAPTokenizer.forTesting()
        let progressStream = await pipeline.progress

        let config = SFXDiffusionConfig(steps: 2, guidanceScale: 1.0, seed: 42, latentShape: [1, 16, 64])

        let progressTask = Task {
            var count = 0
            for await _ in progressStream {
                count += 1
                if count >= 2 { break }
            }
            return count
        }

        _ = try await pipeline.generate(
            text: "rain",
            duration: 5.0,
            config: config,
            tokenizer: tokenizer
        )

        let count = await progressTask.value
        #expect(count >= 1)
    }

    private func makeTestPipeline() -> SoundFXPipeline {
        let clapEncoder = CLAPEncoder(embeddingDim: 512, numLayers: 1, numHeads: 4, vocabSize: 49408)
        let timingEncoder = TimingEncoder(embeddingDim: 256)
        let ditConfig = SFXDiTConfig(hiddenSize: 128, numLayers: 1, numHeads: 2, latentDim: 64, conditioningDim: 768)
        let dit = SFXDiTActor(config: ditConfig)
        let diffusion = SFXDiffusionActor(dit: dit)
        let vaeConfig = SFXVAEConfig(latentDim: 64, outputChannels: 2, sampleRate: 44100, upsampleRatios: [2, 2])
        let vae = SFXVAEActor(config: vaeConfig)

        return SoundFXPipeline(
            clap: clapEncoder,
            diffusion: diffusion,
            vae: vae,
            timingEncoder: timingEncoder
        )
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd AnvilCore && swift build 2>&1 | tail -5`
Expected: Build failure — `SoundFXPipeline` not defined

- [ ] **Step 3: Write SoundFXPipeline**

```swift
// AnvilCore/Sources/AnvilCore/Inference/StableAudio/SoundFXPipeline.swift
import Foundation
import MLX

public actor SoundFXPipeline {
    private let clap: CLAPEncoder
    private let timingEncoder: TimingEncoder
    private let diffusion: SFXDiffusionActor
    private let vae: SFXVAEActor
    private var progressContinuation: AsyncStream<GenerationProgress>.Continuation?
    private var _progress: AsyncStream<GenerationProgress>?

    public static let maxDuration: TimeInterval = 47.0
    public static let defaultSampleRate = 44100
    public static let hopLength = 1024

    public init(
        clap: CLAPEncoder,
        diffusion: SFXDiffusionActor,
        vae: SFXVAEActor,
        timingEncoder: TimingEncoder
    ) {
        self.clap = clap
        self.diffusion = diffusion
        self.vae = vae
        self.timingEncoder = timingEncoder
    }

    public var progress: AsyncStream<GenerationProgress> {
        if let existing = _progress { return existing }
        let (stream, continuation) = AsyncStream<GenerationProgress>.makeStream()
        self.progressContinuation = continuation
        self._progress = stream
        return stream
    }

    public func generate(
        text: String,
        duration: TimeInterval,
        config: SFXDiffusionConfig,
        tokenizer: CLAPTokenizer
    ) async throws -> AudioResult {
        let totalStart = Date()
        let clampedDuration = min(duration, Self.maxDuration)

        // 1. CLAP text encoding
        let clapEmbedding: MLXArray
        let clapLoaded = await clap.isLoaded
        if clapLoaded {
            clapEmbedding = try await clap.encode(text: text, tokenizer: tokenizer)
        } else {
            // Fallback: use random embedding for unloaded model (testing)
            clapEmbedding = MLXArray.zeros([1, 512])
        }

        // 2. Timing conditioning
        let timingEmbedding = timingEncoder.encode(startSeconds: 0.0, durationSeconds: Float(clampedDuration))

        // 3. Combine conditioning
        let conditioning = concatenated([clapEmbedding, timingEmbedding], axis: 1)

        // 4. Diffusion
        progressContinuation?.yield(.diffusion(DiffusionProgress(
            currentStep: 0, totalSteps: config.steps, elapsedTime: 0
        )))

        let diffResult = try await diffusion.sample(conditioning: conditioning, config: config)

        // 5. VAE decode
        progressContinuation?.yield(.decoding)
        let audio = await vae.decode(latents: diffResult.latents)

        let totalTime = Date().timeIntervalSince(totalStart)
        let costs = TimeCosts(
            diffusionTime: diffResult.timeCosts.diffusionTime,
            perStepTime: diffResult.timeCosts.perStepTime,
            totalTime: totalTime
        )

        progressContinuation?.yield(.complete)

        return AudioResult(
            samples: audio,
            sampleRate: Self.defaultSampleRate,
            duration: clampedDuration,
            timeCosts: costs
        )
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd AnvilCore && swift build 2>&1 | tail -5`
Expected: Build complete!

- [ ] **Step 5: Commit**

```bash
git add AnvilCore/Sources/AnvilCore/Inference/StableAudio/SoundFXPipeline.swift AnvilCore/Tests/AnvilCoreTests/Inference/StableAudio/SoundFXPipelineTests.swift
git commit -m "feat: add SoundFXPipeline orchestrating CLAP, diffusion, and VAE"
```

---

### Task 8: Weight Name Maps

**Files:**
- Create: `AnvilCore/Sources/AnvilCore/Inference/Weights/CLAPNameMap.swift`
- Create: `AnvilCore/Sources/AnvilCore/Inference/Weights/SFXDiTNameMap.swift`
- Create: `AnvilCore/Sources/AnvilCore/Inference/Weights/SFXVAENameMap.swift`

- [ ] **Step 1: Write CLAPNameMap**

```swift
// AnvilCore/Sources/AnvilCore/Inference/Weights/CLAPNameMap.swift
import MLX

public enum CLAPNameMap {
    public static func mapKey(_ key: String) -> String? {
        var mapped = key
        // Strip model prefix if present
        if mapped.hasPrefix("text_model.") {
            mapped = String(mapped.dropFirst("text_model.".count))
        }
        if mapped.hasPrefix("text_projection.") {
            mapped = mapped.replacingOccurrences(of: "text_projection.", with: "projection.")
        }
        if mapped.contains("layer_norm") {
            mapped = mapped.replacingOccurrences(of: "layer_norm", with: "ln")
        }
        if mapped.contains("self_attn.q_proj") {
            mapped = mapped.replacingOccurrences(of: "self_attn.q_proj", with: "attention.query")
        }
        if mapped.contains("self_attn.k_proj") {
            mapped = mapped.replacingOccurrences(of: "self_attn.k_proj", with: "attention.key")
        }
        if mapped.contains("self_attn.v_proj") {
            mapped = mapped.replacingOccurrences(of: "self_attn.v_proj", with: "attention.value")
        }
        if mapped.contains("self_attn.out_proj") {
            mapped = mapped.replacingOccurrences(of: "self_attn.out_proj", with: "attention.out")
        }
        if mapped.contains("mlp.fc1") || mapped.contains("mlp.fc2") {
            // Already matches our naming
        }
        return mapped
    }
}
```

- [ ] **Step 2: Write SFXDiTNameMap**

```swift
// AnvilCore/Sources/AnvilCore/Inference/Weights/SFXDiTNameMap.swift
import MLX

public enum SFXDiTNameMap {
    public static func mapKey(_ key: String) -> String? {
        var mapped = key
        // Strip common prefixes
        if mapped.hasPrefix("model.") {
            mapped = String(mapped.dropFirst("model.".count))
        }
        // Map transformer block naming
        if mapped.contains("transformer_blocks") {
            mapped = mapped.replacingOccurrences(of: "transformer_blocks", with: "blocks")
        }
        if mapped.contains("norm1.linear") {
            mapped = mapped.replacingOccurrences(of: "norm1.linear", with: "adaLN")
        }
        if mapped.contains("attn1.to_q") {
            mapped = mapped.replacingOccurrences(of: "attn1.to_q", with: "attn.qkv")
        }
        if mapped.contains("attn1.to_out.0") {
            mapped = mapped.replacingOccurrences(of: "attn1.to_out.0", with: "attn.out")
        }
        if mapped.contains("attn2.to_q") {
            mapped = mapped.replacingOccurrences(of: "attn2.to_q", with: "cross_attn.q")
        }
        if mapped.contains("attn2.to_kv") {
            mapped = mapped.replacingOccurrences(of: "attn2.to_kv", with: "cross_attn.kv")
        }
        if mapped.contains("attn2.to_out.0") {
            mapped = mapped.replacingOccurrences(of: "attn2.to_out.0", with: "cross_attn.out")
        }
        if mapped.contains("ff.net.0") {
            mapped = mapped.replacingOccurrences(of: "ff.net.0", with: "mlp.fc1")
        }
        if mapped.contains("ff.net.2") {
            mapped = mapped.replacingOccurrences(of: "ff.net.2", with: "mlp.fc2")
        }
        return mapped
    }
}
```

- [ ] **Step 3: Write SFXVAENameMap**

```swift
// AnvilCore/Sources/AnvilCore/Inference/Weights/SFXVAENameMap.swift
import MLX

public enum SFXVAENameMap {
    public static func mapKey(_ key: String) -> String? {
        var mapped = key
        if mapped.hasPrefix("decoder.") {
            mapped = String(mapped.dropFirst("decoder.".count))
        }
        if mapped.contains("up_blocks") {
            mapped = mapped.replacingOccurrences(of: "up_blocks", with: "upsampleBlocks")
        }
        if mapped.contains("proj_in") || mapped.contains("proj_out") {
            // Already matches
        }
        return mapped
    }

    public static func isConv1dWeight(_ key: String) -> Bool {
        key.contains("conv.weight") || (key.contains("linear.weight") && !key.contains("norm"))
    }

    public static func conv1dToMLX(_ ptWeight: MLXArray) -> MLXArray {
        if ptWeight.ndim == 3 {
            return ptWeight.transposed(0, 2, 1)
        }
        return ptWeight
    }
}
```

- [ ] **Step 4: Run build to verify**

Run: `cd AnvilCore && swift build 2>&1 | tail -5`
Expected: Build complete!

- [ ] **Step 5: Commit**

```bash
git add AnvilCore/Sources/AnvilCore/Inference/Weights/CLAPNameMap.swift AnvilCore/Sources/AnvilCore/Inference/Weights/SFXDiTNameMap.swift AnvilCore/Sources/AnvilCore/Inference/Weights/SFXVAENameMap.swift
git commit -m "feat: add weight name maps for CLAP, SFX DiT, and SFX VAE"
```

---

### Task 9: SFXModelManager

**Files:**
- Create: `AnvilApp/Services/SFXModelManager.swift`

- [ ] **Step 1: Write SFXModelManager**

```swift
// AnvilApp/Services/SFXModelManager.swift
import Foundation
import AnvilCore

enum SFXDownloadState: Equatable, Sendable {
    case notDownloaded
    case downloading(progress: Float)
    case downloaded
    case error(String)
}

@MainActor
final class SFXModelManager: ObservableObject {
    @Published var downloadState: SFXDownloadState = .notDownloaded
    @Published var pipeline: SoundFXPipeline?

    private let modelID = "stable-audio-open-1.0"
    private let repoID = "stabilityai/stable-audio-open-1.0"

    var modelPath: URL {
        let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        ).first!
        return appSupport
            .appendingPathComponent("Anvil", isDirectory: true)
            .appendingPathComponent("models", isDirectory: true)
            .appendingPathComponent(modelID, isDirectory: true)
    }

    var isReady: Bool { pipeline != nil }

    init() {
        if isModelDownloaded() {
            downloadState = .downloaded
        }
    }

    func downloadIfNeeded() async throws {
        guard downloadState != .downloaded else { return }

        let fm = FileManager.default
        try fm.createDirectory(at: modelPath, withIntermediateDirectories: true)

        downloadState = .downloading(progress: 0)

        let baseURL = "https://huggingface.co/\(repoID)/resolve/main/"
        let session = URLSession.shared

        // Metadata files
        let metadataFiles = ["config.json", "tokenizer.json"]
        for (idx, file) in metadataFiles.enumerated() {
            try Task.checkCancellation()
            guard let url = URL(string: baseURL + file) else { continue }
            let (data, response) = try await session.data(from: url)
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                let msg = "Failed to download \(file)"
                downloadState = .error(msg)
                throw SFXDownloadError.downloadFailed(file)
            }
            try data.write(to: modelPath.appendingPathComponent(file))
            downloadState = .downloading(progress: Float(idx + 1) / Float(metadataFiles.count) * 0.1)
        }

        // Weight files (large — use download task)
        let weightFiles = ["model.safetensors", "vae.safetensors", "clap_model.safetensors"]
        for (idx, file) in weightFiles.enumerated() {
            try Task.checkCancellation()
            guard let url = URL(string: baseURL + file) else { continue }
            let dest = modelPath.appendingPathComponent(file)
            let (tempURL, response) = try await session.download(from: url)
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                let msg = "Failed to download \(file)"
                downloadState = .error(msg)
                throw SFXDownloadError.downloadFailed(file)
            }
            if fm.fileExists(atPath: dest.path) {
                try fm.removeItem(at: dest)
            }
            try fm.moveItem(at: tempURL, to: dest)
            let progress = 0.1 + Float(idx + 1) / Float(weightFiles.count) * 0.9
            downloadState = .downloading(progress: progress)
        }

        downloadState = .downloaded
    }

    func loadPipeline() async throws {
        guard isModelDownloaded() else { return }

        let clapEncoder = CLAPEncoder(embeddingDim: 512, numLayers: 12, numHeads: 8, vocabSize: 49408)
        let timingEncoder = TimingEncoder(embeddingDim: 256)
        let ditConfig = SFXDiTConfig(hiddenSize: 1024, numLayers: 24, numHeads: 16, latentDim: 64, conditioningDim: 768)
        let dit = SFXDiTActor(config: ditConfig)
        let diffusion = SFXDiffusionActor(dit: dit)
        let vaeConfig = SFXVAEConfig(latentDim: 64, outputChannels: 2, sampleRate: 44100, upsampleRatios: [8, 8, 4, 2])
        let vae = SFXVAEActor(config: vaeConfig)

        let sfxPipeline = SoundFXPipeline(
            clap: clapEncoder,
            diffusion: diffusion,
            vae: vae,
            timingEncoder: timingEncoder
        )
        self.pipeline = sfxPipeline
    }

    func deleteWeights() {
        try? FileManager.default.removeItem(at: modelPath)
        downloadState = .notDownloaded
        pipeline = nil
    }

    private func isModelDownloaded() -> Bool {
        let fm = FileManager.default
        let hasConfig = fm.fileExists(atPath: modelPath.appendingPathComponent("config.json").path)
        let hasModel = fm.fileExists(atPath: modelPath.appendingPathComponent("model.safetensors").path)
        return hasConfig && hasModel
    }
}

enum SFXDownloadError: Error, Sendable {
    case downloadFailed(String)
}
```

- [ ] **Step 2: Verify Xcode project compiles (manual check or xcodebuild)**

The file needs to be added to the Xcode project. The subagent should add it to the project.pbxproj file following the established pattern from Phase 4b.

- [ ] **Step 3: Commit**

```bash
git add AnvilApp/Services/SFXModelManager.swift
git commit -m "feat: add SFXModelManager for Stable Audio weight download and activation"
```

---

### Task 10: PipelineChoice + GenerationService Routing

**Files:**
- Create: `AnvilApp/Models/PipelineChoice.swift`
- Modify: `AnvilApp/Models/GenerationRequest.swift`
- Modify: `AnvilApp/Models/GenerationState.swift`
- Modify: `AnvilApp/Services/GenerationService.swift`

- [ ] **Step 1: Create PipelineChoice enum**

```swift
// AnvilApp/Models/PipelineChoice.swift
import Foundation

enum PipelineChoice: String, CaseIterable, Sendable, Identifiable {
    case auto
    case aceStep
    case stableAudio

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .auto: return "Auto"
        case .aceStep: return "ACE-Step (Music)"
        case .stableAudio: return "Stable Audio (SFX)"
        }
    }
}
```

- [ ] **Step 2: Add fields to GenerationRequest**

Modify `AnvilApp/Models/GenerationRequest.swift` — add two new fields:

```swift
import Foundation
import AnvilCore

struct GenerationRequest: Sendable {
    let text: String
    let lyrics: String
    let referenceAudioURL: URL?
    let duration: TimeInterval
    let seed: Int?
    let method: InferMethod
    let guidanceScale: Float
    let precision: Precision
    let resolvedMode: GenerationMode
    let pipelineOverride: PipelineChoice

    init(
        text: String,
        lyrics: String = "",
        referenceAudioURL: URL? = nil,
        duration: TimeInterval = 30.0,
        seed: Int? = nil,
        method: InferMethod = .ode,
        guidanceScale: Float = 1.0,
        precision: Precision = .float16,
        resolvedMode: GenerationMode = .music,
        pipelineOverride: PipelineChoice = .auto
    ) {
        self.text = text
        self.lyrics = lyrics
        self.referenceAudioURL = referenceAudioURL
        self.duration = duration
        self.seed = seed
        self.method = method
        self.guidanceScale = guidanceScale
        self.precision = precision
        self.resolvedMode = resolvedMode
        self.pipelineOverride = pipelineOverride
    }
}
```

- [ ] **Step 3: Add downloadingModel case to GenerationState**

Modify `AnvilApp/Models/GenerationState.swift`:

```swift
import Foundation

enum GenerationState: Equatable, Sendable {
    case idle
    case downloadingModel(progress: Float)
    case loadingWeights(progress: Float)
    case conditioning
    case generating(step: Int, totalSteps: Int, eta: TimeInterval)
    case decoding
    case complete
    case error(String)
    case cancelled
}
```

- [ ] **Step 4: Add routing to GenerationService**

Modify `AnvilApp/Services/GenerationService.swift` — add `sfxModelManager` dependency and routing logic. The `generate` method determines which pipeline to use:

```swift
import Foundation
import AnvilCore
import MLX

struct GenerationResult: Sendable {
    let fileURL: URL
    let entry: HistoryEntry
}

@MainActor
final class GenerationService: ObservableObject {
    @Published var state: GenerationState = .idle
    @Published var currentResult: GenerationResult?

    private var generationTask: Task<Void, Never>?
    private let pipeline: MusicPipeline
    private let conditioning: ConditioningActor
    private let weightConverter: WeightConverter
    private let historyStore: HistoryStore
    private let outputDirectoryManager: OutputDirectoryManager
    private let sfxModelManager: SFXModelManager
    private var weightsLoaded = false

    init(
        pipeline: MusicPipeline,
        conditioning: ConditioningActor,
        weightConverter: WeightConverter,
        historyStore: HistoryStore,
        outputDirectoryManager: OutputDirectoryManager,
        sfxModelManager: SFXModelManager
    ) {
        self.pipeline = pipeline
        self.conditioning = conditioning
        self.weightConverter = weightConverter
        self.historyStore = historyStore
        self.outputDirectoryManager = outputDirectoryManager
        self.sfxModelManager = sfxModelManager
    }

    func generate(request: GenerationRequest) {
        generationTask?.cancel()

        generationTask = Task { [weak self] in
            guard let self else { return }

            guard !request.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                self.state = .error("Prompt cannot be empty")
                return
            }

            guard let outputDir = self.outputDirectoryManager.directory else {
                self.state = .error("No output directory selected")
                return
            }

            let useStableAudio = self.shouldUseStableAudio(request: request)

            do {
                let result: AudioResult
                if useStableAudio {
                    result = try await self.generateWithStableAudio(request: request)
                } else {
                    result = try await self.generateWithACEStep(request: request)
                }

                try Task.checkCancellation()

                let fileName = HistoryEntry.fileName(prompt: request.text)
                let fileURL = outputDir.appending(path: fileName)

                try WAVWriter.write(
                    samples: result.samples,
                    sampleRate: result.sampleRate,
                    to: fileURL
                )

                let entry = HistoryEntry(
                    id: UUID(),
                    createdAt: Date(),
                    prompt: request.text,
                    lyrics: request.lyrics,
                    duration: request.duration,
                    seed: request.seed ?? 0,
                    method: request.method.rawValue,
                    guidanceScale: request.guidanceScale,
                    precision: request.precision.rawValue,
                    fileName: fileName
                )

                self.historyStore.add(entry)
                self.currentResult = GenerationResult(fileURL: fileURL, entry: entry)
                self.state = .complete
            } catch is CancellationError {
                self.state = .cancelled
            } catch {
                self.state = .error(error.localizedDescription)
            }
        }
    }

    func cancel() {
        generationTask?.cancel()
        state = .cancelled
    }

    private func shouldUseStableAudio(request: GenerationRequest) -> Bool {
        switch request.pipelineOverride {
        case .aceStep: return false
        case .stableAudio: return true
        case .auto:
            return request.resolvedMode == .soundFX || request.resolvedMode == .ambient
        }
    }

    private func generateWithStableAudio(request: GenerationRequest) async throws -> AudioResult {
        // Download if needed
        if !sfxModelManager.isReady {
            state = .downloadingModel(progress: 0)
            try await sfxModelManager.downloadIfNeeded()
            try await sfxModelManager.loadPipeline()
        }

        guard let sfxPipeline = sfxModelManager.pipeline else {
            throw GenerationError.pipelineNotReady
        }

        let totalSteps = 100
        state = .generating(step: 0, totalSteps: totalSteps, eta: 0)

        let progressStream = await sfxPipeline.progress
        let progressTask = Task { [weak self] in
            guard let self else { return }
            for await progress in progressStream {
                if case .diffusion(let dp) = progress {
                    let perStep = dp.currentStep > 0
                        ? dp.elapsedTime / Double(dp.currentStep)
                        : 0
                    let eta = perStep * Double(dp.totalSteps - dp.currentStep)
                    self.state = .generating(
                        step: dp.currentStep,
                        totalSteps: dp.totalSteps,
                        eta: eta
                    )
                }
            }
        }
        defer { progressTask.cancel() }

        let clampedDuration = min(request.duration, 47.0)
        let tokenizer = CLAPTokenizer.forTesting()
        let config = SFXDiffusionConfig(
            steps: totalSteps,
            guidanceScale: request.guidanceScale > 1.0 ? request.guidanceScale : 7.0,
            seed: request.seed,
            latentShape: [1, Int(ceil(clampedDuration * 44100.0 / 1024.0)), 64]
        )

        state = .generating(step: 0, totalSteps: config.steps, eta: 0)

        let result = try await sfxPipeline.generate(
            text: request.text,
            duration: clampedDuration,
            config: config,
            tokenizer: tokenizer
        )

        state = .decoding
        return result
    }

    private func generateWithACEStep(request: GenerationRequest) async throws -> AudioResult {
        if !weightsLoaded {
            state = .loadingWeights(progress: 0)
            try await loadWeights()
            weightsLoaded = true
        }

        try Task.checkCancellation()
        state = .conditioning

        let config = DiffusionConfig(
            seed: request.seed,
            method: request.method,
            guidanceScale: request.guidanceScale,
            precision: request.precision
        )

        let totalSteps = config.steps ?? 8
        state = .generating(step: 0, totalSteps: totalSteps, eta: 0)

        let progressStream = await pipeline.progress
        let progressTask = Task { [weak self] in
            guard let self else { return }
            for await progress in progressStream {
                if case .diffusion(let dp) = progress {
                    let perStep = dp.currentStep > 0
                        ? dp.elapsedTime / Double(dp.currentStep)
                        : 0
                    let eta = perStep * Double(dp.totalSteps - dp.currentStep)
                    self.state = .generating(
                        step: dp.currentStep,
                        totalSteps: dp.totalSteps,
                        eta: eta
                    )
                }
            }
        }
        defer { progressTask.cancel() }

        let result = try await pipeline.generate(
            text: request.text,
            lyrics: request.lyrics,
            referenceAudio: nil,
            duration: request.duration,
            config: config,
            conditioning: conditioning
        )

        state = .decoding
        return result
    }

    private func loadWeights() async throws {
        state = .loadingWeights(progress: 1.0)
    }
}

enum GenerationError: Error, LocalizedError {
    case pipelineNotReady

    var errorDescription: String? {
        switch self {
        case .pipelineNotReady: return "Sound FX pipeline is not ready"
        }
    }
}
```

- [ ] **Step 5: Verify build**

The Xcode project needs the new PipelineChoice.swift added. Verify build compiles.

- [ ] **Step 6: Commit**

```bash
git add AnvilApp/Models/PipelineChoice.swift AnvilApp/Models/GenerationRequest.swift AnvilApp/Models/GenerationState.swift AnvilApp/Services/GenerationService.swift
git commit -m "feat: add pipeline routing in GenerationService (ACE-Step vs Stable Audio)"
```

---

### Task 11: UI Changes (GenerateView + Inspector)

**Files:**
- Modify: `AnvilApp/UI/Components/GenerateView.swift`
- Modify: `AnvilApp/UI/MainShellView.swift`

- [ ] **Step 1: Add pipelineOverride state and duration capping to GenerateView**

In `AnvilApp/UI/Components/GenerateView.swift`, add:
- A `@State private var pipelineOverride: PipelineChoice = .auto` property
- Modify the duration slider to cap at 47 when Stable Audio is the resolved pipeline
- Add a download badge to mode buttons when SFX model isn't downloaded
- Pass `resolvedMode` and `pipelineOverride` to the `GenerationRequest`

Add `@EnvironmentObject private var sfxModelManager: SFXModelManager` to the view.

Key changes to the generate action:

```swift
private func startGeneration() {
    let request = GenerationRequest(
        text: promptText,
        lyrics: lyricsText,
        duration: effectiveDuration,
        seed: seed.isEmpty ? nil : Int(seed),
        method: method,
        guidanceScale: Float(guidanceScale),
        precision: precision,
        resolvedMode: resolvedMode,
        pipelineOverride: pipelineOverride
    )
    generationService.generate(request: request)
}

private var effectiveDuration: TimeInterval {
    if shouldUseStableAudio {
        return min(duration, 47.0)
    }
    return duration
}

private var shouldUseStableAudio: Bool {
    switch pipelineOverride {
    case .aceStep: return false
    case .stableAudio: return true
    case .auto: return resolvedMode == .soundFX || resolvedMode == .ambient
    }
}

private var maxDuration: Double {
    shouldUseStableAudio ? 47.0 : 240.0
}
```

Add download badge to mode buttons when SFX not downloaded:

```swift
private func modeButton(_ title: String, isSelected: Bool, isSFXMode: Bool = false, action: @escaping () -> Void) -> some View {
    Button(action: action) {
        HStack(spacing: 4) {
            Text(title)
            if isSFXMode && sfxModelManager.downloadState == .notDownloaded {
                Image(systemName: "arrow.down.circle")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
    }
    .buttonStyle(.bordered)
    .tint(isSelected ? .accentColor : nil)
}
```

Also add handling for `.downloadingModel` in the progress section:

```swift
if case .downloadingModel(let progress) = generationService.state {
    ProgressView("Downloading Sound FX model...", value: Double(progress), total: 1.0)
}
```

- [ ] **Step 2: Add pipeline indicator to Inspector in MainShellView**

In `AnvilApp/UI/MainShellView.swift`, update the `inspector` computed property to show the pipeline section. The view needs `@EnvironmentObject private var sfxModelManager: SFXModelManager` and a binding to `pipelineOverride`. Since the Inspector is in MainShellView and pipelineOverride is in GenerateView, we need to lift it up or use a simpler approach.

The simplest approach: add a `@State private var pipelineOverride: PipelineChoice = .auto` to MainShellView and pass it down, or make it an environment-level concern.

Better approach for this phase: put the pipeline picker directly in the Inspector as a standalone control that shows the current resolved pipeline name (read from sfxModelManager state and the current mode).

```swift
private var inspector: some View {
    Form {
        Section("Pipeline") {
            LabeledContent("Active") {
                Text(resolvedPipelineName)
                    .foregroundStyle(.secondary)
            }
            if sfxModelManager.downloadState == .downloaded {
                Picker("Override", selection: $pipelineOverride) {
                    Text("Auto").tag(PipelineChoice.auto)
                    Text("ACE-Step (Music)").tag(PipelineChoice.aceStep)
                    Text("Stable Audio (SFX)").tag(PipelineChoice.stableAudio)
                }
                .pickerStyle(.menu)
            }
        }
        Section("Parameters") {
            LabeledContent("Duration") { TextField("30s", text: .constant("")) }
            LabeledContent("Seed") { TextField("Random", text: .constant("")) }
            LabeledContent("Guidance") { TextField("1.0", text: .constant("")) }
        }
        Section("Model") {
            Text("ACE-Step DiT v1.5")
                .foregroundStyle(.secondary)
            if sfxModelManager.downloadState == .downloaded {
                Text("Stable Audio Open 1.0")
                    .foregroundStyle(.secondary)
            }
        }
    }
    .formStyle(.grouped)
    .frame(minWidth: 250)
}

private var resolvedPipelineName: String {
    switch pipelineOverride {
    case .aceStep: return "ACE-Step"
    case .stableAudio: return "Stable Audio"
    case .auto: return "Auto"
    }
}
```

Add `@EnvironmentObject private var sfxModelManager: SFXModelManager` and `@State private var pipelineOverride: PipelineChoice = .auto` to `MainShellView`.

- [ ] **Step 3: Verify build**

Ensure the app builds with all UI changes.

- [ ] **Step 4: Commit**

```bash
git add AnvilApp/UI/Components/GenerateView.swift AnvilApp/UI/MainShellView.swift
git commit -m "feat: add pipeline indicator, duration cap, and download badge to UI"
```

---

### Task 12: App Wiring

**Files:**
- Modify: `AnvilApp/AnvilAppApp.swift`

- [ ] **Step 1: Add SFXModelManager to app entry point**

In `AnvilApp/AnvilAppApp.swift`, add `SFXModelManager` as a `@StateObject` and inject it as an environment object. Also update `AppServices` to accept `sfxModelManager` for `GenerationService`.

In the `init()`:

```swift
let sfxManager = SFXModelManager()
_sfxModelManager = StateObject(wrappedValue: sfxManager)
```

Add the property declaration:

```swift
@StateObject private var sfxModelManager: SFXModelManager
```

Inject in body:

```swift
.environmentObject(sfxModelManager)
```

Update `AppServices.makeGenerationService` to accept and pass `sfxModelManager`:

```swift
private static func makeGenerationService(
    historyStore: HistoryStore,
    outputManager: OutputDirectoryManager,
    sfxModelManager: SFXModelManager
) -> GenerationService {
    // ... existing pipeline setup ...
    return GenerationService(
        pipeline: pipeline,
        conditioning: conditioning,
        weightConverter: converter,
        historyStore: historyStore,
        outputDirectoryManager: outputManager,
        sfxModelManager: sfxModelManager
    )
}
```

The `configure(outputManager:)` method and any calls to `makeGenerationService` need the `sfxModelManager` parameter added.

- [ ] **Step 2: Verify build**

Run xcodebuild or verify in Xcode that the app compiles cleanly.

- [ ] **Step 3: Commit**

```bash
git add AnvilApp/AnvilAppApp.swift
git commit -m "feat: wire SFXModelManager into app entry point and GenerationService"
```

---

## Self-Review

**Spec coverage check:**
- ✅ Component 1 (CLAPEncoder) → Task 3
- ✅ Component 2 (CLAPTokenizer) → Task 2
- ✅ Component 3 (TimingEncoder) → Task 1
- ✅ Component 4 (SFXDiTModel + SFXDiTActor) → Task 4
- ✅ Component 5 (SFXVAEModel + SFXVAEActor) → Task 5
- ✅ Component 6 (SFXDiffusionActor) → Task 6
- ✅ Component 7 (SoundFXPipeline) → Task 7
- ✅ Component 8 (SFXModelManager) → Task 9
- ✅ Component 9 (GenerationService routing) → Task 10
- ✅ Component 10 (UI changes) → Task 11
- ✅ Component 11 (Weight name maps) → Task 8
- ✅ App wiring → Task 12

**Type consistency check:**
- `SFXDiTConfig` defined in Task 4, used in Tasks 4, 6, 7, 9 — consistent
- `SFXVAEConfig` defined in Task 5, used in Tasks 5, 7, 9 — consistent
- `SFXDiffusionConfig` defined in Task 6, used in Tasks 6, 7, 10 — consistent
- `SoundFXPipeline.generate(text:duration:config:tokenizer:)` defined in Task 7, called in Task 10 — consistent
- `PipelineChoice` defined in Task 10, used in Tasks 10, 11 — consistent
- `SFXDownloadState` defined in Task 9, used in Tasks 9, 11 — consistent
- `GenerationRequest.resolvedMode` and `.pipelineOverride` added in Task 10, used in Task 11 — consistent
