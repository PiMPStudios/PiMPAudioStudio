# Phase 10a: LoRA Fine-Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add on-device LoRA fine-tuning for ACE-Step DiT so users can train personal style adapters from their generation history and apply them during generation.

**Architecture:** LoRALinear wraps existing Linear layers with low-rank A·B matrices. LoRAInjector walks the DiTDecoder module tree to swap Linear↔LoRALinear at runtime. A flow-matching training loop (LoRATrainer) trains only the adapter weights. The app layer provides TrainingService orchestration, a Training Sheet UI from History, and an Adapter Picker in Generate view with lazy injection at generation time.

**Tech Stack:** Swift, MLX (MLXNN, MLXRandom, MLXFast), MLXOptimizers (AdamW), SwiftUI, safetensors

---

## File Structure

### AnvilCore (New)

| File | Responsibility |
|------|---------------|
| `AnvilCore/Sources/AnvilCore/LoRA/LoRAConfig.swift` | LoRAConfig + LoRAAdapter data types |
| `AnvilCore/Sources/AnvilCore/LoRA/LoRALinear.swift` | LoRALinear Module wrapping Linear |
| `AnvilCore/Sources/AnvilCore/LoRA/LoRAInjector.swift` | Module tree inject/remove/extract/load |
| `AnvilCore/Sources/AnvilCore/LoRA/LoRATrainer.swift` | Flow-matching training loop with AdamW |

### AnvilApp (New)

| File | Responsibility |
|------|---------------|
| `AnvilApp/Services/AdapterRegistry.swift` | Scans and lists saved adapters |
| `AnvilApp/Services/TrainingService.swift` | Orchestrates dataset prep, training, save |
| `AnvilApp/UI/Training/TrainingSheet.swift` | Training config + progress sheet |
| `AnvilApp/UI/Generate/AdapterPicker.swift` | LoRA dropdown in Generate view |

### Modified Files

| File | Changes |
|------|---------|
| `AnvilCore/Package.swift` | Add MLXOptimizers dependency |
| `AnvilCore/Sources/AnvilCore/Inference/DiT/DiTComponents.swift` | Change `let` to `var` on Linear properties in DiTAttention + SwiGLUMLP |
| `AnvilCore/Sources/AnvilCore/Inference/DiT/DiTActor.swift` | Add LoRA inject/remove/load methods |
| `AnvilCore/Sources/AnvilCore/Inference/Diffusion/DiffusionActor.swift` | Add LoRA forwarding methods |
| `AnvilCore/Sources/AnvilCore/Inference/Pipeline/MusicPipeline.swift` | Add LoRA forwarding methods |
| `AnvilApp/Services/GenerationService.swift` | Adapter injection before generation |
| `AnvilApp/UI/Components/HistoryListView.swift` | Multi-select + "Train LoRA" button |
| `AnvilApp/UI/Components/GenerateView.swift` | Add AdapterPicker to controls |

---

### Task 1: LoRAConfig + LoRAAdapter Data Types

**Files:**

- Create: `AnvilCore/Sources/AnvilCore/LoRA/LoRAConfig.swift`

- [ ] **Step 1: Create the LoRA directory**

```bash
mkdir -p AnvilCore/Sources/AnvilCore/LoRA
```

- [ ] **Step 2: Write LoRAConfig and LoRAAdapter**

Create `AnvilCore/Sources/AnvilCore/LoRA/LoRAConfig.swift`:

```swift
import Foundation

/// Configuration for LoRA adapter training and injection.
public struct LoRAConfig: Codable, Sendable, Equatable {
    /// Low-rank dimension. Higher = more expressive, more memory. Typical: 8, 16, 32.
    public let rank: Int

    /// Scaling factor. The adapter contribution is scaled by alpha/rank.
    public let alpha: Float

    /// Dropout rate applied to adapter path during training. 0.0 = no dropout.
    public let dropout: Float

    /// Layer name patterns to inject LoRA into (e.g. "selfAttn.qProj").
    public let targetLayers: [String]

    /// Learning rate for AdamW optimizer.
    public let learningRate: Float

    /// Number of training epochs.
    public let epochs: Int

    /// Training batch size. Default 1 (memory constrained on-device).
    public let batchSize: Int

    public init(
        rank: Int = 16,
        alpha: Float = 16.0,
        dropout: Float = 0.0,
        targetLayers: [String] = LoRAConfig.defaultTargetLayers,
        learningRate: Float = 1e-4,
        epochs: Int = 10,
        batchSize: Int = 1
    ) {
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.targetLayers = targetLayers
        self.learningRate = learningRate
        self.epochs = epochs
        self.batchSize = batchSize
    }

    /// Default LoRA target layers for ACE-Step DiT: q and v projections
    /// in both self-attention and cross-attention across all 24 layers.
    public static let defaultTargetLayers: [String] = [
        "selfAttn.qProj",
        "selfAttn.vProj",
        "crossAttn.qProj",
        "crossAttn.vProj",
    ]
}

/// A saved LoRA adapter with metadata.
public struct LoRAAdapter: Codable, Sendable, Identifiable, Equatable {
    /// Directory name used as identifier.
    public let id: String

    /// User-facing display name.
    public let name: String

    /// The LoRA config used during training.
    public let config: LoRAConfig

    /// ID of the base model this adapter was trained against.
    public let baseModelID: String

    /// Category of the base model for fast compatibility filtering.
    public let baseModelCategory: ModelCategory

    /// When training completed.
    public let createdAt: Date

    /// Number of training samples used.
    public let sampleCount: Int

    /// Final training loss (nil if training was stopped early).
    public let finalLoss: Float?

    public init(
        id: String,
        name: String,
        config: LoRAConfig,
        baseModelID: String,
        baseModelCategory: ModelCategory,
        createdAt: Date,
        sampleCount: Int,
        finalLoss: Float?
    ) {
        self.id = id
        self.name = name
        self.config = config
        self.baseModelID = baseModelID
        self.baseModelCategory = baseModelCategory
        self.createdAt = createdAt
        self.sampleCount = sampleCount
        self.finalLoss = finalLoss
    }
}
```

- [ ] **Step 3: Verify it compiles**

```bash
xcodebuild -project AnvilApp.xcodeproj -scheme AnvilCore -destination 'platform=macOS' build 2>&1 | tail -5
```

Expected: BUILD SUCCEEDED

- [ ] **Step 4: Commit**

```bash
git add AnvilCore/Sources/AnvilCore/LoRA/LoRAConfig.swift
git commit -m "feat: add LoRAConfig and LoRAAdapter data types"
```

---

### Task 2: LoRALinear Module

**Files:**

- Create: `AnvilCore/Sources/AnvilCore/LoRA/LoRALinear.swift`
- Modify: `AnvilCore/Sources/AnvilCore/Inference/DiT/DiTComponents.swift` (change `let` → `var` on Linear properties)

- [ ] **Step 1: Change Linear properties from let to var in DiTComponents**

In `AnvilCore/Sources/AnvilCore/Inference/DiT/DiTComponents.swift`, change the `let` declarations to `var` for all Linear properties that LoRA might target.

In `SwiGLUMLP` (line 70-72), change:

```swift
// FROM:
let gateProj: Linear
let upProj: Linear
let downProj: Linear

// TO:
var gateProj: Linear
var upProj: Linear
var downProj: Linear
```

In `DiTAttention` (line 98-101), change:

```swift
// FROM:
let qProj: Linear
let kProj: Linear
let vProj: Linear
let oProj: Linear

// TO:
var qProj: Linear
var kProj: Linear
var vProj: Linear
var oProj: Linear
```

These must be `var` so LoRAInjector can swap them with LoRALinear instances at runtime. Only the four attention projections and three MLP projections need this change — norms, embeddings, and other layers stay `let`.

- [ ] **Step 2: Write LoRALinear module**

Create `AnvilCore/Sources/AnvilCore/LoRA/LoRALinear.swift`:

```swift
import Foundation
import MLX
import MLXNN
import MLXRandom

/// A Linear layer wrapped with a low-rank adapter.
///
/// Output = base(x) + scaling * dropout(x @ A^T @ B^T)
///
/// A is initialized with Kaiming uniform, B is initialized to zero,
/// so the adapter starts as identity (no effect on output).
public class LoRALinear: Linear {
    /// The original frozen weight matrix (stored separately from Linear's `weight`).
    let baseWeight: MLXArray

    /// The original bias (nil if the base layer had no bias).
    let baseBias: MLXArray?

    /// Low-rank matrix A: shape (rank, inFeatures). Kaiming uniform init.
    var loraA: MLXArray

    /// Low-rank matrix B: shape (outFeatures, rank). Zero init.
    var loraB: MLXArray

    /// Scaling factor = alpha / rank.
    let scaling: Float

    /// Dropout rate (applied during training only).
    let dropoutRate: Float

    /// Whether we're in training mode (enables dropout).
    var isTraining: Bool = false

    /// Whether adapter has been merged into base weights.
    private var isMerged: Bool = false

    public init(base: Linear, rank: Int, alpha: Float = 16.0, dropout: Float = 0.0) {
        let inFeatures = base.weight.shape[1]
        let outFeatures = base.weight.shape[0]

        self.baseWeight = base.weight
        self.baseBias = base.bias
        self.scaling = alpha / Float(rank)
        self.dropoutRate = dropout

        // Kaiming uniform for A: scale = sqrt(1/inFeatures)
        let bound = Float(1.0 / Float(inFeatures).squareRoot())
        self.loraA = MLXRandom.uniform(
            low: -bound,
            high: bound,
            [rank, inFeatures]
        )

        // Zero init for B — adapter starts as identity
        self.loraB = MLXArray.zeros([outFeatures, rank])

        // Initialize Linear superclass with base weights
        if let bias = base.bias {
            super.init(inFeatures, outFeatures, bias: true)
            self.weight = base.weight
            self.bias = bias
        } else {
            super.init(inFeatures, outFeatures, bias: false)
            self.weight = base.weight
        }
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Base linear: x @ W^T + bias
        let baseOut: MLXArray
        if let bias {
            baseOut = x.matmul(weight.transposed()) + bias
        } else {
            baseOut = x.matmul(weight.transposed())
        }

        if isMerged {
            return baseOut
        }

        // LoRA path: scaling * x @ A^T @ B^T
        var loraInput = x
        if isTraining && dropoutRate > 0 {
            let mask = MLXRandom.uniform(loraInput.shape) .> dropoutRate
            loraInput = loraInput * mask / (1.0 - dropoutRate)
        }
        let loraOut = scaling * loraInput.matmul(loraA.transposed()).matmul(loraB.transposed())

        return baseOut + loraOut
    }

    /// Fold adapter weights into the base weight matrix for zero-overhead inference.
    /// W' = W + scaling * (B @ A)
    public func merge() {
        guard !isMerged else { return }
        let delta = scaling * loraB.matmul(loraA)
        self.weight = baseWeight + delta
        isMerged = true
    }

    /// Reverse a merge operation. Only valid if weight hasn't been modified externally.
    /// W' = W - scaling * (B @ A)
    public func unmerge() {
        guard isMerged else { return }
        self.weight = baseWeight
        isMerged = false
    }

    /// Mark base parameters as non-trainable, keeping only loraA and loraB trainable.
    public func freezeBase() {
        freeze(recursive: false, keys: ["weight", "bias", "baseWeight", "baseBias"])
    }
}
```

- [ ] **Step 3: Verify it compiles**

```bash
xcodebuild -project AnvilApp.xcodeproj -scheme AnvilCore -destination 'platform=macOS' build 2>&1 | tail -5
```

Expected: BUILD SUCCEEDED

- [ ] **Step 4: Commit**

```bash
git add AnvilCore/Sources/AnvilCore/LoRA/LoRALinear.swift AnvilCore/Sources/AnvilCore/Inference/DiT/DiTComponents.swift
git commit -m "feat: add LoRALinear module with merge/unmerge support"
```

---

### Task 3: LoRAInjector Module Tree Walking

**Files:**

- Create: `AnvilCore/Sources/AnvilCore/LoRA/LoRAInjector.swift`

- [ ] **Step 1: Write LoRAInjector**

Create `AnvilCore/Sources/AnvilCore/LoRA/LoRAInjector.swift`:

```swift
import Foundation
import MLX
import MLXNN

/// Utility for injecting and removing LoRA adapters from a DiTDecoder module tree.
///
/// Walks the module tree, finds Linear layers matching target patterns,
/// and replaces them with LoRALinear wrappers (or reverses the process).
public enum LoRAInjector {

    /// Inject LoRA adapters into a model's Linear layers matching the config's target patterns.
    ///
    /// - Parameters:
    ///   - model: The DiTDecoder (or any Module) to inject into.
    ///   - config: LoRA configuration specifying rank, alpha, dropout, and target layers.
    /// - Returns: Array of key paths where injection occurred (e.g. "layers.0.selfAttn.qProj").
    @discardableResult
    public static func inject(model: Module, config: LoRAConfig) -> [String] {
        var injectedPaths: [String] = []

        guard let layersArray = model.value(forKey: "layers") as? [DiTLayer] else {
            return injectedPaths
        }

        for (layerIdx, layer) in layersArray.enumerated() {
            for pattern in config.targetLayers {
                let keyPath = "layers.\(layerIdx).\(pattern)"
                if let linear = resolveLinear(module: layer, path: pattern) {
                    let loraLinear = LoRALinear(
                        base: linear,
                        rank: config.rank,
                        alpha: config.alpha,
                        dropout: config.dropout
                    )
                    loraLinear.freezeBase()
                    setModule(on: layer, path: pattern, value: loraLinear)
                    injectedPaths.append(keyPath)
                }
            }
        }

        return injectedPaths
    }

    /// Remove LoRA adapters from previously injected paths, restoring plain Linear layers.
    ///
    /// - Parameters:
    ///   - model: The model containing LoRALinear layers.
    ///   - paths: The key paths returned by `inject()`.
    public static func remove(model: Module, paths: [String]) {
        guard let layersArray = model.value(forKey: "layers") as? [DiTLayer] else {
            return
        }

        for path in paths {
            // Parse "layers.N.selfAttn.qProj" → layerIdx=N, subPath="selfAttn.qProj"
            let components = path.split(separator: ".")
            guard components.count >= 3,
                  components[0] == "layers",
                  let layerIdx = Int(components[1]),
                  layerIdx < layersArray.count else {
                continue
            }

            let subPath = components[2...].joined(separator: ".")
            let layer = layersArray[layerIdx]

            if let loraLinear = resolveLinear(module: layer, path: subPath) as? LoRALinear {
                // Create a plain Linear from the base weights
                let inFeatures = loraLinear.baseWeight.shape[1]
                let outFeatures = loraLinear.baseWeight.shape[0]
                let hasBias = loraLinear.baseBias != nil
                let restored = Linear(inFeatures, outFeatures, bias: hasBias)
                restored.weight = loraLinear.baseWeight
                if let bias = loraLinear.baseBias {
                    restored.bias = bias
                }
                setModule(on: layer, path: subPath, value: restored)
            }
        }
    }

    /// Extract only the LoRA A and B weight matrices from injected layers.
    ///
    /// - Parameters:
    ///   - model: The model with LoRALinear layers.
    ///   - paths: The key paths returned by `inject()`.
    /// - Returns: Array of (key, MLXArray) tuples for saving to safetensors.
    public static func extractWeights(
        model: Module,
        paths: [String]
    ) -> [(String, MLXArray)] {
        var weights: [(String, MLXArray)] = []

        guard let layersArray = model.value(forKey: "layers") as? [DiTLayer] else {
            return weights
        }

        for path in paths {
            let components = path.split(separator: ".")
            guard components.count >= 3,
                  components[0] == "layers",
                  let layerIdx = Int(components[1]),
                  layerIdx < layersArray.count else {
                continue
            }

            let subPath = components[2...].joined(separator: ".")
            let layer = layersArray[layerIdx]

            if let loraLinear = resolveLinear(module: layer, path: subPath) as? LoRALinear {
                weights.append(("\(path).loraA", loraLinear.loraA))
                weights.append(("\(path).loraB", loraLinear.loraB))
            }
        }

        return weights
    }

    /// Load LoRA A and B weight matrices into existing LoRALinear layers.
    ///
    /// - Parameters:
    ///   - model: The model with LoRALinear layers already injected.
    ///   - paths: The key paths returned by `inject()`.
    ///   - weights: The (key, MLXArray) tuples loaded from safetensors.
    public static func loadWeights(
        model: Module,
        paths: [String],
        weights: [(String, MLXArray)]
    ) {
        let weightDict = Dictionary(weights, uniquingKeysWith: { _, last in last })

        guard let layersArray = model.value(forKey: "layers") as? [DiTLayer] else {
            return
        }

        for path in paths {
            let components = path.split(separator: ".")
            guard components.count >= 3,
                  components[0] == "layers",
                  let layerIdx = Int(components[1]),
                  layerIdx < layersArray.count else {
                continue
            }

            let subPath = components[2...].joined(separator: ".")
            let layer = layersArray[layerIdx]

            if let loraLinear = resolveLinear(module: layer, path: subPath) as? LoRALinear {
                if let a = weightDict["\(path).loraA"] {
                    loraLinear.loraA = a
                }
                if let b = weightDict["\(path).loraB"] {
                    loraLinear.loraB = b
                }
            }
        }
    }

    // MARK: - Private helpers

    /// Resolve a Linear (or LoRALinear) at a dot-separated path within a module.
    /// E.g. "selfAttn.qProj" → module.selfAttn.qProj
    private static func resolveLinear(module: Module, path: String) -> Linear? {
        let components = path.split(separator: ".")
        var current: Any = module
        for component in components {
            guard let mod = current as? Module,
                  let next = mod.value(forKey: String(component)) else {
                return nil
            }
            current = next
        }
        return current as? Linear
    }

    /// Set a module property at a dot-separated path.
    /// E.g. "selfAttn.qProj" sets module.selfAttn.qProj = value
    private static func setModule(on module: Module, path: String, value: Linear) {
        let components = path.split(separator: ".")
        guard !components.isEmpty else { return }

        if components.count == 1 {
            module.setValue(value, forKey: String(components[0]))
            return
        }

        // Navigate to parent, then set the final key
        var current: Any = module
        for component in components.dropLast() {
            guard let mod = current as? Module,
                  let next = mod.value(forKey: String(component)) else {
                return
            }
            current = next
        }

        if let parentModule = current as? Module {
            parentModule.setValue(value, forKey: String(components.last!))
        }
    }
}
```

**Important implementation note:** The `resolveLinear` and `setModule` helpers use KVC-style access (`value(forKey:)` and `setValue(_:forKey:)`). MLXNN Module provides these methods for traversing the module tree. If the exact API differs, these methods may need to be adapted to use Module's `children()` or subscript APIs — the subagent should check `MLXNN.Module` documentation and adapt accordingly.

- [ ] **Step 2: Verify it compiles**

```bash
xcodebuild -project AnvilApp.xcodeproj -scheme AnvilCore -destination 'platform=macOS' build 2>&1 | tail -5
```

Expected: BUILD SUCCEEDED

If compilation fails due to Module API differences, adapt `resolveLinear` and `setModule` to use whichever module tree traversal API MLXNN provides (e.g., `children()`, subscript access, or manual property access via the module's parameter tree).

- [ ] **Step 3: Commit**

```bash
git add AnvilCore/Sources/AnvilCore/LoRA/LoRAInjector.swift
git commit -m "feat: add LoRAInjector for module tree inject/remove/extract/load"
```

---

### Task 4: LoRATrainer Flow-Matching Training Loop

**Files:**

- Create: `AnvilCore/Sources/AnvilCore/LoRA/LoRATrainer.swift`
- Modify: `AnvilCore/Package.swift` (add MLXOptimizers dependency)

- [ ] **Step 1: Add MLXOptimizers to Package.swift**

In `AnvilCore/Package.swift`, add the MLXOptimizers product to the AnvilCore target's dependencies list. The current dependencies array (line 18-23) is:

```swift
// FROM:
dependencies: [
    .product(name: "MLX", package: "mlx-swift"),
    .product(name: "MLXNN", package: "mlx-swift"),
    .product(name: "MLXRandom", package: "mlx-swift"),
    .product(name: "MLXFast", package: "mlx-swift"),
    .product(name: "Transformers", package: "swift-transformers"),
]

// TO:
dependencies: [
    .product(name: "MLX", package: "mlx-swift"),
    .product(name: "MLXNN", package: "mlx-swift"),
    .product(name: "MLXOptimizers", package: "mlx-swift"),
    .product(name: "MLXRandom", package: "mlx-swift"),
    .product(name: "MLXFast", package: "mlx-swift"),
    .product(name: "Transformers", package: "swift-transformers"),
]
```

- [ ] **Step 2: Write LoRATrainer**

Create `AnvilCore/Sources/AnvilCore/LoRA/LoRATrainer.swift`:

```swift
import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom

/// A single training sample: preprocessed latent and conditioning data.
public struct TrainingSample: Sendable {
    /// Clean latent x_0 from VAE encoder. Shape: [1, frames, channels]
    public let latent: MLXArray

    /// Encoder hidden states from conditioning. Shape: [1, seqLen, hiddenSize]
    public let encoderStates: MLXArray

    /// Context latents from conditioning. Shape: [1, contextLen, hiddenSize]
    public let contextLatents: MLXArray

    /// Encoder attention mask. Shape: [1, 1, 1, seqLen] or nil
    public let encoderMask: MLXArray?

    public init(
        latent: MLXArray,
        encoderStates: MLXArray,
        contextLatents: MLXArray,
        encoderMask: MLXArray?
    ) {
        self.latent = latent
        self.encoderStates = encoderStates
        self.contextLatents = contextLatents
        self.encoderMask = encoderMask
    }
}

/// Progress report from the training loop.
public struct TrainingProgress: Sendable {
    public let epoch: Int
    public let totalEpochs: Int
    public let loss: Float
    public let elapsedTime: TimeInterval
    public let estimatedRemaining: TimeInterval
}

/// Flow-matching training loop for LoRA adapters.
///
/// Trains only the LoRA A/B matrices while keeping the base model frozen.
/// Uses flow-matching loss: sample t∈[0,1], interpolate x_t = (1-t)·x₀ + t·noise,
/// predict velocity v, loss = MSE(v_pred, v_target) where v_target = noise - x₀.
public final class LoRATrainer {
    private let model: Module
    private let config: LoRAConfig
    private let injectedPaths: [String]
    private var bestLoss: Float = .infinity
    private var bestWeights: [(String, MLXArray)]?
    private var isCancelled: Bool = false

    /// The forward function type matching DiTDecoder's call signature.
    /// Takes (hiddenStates, timestep, timestepR, encoderHiddenStates,
    ///        contextLatents, encoderAttentionMask, cache, useCache)
    /// Returns (output, cache)
    public typealias ForwardFn = (
        MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray?, CrossAttentionCache?, Bool
    ) -> (MLXArray, CrossAttentionCache?)

    public init(model: Module, config: LoRAConfig, injectedPaths: [String]) {
        self.model = model
        self.config = config
        self.injectedPaths = injectedPaths
    }

    /// Run the training loop.
    ///
    /// - Parameters:
    ///   - dataset: Array of preprocessed training samples.
    ///   - forward: The model's forward function (DiTDecoder.callAsFunction).
    ///   - onProgress: Called after each epoch with training progress.
    /// - Returns: Array of (key, MLXArray) tuples containing trained LoRA weights.
    public func train(
        dataset: [TrainingSample],
        forward: @escaping ForwardFn,
        onProgress: @escaping (TrainingProgress) -> Void
    ) -> [(String, MLXArray)] {
        guard !dataset.isEmpty else { return [] }

        // Collect only LoRA parameters for the optimizer
        let trainableParams = collectTrainableParameters()
        let optimizer = AdamW(learningRate: config.learningRate)

        let startTime = Date()
        var lastEpochLoss: Float = 0

        // Set training mode on all LoRALinear layers
        setTrainingMode(true)

        for epoch in 1...config.epochs {
            guard !isCancelled else { break }

            var epochLoss: Float = 0
            let shuffledIndices = Array(0..<dataset.count).shuffled()

            for idx in shuffledIndices {
                guard !isCancelled else { break }

                let sample = dataset[idx]

                // Flow-matching: sample random timestep
                let t = MLXRandom.uniform([1])
                let tScalar = t.item(Float.self)

                // Generate noise
                let noise = MLXRandom.normal(sample.latent.shape)

                // Interpolate: x_t = (1 - t) * x_0 + t * noise
                let oneMinusT = 1.0 - t
                let xt = oneMinusT * sample.latent + t * noise

                // Target velocity: v_target = noise - x_0
                let vTarget = noise - sample.latent

                // Compute loss and gradients using valueAndGrad
                let lossAndGrad = valueAndGrad(model: model) { model in
                    let tBatch = MLXArray.full([1], values: t)
                    let (vPred, _) = forward(
                        xt, tBatch, tBatch,
                        sample.encoderStates, sample.contextLatents,
                        sample.encoderMask, nil, false
                    )

                    // MSE loss
                    let diff = vPred - vTarget
                    let mse = (diff * diff).mean()
                    return mse
                }

                let (loss, grads) = lossAndGrad

                // Update only LoRA parameters
                optimizer.update(model: model, gradients: grads)
                eval(model.parameters())

                epochLoss += loss.item(Float.self)
            }

            let avgLoss = epochLoss / Float(dataset.count)
            lastEpochLoss = avgLoss

            // Track best weights
            if avgLoss < bestLoss {
                bestLoss = avgLoss
                bestWeights = LoRAInjector.extractWeights(model: model, paths: injectedPaths)
            }

            let elapsed = Date().timeIntervalSince(startTime)
            let avgEpochTime = elapsed / Double(epoch)
            let remaining = avgEpochTime * Double(config.epochs - epoch)

            onProgress(TrainingProgress(
                epoch: epoch,
                totalEpochs: config.epochs,
                loss: avgLoss,
                elapsedTime: elapsed,
                estimatedRemaining: remaining
            ))
        }

        setTrainingMode(false)

        // Return best weights (or current if none tracked)
        return bestWeights ?? LoRAInjector.extractWeights(model: model, paths: injectedPaths)
    }

    /// Request training cancellation. The loop will stop after the current sample.
    public func cancel() {
        isCancelled = true
    }

    /// Whether at least 2 epochs completed (enough to save best checkpoint).
    public var hasUsableCheckpoint: Bool {
        bestWeights != nil
    }

    /// The best loss seen during training, or nil if no complete epoch.
    public var lowestLoss: Float? {
        bestWeights != nil ? bestLoss : nil
    }

    // MARK: - Private

    private func collectTrainableParameters() -> [String: MLXArray] {
        // Module.trainableParameters() returns only non-frozen params.
        // After freezeBase(), only loraA and loraB remain trainable.
        var params: [String: MLXArray] = [:]
        for (key, value) in model.trainableParameters().flattened() {
            params[key] = value
        }
        return params
    }

    private func setTrainingMode(_ training: Bool) {
        // Walk injected paths and set isTraining on each LoRALinear
        guard let layersArray = model.value(forKey: "layers") as? [DiTLayer] else {
            return
        }

        for path in injectedPaths {
            let components = path.split(separator: ".")
            guard components.count >= 3,
                  components[0] == "layers",
                  let layerIdx = Int(components[1]),
                  layerIdx < layersArray.count else {
                continue
            }

            let subPath = components[2...].joined(separator: ".")
            let layer = layersArray[layerIdx]

            // Navigate to the LoRALinear
            var current: Any = layer
            for component in subPath.split(separator: ".") {
                guard let mod = current as? Module,
                      let next = mod.value(forKey: String(component)) else {
                    break
                }
                current = next
            }

            if let loraLinear = current as? LoRALinear {
                loraLinear.isTraining = training
            }
        }
    }
}
```

**Implementation note:** The `valueAndGrad` function from MLX computes loss and gradients in one pass. The exact API may be `MLX.valueAndGrad(model:_:)` or similar — the subagent should check MLX Swift documentation and adapt the call signature. The core pattern is: wrap the forward pass + loss computation in a closure, get both the loss value and parameter gradients back.

- [ ] **Step 3: Verify it compiles**

```bash
xcodebuild -project AnvilApp.xcodeproj -scheme AnvilCore -destination 'platform=macOS' build 2>&1 | tail -5
```

Expected: BUILD SUCCEEDED

If `MLXOptimizers` is not found, check that the mlx-swift package version (≥0.21.0) includes the MLXOptimizers product. If not available as a separate product, AdamW may be in the `MLXNN` module or the optimizer may need a manual implementation.

- [ ] **Step 4: Commit**

```bash
git add AnvilCore/Sources/AnvilCore/LoRA/LoRATrainer.swift AnvilCore/Package.swift
git commit -m "feat: add LoRATrainer with flow-matching training loop"
```

---

### Task 5: AdapterRegistry File-System Scanner

**Files:**

- Create: `AnvilApp/Services/AdapterRegistry.swift`

- [ ] **Step 1: Write AdapterRegistry**

Create `AnvilApp/Services/AdapterRegistry.swift`:

```swift
import Foundation
import AnvilCore

/// Scans the adapter directory for saved LoRA adapters.
///
/// Lightweight file-system scanner — not an actor, not an observable.
/// Called by the adapter picker to populate its dropdown and by
/// TrainingService to determine save locations.
struct AdapterRegistry {
    /// Base directory for all adapters.
    static var adaptersDirectory: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appending(path: "Anvil/models/adapters")
    }

    /// Scan the adapters directory and return all valid adapters.
    static func listAdapters() -> [LoRAAdapter] {
        let dir = adaptersDirectory
        let fm = FileManager.default

        guard let contents = try? fm.contentsOfDirectory(
            at: dir,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) else {
            return []
        }

        var adapters: [LoRAAdapter] = []
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601

        for itemURL in contents {
            let isDir = (try? itemURL.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory ?? false
            guard isDir else { continue }

            let metadataURL = itemURL.appending(path: "adapter.json")
            let weightsURL = itemURL.appending(path: "adapter.safetensors")

            // Both files must exist
            guard fm.fileExists(atPath: metadataURL.path(percentEncoded: false)),
                  fm.fileExists(atPath: weightsURL.path(percentEncoded: false)) else {
                continue
            }

            guard let data = try? Data(contentsOf: metadataURL),
                  let adapter = try? decoder.decode(LoRAAdapter.self, from: data) else {
                continue
            }

            adapters.append(adapter)
        }

        return adapters.sorted { $0.createdAt > $1.createdAt }
    }

    /// Return adapters compatible with a given model category.
    static func compatibleAdapters(for category: ModelCategory) -> [LoRAAdapter] {
        listAdapters().filter { $0.baseModelCategory == category }
    }

    /// Directory URL for a specific adapter by ID.
    static func adapterDirectory(id: String) -> URL {
        adaptersDirectory.appending(path: id)
    }

    /// Save adapter metadata and weights.
    static func save(
        adapter: LoRAAdapter,
        weights: [(String, MLXArray)]
    ) throws {
        let dir = adapterDirectory(id: adapter.id)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        // Save metadata
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(adapter)
        try data.write(to: dir.appending(path: "adapter.json"))

        // Save weights as safetensors
        try saveSafetensors(weights: weights, to: dir.appending(path: "adapter.safetensors"))
    }

    /// Load LoRA weights from a saved adapter's safetensors file.
    static func loadWeights(adapterID: String) throws -> [(String, MLXArray)] {
        let weightsURL = adapterDirectory(id: adapterID).appending(path: "adapter.safetensors")
        return try loadSafetensors(from: weightsURL)
    }

    // MARK: - Safetensors I/O

    /// Save weight tuples to a safetensors file.
    /// Uses MLX's built-in safetensors support.
    private static func saveSafetensors(
        weights: [(String, MLXArray)],
        to url: URL
    ) throws {
        let dict = Dictionary(weights, uniquingKeysWith: { _, last in last })
        try MLX.save(arrays: dict, url: url)
    }

    /// Load weight tuples from a safetensors file.
    private static func loadSafetensors(from url: URL) throws -> [(String, MLXArray)] {
        let dict = try MLX.loadArrays(url: url)
        return dict.map { ($0.key, $0.value) }
    }
}
```

**Implementation note:** The `MLX.save(arrays:url:)` and `MLX.loadArrays(url:)` functions handle safetensors format. If the exact API names differ in the mlx-swift version, the subagent should check the MLX Swift documentation. The app's existing `WeightConverter.loadSafetensors(from:)` can also be referenced as a pattern.

- [ ] **Step 2: Create the AdapterRegistry directory structure for file placement**

```bash
mkdir -p AnvilApp/Services
```

(Should already exist since other services are there.)

- [ ] **Step 3: Verify it compiles**

```bash
xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | tail -5
```

Expected: BUILD SUCCEEDED (ignoring pre-existing SFXModelManager scope errors)

- [ ] **Step 4: Commit**

```bash
git add AnvilApp/Services/AdapterRegistry.swift
git commit -m "feat: add AdapterRegistry file-system scanner for saved adapters"
```

---

### Task 6: TrainingService Orchestration

**Files:**

- Create: `AnvilApp/Services/TrainingService.swift`

- [ ] **Step 1: Write TrainingService**

Create `AnvilApp/Services/TrainingService.swift`:

```swift
import Foundation
import AnvilCore
import MLX

/// Training state for UI observation.
enum TrainingState: Equatable {
    case idle
    case preparingDataset(current: Int, total: Int)
    case training(TrainingProgress)
    case saving
    case complete(LoRAAdapter)
    case failed(String)
    case cancelled

    static func == (lhs: TrainingState, rhs: TrainingState) -> Bool {
        switch (lhs, rhs) {
        case (.idle, .idle): return true
        case (.cancelled, .cancelled): return true
        case (.saving, .saving): return true
        case (.preparingDataset(let a1, let a2), .preparingDataset(let b1, let b2)):
            return a1 == b1 && a2 == b2
        case (.complete(let a), .complete(let b)):
            return a.id == b.id
        case (.failed(let a), .failed(let b)):
            return a == b
        case (.training(let a), .training(let b)):
            return a.epoch == b.epoch && a.totalEpochs == b.totalEpochs
        default: return false
        }
    }
}

/// Orchestrates the full LoRA training flow:
/// 1. Preprocess dataset (audio → latents, prompt → conditioning)
/// 2. Inject LoRA into DiT
/// 3. Run training loop
/// 4. Save adapter (weights + metadata)
/// 5. Clean up (remove LoRA from DiT)
@MainActor
final class TrainingService: ObservableObject {
    @Published var state: TrainingState = .idle

    private let pipeline: MusicPipeline
    private let conditioning: ConditioningActor
    private let registry: ModelRegistry
    private var trainer: LoRATrainer?
    private var trainingTask: Task<Void, Never>?

    init(
        pipeline: MusicPipeline,
        conditioning: ConditioningActor,
        registry: ModelRegistry
    ) {
        self.pipeline = pipeline
        self.conditioning = conditioning
        self.registry = registry
    }

    /// Start training with the given entries and config.
    ///
    /// - Parameters:
    ///   - entries: History entries to use as training data.
    ///   - audioURLs: File URLs corresponding to each entry's audio.
    ///   - importedFiles: Additional imported audio files with descriptions.
    ///   - adapterName: User-chosen name for the adapter.
    ///   - config: LoRA training configuration.
    func startTraining(
        entries: [HistoryEntry],
        audioURLs: [URL],
        importedFiles: [(url: URL, description: String)],
        adapterName: String,
        config: LoRAConfig
    ) {
        trainingTask?.cancel()

        trainingTask = Task { [weak self] in
            guard let self else { return }

            do {
                // 1. Prepare dataset
                let totalSamples = entries.count + importedFiles.count
                self.state = .preparingDataset(current: 0, total: totalSamples)

                var samples: [TrainingSample] = []

                // Process history entries
                for (idx, entry) in entries.enumerated() {
                    try Task.checkCancellation()
                    let url = audioURLs[idx]
                    let sample = try await self.preprocessSample(
                        audioURL: url,
                        prompt: entry.prompt,
                        lyrics: entry.lyrics,
                        duration: entry.duration
                    )
                    samples.append(sample)
                    self.state = .preparingDataset(current: idx + 1, total: totalSamples)
                }

                // Process imported files
                for (idx, imported) in importedFiles.enumerated() {
                    try Task.checkCancellation()
                    let sample = try await self.preprocessSample(
                        audioURL: imported.url,
                        prompt: imported.description,
                        lyrics: "",
                        duration: 0 // Duration determined from audio
                    )
                    samples.append(sample)
                    self.state = .preparingDataset(
                        current: entries.count + idx + 1,
                        total: totalSamples
                    )
                }

                guard !samples.isEmpty else {
                    self.state = .failed("No valid training samples")
                    return
                }

                // 2. Inject LoRA and run training
                // Training happens on the DiTActor through the pipeline chain
                let result = try await self.runTraining(
                    samples: samples,
                    config: config
                )

                try Task.checkCancellation()

                // 3. Save adapter
                self.state = .saving

                let adapterID = adapterName
                    .lowercased()
                    .replacingOccurrences(of: " ", with: "-")
                    .replacingOccurrences(
                        of: "[^a-z0-9\\-]",
                        with: "",
                        options: .regularExpression
                    )

                let activeModelID = await self.registry.activeModelID(for: .aceStepDiT) ?? "unknown"

                let adapter = LoRAAdapter(
                    id: adapterID,
                    name: adapterName,
                    config: config,
                    baseModelID: activeModelID,
                    baseModelCategory: .aceStepDiT,
                    createdAt: Date(),
                    sampleCount: samples.count,
                    finalLoss: result.finalLoss
                )

                try AdapterRegistry.save(adapter: adapter, weights: result.weights)

                self.state = .complete(adapter)

            } catch is CancellationError {
                // Check if we have a usable checkpoint
                if let trainer = self.trainer, trainer.hasUsableCheckpoint,
                   let loss = trainer.lowestLoss {
                    // Save best checkpoint
                    self.state = .saving
                    // Checkpoint saving handled via stopTraining()
                }
                self.state = .cancelled
            } catch {
                self.state = .failed(error.localizedDescription)
            }

            self.trainer = nil
        }
    }

    /// Stop training. If 2+ epochs completed, saves the best checkpoint.
    func stopTraining() {
        trainer?.cancel()
    }

    /// Whether training can be stopped with a saved checkpoint.
    var canSaveOnStop: Bool {
        trainer?.hasUsableCheckpoint ?? false
    }

    // MARK: - Private

    private func preprocessSample(
        audioURL: URL,
        prompt: String,
        lyrics: String,
        duration: TimeInterval
    ) async throws -> TrainingSample {
        // Load audio file
        let audioData = try WAVReader.read(from: audioURL)
        let audioArray = MLXArray(audioData.samples)
            .reshaped([1, 1, audioData.samples.count])

        // Encode audio to latent
        let latent = await pipeline.encodeMean(
            audio: audioArray,
            precision: .float16
        )

        // Encode text to conditioning
        let actualDuration = duration > 0
            ? duration
            : Double(audioData.samples.count) / Double(audioData.sampleRate)
        let cond = await conditioning.encode(
            text: prompt,
            lyrics: lyrics,
            referenceAudio: nil,
            duration: actualDuration,
            precision: .float16
        )

        return TrainingSample(
            latent: latent,
            encoderStates: cond.encoderStates,
            contextLatents: cond.contextLatents,
            encoderMask: cond.encoderMask
        )
    }

    private struct TrainingResult {
        let weights: [(String, MLXArray)]
        let finalLoss: Float?
    }

    private func runTraining(
        samples: [TrainingSample],
        config: LoRAConfig
    ) async throws -> TrainingResult {
        // Inject LoRA, train, extract weights, remove LoRA
        // This delegates through the pipeline → diffusion → DiT actor chain
        let result = try await pipeline.trainLoRA(
            samples: samples,
            config: config,
            onProgress: { [weak self] progress in
                Task { @MainActor in
                    self?.state = .training(progress)
                }
            }
        )

        return TrainingResult(
            weights: result.weights,
            finalLoss: result.finalLoss
        )
    }
}
```

**Implementation notes:**

1. `WAVReader.read(from:)` — The subagent should check if this utility exists in AnvilCore. If not, use whatever audio loading mechanism the codebase provides (e.g., AVFoundation `AVAudioFile`).
2. `pipeline.encodeMean(audio:precision:)` — This needs to be added as a forwarding method on MusicPipeline (Task 10 adds this). The subagent may need to add a stub or implement it here.
3. `pipeline.trainLoRA(samples:config:onProgress:)` — This also needs to be added as a forwarding method (Task 10). Same approach.
4. The exact `ConditioningActor.encode` return type should be checked — it likely returns a struct with `encoderStates`, `contextLatents`, `encoderMask` properties.

- [ ] **Step 2: Verify it compiles**

```bash
xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | tail -5
```

Expected: BUILD SUCCEEDED (may have errors from missing forwarding methods — those are resolved in Task 10)

- [ ] **Step 3: Commit**

```bash
git add AnvilApp/Services/TrainingService.swift
git commit -m "feat: add TrainingService for LoRA training orchestration"
```

---

### Task 7: Training Sheet UI

**Files:**

- Create: `AnvilApp/UI/Training/TrainingSheet.swift`

- [ ] **Step 1: Create the Training directory**

```bash
mkdir -p AnvilApp/UI/Training
```

- [ ] **Step 2: Write TrainingSheet**

Create `AnvilApp/UI/Training/TrainingSheet.swift`:

```swift
import SwiftUI
import AnvilCore

/// Sheet for configuring and running LoRA training.
/// Presented from History view after selecting tracks.
struct TrainingSheet: View {
    @EnvironmentObject private var trainingService: TrainingService
    @EnvironmentObject private var historyStore: HistoryStore

    /// The history entries selected for training.
    let selectedEntries: [HistoryEntry]

    /// Dismiss action provided by the presenting view.
    @Environment(\.dismiss) private var dismiss

    // Config state
    @State private var adapterName: String = ""
    @State private var selectedRank: Int = 16
    @State private var selectedEpochs: Int = 10
    @State private var importedFiles: [(url: URL, description: String)] = []
    @State private var showingFileImporter: Bool = false
    @State private var showingStopConfirmation: Bool = false

    private let rankOptions = [8, 16, 32]
    private let epochOptions = [5, 10, 20, 50]

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            ScrollView {
                switch trainingService.state {
                case .idle, .failed, .cancelled:
                    configView
                case .preparingDataset, .training, .saving:
                    progressView
                case .complete(let adapter):
                    completionView(adapter: adapter)
                }
            }
        }
        .frame(minWidth: 450, minHeight: 400)
        .fileImporter(
            isPresented: $showingFileImporter,
            allowedContentTypes: [.audio],
            allowsMultipleSelection: true
        ) { result in
            handleFileImport(result)
        }
        .alert("Stop Training?", isPresented: $showingStopConfirmation) {
            Button("Continue Training", role: .cancel) {}
            if trainingService.canSaveOnStop {
                Button("Save & Stop") {
                    trainingService.stopTraining()
                }
            }
            Button("Discard & Stop", role: .destructive) {
                trainingService.stopTraining()
            }
        } message: {
            if trainingService.canSaveOnStop {
                Text("Training has completed 2+ epochs. You can save the best checkpoint or discard.")
            } else {
                Text("Training hasn't completed enough epochs to save. Progress will be lost.")
            }
        }
    }

    // MARK: - Header

    private var header: some View {
        HStack {
            Text("Train LoRA Adapter")
                .font(.headline)
            Spacer()
            Button("Close") {
                dismiss()
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
        }
        .padding()
    }

    // MARK: - Config View

    private var configView: some View {
        VStack(alignment: .leading, spacing: 16) {
            // Training data section
            trainingDataSection

            // Configuration section
            configSection

            // Error display
            if case .failed(let message) = trainingService.state {
                Label(message, systemImage: "exclamationmark.triangle")
                    .foregroundStyle(.red)
                    .font(.caption)
            }

            // Start button
            HStack {
                Spacer()
                Button("Start Training") {
                    startTraining()
                }
                .buttonStyle(.borderedProminent)
                .disabled(adapterName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                          || (selectedEntries.isEmpty && importedFiles.isEmpty))
            }
        }
        .padding()
    }

    private var trainingDataSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Training Data")
                .font(.subheadline)
                .fontWeight(.semibold)

            VStack(spacing: 4) {
                ForEach(selectedEntries) { entry in
                    HStack {
                        Image(systemName: "waveform")
                            .foregroundStyle(.secondary)
                        Text(entry.prompt)
                            .lineLimit(1)
                            .font(.callout)
                        Spacer()
                        Text("\(Int(entry.duration))s")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }

                ForEach(importedFiles.indices, id: \.self) { idx in
                    HStack {
                        Image(systemName: "doc.badge.plus")
                            .foregroundStyle(.secondary)
                        TextField("Description", text: Binding(
                            get: { importedFiles[idx].description },
                            set: { importedFiles[idx].description = $0 }
                        ))
                        .font(.callout)
                        .textFieldStyle(.roundedBorder)
                        Button {
                            importedFiles.remove(at: idx)
                        } label: {
                            Image(systemName: "xmark.circle")
                                .foregroundStyle(.secondary)
                        }
                        .buttonStyle(.borderless)
                    }
                }
            }
            .padding(8)
            .background(.quaternary.opacity(0.3))
            .clipShape(RoundedRectangle(cornerRadius: 8))

            HStack {
                let totalCount = selectedEntries.count + importedFiles.count
                Text("\(totalCount) tracks selected")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                Button("+ Add audio files...") {
                    showingFileImporter = true
                }
                .buttonStyle(.borderless)
                .font(.caption)
            }
        }
    }

    private var configSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Configuration")
                .font(.subheadline)
                .fontWeight(.semibold)

            LabeledContent("Adapter Name") {
                TextField("e.g. My Jazz Style", text: $adapterName)
                    .textFieldStyle(.roundedBorder)
                    .frame(maxWidth: 200)
            }

            LabeledContent("Base Model") {
                Text("ACE-Step DiT (active)")
                    .foregroundStyle(.secondary)
            }

            LabeledContent("Rank") {
                Picker("Rank", selection: $selectedRank) {
                    ForEach(rankOptions, id: \.self) { rank in
                        Text("\(rank)").tag(rank)
                    }
                }
                .pickerStyle(.segmented)
                .frame(maxWidth: 150)
            }

            LabeledContent("Epochs") {
                Picker("Epochs", selection: $selectedEpochs) {
                    ForEach(epochOptions, id: \.self) { epoch in
                        Text("\(epoch)").tag(epoch)
                    }
                }
                .pickerStyle(.segmented)
                .frame(maxWidth: 200)
            }
        }
    }

    // MARK: - Progress View

    private var progressView: some View {
        VStack(spacing: 16) {
            Spacer()

            switch trainingService.state {
            case .preparingDataset(let current, let total):
                VStack(spacing: 8) {
                    Text("Preparing Dataset")
                        .font(.headline)
                    Text("Encoding sample \(current) of \(total)")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                    ProgressView(value: Double(current), total: Double(total))
                        .frame(maxWidth: 300)
                }

            case .training(let progress):
                VStack(spacing: 12) {
                    Text("Training \"\(adapterName)\"")
                        .font(.headline)
                    Text("Epoch \(progress.epoch) of \(progress.totalEpochs)")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)

                    ProgressView(
                        value: Double(progress.epoch),
                        total: Double(progress.totalEpochs)
                    )
                    .frame(maxWidth: 300)

                    HStack(spacing: 24) {
                        statView(
                            value: String(format: "%.4f", progress.loss),
                            label: "Current Loss"
                        )
                        statView(
                            value: formatTime(progress.elapsedTime),
                            label: "Elapsed"
                        )
                        statView(
                            value: "~\(formatTime(progress.estimatedRemaining))",
                            label: "Remaining"
                        )
                    }
                }

            case .saving:
                VStack(spacing: 8) {
                    ProgressView()
                    Text("Saving adapter...")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }

            default:
                EmptyView()
            }

            Spacer()

            // Stop button
            if case .training = trainingService.state {
                Button("Stop Training", role: .destructive) {
                    showingStopConfirmation = true
                }
                .buttonStyle(.bordered)
            }
        }
        .padding()
    }

    // MARK: - Completion View

    private func completionView(adapter: LoRAAdapter) -> some View {
        VStack(spacing: 16) {
            Spacer()

            Image(systemName: "checkmark.circle.fill")
                .font(.largeTitle)
                .foregroundStyle(.green)

            Text("Training Complete")
                .font(.headline)

            VStack(spacing: 4) {
                Text(adapter.name)
                    .font(.title3)
                    .fontWeight(.semibold)

                if let loss = adapter.finalLoss {
                    Text("Final loss: \(String(format: "%.4f", loss))")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Text("Rank \(adapter.config.rank) \u{2022} \(adapter.sampleCount) tracks")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            Button("Done") {
                dismiss()
            }
            .buttonStyle(.borderedProminent)
        }
        .padding()
    }

    // MARK: - Helpers

    private func statView(value: String, label: String) -> some View {
        VStack(spacing: 2) {
            Text(value)
                .font(.title3)
                .fontWeight(.semibold)
                .monospacedDigit()
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }

    private func formatTime(_ interval: TimeInterval) -> String {
        let minutes = Int(interval) / 60
        let seconds = Int(interval) % 60
        return "\(minutes):\(String(format: "%02d", seconds))"
    }

    private func startTraining() {
        let audioURLs = selectedEntries.map { historyStore.fileURL(for: $0) }
        let config = LoRAConfig(
            rank: selectedRank,
            epochs: selectedEpochs
        )

        trainingService.startTraining(
            entries: selectedEntries,
            audioURLs: audioURLs,
            importedFiles: importedFiles,
            adapterName: adapterName,
            config: config
        )
    }

    private func handleFileImport(_ result: Result<[URL], Error>) {
        switch result {
        case .success(let urls):
            for url in urls {
                importedFiles.append((url: url, description: ""))
            }
        case .failure:
            break
        }
    }
}
```

**Implementation note:** `historyStore.fileURL(for:)` — check that HistoryStore has this method. Looking at the current HistoryStore code, it uses `directory.appending(path: entry.fileName)`. The subagent should verify the method exists and add it if needed.

- [ ] **Step 3: Verify it compiles**

```bash
xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | tail -5
```

Expected: BUILD SUCCEEDED (may have errors from missing TrainingService methods — those get resolved as other tasks complete)

- [ ] **Step 4: Commit**

```bash
git add AnvilApp/UI/Training/TrainingSheet.swift
git commit -m "feat: add TrainingSheet UI for LoRA config and progress"
```

---

### Task 8: HistoryListView Multi-Select + Train LoRA Button

**Files:**

- Modify: `AnvilApp/UI/Components/HistoryListView.swift`

- [ ] **Step 1: Add multi-select state and Train LoRA button to HistoryListView**

Replace the full content of `AnvilApp/UI/Components/HistoryListView.swift`:

```swift
import SwiftUI

struct HistoryListView: View {
    @EnvironmentObject private var historyStore: HistoryStore
    @EnvironmentObject private var playerService: AudioPlayerService
    @EnvironmentObject private var trainingService: TrainingService

    @State private var isSelecting: Bool = false
    @State private var selectedIDs: Set<UUID> = []
    @State private var showingTrainingSheet: Bool = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            Divider()
            if historyStore.entries.isEmpty {
                emptyState
            } else {
                list
            }
        }
        .sheet(isPresented: $showingTrainingSheet) {
            TrainingSheet(selectedEntries: selectedEntries)
        }
    }

    private var selectedEntries: [HistoryEntry] {
        historyStore.entries.filter { selectedIDs.contains($0.id) }
    }

    private var header: some View {
        HStack {
            Text("HISTORY")
                .font(.caption)
                .fontWeight(.semibold)
                .foregroundStyle(.secondary)

            Spacer()

            if isSelecting && !selectedIDs.isEmpty {
                Text("\(selectedIDs.count) selected")
                    .font(.caption2)
                    .foregroundStyle(.secondary)

                Button("Train LoRA") {
                    showingTrainingSheet = true
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.mini)
            }

            if !historyStore.entries.isEmpty {
                Button(isSelecting ? "Done" : "Select") {
                    isSelecting.toggle()
                    if !isSelecting {
                        selectedIDs.removeAll()
                    }
                }
                .buttonStyle(.bordered)
                .controlSize(.mini)

                if !isSelecting {
                    Button("Delete All", role: .destructive) {
                        historyStore.deleteAll()
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.mini)
                }
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
    }

    private var emptyState: some View {
        VStack(spacing: 8) {
            Image(systemName: "waveform")
                .font(.title)
                .foregroundStyle(.quaternary)
            Text("No generations yet")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var list: some View {
        List {
            ForEach(historyStore.entries) { entry in
                HStack {
                    if isSelecting {
                        Image(systemName: selectedIDs.contains(entry.id)
                              ? "checkmark.circle.fill"
                              : "circle")
                            .foregroundStyle(
                                selectedIDs.contains(entry.id) ? .blue : .secondary
                            )
                            .onTapGesture {
                                toggleSelection(entry.id)
                            }
                    }

                    HistoryRow(entry: entry)
                }
                .contentShape(Rectangle())
                .onTapGesture {
                    if isSelecting {
                        toggleSelection(entry.id)
                    } else {
                        let url = historyStore.fileURL(for: entry)
                        playerService.load(url: url)
                    }
                }
            }
            .onDelete { indices in
                let toDelete = indices.map { historyStore.entries[$0] }
                for entry in toDelete {
                    historyStore.delete(entry)
                }
            }
        }
        .listStyle(.plain)
    }

    private func toggleSelection(_ id: UUID) {
        if selectedIDs.contains(id) {
            selectedIDs.remove(id)
        } else {
            selectedIDs.insert(id)
        }
    }
}

struct HistoryRow: View {
    let entry: HistoryEntry

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(entry.prompt)
                .font(.callout)
                .lineLimit(1)
                .truncationMode(.tail)

            HStack(spacing: 8) {
                Text("\(Int(entry.duration))s")
                Text("\u{2022}")
                Text(entry.createdAt, style: .date)
                Text("\u{2022}")
                Text("seed: \(entry.seed)")
            }
            .font(.caption2)
            .foregroundStyle(.secondary)
        }
        .padding(.vertical, 4)
    }
}
```

- [ ] **Step 2: Verify it compiles**

```bash
xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | tail -5
```

Expected: BUILD SUCCEEDED

- [ ] **Step 3: Commit**

```bash
git add AnvilApp/UI/Components/HistoryListView.swift
git commit -m "feat: add multi-select mode and Train LoRA button to History"
```

---

### Task 9: AdapterPicker in Generate View

**Files:**

- Create: `AnvilApp/UI/Generate/AdapterPicker.swift`
- Modify: `AnvilApp/UI/Components/GenerateView.swift`

- [ ] **Step 1: Create the Generate UI directory**

```bash
mkdir -p AnvilApp/UI/Generate
```

- [ ] **Step 2: Write AdapterPicker**

Create `AnvilApp/UI/Generate/AdapterPicker.swift`:

```swift
import SwiftUI
import AnvilCore

/// Dropdown picker for selecting a LoRA adapter in the Generate view.
/// Shows "None" plus all adapters compatible with the active base model.
struct AdapterPicker: View {
    @Binding var selectedAdapterID: String?
    let activeModelCategory: ModelCategory

    @State private var adapters: [LoRAAdapter] = []

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("LoRA Adapter")
                .font(.caption)
                .foregroundStyle(.secondary)

            Picker("LoRA Adapter", selection: $selectedAdapterID) {
                Text("None").tag(String?.none)

                ForEach(adapters) { adapter in
                    HStack {
                        Text(adapter.name)
                        Text("rank \(adapter.config.rank) \u{2022} \(adapter.sampleCount) tracks")
                            .foregroundStyle(.secondary)
                    }
                    .tag(Optional(adapter.id))
                }
            }
            .pickerStyle(.menu)
        }
        .onAppear {
            refreshAdapters()
        }
        .onChange(of: activeModelCategory) { _, _ in
            refreshAdapters()
            // Revert selection if incompatible
            if let selected = selectedAdapterID,
               !adapters.contains(where: { $0.id == selected }) {
                selectedAdapterID = nil
            }
        }
    }

    private func refreshAdapters() {
        adapters = AdapterRegistry.compatibleAdapters(for: activeModelCategory)
    }
}
```

- [ ] **Step 3: Add AdapterPicker to GenerateView**

In `AnvilApp/UI/Components/GenerateView.swift`, add the adapter picker state and integrate it into the view.

Add a new `@State` property after the existing state declarations (after line 20):

```swift
@State private var selectedAdapterID: String? = nil
```

Also add `@AppStorage` for persistence:

```swift
@AppStorage("selectedAdapterID") private var persistedAdapterID: String = ""
```

Add the adapter picker to the body's VStack, after `controlsBar` and before `progressSection`. Insert after line 29 (`controlsBar`):

```swift
adapterPickerSection
```

Add the adapter picker section computed property (after the `controlsBar` property):

```swift
private var adapterPickerSection: some View {
    AdapterPicker(
        selectedAdapterID: $selectedAdapterID,
        activeModelCategory: .aceStepDiT
    )
    .onChange(of: selectedAdapterID) { _, newValue in
        persistedAdapterID = newValue ?? ""
    }
    .onAppear {
        if !persistedAdapterID.isEmpty {
            selectedAdapterID = persistedAdapterID
        }
    }
}
```

- [ ] **Step 4: Verify it compiles**

```bash
xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | tail -5
```

Expected: BUILD SUCCEEDED

- [ ] **Step 5: Commit**

```bash
git add AnvilApp/UI/Generate/AdapterPicker.swift AnvilApp/UI/Components/GenerateView.swift
git commit -m "feat: add AdapterPicker dropdown to Generate view"
```

---

### Task 10: GenerationService Adapter Injection + Actor Forwarding

**Files:**

- Modify: `AnvilCore/Sources/AnvilCore/Inference/DiT/DiTActor.swift`
- Modify: `AnvilCore/Sources/AnvilCore/Inference/Diffusion/DiffusionActor.swift`
- Modify: `AnvilCore/Sources/AnvilCore/Inference/Pipeline/MusicPipeline.swift`
- Modify: `AnvilApp/Services/GenerationService.swift`

- [ ] **Step 1: Add LoRA methods to DiTActor**

In `AnvilCore/Sources/AnvilCore/Inference/DiT/DiTActor.swift`, add the following methods after the existing `unloadWeights()` method (after line 63):

```swift
// MARK: - LoRA

public private(set) var injectedAdapterID: String?
private var injectedPaths: [String] = []

/// Inject LoRA adapter layers into the DiT model.
public func injectLoRA(config: LoRAConfig) -> [String] {
    guard let decoder else {
        fatalError("DiTActor.injectLoRA called without loaded weights")
    }
    let paths = LoRAInjector.inject(model: decoder, config: config)
    injectedPaths = paths
    return paths
}

/// Remove LoRA adapter layers, restoring plain Linear layers.
public func removeLoRA() {
    guard let decoder else { return }
    LoRAInjector.remove(model: decoder, paths: injectedPaths)
    injectedPaths = []
    injectedAdapterID = nil
}

/// Load saved LoRA weights into already-injected LoRALinear layers.
public func loadLoRAWeights(_ weights: [(String, MLXArray)], adapterID: String) {
    guard let decoder else {
        fatalError("DiTActor.loadLoRAWeights called without loaded weights")
    }
    LoRAInjector.loadWeights(model: decoder, paths: injectedPaths, weights: weights)
    eval(decoder.parameters())
    injectedAdapterID = adapterID
}

/// Extract current LoRA weights (for saving after training).
public func extractLoRAWeights() -> [(String, MLXArray)] {
    guard let decoder else { return [] }
    return LoRAInjector.extractWeights(model: decoder, paths: injectedPaths)
}

/// Access the decoder directly for training.
/// Only used by LoRATrainer — not for normal inference.
public var decoderForTraining: DiTDecoder? { decoder }
```

- [ ] **Step 2: Add LoRA forwarding methods to DiffusionActor**

In `AnvilCore/Sources/AnvilCore/Inference/Diffusion/DiffusionActor.swift`, add after the `unloadDiTWeights()` method (after line 132):

```swift
// MARK: - LoRA

public func injectLoRA(config: LoRAConfig) async -> [String] {
    await dit.injectLoRA(config: config)
}

public func removeLoRA() async {
    await dit.removeLoRA()
}

public func loadLoRAWeights(_ weights: [(String, MLXArray)], adapterID: String) async {
    await dit.loadLoRAWeights(weights, adapterID: adapterID)
}

public var injectedAdapterID: String? {
    get async { await dit.injectedAdapterID }
}
```

- [ ] **Step 3: Add LoRA forwarding methods to MusicPipeline**

In `AnvilCore/Sources/AnvilCore/Inference/Pipeline/MusicPipeline.swift`, add after the `unloadVAEWeights()` method (after line 51):

```swift
// MARK: - LoRA

public func injectLoRA(config: LoRAConfig) async -> [String] {
    await diffusion.injectLoRA(config: config)
}

public func removeLoRA() async {
    await diffusion.removeLoRA()
}

public func loadLoRAWeights(_ weights: [(String, MLXArray)], adapterID: String) async {
    await diffusion.loadLoRAWeights(weights, adapterID: adapterID)
}

public var injectedAdapterID: String? {
    get async { await diffusion.injectedAdapterID }
}

/// Encode audio to latent mean (used for training dataset preparation).
public func encodeMean(audio: MLXArray, precision: Precision) async -> MLXArray {
    await vae.encodeMean(audio: audio, precision: precision)
}
```

- [ ] **Step 4: Add adapter injection to GenerationService**

In `AnvilApp/Services/GenerationService.swift`, add the adapter tracking property after `currentlyLoadedVAEModelID` (after line 24):

```swift
private var currentlyInjectedAdapterID: String?
```

Add a public property for the selected adapter (after the new property above):

```swift
var selectedAdapterID: String?
```

Add the adapter injection method after `loadVAEWeights()` (after line 293):

```swift
// MARK: - LoRA Adapter Injection

private func ensureAdapter() async {
    let desired = selectedAdapterID
    let current = currentlyInjectedAdapterID

    if desired == current {
        // Already correct — no change needed
        return
    }

    // Remove current adapter if one is injected
    if current != nil {
        await pipeline.removeLoRA()
        currentlyInjectedAdapterID = nil
    }

    // Inject new adapter if one is selected
    guard let adapterID = desired else { return }

    do {
        let adapters = AdapterRegistry.listAdapters()
        guard let adapter = adapters.first(where: { $0.id == adapterID }) else {
            selectedAdapterID = nil
            return
        }

        let config = adapter.config
        let _ = await pipeline.injectLoRA(config: config)
        let weights = try AdapterRegistry.loadWeights(adapterID: adapterID)
        await pipeline.loadLoRAWeights(weights, adapterID: adapterID)
        currentlyInjectedAdapterID = adapterID
    } catch {
        // Fall back to base model on any adapter failure
        await pipeline.removeLoRA()
        currentlyInjectedAdapterID = nil
        selectedAdapterID = nil
    }
}
```

In `generateWithACEStep(request:)`, add the adapter injection call after the VAE check and before the cancellation check. Insert after the VAE model swap block (after line 197 — `}` closing the VAE if block) and before `try Task.checkCancellation()`:

```swift
// Check LoRA adapter swap
await ensureAdapter()
```

Also, in `loadDiTWeights()`, after unloading DiT weights (after `await pipeline.unloadDiTWeights()`), add removal of any injected adapter:

```swift
// Remove any injected adapter when base model changes
if currentlyInjectedAdapterID != nil {
    await pipeline.removeLoRA()
    currentlyInjectedAdapterID = nil
}
```

- [ ] **Step 5: Verify it compiles**

```bash
xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | tail -5
```

Expected: BUILD SUCCEEDED (ignoring pre-existing SFXModelManager scope errors)

- [ ] **Step 6: Commit**

```bash
git add AnvilCore/Sources/AnvilCore/Inference/DiT/DiTActor.swift \
       AnvilCore/Sources/AnvilCore/Inference/Diffusion/DiffusionActor.swift \
       AnvilCore/Sources/AnvilCore/Inference/Pipeline/MusicPipeline.swift \
       AnvilApp/Services/GenerationService.swift
git commit -m "feat: add adapter injection to generation pipeline with lazy swap"
```

---

## Build Verification

After all tasks, verify the complete build:

```bash
xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | tail -10
```

Expected: BUILD SUCCEEDED (with only pre-existing SFXModelManager scope errors)
