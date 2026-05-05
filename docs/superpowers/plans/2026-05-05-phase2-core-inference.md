# Phase 2: Core Inference Stack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port ACE-Step's MLX Python inference layer to native MLX-Swift as testable actors in AnvilCore.

**Architecture:** Bottom-up build: shared types → weight converter → VAE → DiT → diffusion sampler → pipeline coordinator. Each component is an actor with async API, tested in isolation against the Python reference using deterministic fixtures.

**Tech Stack:** Swift 5.10, macOS 14+, MLX-Swift (ml-explore/mlx-swift), Swift Testing framework

---

## File Structure

```text
AnvilCore/
├── Package.swift                                    (modify: add mlx-swift dep)
├── Sources/AnvilCore/
│   ├── Inference/
│   │   ├── Types.swift                             (create: Precision, InferMethod, etc.)
│   │   ├── Pipeline/
│   │   │   └── MusicPipeline.swift                 (create)
│   │   ├── VAE/
│   │   │   ├── VAEComponents.swift                 (create: Snake1d, ResidualUnit, blocks)
│   │   │   ├── VAEModel.swift                      (create: OobleckEncoder/Decoder/VAE)
│   │   │   ├── VAETiledDecode.swift                (create: tiling logic)
│   │   │   └── VAEActor.swift                      (create: actor wrapper)
│   │   ├── DiT/
│   │   │   ├── DiTComponents.swift                 (create: RoPE, Attention, SwiGLU, etc.)
│   │   │   ├── DiTModel.swift                      (create: full decoder)
│   │   │   ├── CrossAttentionCache.swift           (create)
│   │   │   └── DiTActor.swift                      (create: actor wrapper)
│   │   ├── Diffusion/
│   │   │   ├── TimestepSchedule.swift              (create)
│   │   │   ├── DiffusionSampler.swift              (create: ODE/SDE + CFG/APG)
│   │   │   └── DiffusionActor.swift                (create: actor wrapper)
│   │   └── Weights/
│   │       ├── WeightConverter.swift               (create)
│   │       ├── TensorNameMap.swift                  (create: DiT key mapping)
│   │       └── VAENameMap.swift                     (create: VAE key mapping + weight_norm)
│   └── ...existing files...
├── Tests/AnvilCoreTests/
│   ├── Inference/
│   │   ├── TypesTests.swift                        (create)
│   │   ├── WeightConverterTests.swift              (create)
│   │   ├── VAEComponentsTests.swift                (create)
│   │   ├── VAEModelTests.swift                     (create)
│   │   ├── VAETiledDecodeTests.swift               (create)
│   │   ├── DiTComponentsTests.swift                (create)
│   │   ├── DiTModelTests.swift                     (create)
│   │   ├── CrossAttentionCacheTests.swift          (create)
│   │   ├── TimestepScheduleTests.swift             (create)
│   │   ├── DiffusionSamplerTests.swift             (create)
│   │   └── MusicPipelineTests.swift                (create)
│   └── ...existing tests...
```

---

### Task 1: Add MLX-Swift Dependency and Shared Types

**Files:**
- Modify: `AnvilCore/Package.swift`
- Create: `AnvilCore/Sources/AnvilCore/Inference/Types.swift`
- Create: `AnvilCore/Tests/AnvilCoreTests/Inference/TypesTests.swift`

- [ ] **Step 1: Write the failing test for shared types**

```swift
// AnvilCore/Tests/AnvilCoreTests/Inference/TypesTests.swift
import Testing
@testable import AnvilCore

@Suite("Inference Types")
struct TypesTests {
    @Test("Precision raw values")
    func precisionRawValues() {
        #expect(Precision.float32.rawValue == "float32")
        #expect(Precision.float16.rawValue == "float16")
    }

    @Test("InferMethod raw values")
    func inferMethodRawValues() {
        #expect(InferMethod.ode.rawValue == "ode")
        #expect(InferMethod.sde.rawValue == "sde")
    }

    @Test("ModelComponent cases exist")
    func modelComponentCases() {
        let vae = ModelComponent.vae
        let dit = ModelComponent.dit
        #expect(vae != dit)
    }

    @Test("DiffusionConfig defaults")
    func diffusionConfigDefaults() {
        let config = DiffusionConfig()
        #expect(config.seed == nil)
        #expect(config.method == .ode)
        #expect(config.shift == 3.0)
        #expect(config.steps == nil)
        #expect(config.guidanceScale == 1.0)
        #expect(config.cfgIntervalStart == 0.0)
        #expect(config.cfgIntervalEnd == 1.0)
        #expect(config.audioCoverStrength == 1.0)
        #expect(config.precision == .float32)
    }

    @Test("TimeCosts stores values")
    func timeCosts() {
        let costs = TimeCosts(diffusionTime: 1.5, perStepTime: 0.2, totalTime: 2.0)
        #expect(costs.diffusionTime == 1.5)
        #expect(costs.perStepTime == 0.2)
        #expect(costs.totalTime == 2.0)
    }

    @Test("DiffusionProgress stores values")
    func diffusionProgress() {
        let progress = DiffusionProgress(currentStep: 3, totalSteps: 8, elapsedTime: 0.6)
        #expect(progress.currentStep == 3)
        #expect(progress.totalSteps == 8)
    }

    @Test("PipelineState cases")
    func pipelineStateCases() {
        let state = PipelineState.generating(step: 2, totalSteps: 8)
        if case .generating(let step, let total) = state {
            #expect(step == 2)
            #expect(total == 8)
        } else {
            Issue.record("Expected .generating")
        }
    }

    @Test("ConversionProgress stores values")
    func conversionProgress() {
        let p = ConversionProgress(current: 5, total: 100, currentTensorName: "layers.0.weight")
        #expect(p.current == 5)
        #expect(p.total == 100)
        #expect(p.currentTensorName == "layers.0.weight")
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/tweaver/Developer/GitRepos/AnvilApp/AnvilCore && swift test --filter TypesTests 2>&1 | tail -20`
Expected: FAIL — types not defined

- [ ] **Step 3: Add mlx-swift dependency to Package.swift**

```swift
// AnvilCore/Package.swift
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
    ],
    targets: [
        .target(
            name: "AnvilCore",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ]
        ),
        .testTarget(name: "AnvilCoreTests", dependencies: ["AnvilCore"]),
    ]
)
```

- [ ] **Step 4: Create the shared types file**

```swift
// AnvilCore/Sources/AnvilCore/Inference/Types.swift
import MLX

public enum Precision: String, Sendable, CaseIterable {
    case float32
    case float16

    public var mlxDtype: DType {
        switch self {
        case .float32: return .float32
        case .float16: return .float16
        }
    }
}

public enum InferMethod: String, Sendable {
    case ode
    case sde
}

public enum ModelComponent: Sendable {
    case vae
    case dit
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

    public init(
        seed: Int? = nil,
        method: InferMethod = .ode,
        shift: Float = 3.0,
        steps: Int? = nil,
        guidanceScale: Float = 1.0,
        cfgIntervalStart: Float = 0.0,
        cfgIntervalEnd: Float = 1.0,
        audioCoverStrength: Float = 1.0,
        precision: Precision = .float32
    ) {
        self.seed = seed
        self.method = method
        self.shift = shift
        self.steps = steps
        self.guidanceScale = guidanceScale
        self.cfgIntervalStart = cfgIntervalStart
        self.cfgIntervalEnd = cfgIntervalEnd
        self.audioCoverStrength = audioCoverStrength
        self.precision = precision
    }
}

public struct TimeCosts: Sendable {
    public let diffusionTime: TimeInterval
    public let perStepTime: TimeInterval
    public let totalTime: TimeInterval

    public init(diffusionTime: TimeInterval, perStepTime: TimeInterval, totalTime: TimeInterval) {
        self.diffusionTime = diffusionTime
        self.perStepTime = perStepTime
        self.totalTime = totalTime
    }
}

public struct DiffusionProgress: Sendable {
    public let currentStep: Int
    public let totalSteps: Int
    public let elapsedTime: TimeInterval

    public init(currentStep: Int, totalSteps: Int, elapsedTime: TimeInterval) {
        self.currentStep = currentStep
        self.totalSteps = totalSteps
        self.elapsedTime = elapsedTime
    }
}

public struct ConversionProgress: Sendable {
    public let current: Int
    public let total: Int
    public let currentTensorName: String

    public init(current: Int, total: Int, currentTensorName: String) {
        self.current = current
        self.total = total
        self.currentTensorName = currentTensorName
    }
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

public struct AudioResult: Sendable {
    public let samples: MLXArray
    public let sampleRate: Int
    public let duration: TimeInterval
    public let timeCosts: TimeCosts

    public init(samples: MLXArray, sampleRate: Int = 44100, duration: TimeInterval, timeCosts: TimeCosts) {
        self.samples = samples
        self.sampleRate = sampleRate
        self.duration = duration
        self.timeCosts = timeCosts
    }
}

public struct DiffusionResult: Sendable {
    public let latents: MLXArray
    public let timeCosts: TimeCosts

    public init(latents: MLXArray, timeCosts: TimeCosts) {
        self.latents = latents
        self.timeCosts = timeCosts
    }
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/tweaver/Developer/GitRepos/AnvilApp/AnvilCore && swift test --filter TypesTests 2>&1 | tail -20`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
cd /Users/tweaver/Developer/GitRepos/AnvilApp
git add AnvilCore/Package.swift AnvilCore/Sources/AnvilCore/Inference/Types.swift AnvilCore/Tests/AnvilCoreTests/Inference/TypesTests.swift
git commit -m "feat: add mlx-swift dependency and shared inference types"
```

---

### Task 2: Timestep Schedule

**Files:**
- Create: `AnvilCore/Sources/AnvilCore/Inference/Diffusion/TimestepSchedule.swift`
- Create: `AnvilCore/Tests/AnvilCoreTests/Inference/TimestepScheduleTests.swift`

- [ ] **Step 1: Write the failing tests**

```swift
// AnvilCore/Tests/AnvilCoreTests/Inference/TimestepScheduleTests.swift
import Testing
@testable import AnvilCore

@Suite("TimestepSchedule")
struct TimestepScheduleTests {
    @Test("Lookup table for shift 3.0 returns 8 steps")
    func lookupShift3() {
        let schedule = TimestepSchedule.compute(shift: 3.0)
        #expect(schedule.count == 8)
        #expect(schedule[0] == 1.0)
        #expect(schedule[7] == 0.3)
    }

    @Test("Lookup table for shift 1.0")
    func lookupShift1() {
        let schedule = TimestepSchedule.compute(shift: 1.0)
        #expect(schedule.count == 8)
        #expect(schedule[0] == 1.0)
        #expect(schedule[7] == 0.125)
    }

    @Test("Lookup table for shift 2.0")
    func lookupShift2() {
        let schedule = TimestepSchedule.compute(shift: 2.0)
        #expect(schedule.count == 8)
        #expect(schedule[0] == 1.0)
    }

    @Test("Shift rounds to nearest valid")
    func shiftRounding() {
        let schedule = TimestepSchedule.compute(shift: 2.7)
        // 2.7 rounds to 3.0
        #expect(schedule.count == 8)
        #expect(schedule[0] == 1.0)
        #expect(schedule[7] == 0.3)
    }

    @Test("Linspace with inferSteps overrides lookup")
    func linspaceSteps() {
        let schedule = TimestepSchedule.compute(shift: 3.0, inferSteps: 4)
        #expect(schedule.count == 4)
        // First value is always 1.0 after shift transform
        #expect(schedule[0] == 1.0)
        // Values are descending
        for i in 1..<schedule.count {
            #expect(schedule[i] < schedule[i - 1])
        }
    }

    @Test("Custom timesteps override everything")
    func customTimesteps() {
        let custom: [Float] = [1.0, 0.75, 0.5, 0.25]
        let schedule = TimestepSchedule.compute(shift: 3.0, timesteps: custom)
        #expect(schedule.count == 4)
        // Custom timesteps get snapped to nearest valid
        #expect(schedule[0] == 1.0)
    }

    @Test("Custom timesteps trailing zeros removed")
    func trailingZeros() {
        let custom: [Float] = [1.0, 0.5, 0.0, 0.0]
        let schedule = TimestepSchedule.compute(shift: 3.0, timesteps: custom)
        #expect(schedule.count == 2)
    }

    @Test("Linspace with shift 1.0 produces evenly spaced")
    func linspaceNoShift() {
        let schedule = TimestepSchedule.compute(shift: 1.0, inferSteps: 4)
        #expect(schedule.count == 4)
        // With shift=1.0, no transformation applied: [1.0, 0.75, 0.5, 0.25]
        #expect(abs(schedule[0] - 1.0) < 1e-6)
        #expect(abs(schedule[1] - 0.75) < 1e-6)
        #expect(abs(schedule[2] - 0.5) < 1e-6)
        #expect(abs(schedule[3] - 0.25) < 1e-6)
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/tweaver/Developer/GitRepos/AnvilApp/AnvilCore && swift test --filter TimestepScheduleTests 2>&1 | tail -20`
Expected: FAIL — TimestepSchedule not defined

- [ ] **Step 3: Implement TimestepSchedule**

```swift
// AnvilCore/Sources/AnvilCore/Inference/Diffusion/TimestepSchedule.swift

public enum TimestepSchedule {
    static let validTimesteps: [Float] = [
        1.0, 0.9545454545454546, 0.9333333333333333, 0.9, 0.875,
        0.8571428571428571, 0.8333333333333334, 0.7692307692307693, 0.75,
        0.6666666666666666, 0.6428571428571429, 0.625, 0.5454545454545454,
        0.5, 0.4, 0.375, 0.3, 0.25, 0.2222222222222222, 0.125,
    ]

    static let validShifts: [Float] = [1.0, 2.0, 3.0]

    static let shiftTimesteps: [Float: [Float]] = [
        1.0: [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125],
        2.0: [1.0, 0.9333333333333333, 0.8571428571428571, 0.7692307692307693,
              0.6666666666666666, 0.5454545454545454, 0.4, 0.2222222222222222],
        3.0: [1.0, 0.9545454545454546, 0.9, 0.8333333333333334, 0.75,
              0.6428571428571429, 0.5, 0.3],
    ]

    public static func compute(
        shift: Float = 3.0,
        timesteps: [Float]? = nil,
        inferSteps: Int? = nil
    ) -> [Float] {
        if let timesteps = timesteps {
            var ts = timesteps
            while let last = ts.last, last == 0 {
                ts.removeLast()
            }
            guard !ts.isEmpty else {
                return computeFallback(shift: shift)
            }
            if ts.count > 20 {
                ts = Array(ts.prefix(20))
            }
            return ts.map { t in
                Self.validTimesteps.min(by: { abs($0 - t) < abs($1 - t) })!
            }
        }

        if let inferSteps = inferSteps, inferSteps > 0 {
            var raw = (0..<inferSteps).map { i in
                1.0 - Float(i) / Float(inferSteps)
            }
            if shift != 1.0 {
                raw = raw.map { t in
                    shift * t / (1.0 + (shift - 1.0) * t)
                }
            }
            return raw
        }

        return computeFallback(shift: shift)
    }

    private static func computeFallback(shift: Float) -> [Float] {
        let nearestShift = Self.validShifts.min(by: { abs($0 - shift) < abs($1 - shift) })!
        return Self.shiftTimesteps[nearestShift]!
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/tweaver/Developer/GitRepos/AnvilApp/AnvilCore && swift test --filter TimestepScheduleTests 2>&1 | tail -20`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/tweaver/Developer/GitRepos/AnvilApp
git add AnvilCore/Sources/AnvilCore/Inference/Diffusion/TimestepSchedule.swift AnvilCore/Tests/AnvilCoreTests/Inference/TimestepScheduleTests.swift
git commit -m "feat: implement timestep schedule with lookup tables and linspace"
```

---

### Task 3: Cross-Attention KV Cache

**Files:**
- Create: `AnvilCore/Sources/AnvilCore/Inference/DiT/CrossAttentionCache.swift`
- Create: `AnvilCore/Tests/AnvilCoreTests/Inference/CrossAttentionCacheTests.swift`

- [ ] **Step 1: Write the failing tests**

```swift
// AnvilCore/Tests/AnvilCoreTests/Inference/CrossAttentionCacheTests.swift
import Testing
import MLX
@testable import AnvilCore

@Suite("CrossAttentionCache")
struct CrossAttentionCacheTests {
    @Test("Empty cache reports not updated")
    func emptyCache() {
        let cache = CrossAttentionCache()
        #expect(!cache.isUpdated(layerIdx: 0))
        #expect(cache.get(layerIdx: 0) == nil)
    }

    @Test("Update and retrieve")
    func updateAndRetrieve() {
        let cache = CrossAttentionCache()
        let key = MLXArray.zeros([1, 4, 10, 128])
        let value = MLXArray.ones([1, 4, 10, 128])
        cache.update(key: key, value: value, layerIdx: 5)

        #expect(cache.isUpdated(layerIdx: 5))
        #expect(!cache.isUpdated(layerIdx: 0))

        let result = cache.get(layerIdx: 5)
        #expect(result != nil)
        #expect(result!.key.shape == [1, 4, 10, 128])
        #expect(result!.value.shape == [1, 4, 10, 128])
    }

    @Test("Reset clears all entries")
    func resetClears() {
        let cache = CrossAttentionCache()
        let key = MLXArray.zeros([1, 4, 10, 128])
        let value = MLXArray.ones([1, 4, 10, 128])
        cache.update(key: key, value: value, layerIdx: 0)
        cache.update(key: key, value: value, layerIdx: 1)

        cache.reset()

        #expect(!cache.isUpdated(layerIdx: 0))
        #expect(!cache.isUpdated(layerIdx: 1))
        #expect(cache.get(layerIdx: 0) == nil)
    }

    @Test("Multiple layers independent")
    func multipleLayers() {
        let cache = CrossAttentionCache()
        let k0 = MLXArray.zeros([1, 4, 10, 128])
        let v0 = MLXArray.zeros([1, 4, 10, 128])
        let k1 = MLXArray.ones([1, 4, 20, 128])
        let v1 = MLXArray.ones([1, 4, 20, 128])

        cache.update(key: k0, value: v0, layerIdx: 0)
        cache.update(key: k1, value: v1, layerIdx: 1)

        let r0 = cache.get(layerIdx: 0)!
        let r1 = cache.get(layerIdx: 1)!
        #expect(r0.key.shape[2] == 10)
        #expect(r1.key.shape[2] == 20)
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/tweaver/Developer/GitRepos/AnvilApp/AnvilCore && swift test --filter CrossAttentionCacheTests 2>&1 | tail -20`
Expected: FAIL — CrossAttentionCache not defined

- [ ] **Step 3: Implement CrossAttentionCache**

```swift
// AnvilCore/Sources/AnvilCore/Inference/DiT/CrossAttentionCache.swift
import MLX

public final class CrossAttentionCache: @unchecked Sendable {
    private var keys: [Int: MLXArray] = [:]
    private var values: [Int: MLXArray] = [:]

    public init() {}

    public func update(key: MLXArray, value: MLXArray, layerIdx: Int) {
        keys[layerIdx] = key
        values[layerIdx] = value
    }

    public func get(layerIdx: Int) -> (key: MLXArray, value: MLXArray)? {
        guard let k = keys[layerIdx], let v = values[layerIdx] else {
            return nil
        }
        return (k, v)
    }

    public func isUpdated(layerIdx: Int) -> Bool {
        keys[layerIdx] != nil
    }

    public func reset() {
        keys.removeAll()
        values.removeAll()
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/tweaver/Developer/GitRepos/AnvilApp/AnvilCore && swift test --filter CrossAttentionCacheTests 2>&1 | tail -20`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/tweaver/Developer/GitRepos/AnvilApp
git add AnvilCore/Sources/AnvilCore/Inference/DiT/CrossAttentionCache.swift AnvilCore/Tests/AnvilCoreTests/Inference/CrossAttentionCacheTests.swift
git commit -m "feat: implement cross-attention KV cache for DiT"
```

---

### Task 4: VAE Components (Snake1d, ResidualUnit, Blocks)

**Files:**
- Create: `AnvilCore/Sources/AnvilCore/Inference/VAE/VAEComponents.swift`
- Create: `AnvilCore/Tests/AnvilCoreTests/Inference/VAEComponentsTests.swift`

- [ ] **Step 1: Write the failing tests**

```swift
// AnvilCore/Tests/AnvilCoreTests/Inference/VAEComponentsTests.swift
import Testing
import MLX
import MLXNN
@testable import AnvilCore

@Suite("VAE Components")
struct VAEComponentsTests {
    @Test("Snake1d output shape matches input")
    func snake1dShape() {
        let snake = Snake1d(hiddenDim: 16)
        let input = MLXArray.zeros([1, 100, 16])
        let output = snake(input)
        #expect(output.shape == [1, 100, 16])
    }

    @Test("Snake1d is not identity (alpha/beta initialized to zero gives sin²(x)/1 offset)")
    func snake1dNotIdentity() {
        let snake = Snake1d(hiddenDim: 4)
        let input = MLXArray(converting: [[[1.0, 2.0, 3.0, 4.0]]], type: Float.self)
        let output = snake(input)
        eval(output)
        // With alpha=0, beta=0: exp(alpha)=1, exp(beta)=1
        // output = x + (1/1) * sin(1*x)^2 = x + sin(x)^2
        // sin(1)^2 ≈ 0.708, so output[0] ≈ 1.708
        let vals = output.asArray(Float.self)
        #expect(vals[0] > 1.5) // x=1 + sin(1)^2 ≈ 1.708
    }

    @Test("OobleckResidualUnit preserves shape")
    func residualUnitShape() {
        let unit = OobleckResidualUnit(dimension: 16, dilation: 1)
        let input = MLXArray.zeros([1, 64, 16])
        let output = unit(input)
        #expect(output.shape == [1, 64, 16])
    }

    @Test("OobleckResidualUnit with dilation 3")
    func residualUnitDilation3() {
        let unit = OobleckResidualUnit(dimension: 16, dilation: 3)
        let input = MLXArray.zeros([1, 64, 16])
        let output = unit(input)
        #expect(output.shape == [1, 64, 16])
    }

    @Test("OobleckEncoderBlock downsamples by stride")
    func encoderBlockDownsample() {
        let block = OobleckEncoderBlock(inputDim: 16, outputDim: 32, stride: 4)
        let input = MLXArray.zeros([1, 256, 16])
        let output = block(input)
        // Strided conv downsamples: 256/4 = 64
        #expect(output.shape[0] == 1)
        #expect(output.shape[2] == 32)
        // Length depends on padding but should be approximately 256/4
        #expect(output.shape[1] >= 63 && output.shape[1] <= 65)
    }

    @Test("OobleckDecoderBlock upsamples by stride")
    func decoderBlockUpsample() {
        let block = OobleckDecoderBlock(inputDim: 32, outputDim: 16, stride: 4)
        let input = MLXArray.zeros([1, 64, 32])
        let output = block(input)
        #expect(output.shape[0] == 1)
        #expect(output.shape[2] == 16)
        // ConvTranspose upsamples: 64*4 = 256 approximately
        #expect(output.shape[1] >= 255 && output.shape[1] <= 257)
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/tweaver/Developer/GitRepos/AnvilApp/AnvilCore && swift test --filter VAEComponentsTests 2>&1 | tail -20`
Expected: FAIL — types not defined

- [ ] **Step 3: Implement VAE components**

```swift
// AnvilCore/Sources/AnvilCore/Inference/VAE/VAEComponents.swift
import Foundation
import MLX
import MLXNN

public class Snake1d: Module, UnaryLayer {
    let alpha: MLXArray
    let beta: MLXArray
    let logscale: Bool

    public init(hiddenDim: Int, logscale: Bool = true) {
        self.alpha = MLXArray.zeros([hiddenDim])
        self.beta = MLXArray.zeros([hiddenDim])
        self.logscale = logscale
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let a = logscale ? exp(alpha) : alpha
        let b = logscale ? exp(beta) : beta
        let sinTerm = sin(a * x)
        let sinSquared = sinTerm * sinTerm
        return x + (1.0 / (b + 1e-9)) * sinSquared
    }
}

public class OobleckResidualUnit: Module, UnaryLayer {
    let snake1: Snake1d
    let conv1: Conv1d
    let snake2: Snake1d
    let conv2: Conv1d

    public init(dimension: Int = 16, dilation: Int = 1) {
        let pad = ((7 - 1) * dilation) / 2
        self.snake1 = Snake1d(hiddenDim: dimension)
        self.conv1 = Conv1d(
            inputChannels: dimension, outputChannels: dimension,
            kernelSize: 7, stride: 1, padding: pad, dilation: dilation
        )
        self.snake2 = Snake1d(hiddenDim: dimension)
        self.conv2 = Conv1d(
            inputChannels: dimension, outputChannels: dimension,
            kernelSize: 1
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var output = snake1(x)
        output = conv1(output)
        output = snake2(output)
        output = conv2(output)

        // Safety trim for padding mismatches
        let padding = (x.shape[1] - output.shape[1]) / 2
        var residual = x
        if padding > 0 {
            residual = x[0..., padding ..< (x.shape[1] - padding), 0...]
        }
        return residual + output
    }
}

public class OobleckEncoderBlock: Module, UnaryLayer {
    let resUnit1: OobleckResidualUnit
    let resUnit2: OobleckResidualUnit
    let resUnit3: OobleckResidualUnit
    let snake1: Snake1d
    let conv1: Conv1d

    public init(inputDim: Int, outputDim: Int, stride: Int = 1) {
        self.resUnit1 = OobleckResidualUnit(dimension: inputDim, dilation: 1)
        self.resUnit2 = OobleckResidualUnit(dimension: inputDim, dilation: 3)
        self.resUnit3 = OobleckResidualUnit(dimension: inputDim, dilation: 9)
        self.snake1 = Snake1d(hiddenDim: inputDim)
        let kernelSize = 2 * stride
        let pad = Int(ceil(Double(stride) / 2.0))
        self.conv1 = Conv1d(
            inputChannels: inputDim, outputChannels: outputDim,
            kernelSize: kernelSize, stride: stride, padding: pad
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = resUnit1(x)
        h = resUnit2(h)
        h = resUnit3(h)
        h = snake1(h)
        h = conv1(h)
        return h
    }
}

public class OobleckDecoderBlock: Module, UnaryLayer {
    let snake1: Snake1d
    let convT1: ConvTranspose1d
    let resUnit1: OobleckResidualUnit
    let resUnit2: OobleckResidualUnit
    let resUnit3: OobleckResidualUnit

    public init(inputDim: Int, outputDim: Int, stride: Int = 1) {
        self.snake1 = Snake1d(hiddenDim: inputDim)
        let kernelSize = 2 * stride
        let pad = Int(ceil(Double(stride) / 2.0))
        self.convT1 = ConvTranspose1d(
            inputChannels: inputDim, outputChannels: outputDim,
            kernelSize: kernelSize, stride: stride, padding: pad
        )
        self.resUnit1 = OobleckResidualUnit(dimension: outputDim, dilation: 1)
        self.resUnit2 = OobleckResidualUnit(dimension: outputDim, dilation: 3)
        self.resUnit3 = OobleckResidualUnit(dimension: outputDim, dilation: 9)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = snake1(x)
        h = convT1(h)
        h = resUnit1(h)
        h = resUnit2(h)
        h = resUnit3(h)
        return h
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/tweaver/Developer/GitRepos/AnvilApp/AnvilCore && swift test --filter VAEComponentsTests 2>&1 | tail -20`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/tweaver/Developer/GitRepos/AnvilApp
git add AnvilCore/Sources/AnvilCore/Inference/VAE/VAEComponents.swift AnvilCore/Tests/AnvilCoreTests/Inference/VAEComponentsTests.swift
git commit -m "feat: implement VAE components (Snake1d, ResidualUnit, Encoder/DecoderBlock)"
```

---

### Task 5: VAE Model and Tiled Decode

**Files:**
- Create: `AnvilCore/Sources/AnvilCore/Inference/VAE/VAEModel.swift`
- Create: `AnvilCore/Sources/AnvilCore/Inference/VAE/VAETiledDecode.swift`
- Create: `AnvilCore/Tests/AnvilCoreTests/Inference/VAEModelTests.swift`
- Create: `AnvilCore/Tests/AnvilCoreTests/Inference/VAETiledDecodeTests.swift`

- [ ] **Step 1: Write the failing tests for VAEModel**

```swift
// AnvilCore/Tests/AnvilCoreTests/Inference/VAEModelTests.swift
import Testing
import MLX
import MLXNN
@testable import AnvilCore

@Suite("VAE Model")
struct VAEModelTests {
    @Test("OobleckEncoder output shape")
    func encoderShape() {
        let encoder = OobleckEncoder(
            encoderHiddenSize: 128,
            audioChannels: 2,
            downsamplingRatios: [2, 4, 4, 8, 8],
            channelMultiples: [1, 2, 4, 8, 16]
        )
        // 2048 samples -> 2048/(2*4*4*8*8) = 1 latent frame
        let audio = MLXArray.zeros([1, 2048, 2])
        let latent = encoder(audio)
        #expect(latent.shape[0] == 1)
        #expect(latent.shape[2] == 128) // encoder_hidden_size
        #expect(latent.shape[1] == 1)   // 2048 / hop_length = 1
    }

    @Test("OobleckDecoder output shape")
    func decoderShape() {
        let decoder = OobleckDecoder(
            channels: 128,
            inputChannels: 64,
            audioChannels: 2,
            upsamplingRatios: [8, 8, 4, 4, 2],
            channelMultiples: [1, 2, 4, 8, 16]
        )
        let latent = MLXArray.zeros([1, 4, 64])
        let audio = decoder(latent)
        #expect(audio.shape[0] == 1)
        #expect(audio.shape[2] == 2)
        // 4 * 2048 = 8192 samples approximately
        #expect(audio.shape[1] >= 8100 && audio.shape[1] <= 8300)
    }

    @Test("AutoEncoderOobleck encode_and_sample shape")
    func vaeEncodeSampleShape() {
        let vae = AutoEncoderOobleck()
        let audio = MLXArray.zeros([1, 4096, 2])
        let z = vae.encodeSample(audio)
        #expect(z.shape[0] == 1)
        #expect(z.shape[2] == 64) // decoder_input_channels (latent dim)
        #expect(z.shape[1] == 2)  // 4096 / 2048 = 2
    }

    @Test("AutoEncoderOobleck decode shape")
    func vaeDecodeShape() {
        let vae = AutoEncoderOobleck()
        let latent = MLXArray.zeros([1, 2, 64])
        let audio = vae.decode(latent)
        #expect(audio.shape[0] == 1)
        #expect(audio.shape[2] == 2) // stereo
        // 2 latent frames * 2048 = 4096 samples approximately
        #expect(audio.shape[1] >= 4000 && audio.shape[1] <= 4200)
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/tweaver/Developer/GitRepos/AnvilApp/AnvilCore && swift test --filter VAEModelTests 2>&1 | tail -20`
Expected: FAIL — OobleckEncoder not defined

- [ ] **Step 3: Implement VAE Model**

```swift
// AnvilCore/Sources/AnvilCore/Inference/VAE/VAEModel.swift
import Foundation
import MLX
import MLXNN

public class OobleckEncoder: Module, UnaryLayer {
    let conv1: Conv1d
    let blocks: [OobleckEncoderBlock]
    let snake1: Snake1d
    let conv2: Conv1d

    public init(
        encoderHiddenSize: Int = 128,
        audioChannels: Int = 2,
        downsamplingRatios: [Int] = [2, 4, 4, 8, 8],
        channelMultiples: [Int] = [1, 2, 4, 8, 16]
    ) {
        let cm = [1] + channelMultiples
        self.conv1 = Conv1d(
            inputChannels: audioChannels, outputChannels: encoderHiddenSize,
            kernelSize: 7, padding: 3
        )
        var blocks: [OobleckEncoderBlock] = []
        for (i, stride) in downsamplingRatios.enumerated() {
            blocks.append(OobleckEncoderBlock(
                inputDim: encoderHiddenSize * cm[i],
                outputDim: encoderHiddenSize * cm[i + 1],
                stride: stride
            ))
        }
        self.blocks = blocks
        let dModel = encoderHiddenSize * cm.last!
        self.snake1 = Snake1d(hiddenDim: dModel)
        self.conv2 = Conv1d(
            inputChannels: dModel, outputChannels: encoderHiddenSize,
            kernelSize: 3, padding: 1
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv1(x)
        for block in blocks {
            h = block(h)
        }
        h = snake1(h)
        h = conv2(h)
        return h
    }
}

public class OobleckDecoder: Module, UnaryLayer {
    let conv1: Conv1d
    let blocks: [OobleckDecoderBlock]
    let snake1: Snake1d
    let conv2: Conv1d

    public init(
        channels: Int = 128,
        inputChannels: Int = 64,
        audioChannels: Int = 2,
        upsamplingRatios: [Int] = [8, 8, 4, 4, 2],
        channelMultiples: [Int] = [1, 2, 4, 8, 16]
    ) {
        let cm = [1] + channelMultiples
        self.conv1 = Conv1d(
            inputChannels: inputChannels, outputChannels: channels * cm.last!,
            kernelSize: 7, padding: 3
        )
        var blocks: [OobleckDecoderBlock] = []
        for (i, stride) in upsamplingRatios.enumerated() {
            let idx = upsamplingRatios.count - i
            blocks.append(OobleckDecoderBlock(
                inputDim: channels * cm[idx],
                outputDim: channels * cm[idx - 1],
                stride: stride
            ))
        }
        self.blocks = blocks
        self.snake1 = Snake1d(hiddenDim: channels)
        self.conv2 = Conv1d(
            inputChannels: channels, outputChannels: audioChannels,
            kernelSize: 7, padding: 3, bias: false
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv1(x)
        for block in blocks {
            h = block(h)
        }
        h = snake1(h)
        h = conv2(h)
        return h
    }
}

public class AutoEncoderOobleck: Module {
    let encoder: OobleckEncoder
    let decoder: OobleckDecoder
    public let encoderHiddenSize: Int
    public let decoderInputChannels: Int

    public init(
        encoderHiddenSize: Int = 128,
        downsamplingRatios: [Int] = [2, 4, 4, 8, 8],
        channelMultiples: [Int] = [1, 2, 4, 8, 16],
        decoderChannels: Int = 128,
        decoderInputChannels: Int = 64,
        audioChannels: Int = 2
    ) {
        self.encoderHiddenSize = encoderHiddenSize
        self.decoderInputChannels = decoderInputChannels
        self.encoder = OobleckEncoder(
            encoderHiddenSize: encoderHiddenSize,
            audioChannels: audioChannels,
            downsamplingRatios: downsamplingRatios,
            channelMultiples: channelMultiples
        )
        self.decoder = OobleckDecoder(
            channels: decoderChannels,
            inputChannels: decoderInputChannels,
            audioChannels: audioChannels,
            upsamplingRatios: downsamplingRatios.reversed(),
            channelMultiples: channelMultiples
        )
    }

    public func encodeSample(_ audioNLC: MLXArray) -> MLXArray {
        let h = encoder(audioNLC)
        let parts = split(h, parts: 2, axis: -1)
        let mean = parts[0]
        let scale = parts[1]
        let std = log(1.0 + exp(where(scale .> 20.0, scale, scale))) + 1e-4
        let noise = MLXRandom.normal(mean.shape)
        return mean + std * noise
    }

    public func encodeMean(_ audioNLC: MLXArray) -> MLXArray {
        let h = encoder(audioNLC)
        let parts = split(h, parts: 2, axis: -1)
        return parts[0]
    }

    public func decode(_ latentsNLC: MLXArray) -> MLXArray {
        decoder(latentsNLC)
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/tweaver/Developer/GitRepos/AnvilApp/AnvilCore && swift test --filter VAEModelTests 2>&1 | tail -20`
Expected: All tests PASS

- [ ] **Step 5: Write failing tests for tiled decode**

```swift
// AnvilCore/Tests/AnvilCoreTests/Inference/VAETiledDecodeTests.swift
import Testing
import MLX
@testable import AnvilCore

@Suite("VAE Tiled Decode")
struct VAETiledDecodeTests {
    @Test("Short sequence does not tile")
    func shortSequenceNoTile() {
        let vae = AutoEncoderOobleck()
        let latent = MLXArray.zeros([1, 100, 64]) // 100 < 2048, no tiling
        let audio = VAETiledDecode.decode(latent, using: vae)
        #expect(audio.shape[0] == 1)
        #expect(audio.shape[2] == 2)
    }

    @Test("Long sequence tiles correctly")
    func longSequenceTiles() {
        let vae = AutoEncoderOobleck()
        // 3000 > 2048, should trigger tiling
        let latent = MLXArray.zeros([1, 3000, 64])
        let audio = VAETiledDecode.decode(latent, using: vae)
        #expect(audio.shape[0] == 1)
        #expect(audio.shape[2] == 2)
        // 3000 latent frames * 2048 samples/frame ≈ 6144000 audio samples
        #expect(audio.shape[1] > 5_000_000)
    }

    @Test("Tile count calculation")
    func tileCount() {
        // chunk=2048, overlap=64 each side, stride=2048-128=1920
        // For 3000 frames: ceil(3000/1920) = 2 tiles
        let tiles = VAETiledDecode.computeTileCount(latentFrames: 3000)
        #expect(tiles == 2)
    }

    @Test("Exactly at threshold does not tile")
    func atThresholdNoTile() {
        let tiles = VAETiledDecode.computeTileCount(latentFrames: 2048)
        #expect(tiles == 1)
    }
}
```

- [ ] **Step 6: Implement tiled decode**

```swift
// AnvilCore/Sources/AnvilCore/Inference/VAE/VAETiledDecode.swift
import Foundation
import MLX

public enum VAETiledDecode {
    public static let chunkSize = 2048
    public static let overlap = 64
    public static let stride = chunkSize - 2 * overlap // 1920

    public static func computeTileCount(latentFrames: Int) -> Int {
        if latentFrames <= chunkSize { return 1 }
        return Int(ceil(Double(latentFrames) / Double(stride)))
    }

    public static func decode(_ latents: MLXArray, using vae: AutoEncoderOobleck) -> MLXArray {
        let latentFrames = latents.shape[1]
        if latentFrames <= chunkSize {
            return vae.decode(latents)
        }

        let numSteps = computeTileCount(latentFrames: latentFrames)
        var decodedParts: [MLXArray] = []
        var upsampleFactor: Double? = nil

        for idx in 0..<numSteps {
            let coreStart = idx * stride
            let coreEnd = min(coreStart + stride, latentFrames)
            let winStart = max(0, coreStart - overlap)
            let winEnd = min(latentFrames, coreEnd + overlap)

            let chunk = latents[0..., winStart..<winEnd, 0...]
            let audioChunk = vae.decode(chunk)
            eval(audioChunk)

            if upsampleFactor == nil {
                upsampleFactor = Double(audioChunk.shape[1]) / Double(chunk.shape[1])
            }
            let factor = upsampleFactor!

            let trimStart = Int((Double(coreStart - winStart) * factor).rounded())
            let trimEnd = Int((Double(winEnd - coreEnd) * factor).rounded())
            let audioLen = audioChunk.shape[1]
            let endIdx = trimEnd > 0 ? audioLen - trimEnd : audioLen
            decodedParts.append(audioChunk[0..., trimStart..<endIdx, 0...])
        }

        return concatenated(decodedParts, axis: 1)
    }
}
```

- [ ] **Step 7: Run all VAE tests**

Run: `cd /Users/tweaver/Developer/GitRepos/AnvilApp/AnvilCore && swift test --filter "VAE" 2>&1 | tail -20`
Expected: All tests PASS

- [ ] **Step 8: Commit**

```bash
cd /Users/tweaver/Developer/GitRepos/AnvilApp
git add AnvilCore/Sources/AnvilCore/Inference/VAE/VAEModel.swift AnvilCore/Sources/AnvilCore/Inference/VAE/VAETiledDecode.swift AnvilCore/Tests/AnvilCoreTests/Inference/VAEModelTests.swift AnvilCore/Tests/AnvilCoreTests/Inference/VAETiledDecodeTests.swift
git commit -m "feat: implement VAE model (OobleckEncoder/Decoder) with tiled decode"
```

---

### Task 6: VAE Actor

**Files:**
- Create: `AnvilCore/Sources/AnvilCore/Inference/VAE/VAEActor.swift`
- Create: `AnvilCore/Tests/AnvilCoreTests/Inference/VAEActorTests.swift`

- [ ] **Step 1: Write the failing tests**

```swift
// AnvilCore/Tests/AnvilCoreTests/Inference/VAEActorTests.swift
import Testing
import MLX
@testable import AnvilCore

@Suite("VAE Actor")
struct VAEActorTests {
    @Test("Decode produces audio from latents")
    func decodeProducesAudio() async {
        let actor = VAEActor()
        let latents = MLXArray.zeros([1, 4, 64])
        let audio = await actor.decode(latents: latents, precision: .float32)
        #expect(audio.shape[0] == 1)
        #expect(audio.shape[2] == 2) // stereo
    }

    @Test("Encode produces latents from audio")
    func encodeProducesLatents() async {
        let actor = VAEActor()
        let audio = MLXArray.zeros([1, 4096, 2])
        let latents = await actor.encodeSample(audio: audio, precision: .float32)
        #expect(latents.shape[0] == 1)
        #expect(latents.shape[2] == 64)
    }

    @Test("Decode with fp16 precision")
    func decodeFp16() async {
        let actor = VAEActor()
        let latents = MLXArray.zeros([1, 4, 64])
        let audio = await actor.decode(latents: latents, precision: .float16)
        #expect(audio.shape[0] == 1)
        #expect(audio.shape[2] == 2)
    }

    @Test("Long latents trigger tiled decode")
    func longLatentsTiled() async {
        let actor = VAEActor()
        let latents = MLXArray.zeros([1, 3000, 64])
        let audio = await actor.decode(latents: latents, precision: .float32)
        #expect(audio.shape[0] == 1)
        #expect(audio.shape[2] == 2)
        #expect(audio.shape[1] > 5_000_000)
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/tweaver/Developer/GitRepos/AnvilApp/AnvilCore && swift test --filter VAEActorTests 2>&1 | tail -20`
Expected: FAIL — VAEActor not defined

- [ ] **Step 3: Implement VAEActor**

```swift
// AnvilCore/Sources/AnvilCore/Inference/VAE/VAEActor.swift
import Foundation
import MLX

public actor VAEActor {
    private var vae: AutoEncoderOobleck

    public init(vae: AutoEncoderOobleck = AutoEncoderOobleck()) {
        self.vae = vae
    }

    public func decode(latents: MLXArray, precision: Precision) -> MLXArray {
        let input = precision == .float16 ? latents.asType(.float16) : latents
        let audio = VAETiledDecode.decode(input, using: vae)
        return precision == .float16 ? audio.asType(.float32) : audio
    }

    public func encodeSample(audio: MLXArray, precision: Precision) -> MLXArray {
        let input = precision == .float16 ? audio.asType(.float16) : audio
        let latents = vae.encodeSample(input)
        return precision == .float16 ? latents.asType(.float32) : latents
    }

    public func loadWeights(_ weights: [(String, MLXArray)]) {
        vae.update(parameters: ModuleParameters.unflattened(weights))
        eval(vae.parameters())
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/tweaver/Developer/GitRepos/AnvilApp/AnvilCore && swift test --filter VAEActorTests 2>&1 | tail -20`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/tweaver/Developer/GitRepos/AnvilApp
git add AnvilCore/Sources/AnvilCore/Inference/VAE/VAEActor.swift AnvilCore/Tests/AnvilCoreTests/Inference/VAEActorTests.swift
git commit -m "feat: implement VAEActor with precision switching and tiled decode"
```

---

### Task 7: DiT Components (RoPE, Attention, SwiGLU, TimestepEmbedding, DiTLayer)

**Files:**
- Create: `AnvilCore/Sources/AnvilCore/Inference/DiT/DiTComponents.swift`
- Create: `AnvilCore/Tests/AnvilCoreTests/Inference/DiTComponentsTests.swift`

- [ ] **Step 1: Write the failing tests**

```swift
// AnvilCore/Tests/AnvilCoreTests/Inference/DiTComponentsTests.swift
import Testing
import MLX
import MLXNN
import MLXFast
@testable import AnvilCore

@Suite("DiT Components")
struct DiTComponentsTests {
    @Test("RotaryEmbedding produces correct shapes")
    func rotaryEmbeddingShape() {
        let rope = RotaryEmbedding(headDim: 128, maxLen: 32768, base: 1_000_000.0)
        let (cos, sin) = rope.forward(seqLen: 100)
        #expect(cos.shape == [1, 1, 100, 128])
        #expect(sin.shape == [1, 1, 100, 128])
    }

    @Test("SwiGLUMLP output shape matches input")
    func swiGLUShape() {
        let mlp = SwiGLUMLP(hiddenSize: 2048, intermediateSize: 6144)
        let input = MLXArray.zeros([1, 10, 2048])
        let output = mlp(input)
        #expect(output.shape == [1, 10, 2048])
    }

    @Test("DiTAttention self-attention shape")
    func selfAttentionShape() {
        let attn = DiTAttention(
            hiddenSize: 2048,
            numAttentionHeads: 16,
            numKeyValueHeads: 8,
            headDim: 128,
            rmsNormEps: 1e-6,
            attentionBias: false,
            layerIdx: 0,
            isCrossAttention: false,
            slidingWindow: nil
        )
        let input = MLXArray.zeros([1, 10, 2048])
        let rope = RotaryEmbedding(headDim: 128, maxLen: 32768, base: 1_000_000.0)
        let (cos, sin) = rope.forward(seqLen: 10)
        let output = attn(
            hiddenStates: input,
            positionCosSin: (cos, sin),
            attentionMask: nil,
            encoderHiddenStates: nil,
            cache: nil,
            useCache: false
        )
        #expect(output.shape == [1, 10, 2048])
    }

    @Test("DiTAttention cross-attention shape")
    func crossAttentionShape() {
        let attn = DiTAttention(
            hiddenSize: 2048,
            numAttentionHeads: 16,
            numKeyValueHeads: 8,
            headDim: 128,
            rmsNormEps: 1e-6,
            attentionBias: false,
            layerIdx: 0,
            isCrossAttention: true,
            slidingWindow: nil
        )
        let input = MLXArray.zeros([1, 10, 2048])
        let encoderStates = MLXArray.zeros([1, 50, 2048])
        let output = attn(
            hiddenStates: input,
            positionCosSin: nil,
            attentionMask: nil,
            encoderHiddenStates: encoderStates,
            cache: nil,
            useCache: false
        )
        #expect(output.shape == [1, 10, 2048])
    }

    @Test("DiTAttention cross-attention uses cache on second call")
    func crossAttentionCache() {
        let attn = DiTAttention(
            hiddenSize: 2048,
            numAttentionHeads: 16,
            numKeyValueHeads: 8,
            headDim: 128,
            rmsNormEps: 1e-6,
            attentionBias: false,
            layerIdx: 3,
            isCrossAttention: true,
            slidingWindow: nil
        )
        let input = MLXArray.zeros([1, 10, 2048])
        let encoderStates = MLXArray.zeros([1, 50, 2048])
        let cache = CrossAttentionCache()

        // First call populates cache
        _ = attn(
            hiddenStates: input, positionCosSin: nil, attentionMask: nil,
            encoderHiddenStates: encoderStates, cache: cache, useCache: true
        )
        #expect(cache.isUpdated(layerIdx: 3))

        // Second call uses cache (no encoder states needed)
        let output2 = attn(
            hiddenStates: input, positionCosSin: nil, attentionMask: nil,
            encoderHiddenStates: encoderStates, cache: cache, useCache: true
        )
        #expect(output2.shape == [1, 10, 2048])
    }

    @Test("TimestepEmbedding output shapes")
    func timestepEmbeddingShape() {
        let embed = TimestepEmbedding(inChannels: 256, timeEmbedDim: 2048)
        let t = MLXArray(converting: [0.5, 0.3], type: Float.self)
        let (temb, proj) = embed(t)
        #expect(temb.shape == [2, 2048])
        #expect(proj.shape == [2, 6, 2048])
    }

    @Test("DiTLayer output shape")
    func ditLayerShape() {
        let layer = DiTLayer(
            hiddenSize: 2048,
            intermediateSize: 6144,
            numAttentionHeads: 16,
            numKeyValueHeads: 8,
            headDim: 128,
            rmsNormEps: 1e-6,
            attentionBias: false,
            layerIdx: 0,
            layerType: "full_attention",
            slidingWindow: 128
        )
        let input = MLXArray.zeros([1, 10, 2048])
        let temb = MLXArray.zeros([1, 6, 2048])
        let encoder = MLXArray.zeros([1, 50, 2048])
        let rope = RotaryEmbedding(headDim: 128, maxLen: 32768, base: 1_000_000.0)
        let (cos, sin) = rope.forward(seqLen: 10)
        let output = layer(
            hiddenStates: input,
            positionCosSin: (cos, sin),
            temb: temb,
            selfAttnMask: nil,
            encoderHiddenStates: encoder,
            encoderAttentionMask: nil,
            cache: nil,
            useCache: false
        )
        #expect(output.shape == [1, 10, 2048])
    }

    @Test("createSlidingWindowMask shape and values")
    func slidingWindowMask() {
        let mask = createSlidingWindowMask(seqLen: 8, windowSize: 2)
        #expect(mask.shape == [1, 1, 8, 8])
        eval(mask)
        // Position (0,0) should be 0 (within window)
        // Position (0,4) should be -1e9 (outside window, |0-4|=4 > 2)
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/tweaver/Developer/GitRepos/AnvilApp/AnvilCore && swift test --filter DiTComponentsTests 2>&1 | tail -20`
Expected: FAIL — types not defined

- [ ] **Step 3: Implement DiT components**

```swift
// AnvilCore/Sources/AnvilCore/Inference/DiT/DiTComponents.swift
import Foundation
import MLX
import MLXNN
import MLXFast

// MARK: - Rotary Position Embedding

public class RotaryEmbedding: Module {
    let headDim: Int
    let maxLen: Int
    let cosTable: MLXArray
    let sinTable: MLXArray

    public init(headDim: Int, maxLen: Int = 32768, base: Float = 1_000_000.0) {
        self.headDim = headDim
        self.maxLen = maxLen

        let invFreq = 1.0 / pow(
            MLXArray(base),
            MLXArray(stride(from: 0, to: headDim, by: 2).map { Float($0) / Float(headDim) })
        )
        let positions = MLXArray(Array(0..<maxLen).map { Float($0) })
        // [maxLen, headDim/2]
        let freqs = positions.expandedDimensions(axis: 1) * invFreq.expandedDimensions(axis: 0)
        // [maxLen, headDim]
        let freqsFull = concatenated([freqs, freqs], axis: 1)
        self.cosTable = cos(freqsFull)
        self.sinTable = sin(freqsFull)
    }

    public func forward(seqLen: Int) -> (cos: MLXArray, sin: MLXArray) {
        let c = cosTable[0..<seqLen].reshaped([1, 1, seqLen, headDim])
        let s = sinTable[0..<seqLen].reshaped([1, 1, seqLen, headDim])
        return (c, s)
    }
}

// MARK: - Rotary helpers

func rotateHalf(_ x: MLXArray) -> MLXArray {
    let half = x.shape.last! / 2
    let x1 = x[.ellipsis, 0..<half]
    let x2 = x[.ellipsis, half...]
    return concatenated([-x2, x1], axis: -1)
}

func applyRotaryPosEmb(
    q: MLXArray, k: MLXArray, cos: MLXArray, sin: MLXArray
) -> (MLXArray, MLXArray) {
    let qEmbed = (q * cos) + (rotateHalf(q) * sin)
    let kEmbed = (k * cos) + (rotateHalf(k) * sin)
    return (qEmbed, kEmbed)
}

// MARK: - Sliding Window Mask

public func createSlidingWindowMask(seqLen: Int, windowSize: Int, dtype: DType = .float32) -> MLXArray {
    let indices = MLXArray(Array(0..<seqLen).map { Float($0) })
    let diff = abs(indices.expandedDimensions(axis: 1) - indices.expandedDimensions(axis: 0))
    let zeros = MLXArray.zeros([seqLen, seqLen])
    let negInf = MLXArray.full([seqLen, seqLen], values: Float(-1e9))
    let mask = which(diff .< Float(windowSize + 1), zeros, negInf)
    return mask.reshaped([1, 1, seqLen, seqLen])
}

// MARK: - SwiGLU MLP

public class SwiGLUMLP: Module, UnaryLayer {
    let gateProj: Linear
    let upProj: Linear
    let downProj: Linear

    public init(hiddenSize: Int, intermediateSize: Int) {
        self.gateProj = Linear(hiddenSize, intermediateSize, bias: false)
        self.upProj = Linear(hiddenSize, intermediateSize, bias: false)
        self.downProj = Linear(intermediateSize, hiddenSize, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - Attention

public class DiTAttention: Module {
    let hiddenSize: Int
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let nRep: Int
    let scale: Float
    let layerIdx: Int
    let isCrossAttention: Bool
    let slidingWindow: Int?

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
        rmsNormEps: Float,
        attentionBias: Bool,
        layerIdx: Int,
        isCrossAttention: Bool = false,
        slidingWindow: Int? = nil
    ) {
        self.hiddenSize = hiddenSize
        self.numHeads = numAttentionHeads
        self.numKVHeads = numKeyValueHeads
        self.headDim = headDim
        self.nRep = numAttentionHeads / numKeyValueHeads
        self.scale = pow(Float(headDim), -0.5)
        self.layerIdx = layerIdx
        self.isCrossAttention = isCrossAttention
        self.slidingWindow = slidingWindow

        self.qProj = Linear(hiddenSize, numAttentionHeads * headDim, bias: attentionBias)
        self.kProj = Linear(hiddenSize, numKeyValueHeads * headDim, bias: attentionBias)
        self.vProj = Linear(hiddenSize, numKeyValueHeads * headDim, bias: attentionBias)
        self.oProj = Linear(numAttentionHeads * headDim, hiddenSize, bias: attentionBias)
        self.qNorm = RMSNorm(dimensions: headDim, eps: rmsNormEps)
        self.kNorm = RMSNorm(dimensions: headDim, eps: rmsNormEps)
    }

    private func repeatKV(_ x: MLXArray, nRep: Int) -> MLXArray {
        if nRep == 1 { return x }
        // x: [B, nKV, L, D] -> [B, nKV*nRep, L, D]
        let (b, nKV, l, d) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        let expanded = x.expandedDimensions(axis: 2) // [B, nKV, 1, L, D]
        let broadcasted = broadcast(expanded, to: [b, nKV, nRep, l, d])
        return broadcasted.reshaped([b, nKV * nRep, l, d])
    }

    public func callAsFunction(
        hiddenStates: MLXArray,
        positionCosSin: (cos: MLXArray, sin: MLXArray)?,
        attentionMask: MLXArray?,
        encoderHiddenStates: MLXArray?,
        cache: CrossAttentionCache?,
        useCache: Bool
    ) -> MLXArray {
        let (b, l, _) = (hiddenStates.shape[0], hiddenStates.shape[1], hiddenStates.shape[2])

        // Project queries
        var q = qProj(hiddenStates)
        q = qNorm(q.reshaped([b, l, numHeads, headDim]))
        q = q.transposed(0, 2, 1, 3) // [B, nHeads, L, D]

        var k: MLXArray
        var v: MLXArray

        if isCrossAttention, let enc = encoderHiddenStates {
            if let cache = cache, cache.isUpdated(layerIdx: layerIdx) {
                let cached = cache.get(layerIdx: layerIdx)!
                k = cached.key
                v = cached.value
            } else {
                let encL = enc.shape[1]
                k = kProj(enc)
                k = kNorm(k.reshaped([b, encL, numKVHeads, headDim]))
                k = k.transposed(0, 2, 1, 3)
                v = vProj(enc).reshaped([b, encL, numKVHeads, headDim]).transposed(0, 2, 1, 3)
                if let cache = cache, useCache {
                    cache.update(key: k, value: v, layerIdx: layerIdx)
                }
            }
        } else {
            k = kProj(hiddenStates)
            k = kNorm(k.reshaped([b, l, numKVHeads, headDim]))
            k = k.transposed(0, 2, 1, 3)
            v = vProj(hiddenStates).reshaped([b, l, numKVHeads, headDim]).transposed(0, 2, 1, 3)

            if let (cos, sin) = positionCosSin {
                (q, k) = applyRotaryPosEmb(q: q, k: k, cos: cos, sin: sin)
            }
        }

        k = repeatKV(k, nRep: nRep)
        v = repeatKV(v, nRep: nRep)

        let attnOut = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: attentionMask
        )

        let merged = attnOut.transposed(0, 2, 1, 3).reshaped([b, l, -1])
        return oProj(merged)
    }
}

// MARK: - Timestep Embedding

public class TimestepEmbedding: Module {
    let inChannels: Int
    let scale: Float
    let linear1: Linear
    let linear2: Linear
    let timeProj: Linear

    public init(inChannels: Int = 256, timeEmbedDim: Int = 2048, scale: Float = 1000.0) {
        self.inChannels = inChannels
        self.scale = scale
        self.linear1 = Linear(inChannels, timeEmbedDim, bias: true)
        self.linear2 = Linear(timeEmbedDim, timeEmbedDim, bias: true)
        self.timeProj = Linear(timeEmbedDim, timeEmbedDim * 6, bias: true)
    }

    private func sinusoidalEmbedding(_ t: MLXArray, dim: Int, maxPeriod: Int = 10000) -> MLXArray {
        let scaled = t * scale
        let half = dim / 2
        let freqs = exp(
            -log(Float(maxPeriod)) * MLXArray(Array(0..<half).map { Float($0) / Float(half) })
        )
        let args = scaled.expandedDimensions(axis: 1) * freqs.expandedDimensions(axis: 0)
        var embedding = concatenated([cos(args), sin(args)], axis: -1)
        if dim % 2 != 0 {
            embedding = concatenated([embedding, MLXArray.zeros([embedding.shape[0], 1])], axis: -1)
        }
        return embedding
    }

    public func callAsFunction(_ t: MLXArray) -> (temb: MLXArray, timestepProj: MLXArray) {
        let tFreq = sinusoidalEmbedding(t, dim: inChannels)
        var temb = linear1(tFreq)
        temb = silu(temb)
        temb = linear2(temb)
        let proj = timeProj(silu(temb))
        let timestepProj = proj.reshaped([proj.shape[0], 6, -1])
        return (temb, timestepProj)
    }
}

// MARK: - DiT Layer

public class DiTLayer: Module {
    let layerType: String
    let selfAttnNorm: RMSNorm
    let selfAttn: DiTAttention
    let crossAttnNorm: RMSNorm
    let crossAttn: DiTAttention
    let mlpNorm: RMSNorm
    let mlp: SwiGLUMLP
    let scaleShiftTable: MLXArray

    public init(
        hiddenSize: Int,
        intermediateSize: Int,
        numAttentionHeads: Int,
        numKeyValueHeads: Int,
        headDim: Int,
        rmsNormEps: Float,
        attentionBias: Bool,
        layerIdx: Int,
        layerType: String,
        slidingWindow: Int = 128
    ) {
        self.layerType = layerType
        let sw: Int? = layerType == "sliding_attention" ? slidingWindow : nil

        self.selfAttnNorm = RMSNorm(dimensions: hiddenSize, eps: rmsNormEps)
        self.selfAttn = DiTAttention(
            hiddenSize: hiddenSize, numAttentionHeads: numAttentionHeads,
            numKeyValueHeads: numKeyValueHeads, headDim: headDim,
            rmsNormEps: rmsNormEps, attentionBias: attentionBias,
            layerIdx: layerIdx, isCrossAttention: false, slidingWindow: sw
        )
        self.crossAttnNorm = RMSNorm(dimensions: hiddenSize, eps: rmsNormEps)
        self.crossAttn = DiTAttention(
            hiddenSize: hiddenSize, numAttentionHeads: numAttentionHeads,
            numKeyValueHeads: numKeyValueHeads, headDim: headDim,
            rmsNormEps: rmsNormEps, attentionBias: attentionBias,
            layerIdx: layerIdx, isCrossAttention: true, slidingWindow: nil
        )
        self.mlpNorm = RMSNorm(dimensions: hiddenSize, eps: rmsNormEps)
        self.mlp = SwiGLUMLP(hiddenSize: hiddenSize, intermediateSize: intermediateSize)
        self.scaleShiftTable = MLXArray.zeros([1, 6, hiddenSize])
    }

    public func callAsFunction(
        hiddenStates: MLXArray,
        positionCosSin: (cos: MLXArray, sin: MLXArray),
        temb: MLXArray,
        selfAttnMask: MLXArray?,
        encoderHiddenStates: MLXArray?,
        encoderAttentionMask: MLXArray?,
        cache: CrossAttentionCache?,
        useCache: Bool
    ) -> MLXArray {
        // AdaLN modulation
        let modulation = scaleShiftTable + temb // [B, 6, D]
        let parts = split(modulation, parts: 6, axis: 1) // each [B, 1, D]
        let (shiftMSA, scaleMSA, gateMSA) = (parts[0], parts[1], parts[2])
        let (cShiftMSA, cScaleMSA, cGateMSA) = (parts[3], parts[4], parts[5])

        // Self-attention with AdaLN
        var h = hiddenStates
        var normed = selfAttnNorm(h)
        normed = normed * (1.0 + scaleMSA) + shiftMSA
        let attnOut = selfAttn(
            hiddenStates: normed, positionCosSin: positionCosSin,
            attentionMask: selfAttnMask, encoderHiddenStates: nil,
            cache: nil, useCache: false
        )
        h = h + attnOut * gateMSA

        // Cross-attention
        normed = crossAttnNorm(h)
        let crossOut = crossAttn(
            hiddenStates: normed, positionCosSin: nil,
            attentionMask: encoderAttentionMask,
            encoderHiddenStates: encoderHiddenStates,
            cache: cache, useCache: useCache
        )
        h = h + crossOut

        // MLP with AdaLN
        normed = mlpNorm(h)
        normed = normed * (1.0 + cScaleMSA) + cShiftMSA
        let ffOut = mlp(normed)
        h = h + ffOut * cGateMSA

        return h
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/tweaver/Developer/GitRepos/AnvilApp/AnvilCore && swift test --filter DiTComponentsTests 2>&1 | tail -20`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/tweaver/Developer/GitRepos/AnvilApp
git add AnvilCore/Sources/AnvilCore/Inference/DiT/DiTComponents.swift AnvilCore/Tests/AnvilCoreTests/Inference/DiTComponentsTests.swift
git commit -m "feat: implement DiT components (RoPE, Attention, SwiGLU, TimestepEmbed, DiTLayer)"
```

---

### Task 8: DiT Model and Actor

**Files:**
- Create: `AnvilCore/Sources/AnvilCore/Inference/DiT/DiTModel.swift`
- Create: `AnvilCore/Sources/AnvilCore/Inference/DiT/DiTActor.swift`
- Create: `AnvilCore/Tests/AnvilCoreTests/Inference/DiTModelTests.swift`

- [ ] **Step 1: Write the failing tests**

```swift
// AnvilCore/Tests/AnvilCoreTests/Inference/DiTModelTests.swift
import Testing
import MLX
@testable import AnvilCore

@Suite("DiT Model")
struct DiTModelTests {
    @Test("DiTDecoder forward pass shape")
    func forwardPassShape() {
        let decoder = DiTDecoder(
            hiddenSize: 256,      // small for testing
            intermediateSize: 512,
            numHiddenLayers: 2,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            headDim: 64,
            rmsNormEps: 1e-6,
            attentionBias: false,
            inChannels: 192,
            audioAcousticHiddenDim: 64,
            patchSize: 2,
            slidingWindow: 128,
            layerTypes: ["sliding_attention", "full_attention"],
            ropeTheta: 1_000_000.0,
            maxPositionEmbeddings: 32768
        )
        let hiddenStates = MLXArray.zeros([1, 20, 64])
        let timestep = MLXArray(converting: [0.5], type: Float.self)
        let timestepR = MLXArray(converting: [0.5], type: Float.self)
        let encoderStates = MLXArray.zeros([1, 50, 256])
        let contextLatents = MLXArray.zeros([1, 20, 128])

        let (output, _) = decoder(
            hiddenStates: hiddenStates,
            timestep: timestep,
            timestepR: timestepR,
            encoderHiddenStates: encoderStates,
            contextLatents: contextLatents,
            cache: nil,
            useCache: false
        )
        #expect(output.shape == [1, 20, 64])
    }

    @Test("DiTDecoder with cache returns cache")
    func forwardWithCache() {
        let decoder = DiTDecoder(
            hiddenSize: 256,
            intermediateSize: 512,
            numHiddenLayers: 2,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            headDim: 64,
            rmsNormEps: 1e-6,
            attentionBias: false,
            inChannels: 192,
            audioAcousticHiddenDim: 64,
            patchSize: 2,
            slidingWindow: 128,
            layerTypes: ["sliding_attention", "full_attention"],
            ropeTheta: 1_000_000.0,
            maxPositionEmbeddings: 32768
        )
        let hiddenStates = MLXArray.zeros([1, 20, 64])
        let timestep = MLXArray(converting: [0.5], type: Float.self)
        let timestepR = MLXArray(converting: [0.5], type: Float.self)
        let encoderStates = MLXArray.zeros([1, 50, 256])
        let contextLatents = MLXArray.zeros([1, 20, 128])
        let cache = CrossAttentionCache()

        let (_, returnedCache) = decoder(
            hiddenStates: hiddenStates,
            timestep: timestep,
            timestepR: timestepR,
            encoderHiddenStates: encoderStates,
            contextLatents: contextLatents,
            cache: cache,
            useCache: true
        )
        #expect(returnedCache != nil)
        // Both layers should have cached cross-attention KV
        #expect(cache.isUpdated(layerIdx: 0))
        #expect(cache.isUpdated(layerIdx: 1))
    }

    @Test("DiTDecoder handles odd sequence lengths (padding)")
    func oddSequenceLength() {
        let decoder = DiTDecoder(
            hiddenSize: 256,
            intermediateSize: 512,
            numHiddenLayers: 2,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            headDim: 64,
            rmsNormEps: 1e-6,
            attentionBias: false,
            inChannels: 192,
            audioAcousticHiddenDim: 64,
            patchSize: 2,
            slidingWindow: 128,
            layerTypes: ["sliding_attention", "full_attention"],
            ropeTheta: 1_000_000.0,
            maxPositionEmbeddings: 32768
        )
        // 21 is odd, needs padding to 22 for patch_size=2
        let hiddenStates = MLXArray.zeros([1, 21, 64])
        let timestep = MLXArray(converting: [0.5], type: Float.self)
        let timestepR = MLXArray(converting: [0.5], type: Float.self)
        let encoderStates = MLXArray.zeros([1, 50, 256])
        let contextLatents = MLXArray.zeros([1, 21, 128])

        let (output, _) = decoder(
            hiddenStates: hiddenStates,
            timestep: timestep,
            timestepR: timestepR,
            encoderHiddenStates: encoderStates,
            contextLatents: contextLatents,
            cache: nil,
            useCache: false
        )
        // Output should be cropped back to original 21
        #expect(output.shape == [1, 21, 64])
    }

    @Test("DiTActor step returns correct shape")
    func actorStep() async {
        let decoder = DiTDecoder(
            hiddenSize: 256,
            intermediateSize: 512,
            numHiddenLayers: 2,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            headDim: 64,
            rmsNormEps: 1e-6,
            attentionBias: false,
            inChannels: 192,
            audioAcousticHiddenDim: 64,
            patchSize: 2,
            slidingWindow: 128,
            layerTypes: ["sliding_attention", "full_attention"],
            ropeTheta: 1_000_000.0,
            maxPositionEmbeddings: 32768
        )
        let actor = DiTActor(decoder: decoder)
        let noisyLatents = MLXArray.zeros([1, 20, 64])
        let encoderStates = MLXArray.zeros([1, 50, 256])
        let contextLatents = MLXArray.zeros([1, 20, 128])

        let (output, cache) = await actor.step(
            noisyLatents: noisyLatents,
            timestep: 0.5,
            timestepR: 0.5,
            encoderStates: encoderStates,
            contextLatents: contextLatents,
            cache: nil,
            useCache: true
        )
        #expect(output.shape == [1, 20, 64])
        #expect(cache != nil)
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/tweaver/Developer/GitRepos/AnvilApp/AnvilCore && swift test --filter DiTModelTests 2>&1 | tail -20`
Expected: FAIL — DiTDecoder not defined

- [ ] **Step 3: Implement DiTDecoder**

```swift
// AnvilCore/Sources/AnvilCore/Inference/DiT/DiTModel.swift
import Foundation
import MLX
import MLXNN

public class DiTDecoder: Module {
    let hiddenSize: Int
    let patchSize: Int
    let rotaryEmb: RotaryEmbedding
    let projIn: Conv1d
    let timeEmbed: TimestepEmbedding
    let timeEmbedR: TimestepEmbedding
    let conditionEmbedder: Linear
    let layers: [DiTLayer]
    let normOut: RMSNorm
    let projOut: ConvTranspose1d
    let scaleShiftTable: MLXArray
    let layerTypes: [String]
    let slidingWindow: Int

    private var slidingMasks: [Int: MLXArray] = [:]

    public init(
        hiddenSize: Int = 2048,
        intermediateSize: Int = 6144,
        numHiddenLayers: Int = 24,
        numAttentionHeads: Int = 16,
        numKeyValueHeads: Int = 8,
        headDim: Int = 128,
        rmsNormEps: Float = 1e-6,
        attentionBias: Bool = false,
        inChannels: Int = 192,
        audioAcousticHiddenDim: Int = 64,
        patchSize: Int = 2,
        slidingWindow: Int = 128,
        layerTypes: [String]? = nil,
        ropeTheta: Float = 1_000_000.0,
        maxPositionEmbeddings: Int = 32768
    ) {
        self.hiddenSize = hiddenSize
        self.patchSize = patchSize
        self.slidingWindow = slidingWindow

        let resolvedLayerTypes = layerTypes ?? (0..<numHiddenLayers).map { i in
            (i + 1) % 2 != 0 ? "sliding_attention" : "full_attention"
        }
        self.layerTypes = resolvedLayerTypes

        self.rotaryEmb = RotaryEmbedding(
            headDim: headDim, maxLen: maxPositionEmbeddings, base: ropeTheta
        )
        self.projIn = Conv1d(
            inputChannels: inChannels, outputChannels: hiddenSize,
            kernelSize: patchSize, stride: patchSize, padding: 0
        )
        self.timeEmbed = TimestepEmbedding(inChannels: 256, timeEmbedDim: hiddenSize)
        self.timeEmbedR = TimestepEmbedding(inChannels: 256, timeEmbedDim: hiddenSize)
        self.conditionEmbedder = Linear(hiddenSize, hiddenSize, bias: true)

        self.layers = (0..<numHiddenLayers).map { i in
            DiTLayer(
                hiddenSize: hiddenSize,
                intermediateSize: intermediateSize,
                numAttentionHeads: numAttentionHeads,
                numKeyValueHeads: numKeyValueHeads,
                headDim: headDim,
                rmsNormEps: rmsNormEps,
                attentionBias: attentionBias,
                layerIdx: i,
                layerType: resolvedLayerTypes[i],
                slidingWindow: slidingWindow
            )
        }

        self.normOut = RMSNorm(dimensions: hiddenSize, eps: rmsNormEps)
        self.projOut = ConvTranspose1d(
            inputChannels: hiddenSize, outputChannels: audioAcousticHiddenDim,
            kernelSize: patchSize, stride: patchSize, padding: 0
        )
        self.scaleShiftTable = MLXArray.zeros([1, 2, hiddenSize])
    }

    private func getSlidingMask(seqLen: Int) -> MLXArray {
        if let cached = slidingMasks[seqLen] { return cached }
        let mask = createSlidingWindowMask(seqLen: seqLen, windowSize: slidingWindow)
        slidingMasks[seqLen] = mask
        return mask
    }

    public func callAsFunction(
        hiddenStates: MLXArray,
        timestep: MLXArray,
        timestepR: MLXArray,
        encoderHiddenStates: MLXArray,
        contextLatents: MLXArray,
        cache: CrossAttentionCache?,
        useCache: Bool
    ) -> (MLXArray, CrossAttentionCache?) {
        // Timestep embeddings
        let (tembT, projT) = timeEmbed(timestep)
        let (tembR, projR) = timeEmbedR(timestep - timestepR)
        let temb = tembT + tembR
        let timestepProj = projT + projR

        // Concatenate context with hidden states
        var h = concatenated([contextLatents, hiddenStates], axis: -1)
        let originalSeqLen = h.shape[1]

        // Pad to multiple of patchSize
        var padLength = 0
        if h.shape[1] % patchSize != 0 {
            padLength = patchSize - (h.shape[1] % patchSize)
            let padding = MLXArray.zeros([h.shape[0], padLength, h.shape[2]])
            h = concatenated([h, padding], axis: 1)
        }

        // Patch embedding
        h = projIn(h)

        // Project encoder states
        let encStates = conditionEmbedder(encoderHiddenStates)

        let seqLen = h.shape[1]

        // Position embeddings
        let (cos, sin) = rotaryEmb.forward(seqLen: seqLen)

        // Sliding mask
        let hasSlidingLayer = layerTypes.contains("sliding_attention")
        let slidingMask: MLXArray? = hasSlidingLayer ? getSlidingMask(seqLen: seqLen) : nil

        // Process through transformer layers
        for layer in layers {
            let selfAttnMask = layer.layerType == "sliding_attention" ? slidingMask : nil
            h = layer(
                hiddenStates: h,
                positionCosSin: (cos, sin),
                temb: timestepProj,
                selfAttnMask: selfAttnMask,
                encoderHiddenStates: encStates,
                encoderAttentionMask: nil,
                cache: cache,
                useCache: useCache
            )
        }

        // Output adaptive layer norm
        let expandedTemb = temb.expandedDimensions(axis: 1)
        let outMod = scaleShiftTable + expandedTemb
        let outParts = split(outMod, parts: 2, axis: 1)
        let (shift, scale) = (outParts[0], outParts[1])
        h = normOut(h) * (1.0 + scale) + shift

        // De-patchify
        h = projOut(h)

        // Crop back to original sequence length
        h = h[0..., 0..<originalSeqLen, 0...]

        return (h, cache)
    }
}
```

- [ ] **Step 4: Implement DiTActor**

```swift
// AnvilCore/Sources/AnvilCore/Inference/DiT/DiTActor.swift
import Foundation
import MLX

public actor DiTActor {
    private let decoder: DiTDecoder

    public init(decoder: DiTDecoder) {
        self.decoder = decoder
    }

    public func step(
        noisyLatents: MLXArray,
        timestep: Float,
        timestepR: Float,
        encoderStates: MLXArray,
        contextLatents: MLXArray,
        cache: CrossAttentionCache?,
        useCache: Bool
    ) -> (output: MLXArray, cache: CrossAttentionCache?) {
        let batchSize = noisyLatents.shape[0]
        let tArray = MLXArray.full([batchSize], values: timestep)
        let trArray = MLXArray.full([batchSize], values: timestepR)

        let (output, returnedCache) = decoder(
            hiddenStates: noisyLatents,
            timestep: tArray,
            timestepR: trArray,
            encoderHiddenStates: encoderStates,
            contextLatents: contextLatents,
            cache: cache,
            useCache: useCache
        )
        eval(output)
        return (output, returnedCache)
    }

    public func loadWeights(_ weights: [(String, MLXArray)]) {
        decoder.update(parameters: ModuleParameters.unflattened(weights))
        eval(decoder.parameters())
    }
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/tweaver/Developer/GitRepos/AnvilApp/AnvilCore && swift test --filter DiTModelTests 2>&1 | tail -20`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
cd /Users/tweaver/Developer/GitRepos/AnvilApp
git add AnvilCore/Sources/AnvilCore/Inference/DiT/DiTModel.swift AnvilCore/Sources/AnvilCore/Inference/DiT/DiTActor.swift AnvilCore/Tests/AnvilCoreTests/Inference/DiTModelTests.swift
git commit -m "feat: implement DiT decoder model and actor"
```

---

### Task 9: Diffusion Sampler (ODE/SDE + CFG/APG)

**Files:**
- Create: `AnvilCore/Sources/AnvilCore/Inference/Diffusion/DiffusionSampler.swift`
- Create: `AnvilCore/Tests/AnvilCoreTests/Inference/DiffusionSamplerTests.swift`

- [ ] **Step 1: Write the failing tests**

```swift
// AnvilCore/Tests/AnvilCoreTests/Inference/DiffusionSamplerTests.swift
import Testing
import MLX
import MLXRandom
@testable import AnvilCore

@Suite("Diffusion Sampler")
struct DiffusionSamplerTests {
    @Test("ODE step reduces noise")
    func odeStepBasic() {
        // x_{t-1} = x_t - v_t * dt
        let xt = MLXArray(converting: [[[1.0, 2.0, 3.0]]], type: Float.self) // [1,1,3]
        let vt = MLXArray(converting: [[[0.5, 0.5, 0.5]]], type: Float.self)
        let result = DiffusionSampler.odeStep(xt: xt, vt: vt, currentT: 1.0, nextT: 0.5)
        eval(result)
        // dt = 1.0 - 0.5 = 0.5, result = [1-0.25, 2-0.25, 3-0.25] = [0.75, 1.75, 2.75]
        let vals = result.asArray(Float.self)
        #expect(abs(vals[0] - 0.75) < 1e-5)
        #expect(abs(vals[1] - 1.75) < 1e-5)
        #expect(abs(vals[2] - 2.75) < 1e-5)
    }

    @Test("Final step computes x0 correctly")
    func finalStep() {
        let xt = MLXArray(converting: [[[2.0, 4.0]]], type: Float.self)
        let vt = MLXArray(converting: [[[1.0, 2.0]]], type: Float.self)
        let result = DiffusionSampler.finalStep(xt: xt, vt: vt, currentT: 0.5)
        eval(result)
        // x0 = xt - vt * t = [2-0.5, 4-1.0] = [1.5, 3.0]
        let vals = result.asArray(Float.self)
        #expect(abs(vals[0] - 1.5) < 1e-5)
        #expect(abs(vals[1] - 3.0) < 1e-5)
    }

    @Test("APG with scale 1.0 returns pred_cond unchanged")
    func apgNoGuidance() {
        let predCond = MLXArray(converting: [[[1.0, 2.0, 3.0]]], type: Float.self)
        let predUncond = MLXArray(converting: [[[0.5, 1.0, 1.5]]], type: Float.self)
        let result = DiffusionSampler.apgForward(
            predCond: predCond, predUncond: predUncond,
            guidanceScale: 1.0, momentum: nil
        )
        eval(result)
        let vals = result.asArray(Float.self)
        // scale=1.0 means (scale-1)*orthogonal = 0, result = predCond
        #expect(abs(vals[0] - 1.0) < 1e-4)
        #expect(abs(vals[1] - 2.0) < 1e-4)
        #expect(abs(vals[2] - 3.0) < 1e-4)
    }

    @Test("APG with scale > 1.0 differs from pred_cond")
    func apgWithGuidance() {
        let predCond = MLXArray(converting: [[[1.0, 0.0, 0.0]]], type: Float.self)
        let predUncond = MLXArray(converting: [[[0.0, 1.0, 0.0]]], type: Float.self)
        let result = DiffusionSampler.apgForward(
            predCond: predCond, predUncond: predUncond,
            guidanceScale: 3.0, momentum: nil
        )
        eval(result)
        let vals = result.asArray(Float.self)
        // With guidance, the result should differ from predCond
        let diffFromCond = abs(vals[0] - 1.0) + abs(vals[1]) + abs(vals[2])
        #expect(diffFromCond > 0.01)
    }

    @Test("Noise generation with seed is deterministic")
    func deterministicNoise() {
        let noise1 = DiffusionSampler.generateNoise(shape: [1, 10, 64], seed: 42)
        let noise2 = DiffusionSampler.generateNoise(shape: [1, 10, 64], seed: 42)
        eval(noise1, noise2)
        let diff = abs(noise1 - noise2).sum()
        eval(diff)
        #expect(diff.item(Float.self) < 1e-6)
    }

    @Test("Noise generation without seed is random")
    func randomNoise() {
        let noise1 = DiffusionSampler.generateNoise(shape: [1, 10, 64], seed: nil)
        let noise2 = DiffusionSampler.generateNoise(shape: [1, 10, 64], seed: nil)
        eval(noise1, noise2)
        let diff = abs(noise1 - noise2).sum()
        eval(diff)
        #expect(diff.item(Float.self) > 0.1)
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/tweaver/Developer/GitRepos/AnvilApp/AnvilCore && swift test --filter DiffusionSamplerTests 2>&1 | tail -20`
Expected: FAIL — DiffusionSampler not defined

- [ ] **Step 3: Implement DiffusionSampler**

```swift
// AnvilCore/Sources/AnvilCore/Inference/Diffusion/DiffusionSampler.swift
import Foundation
import MLX
import MLXRandom

public enum DiffusionSampler {

    // MARK: - Stepping

    public static func odeStep(xt: MLXArray, vt: MLXArray, currentT: Float, nextT: Float) -> MLXArray {
        let dt = currentT - nextT
        let bsz = xt.shape[0]
        let dtArr = MLXArray.full([bsz, 1, 1], values: dt)
        return xt - vt * dtArr
    }

    public static func sdeStep(xt: MLXArray, vt: MLXArray, currentT: Float, nextT: Float) -> MLXArray {
        let bsz = xt.shape[0]
        let tUnsq = MLXArray.full([bsz, 1, 1], values: currentT)
        let predClean = xt - vt * tUnsq
        let newNoise = MLXRandom.normal(xt.shape)
        let ntUnsq = MLXArray.full([bsz, 1, 1], values: nextT)
        return ntUnsq * newNoise + (1.0 - ntUnsq) * predClean
    }

    public static func finalStep(xt: MLXArray, vt: MLXArray, currentT: Float) -> MLXArray {
        let bsz = xt.shape[0]
        let tUnsq = MLXArray.full([bsz, 1, 1], values: currentT)
        return xt - vt * tUnsq
    }

    // MARK: - CFG / APG

    public static func apgForward(
        predCond: MLXArray,
        predUncond: MLXArray,
        guidanceScale: Float,
        momentum: APGMomentum?,
        normThreshold: Float = 2.5
    ) -> MLXArray {
        var diff = predCond - predUncond
        if let momentum = momentum {
            diff = diff + momentum.running
            momentum.running = diff
        }

        if normThreshold > 0 {
            let diffNorm = sqrt((diff * diff).sum(axis: 1, keepDims: true))
            let scaleFactor = minimum(MLXArray.ones(like: diffNorm), MLXArray(normThreshold) / (diffNorm + 1e-8))
            diff = diff * scaleFactor
        }

        let condNorm = sqrt((predCond * predCond).sum(axis: 1, keepDims: true)) + 1e-8
        let v1 = predCond / condNorm
        let parallel = (diff * v1).sum(axis: 1, keepDims: true) * v1
        let orthogonal = diff - parallel

        return predCond + (guidanceScale - 1.0) * orthogonal
    }

    // MARK: - Noise Generation

    public static func generateNoise(shape: [Int], seed: Int?) -> MLXArray {
        if let seed = seed {
            let key = MLXRandom.key(UInt64(seed))
            return MLXRandom.normal(shape, key: key)
        }
        return MLXRandom.normal(shape)
    }
}

public class APGMomentum {
    var running: MLXArray

    public init() {
        self.running = MLXArray(Float(0))
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/tweaver/Developer/GitRepos/AnvilApp/AnvilCore && swift test --filter DiffusionSamplerTests 2>&1 | tail -20`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/tweaver/Developer/GitRepos/AnvilApp
git add AnvilCore/Sources/AnvilCore/Inference/Diffusion/DiffusionSampler.swift AnvilCore/Tests/AnvilCoreTests/Inference/DiffusionSamplerTests.swift
git commit -m "feat: implement diffusion sampler with ODE/SDE stepping and APG guidance"
```

---

### Task 10: Diffusion Actor (Full Generation Loop)

**Files:**
- Create: `AnvilCore/Sources/AnvilCore/Inference/Diffusion/DiffusionActor.swift`
- Create: `AnvilCore/Tests/AnvilCoreTests/Inference/DiffusionActorTests.swift`

- [ ] **Step 1: Write the failing tests**

```swift
// AnvilCore/Tests/AnvilCoreTests/Inference/DiffusionActorTests.swift
import Testing
import MLX
@testable import AnvilCore

@Suite("Diffusion Actor")
struct DiffusionActorTests {
    private func makeSmallDiT() -> DiTActor {
        let decoder = DiTDecoder(
            hiddenSize: 256,
            intermediateSize: 512,
            numHiddenLayers: 2,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            headDim: 64,
            rmsNormEps: 1e-6,
            attentionBias: false,
            inChannels: 192,
            audioAcousticHiddenDim: 64,
            patchSize: 2,
            slidingWindow: 128,
            layerTypes: ["sliding_attention", "full_attention"],
            ropeTheta: 1_000_000.0,
            maxPositionEmbeddings: 32768
        )
        return DiTActor(decoder: decoder)
    }

    @Test("Generate returns latents with correct shape")
    func generateShape() async {
        let dit = makeSmallDiT()
        let actor = DiffusionActor(dit: dit)
        let encoderStates = MLXArray.zeros([1, 50, 256])
        let contextLatents = MLXArray.zeros([1, 20, 128])
        let config = DiffusionConfig(seed: 42, steps: 4)

        let result = await actor.generate(
            encoderStates: encoderStates,
            contextLatents: contextLatents,
            shape: (batch: 1, frames: 20, channels: 64),
            config: config
        )
        #expect(result.latents.shape == [1, 20, 64])
        #expect(result.timeCosts.totalTime > 0)
    }

    @Test("Deterministic with same seed")
    func deterministic() async {
        let dit = makeSmallDiT()
        let actor = DiffusionActor(dit: dit)
        let encoderStates = MLXArray.zeros([1, 20, 256])
        let contextLatents = MLXArray.zeros([1, 10, 128])
        let config = DiffusionConfig(seed: 123, steps: 2)

        let r1 = await actor.generate(
            encoderStates: encoderStates, contextLatents: contextLatents,
            shape: (batch: 1, frames: 10, channels: 64), config: config
        )
        let r2 = await actor.generate(
            encoderStates: encoderStates, contextLatents: contextLatents,
            shape: (batch: 1, frames: 10, channels: 64), config: config
        )
        eval(r1.latents, r2.latents)
        let diff = abs(r1.latents - r2.latents).sum()
        eval(diff)
        #expect(diff.item(Float.self) < 1e-4)
    }

    @Test("SDE method produces different results each run without seed")
    func sdeVariation() async {
        let dit = makeSmallDiT()
        let actor = DiffusionActor(dit: dit)
        let encoderStates = MLXArray.zeros([1, 20, 256])
        let contextLatents = MLXArray.zeros([1, 10, 128])
        let config = DiffusionConfig(seed: nil, method: .sde, steps: 2)

        let r1 = await actor.generate(
            encoderStates: encoderStates, contextLatents: contextLatents,
            shape: (batch: 1, frames: 10, channels: 64), config: config
        )
        let r2 = await actor.generate(
            encoderStates: encoderStates, contextLatents: contextLatents,
            shape: (batch: 1, frames: 10, channels: 64), config: config
        )
        eval(r1.latents, r2.latents)
        let diff = abs(r1.latents - r2.latents).sum()
        eval(diff)
        #expect(diff.item(Float.self) > 0.01)
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/tweaver/Developer/GitRepos/AnvilApp/AnvilCore && swift test --filter DiffusionActorTests 2>&1 | tail -20`
Expected: FAIL — DiffusionActor not defined

- [ ] **Step 3: Implement DiffusionActor**

```swift
// AnvilCore/Sources/AnvilCore/Inference/Diffusion/DiffusionActor.swift
import Foundation
import MLX
import MLXRandom

public actor DiffusionActor {
    private let dit: DiTActor
    private var progressContinuation: AsyncStream<DiffusionProgress>.Continuation?
    private var _progress: AsyncStream<DiffusionProgress>?

    public init(dit: DiTActor) {
        self.dit = dit
    }

    public var progress: AsyncStream<DiffusionProgress> {
        if let existing = _progress { return existing }
        let (stream, continuation) = AsyncStream<DiffusionProgress>.makeStream()
        self.progressContinuation = continuation
        self._progress = stream
        return stream
    }

    public func generate(
        encoderStates: MLXArray,
        contextLatents: MLXArray,
        shape: (batch: Int, frames: Int, channels: Int),
        config: DiffusionConfig
    ) async -> DiffusionResult {
        let totalStart = Date()

        let bsz = shape.batch
        let T = shape.frames
        let C = shape.channels

        // Generate initial noise
        var xt = DiffusionSampler.generateNoise(shape: [bsz, T, C], seed: config.seed)

        // Compute timestep schedule
        let schedule = TimestepSchedule.compute(
            shift: config.shift, timesteps: nil, inferSteps: config.steps
        )
        let numSteps = schedule.count

        // CFG setup
        let doCFG = config.guidanceScale > 1.0
        var encHS = encoderStates
        var ctx = contextLatents
        if doCFG {
            let nullExpanded = MLXArray.zeros(like: encHS)
            encHS = concatenated([encHS, nullExpanded], axis: 0)
            ctx = concatenated([ctx, ctx], axis: 0)
        }
        let momentum: APGMomentum? = doCFG ? APGMomentum() : nil

        // Cover step tracking
        let coverSteps = Int(Float(numSteps) * config.audioCoverStrength)

        let cache = CrossAttentionCache()
        let diffStart = Date()

        for stepIdx in 0..<numSteps {
            let currentT = schedule[stepIdx]

            // Invalidate cache when switching from cover to non-cover
            if stepIdx >= coverSteps {
                cache.reset()
            }

            // Build input
            let xIn = doCFG ? concatenated([xt, xt], axis: 0) : xt

            let (vt, _) = await dit.step(
                noisyLatents: xIn,
                timestep: currentT,
                timestepR: currentT,
                encoderStates: encHS,
                contextLatents: ctx,
                cache: cache,
                useCache: !doCFG
            )

            var velocity = vt

            // Apply CFG
            if doCFG {
                let predCond = vt[0..<bsz]
                let predUncond = vt[bsz...]
                let applyCFG = currentT >= config.cfgIntervalStart && currentT <= config.cfgIntervalEnd
                if applyCFG {
                    velocity = DiffusionSampler.apgForward(
                        predCond: predCond, predUncond: predUncond,
                        guidanceScale: config.guidanceScale, momentum: momentum
                    )
                } else {
                    velocity = predCond
                }
            }

            // Step
            if stepIdx == numSteps - 1 {
                xt = DiffusionSampler.finalStep(xt: xt, vt: velocity, currentT: currentT)
            } else {
                let nextT = schedule[stepIdx + 1]
                switch config.method {
                case .ode:
                    xt = DiffusionSampler.odeStep(xt: xt, vt: velocity, currentT: currentT, nextT: nextT)
                case .sde:
                    xt = DiffusionSampler.sdeStep(xt: xt, vt: velocity, currentT: currentT, nextT: nextT)
                }
            }
            eval(xt)

            let elapsed = Date().timeIntervalSince(diffStart)
            progressContinuation?.yield(DiffusionProgress(
                currentStep: stepIdx + 1, totalSteps: numSteps, elapsedTime: elapsed
            ))
        }

        let diffTime = Date().timeIntervalSince(diffStart)
        let totalTime = Date().timeIntervalSince(totalStart)
        let costs = TimeCosts(
            diffusionTime: diffTime,
            perStepTime: diffTime / Double(max(numSteps, 1)),
            totalTime: totalTime
        )

        return DiffusionResult(latents: xt, timeCosts: costs)
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/tweaver/Developer/GitRepos/AnvilApp/AnvilCore && swift test --filter DiffusionActorTests 2>&1 | tail -20`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/tweaver/Developer/GitRepos/AnvilApp
git add AnvilCore/Sources/AnvilCore/Inference/Diffusion/DiffusionActor.swift AnvilCore/Tests/AnvilCoreTests/Inference/DiffusionActorTests.swift
git commit -m "feat: implement DiffusionActor with full generation loop, CFG, and progress"
```

---

### Task 11: Weight Converter (VAE Name Map)

**Files:**
- Create: `AnvilCore/Sources/AnvilCore/Inference/Weights/VAENameMap.swift`
- Create: `AnvilCore/Tests/AnvilCoreTests/Inference/WeightConverterTests.swift`

- [ ] **Step 1: Write the failing tests**

```swift
// AnvilCore/Tests/AnvilCoreTests/Inference/WeightConverterTests.swift
import Testing
import MLX
@testable import AnvilCore

@Suite("Weight Converter")
struct WeightConverterTests {
    @Test("VAE weight_norm fusion produces correct shape")
    func vaeWeightNormFusion() {
        // weight_g: [out, 1, 1], weight_v: [out, in, K]
        let g = MLXArray.ones([16, 1, 1])
        let v = MLXArray.ones([16, 8, 7])
        let fused = VAENameMap.fuseWeightNorm(weightG: g, weightV: v)
        #expect(fused.shape == [16, 8, 7])
    }

    @Test("VAE Conv1d axis swap: [out, in, K] -> [out, K, in]")
    func vaeConv1dSwap() {
        // Create identifiable tensor: shape [2, 3, 5]
        var data: [Float] = []
        for o in 0..<2 { for i in 0..<3 { for k in 0..<5 {
            data.append(Float(o * 100 + i * 10 + k))
        }}}
        let pt = MLXArray(data, [2, 3, 5])
        let mlx = VAENameMap.conv1dToMLX(pt)
        #expect(mlx.shape == [2, 5, 3]) // [out, K, in]
        eval(mlx)
    }

    @Test("VAE ConvTranspose1d: [in, out, K] -> [out, K, in]")
    func vaeConvTranspose1dSwap() {
        let pt = MLXArray.zeros([3, 2, 5]) // [in=3, out=2, K=5]
        let mlx = VAENameMap.convTranspose1dToMLX(pt)
        #expect(mlx.shape == [2, 5, 3]) // [out, K, in]
    }

    @Test("VAE Snake1d squeeze: [1, C, 1] -> [C]")
    func vaeSnakeSqueeze() {
        let pt = MLXArray.ones([1, 32, 1])
        let mlx = VAENameMap.squeezeSnakeParam(pt)
        #expect(mlx.shape == [32])
    }

    @Test("DiT proj_in key remapping")
    func ditProjInRemap() {
        let key = "proj_in.1.weight"
        let mapped = TensorNameMap.mapDiTKey(key)
        #expect(mapped == "proj_in.weight")
    }

    @Test("DiT proj_out key remapping")
    func ditProjOutRemap() {
        let key = "proj_out.1.weight"
        let mapped = TensorNameMap.mapDiTKey(key)
        #expect(mapped == "proj_out.weight")
    }

    @Test("DiT rotary_emb keys are skipped")
    func ditRotarySkipped() {
        let key = "rotary_emb.inv_freq"
        let mapped = TensorNameMap.mapDiTKey(key)
        #expect(mapped == nil) // should be skipped
    }

    @Test("DiT normal keys pass through")
    func ditPassthrough() {
        let key = "layers.0.self_attn.q_proj.weight"
        let mapped = TensorNameMap.mapDiTKey(key)
        #expect(mapped == "layers.0.self_attn.q_proj.weight")
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/tweaver/Developer/GitRepos/AnvilApp/AnvilCore && swift test --filter WeightConverterTests 2>&1 | tail -20`
Expected: FAIL — VAENameMap not defined

- [ ] **Step 3: Implement VAENameMap**

```swift
// AnvilCore/Sources/AnvilCore/Inference/Weights/VAENameMap.swift
import MLX

public enum VAENameMap {
    public static func fuseWeightNorm(weightG: MLXArray, weightV: MLXArray) -> MLXArray {
        let vFlat = weightV.reshaped([weightV.shape[0], -1])
        let norm = sqrt((vFlat * vFlat).sum(axis: 1)).reshaped(weightG.shape)
        return weightG * weightV / (norm + 1e-9)
    }

    public static func conv1dToMLX(_ ptWeight: MLXArray) -> MLXArray {
        // PT [out, in, K] -> MLX [out, K, in]
        ptWeight.transposed(0, 2, 1)
    }

    public static func convTranspose1dToMLX(_ ptWeight: MLXArray) -> MLXArray {
        // PT [in, out, K] -> MLX [out, K, in]
        ptWeight.transposed(1, 2, 0)
    }

    public static func squeezeSnakeParam(_ ptParam: MLXArray) -> MLXArray {
        ptParam.squeezed()
    }

    public static func isWeightNormG(_ key: String) -> Bool {
        key.hasSuffix(".weight_g")
    }

    public static func isWeightNormV(_ key: String) -> Bool {
        key.hasSuffix(".weight_v")
    }

    public static func isSnakeParam(_ key: String) -> Bool {
        key.hasSuffix(".alpha") || key.hasSuffix(".beta")
    }

    public static func isConvTranspose(_ key: String) -> Bool {
        key.contains("conv_t1")
    }

    public static func weightNormCompanion(_ gKey: String) -> String {
        String(gKey.dropLast(".weight_g".count)) + ".weight_v"
    }

    public static func fusedKeyName(_ gKey: String) -> String {
        String(gKey.dropLast(".weight_g".count)) + ".weight"
    }
}
```

- [ ] **Step 4: Implement TensorNameMap**

```swift
// AnvilCore/Sources/AnvilCore/Inference/Weights/TensorNameMap.swift
import MLX

public enum TensorNameMap {
    public static func mapDiTKey(_ key: String) -> String? {
        if key.contains("rotary_emb") {
            return nil
        }
        if key.hasPrefix("proj_in.1.") {
            return key.replacingOccurrences(of: "proj_in.1.", with: "proj_in.")
        }
        if key.hasPrefix("proj_out.1.") {
            return key.replacingOccurrences(of: "proj_out.1.", with: "proj_out.")
        }
        return key
    }

    public static func isDiTConv1dWeight(_ key: String) -> Bool {
        key.contains("proj_in.weight")
    }

    public static func isDiTConvTransposeWeight(_ key: String) -> Bool {
        key.contains("proj_out.weight")
    }

    public static func conv1dToMLX(_ ptWeight: MLXArray) -> MLXArray {
        // PT Conv1d: [out, in, K] -> MLX: [out, K, in]
        ptWeight.transposed(0, 2, 1)
    }

    public static func convTranspose1dToMLX(_ ptWeight: MLXArray) -> MLXArray {
        // PT ConvTranspose1d: [in, out, K] -> MLX: [out, K, in]
        ptWeight.transposed(1, 2, 0)
    }
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/tweaver/Developer/GitRepos/AnvilApp/AnvilCore && swift test --filter WeightConverterTests 2>&1 | tail -20`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
cd /Users/tweaver/Developer/GitRepos/AnvilApp
git add AnvilCore/Sources/AnvilCore/Inference/Weights/VAENameMap.swift AnvilCore/Sources/AnvilCore/Inference/Weights/TensorNameMap.swift AnvilCore/Tests/AnvilCoreTests/Inference/WeightConverterTests.swift
git commit -m "feat: implement weight name maps for VAE and DiT conversion"
```

---

### Task 12: Weight Converter Actor

**Files:**
- Create: `AnvilCore/Sources/AnvilCore/Inference/Weights/WeightConverter.swift`
- Modify: `AnvilCore/Tests/AnvilCoreTests/Inference/WeightConverterTests.swift`

- [ ] **Step 1: Write additional failing tests**

Add to the existing `WeightConverterTests.swift`:

```swift
@Test("WeightConverter converts VAE state dict")
func convertVAEStateDict() async throws {
    // Create a minimal fake state dict (as if loaded from .bin)
    let stateDict: [(String, MLXArray)] = [
        ("encoder.conv1.weight_g", MLXArray.ones([128, 1, 1])),
        ("encoder.conv1.weight_v", MLXArray.ones([128, 2, 7])),
        ("encoder.conv1.bias", MLXArray.zeros([128])),
        ("encoder.block.0.res_unit1.snake1.alpha", MLXArray.zeros([1, 128, 1])),
        ("encoder.block.0.res_unit1.snake1.beta", MLXArray.zeros([1, 128, 1])),
    ]
    let converter = WeightConverter()
    let result = await converter.convertVAEWeights(stateDict: stateDict)

    // Should have fused weight_g+v into .weight, squeezed snake params
    let keys = result.map { $0.0 }
    #expect(keys.contains("encoder.conv1.weight"))
    #expect(keys.contains("encoder.conv1.bias"))
    #expect(keys.contains("encoder.block.0.res_unit1.snake1.alpha"))
    #expect(!keys.contains("encoder.conv1.weight_g"))
    #expect(!keys.contains("encoder.conv1.weight_v"))

    // Check fused weight shape: [out, K, in] = [128, 7, 2]
    let fusedWeight = result.first(where: { $0.0 == "encoder.conv1.weight" })!.1
    #expect(fusedWeight.shape == [128, 7, 2])

    // Check snake param shape: [128]
    let alpha = result.first(where: { $0.0 == "encoder.block.0.res_unit1.snake1.alpha" })!.1
    #expect(alpha.shape == [128])
}

@Test("WeightConverter converts DiT state dict")
func convertDiTStateDict() async throws {
    let stateDict: [(String, MLXArray)] = [
        ("proj_in.1.weight", MLXArray.ones([2048, 192, 2])),
        ("proj_in.1.bias", MLXArray.zeros([2048])),
        ("proj_out.1.weight", MLXArray.ones([2048, 64, 2])),
        ("proj_out.1.bias", MLXArray.zeros([64])),
        ("layers.0.self_attn.q_proj.weight", MLXArray.ones([2048, 2048])),
        ("rotary_emb.inv_freq", MLXArray.ones([64])),
    ]
    let converter = WeightConverter()
    let result = await converter.convertDiTWeights(stateDict: stateDict)

    let keys = result.map { $0.0 }
    #expect(keys.contains("proj_in.weight"))
    #expect(keys.contains("proj_out.weight"))
    #expect(keys.contains("layers.0.self_attn.q_proj.weight"))
    #expect(!keys.contains("rotary_emb.inv_freq"))

    // proj_in Conv1d: [out, in, K] -> [out, K, in] = [2048, 2, 192]
    let projIn = result.first(where: { $0.0 == "proj_in.weight" })!.1
    #expect(projIn.shape == [2048, 2, 192])

    // proj_out ConvTranspose1d: [in, out, K] -> [out, K, in] = [64, 2, 2048]
    let projOut = result.first(where: { $0.0 == "proj_out.weight" })!.1
    #expect(projOut.shape == [64, 2, 2048])
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/tweaver/Developer/GitRepos/AnvilApp/AnvilCore && swift test --filter WeightConverterTests 2>&1 | tail -20`
Expected: FAIL — WeightConverter.convertVAEWeights not defined

- [ ] **Step 3: Implement WeightConverter**

```swift
// AnvilCore/Sources/AnvilCore/Inference/Weights/WeightConverter.swift
import Foundation
import MLX

public actor WeightConverter {
    private var progressContinuation: AsyncStream<ConversionProgress>.Continuation?
    private var _progress: AsyncStream<ConversionProgress>?

    public init() {}

    public var progress: AsyncStream<ConversionProgress> {
        if let existing = _progress { return existing }
        let (stream, continuation) = AsyncStream<ConversionProgress>.makeStream()
        self.progressContinuation = continuation
        self._progress = stream
        return stream
    }

    public func convertVAEWeights(stateDict: [(String, MLXArray)]) -> [(String, MLXArray)] {
        var result: [(String, MLXArray)] = []
        var processed: Set<String> = []
        let total = stateDict.count

        let dict = Dictionary(stateDict, uniquingKeysWith: { first, _ in first })
        let allKeys = stateDict.map { $0.0 }.sorted()

        for (idx, key) in allKeys.enumerated() {
            if processed.contains(key) { continue }

            progressContinuation?.yield(ConversionProgress(
                current: idx, total: total, currentTensorName: key
            ))

            if VAENameMap.isWeightNormG(key) {
                let vKey = VAENameMap.weightNormCompanion(key)
                guard let g = dict[key], let v = dict[vKey] else {
                    processed.insert(key)
                    continue
                }
                var fused = VAENameMap.fuseWeightNorm(weightG: g, weightV: v)
                let fusedKey = VAENameMap.fusedKeyName(key)
                if VAENameMap.isConvTranspose(fusedKey) {
                    fused = VAENameMap.convTranspose1dToMLX(fused)
                } else {
                    fused = VAENameMap.conv1dToMLX(fused)
                }
                result.append((fusedKey, fused))
                processed.insert(key)
                processed.insert(vKey)
                continue
            }

            if VAENameMap.isWeightNormV(key) { continue }

            guard let value = dict[key] else { continue }

            if VAENameMap.isSnakeParam(key) {
                result.append((key, VAENameMap.squeezeSnakeParam(value)))
            } else {
                result.append((key, value))
            }
            processed.insert(key)
        }

        return result
    }

    public func convertDiTWeights(stateDict: [(String, MLXArray)]) -> [(String, MLXArray)] {
        var result: [(String, MLXArray)] = []
        let total = stateDict.count

        for (idx, (key, value)) in stateDict.enumerated() {
            progressContinuation?.yield(ConversionProgress(
                current: idx, total: total, currentTensorName: key
            ))

            guard let mappedKey = TensorNameMap.mapDiTKey(key) else { continue }

            if TensorNameMap.isDiTConv1dWeight(mappedKey) {
                result.append((mappedKey, TensorNameMap.conv1dToMLX(value)))
            } else if TensorNameMap.isDiTConvTransposeWeight(mappedKey) {
                result.append((mappedKey, TensorNameMap.convTranspose1dToMLX(value)))
            } else {
                result.append((mappedKey, value))
            }
        }

        return result
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/tweaver/Developer/GitRepos/AnvilApp/AnvilCore && swift test --filter WeightConverterTests 2>&1 | tail -20`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/tweaver/Developer/GitRepos/AnvilApp
git add AnvilCore/Sources/AnvilCore/Inference/Weights/WeightConverter.swift AnvilCore/Tests/AnvilCoreTests/Inference/WeightConverterTests.swift
git commit -m "feat: implement WeightConverter actor with VAE and DiT conversion"
```

---

### Task 13: Music Pipeline (Coordinator Actor)

**Files:**
- Create: `AnvilCore/Sources/AnvilCore/Inference/Pipeline/MusicPipeline.swift`
- Create: `AnvilCore/Tests/AnvilCoreTests/Inference/MusicPipelineTests.swift`

- [ ] **Step 1: Write the failing tests**

```swift
// AnvilCore/Tests/AnvilCoreTests/Inference/MusicPipelineTests.swift
import Testing
import MLX
@testable import AnvilCore

@Suite("MusicPipeline")
struct MusicPipelineTests {
    private func makeSmallPipeline() -> MusicPipeline {
        let ditDecoder = DiTDecoder(
            hiddenSize: 256,
            intermediateSize: 512,
            numHiddenLayers: 2,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            headDim: 64,
            rmsNormEps: 1e-6,
            attentionBias: false,
            inChannels: 192,
            audioAcousticHiddenDim: 64,
            patchSize: 2,
            slidingWindow: 128,
            layerTypes: ["sliding_attention", "full_attention"],
            ropeTheta: 1_000_000.0,
            maxPositionEmbeddings: 32768
        )
        let ditActor = DiTActor(decoder: ditDecoder)
        let vaeActor = VAEActor()
        let diffusionActor = DiffusionActor(dit: ditActor)
        return MusicPipeline(vae: vaeActor, diffusion: diffusionActor)
    }

    @Test("Initial state is ready when constructed with actors")
    func initialState() async {
        let pipeline = makeSmallPipeline()
        let state = await pipeline.state
        #expect(state == .ready)
    }

    @Test("Generate returns audio with correct sample rate")
    func generateAudio() async throws {
        let pipeline = makeSmallPipeline()
        let encoderStates = MLXArray.zeros([1, 20, 256])
        let contextLatents = MLXArray.zeros([1, 10, 128])
        let config = DiffusionConfig(seed: 42, steps: 2)

        let result = try await pipeline.generate(
            encoderStates: encoderStates,
            contextLatents: contextLatents,
            duration: 0.5, // 0.5 seconds
            config: config
        )
        #expect(result.sampleRate == 44100)
        #expect(result.samples.shape[0] == 1)
        #expect(result.samples.shape[2] == 2) // stereo
        #expect(result.timeCosts.totalTime > 0)
    }

    @Test("Duration maps to correct number of latent frames")
    func durationToFrames() {
        // 44100 samples/sec * duration / hop_length(2048) = latent frames
        let frames = MusicPipeline.latentFrames(forDuration: 1.0, sampleRate: 44100, hopLength: 2048)
        // 44100 / 2048 ≈ 21.5 -> 22 frames
        #expect(frames == 22)
    }

    @Test("Duration 4 minutes")
    func fourMinutes() {
        let frames = MusicPipeline.latentFrames(forDuration: 240.0, sampleRate: 44100, hopLength: 2048)
        // 240 * 44100 / 2048 ≈ 5168
        #expect(frames >= 5160 && frames <= 5180)
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/tweaver/Developer/GitRepos/AnvilApp/AnvilCore && swift test --filter MusicPipelineTests 2>&1 | tail -20`
Expected: FAIL — MusicPipeline not defined

- [ ] **Step 3: Implement MusicPipeline**

```swift
// AnvilCore/Sources/AnvilCore/Inference/Pipeline/MusicPipeline.swift
import Foundation
import MLX

public actor MusicPipeline {
    private let vae: VAEActor
    private let diffusion: DiffusionActor
    private var _state: PipelineState = .ready
    private var progressContinuation: AsyncStream<GenerationProgress>.Continuation?
    private var _progress: AsyncStream<GenerationProgress>?

    public static let defaultSampleRate = 44100
    public static let hopLength = 2048

    public init(vae: VAEActor, diffusion: DiffusionActor) {
        self.vae = vae
        self.diffusion = diffusion
    }

    public var state: PipelineState { _state }

    public var progress: AsyncStream<GenerationProgress> {
        if let existing = _progress { return existing }
        let (stream, continuation) = AsyncStream<GenerationProgress>.makeStream()
        self.progressContinuation = continuation
        self._progress = stream
        return stream
    }

    public static func latentFrames(
        forDuration duration: TimeInterval,
        sampleRate: Int = defaultSampleRate,
        hopLength: Int = hopLength
    ) -> Int {
        Int(ceil(Double(sampleRate) * duration / Double(hopLength)))
    }

    public func generate(
        encoderStates: MLXArray,
        contextLatents: MLXArray,
        duration: TimeInterval,
        config: DiffusionConfig
    ) async throws -> AudioResult {
        let totalStart = Date()
        _state = .generating(step: 0, totalSteps: config.steps ?? 8)

        let frames = Self.latentFrames(forDuration: duration)
        let batchSize = encoderStates.shape[0]

        // Run diffusion
        progressContinuation?.yield(.diffusion(DiffusionProgress(
            currentStep: 0, totalSteps: config.steps ?? 8, elapsedTime: 0
        )))

        let diffResult = await diffusion.generate(
            encoderStates: encoderStates,
            contextLatents: contextLatents,
            shape: (batch: batchSize, frames: frames, channels: 64),
            config: config
        )

        // Decode latents to audio
        progressContinuation?.yield(.decoding)
        let audio = await vae.decode(latents: diffResult.latents, precision: config.precision)

        let totalTime = Date().timeIntervalSince(totalStart)
        let costs = TimeCosts(
            diffusionTime: diffResult.timeCosts.diffusionTime,
            perStepTime: diffResult.timeCosts.perStepTime,
            totalTime: totalTime
        )

        _state = .ready
        progressContinuation?.yield(.complete)

        return AudioResult(
            samples: audio,
            sampleRate: Self.defaultSampleRate,
            duration: duration,
            timeCosts: costs
        )
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/tweaver/Developer/GitRepos/AnvilApp/AnvilCore && swift test --filter MusicPipelineTests 2>&1 | tail -20`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/tweaver/Developer/GitRepos/AnvilApp
git add AnvilCore/Sources/AnvilCore/Inference/Pipeline/MusicPipeline.swift AnvilCore/Tests/AnvilCoreTests/Inference/MusicPipelineTests.swift
git commit -m "feat: implement MusicPipeline coordinator actor"
```

---

### Task 14: Final Integration Test and Cleanup

**Files:**
- Modify: `AnvilCore/Tests/AnvilCoreTests/Inference/MusicPipelineTests.swift`
- Verify: All existing tests still pass

- [ ] **Step 1: Run the full test suite**

Run: `cd /Users/tweaver/Developer/GitRepos/AnvilApp/AnvilCore && swift test 2>&1 | tail -30`
Expected: All tests PASS (both existing Phase 1 tests and new Phase 2 tests)

- [ ] **Step 2: Verify build succeeds**

Run: `cd /Users/tweaver/Developer/GitRepos/AnvilApp && xcodegen generate && xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | grep -E "(BUILD|error:)" | tail -10`
Expected: BUILD SUCCEEDED

- [ ] **Step 3: Run swift package resolve to verify dependency resolution**

Run: `cd /Users/tweaver/Developer/GitRepos/AnvilApp/AnvilCore && swift package resolve 2>&1 | tail -10`
Expected: Clean resolution with mlx-swift package

- [ ] **Step 4: Final commit if any cleanup needed**

```bash
cd /Users/tweaver/Developer/GitRepos/AnvilApp
git status
# If there are any remaining changes:
git add -A
git commit -m "chore: final Phase 2 integration cleanup"
```

- [ ] **Step 5: Push to remote**

```bash
cd /Users/tweaver/Developer/GitRepos/AnvilApp
git push origin main
```
