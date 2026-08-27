import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom

public struct TrainingValidationPoint: Codable, Sendable, Equatable, Hashable {
    public let epoch: Int
    public let step: Int
    public let loss: Float

    public init(epoch: Int, step: Int, loss: Float) {
        self.epoch = epoch
        self.step = step
        self.loss = loss
    }
}

public struct AdapterTrainingResult: @unchecked Sendable {
    /// Adapter artifact to export on successful completion (best snapshot when available).
    public let weights: [(String, MLXArray)]
    /// Model weights at the end of `train()` — used to resume optimization.
    public let currentWeights: [(String, MLXArray)]
    /// Best selection-loss snapshot across completed epochs, if any.
    public let bestWeights: [(String, MLXArray)]?
    /// Best selection loss (validation when present, else train avg). Nil if no epoch finished.
    public let finalLoss: Float?
    public let trainedPaths: [String]
    public let optimizerState: [(String, MLXArray)]
    public let wasCancelled: Bool

    public init(
        weights: [(String, MLXArray)],
        currentWeights: [(String, MLXArray)]? = nil,
        bestWeights: [(String, MLXArray)]? = nil,
        finalLoss: Float?,
        trainedPaths: [String],
        optimizerState: [(String, MLXArray)],
        wasCancelled: Bool
    ) {
        self.weights = weights
        self.currentWeights = currentWeights ?? weights
        self.bestWeights = bestWeights
        self.finalLoss = finalLoss
        self.trainedPaths = trainedPaths
        self.optimizerState = optimizerState
        self.wasCancelled = wasCancelled
    }
}

public struct AdapterTrainingSplit {
    let trainingIndices: [Int]
    let validationIndices: [Int]

    public static func trainingSampleCount(total: Int) -> Int {
        total - validationSampleCount(total: total)
    }

    static func make(sampleCount: Int, seed: UInt64) -> AdapterTrainingSplit {
        guard sampleCount > 0 else {
            return AdapterTrainingSplit(trainingIndices: [], validationIndices: [])
        }

        var indices = Array(0 ..< sampleCount)
        var rng = SeededRandomNumberGenerator(seed: AdapterTrainingRandom.mix(seed, 0x7A7C_72A1_51D5_EED5))
        indices.shuffle(using: &rng)

        let validationCount = validationSampleCount(total: sampleCount)
        let validation = Array(indices.prefix(validationCount)).sorted()
        let training = Array(indices.dropFirst(validationCount))
        return AdapterTrainingSplit(
            trainingIndices: training.isEmpty ? Array(0 ..< sampleCount) : training,
            validationIndices: training.isEmpty ? [] : validation
        )
    }

    private static func validationSampleCount(total: Int) -> Int {
        guard total >= 10 else { return 0 }
        return min(max(1, Int((Double(total) * 0.10).rounded())), 16)
    }
}

struct SeededRandomNumberGenerator: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) {
        state = seed == 0 ? 0x9E37_79B9_7F4A_7C15 : seed
    }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var value = state
        value = (value ^ (value >> 30)) &* 0xBF58_476D_1CE4_E5B9
        value = (value ^ (value >> 27)) &* 0x94D0_49BB_1331_11EB
        return value ^ (value >> 31)
    }
}

enum AdapterTrainingRandom {
    static func sampleKeys(seed: UInt64, epoch: Int, sampleIndex: Int, globalStep: Int) -> [MLXArray] {
        let sampleSeed = deriveSeed(
            seed,
            UInt64(truncatingIfNeeded: epoch),
            UInt64(truncatingIfNeeded: sampleIndex),
            UInt64(truncatingIfNeeded: globalStep)
        )
        let keys = MLXRandom.split(key: MLXRandom.key(sampleSeed), into: 4)
        eval(keys)
        return keys
    }

    static func dropoutSeed(
        baseSeed: UInt64,
        epoch: Int,
        globalStep: Int,
        sampleIndex: Int,
        pathCount: Int
    ) -> UInt64 {
        deriveSeed(
            baseSeed,
            UInt64(truncatingIfNeeded: epoch),
            UInt64(truncatingIfNeeded: globalStep),
            UInt64(truncatingIfNeeded: sampleIndex),
            UInt64(truncatingIfNeeded: pathCount)
        )
    }

    /// Fixed validation-noise seed for a training run.
    ///
    /// Must **not** incorporate the epoch index — epoch validation scores are
    /// only comparable when noise is identical across epochs.
    static func validationSeed(runSeed: UInt64) -> UInt64 {
        mix(runSeed, 0xFA11_D47E_F1ED_5EED)
    }

    static func validationNoiseKey(seed: UInt64, sampleIndex: Int, timestepIndex: Int) -> MLXArray {
        MLXRandom.key(deriveSeed(
            seed,
            0xD1A6_90B1_3F01_F00D,
            UInt64(truncatingIfNeeded: sampleIndex),
            UInt64(truncatingIfNeeded: timestepIndex)
        ))
    }

    static func mix(_ lhs: UInt64, _ rhs: UInt64) -> UInt64 {
        deriveSeed(lhs, rhs)
    }

    static func deriveSeed(_ values: UInt64...) -> UInt64 {
        var hash: UInt64 = 0xCBF2_9CE4_8422_2325
        for value in values {
            hash ^= value &+ 0x9E37_79B9_7F4A_7C15
            hash &*= 0x0000_0100_0000_01B3
            hash ^= hash >> 32
        }
        return hash == 0 ? 1 : hash
    }
}

enum AdapterOptimizerState {
    static func initializeAdamState(
        optimizer: AdamW,
        model: DiTDecoder,
        trainableParameterKeys: [String]
    ) {
        let trainableKeySet = Set(trainableParameterKeys)
        let zeroGradients = model.trainableParameters().flattened().map { key, value in
            (key, MLXArray.zeros(like: value))
        }
        let filtered = zeroGradients.filter { key, _ in trainableKeySet.contains(key) }
        optimizer.update(model: model, gradients: ModuleParameters.unflattened(filtered))
        eval(model.trainableParameters(), optimizer.innerState())
    }

    static func namedState(
        optimizer: AdamW,
        trainableParameterKeys: [String]
    ) -> [(String, MLXArray)] {
        let state = optimizer.innerState()
        guard state.count == trainableParameterKeys.count * 2 else {
            return state.enumerated().map { ("state.\(String(format: "%05d", $0.offset))", $0.element) }
        }

        var result: [(String, MLXArray)] = []
        result.reserveCapacity(state.count)
        for (index, key) in trainableParameterKeys.enumerated() {
            result.append(("\(key).adam.m", state[index * 2]))
            result.append(("\(key).adam.v", state[index * 2 + 1]))
        }
        return result
    }

    static func restoreAdamStateIfPresent(
        optimizer: AdamW,
        trainableParameterKeys: [String],
        checkpoint: TrainingCheckpoint?,
        kind: String
    ) {
        guard let checkpoint else { return }
        let url = checkpoint.optimizerStateURL
        guard url.pathExtension == "safetensors",
              FileManager.default.fileExists(atPath: url.path(percentEncoded: false)) else {
            AudioDiagnostics.note(
                "training/trainer/optimizerState",
                "kind=\(kind) restore=missing path=\(url.path(percentEncoded: false))"
            )
            return
        }

        do {
            let arrays = try MLX.loadArrays(url: url)
            let state = optimizer.innerState()
            guard state.count == trainableParameterKeys.count * 2 else {
                AudioDiagnostics.note(
                    "training/trainer/optimizerState",
                    "kind=\(kind) restore=skipped reason=stateCountMismatch expected=\(trainableParameterKeys.count * 2) actual=\(state.count)"
                )
                return
            }

            for (index, key) in trainableParameterKeys.enumerated() {
                let namedM = arrays["\(key).adam.m"]
                let namedV = arrays["\(key).adam.v"]
                let fallbackM = arrays["state.\(String(format: "%05d", index * 2))"]
                let fallbackV = arrays["state.\(String(format: "%05d", index * 2 + 1))"]
                guard let m = namedM ?? fallbackM,
                      let v = namedV ?? fallbackV else {
                    AudioDiagnostics.note(
                        "training/trainer/optimizerState",
                        "kind=\(kind) restore=skipped reason=missingKey key=\(key)"
                    )
                    return
                }
                state[index * 2]._updateInternal(m)
                state[index * 2 + 1]._updateInternal(v)
            }
            eval(state)
            AudioDiagnostics.note(
                "training/trainer/optimizerState",
                "kind=\(kind) restore=loaded tensors=\(arrays.count) path=\(url.path(percentEncoded: false))"
            )
        } catch {
            AudioDiagnostics.note(
                "training/trainer/optimizerState",
                "kind=\(kind) restore=failed error=\(error.localizedDescription)"
            )
        }
    }
}

enum AdapterValidationEvaluator {
    private static let timesteps: [Float] = [0.20, 0.45, 0.70, 0.95]

    static func validationLoss(
        model: DiTDecoder,
        samples: [TrainingSample],
        validationIndices: [Int],
        seed: UInt64
    ) -> Float? {
        guard !validationIndices.isEmpty else { return nil }

        var losses: [MLXArray] = []
        losses.reserveCapacity(validationIndices.count * timesteps.count)

        for (sampleOffset, sampleIndex) in validationIndices.enumerated() {
            guard samples.indices.contains(sampleIndex) else { continue }
            let sample = samples[sampleIndex]
            for (timestepIndex, timestep) in timesteps.enumerated() {
                let t = MLXArray([timestep])
                let noise = MLXRandom.normal(
                    sample.latent.shape,
                    key: AdapterTrainingRandom.validationNoiseKey(
                        seed: seed,
                        sampleIndex: sampleOffset,
                        timestepIndex: timestepIndex
                    )
                )
                let xt = (1.0 - t) * sample.latent + t * noise
                let vTarget = noise - sample.latent
                let (vPred, _) = model(
                    hiddenStates: xt,
                    timestep: t,
                    timestepR: t,
                    encoderHiddenStates: sample.encoderStates,
                    contextLatents: sample.contextLatents,
                    encoderAttentionMask: sample.encoderMask,
                    cache: nil,
                    useCache: false
                )
                let diff = (vPred - vTarget).asType(.float32)
                losses.append((diff * diff).mean())
            }
        }

        guard !losses.isEmpty else { return nil }
        eval(losses)
        let total = losses.reduce(Float(0)) { partial, loss in
            partial + loss.item(Float.self)
        }
        return total / Float(losses.count)
    }
}
