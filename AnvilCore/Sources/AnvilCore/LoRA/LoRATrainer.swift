import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom

/// A single training sample: preprocessed latent and conditioning data.
public struct TrainingSample: Sendable {
    /// Clean latent from VAE encoder [1, frames, channels].
    public let latent: MLXArray

    /// Text encoder output [1, seqLen, hiddenSize].
    public let encoderStates: MLXArray

    /// Context latent [1, contextLen, hiddenSize].
    public let contextLatents: MLXArray

    /// Attention mask or nil.
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
    public let step: Int
    public let loss: Float
    public let learningRate: Float
    public let estimatedTimeRemaining: TimeInterval?
    public let lossHistory: [Float]
    public let elapsedTime: TimeInterval

    /// Legacy accessor for backward compatibility with TrainingService.
    public var estimatedRemaining: TimeInterval {
        estimatedTimeRemaining ?? 0
    }
}

/// Flow-matching training loop for LoRA adapters with cosine LR scheduling,
/// gradient accumulation, and checkpoint/resume support.
public final class LoRATrainer {
    private let model: DiTDecoder
    private let config: LoRAConfig
    private let injectedPaths: [String]
    private var bestLoss: Float = .infinity
    private var bestWeights: [(String, MLXArray)]?
    private var isCancelled: Bool = false

    // State for checkpoint/resume
    private var globalStep: Int = 0
    private var currentEpoch: Int = 0
    private var lossHistory: [Float] = []
    private let scheduler: CosineScheduler
    private let totalSteps: Int

    /// Designated initializer.
    /// - Parameters:
    ///   - model: DiT decoder with LoRA layers already injected.
    ///   - config: Training configuration.
    ///   - injectedPaths: Key paths of injected LoRA layers.
    ///   - datasetSize: Number of training samples (for scheduler computation).
    ///   - resumeFrom: Optional checkpoint to resume from.
    public init(
        model: DiTDecoder,
        config: LoRAConfig,
        injectedPaths: [String],
        datasetSize: Int,
        resumeFrom checkpoint: TrainingCheckpoint? = nil
    ) {
        self.model = model
        self.config = config
        self.injectedPaths = injectedPaths

        // Each epoch processes all samples; optimizer steps once per
        // accumulation window, including a final partial window.
        let updatesPerEpoch = Self.optimizerStepsPerEpoch(
            datasetSize: datasetSize,
            gradientAccumulationSteps: config.gradientAccumulationSteps
        )
        self.totalSteps = updatesPerEpoch * config.maxEpochs

        self.scheduler = CosineScheduler(
            baseLR: config.learningRate,
            warmupFraction: config.warmupFraction,
            totalSteps: totalSteps
        )

        // Restore from checkpoint if provided
        if let checkpoint {
            self.globalStep = checkpoint.globalStep
            self.currentEpoch = checkpoint.epoch
            self.lossHistory = checkpoint.lossHistory
            self.bestLoss = checkpoint.bestLoss ?? .infinity
        }
    }

    /// Run the training loop. Returns the best LoRA weights (lowest loss across epochs).
    ///
    /// - Throws: `CancellationError` when the enclosing Swift `Task` is cancelled.
    ///   The loop also exits early when `cancel()` is called on this instance.
    public func train(
        dataset: [TrainingSample],
        onProgress: @escaping @Sendable (TrainingProgress) -> Void
    ) async throws -> [(String, MLXArray)] {
        print("[LoRATrainer] train entered dataset=\(dataset.count)")
        guard !dataset.isEmpty else { return [] }

        let accumSteps = max(1, min(config.gradientAccumulationSteps, dataset.count))
        print("[LoRATrainer] creating optimizer")
        let optimizer = AdamW(learningRate: scheduler.learningRate(atStep: globalStep))
        let startTime = Date()

        print("[LoRATrainer] setTrainingMode begin")
        setTrainingMode(true)
        print("[LoRATrainer] setTrainingMode complete")

        // Build the valueAndGrad closure
        print("[LoRATrainer] building valueAndGrad")
        var hasLoggedGradientClosure = false
        let trainingModel = model
        let trainableParameterKeys = model.trainableParameters().flattened().map(\.0)
        let trainableArgumentCount = trainableParameterKeys.count
        let gradientArgumentNumbers = Array(0..<trainableArgumentCount)
        let lossGrad = valueAndGrad({ (allArgs: [MLXArray]) -> [MLXArray] in
            let shouldLogGradientClosure = !hasLoggedGradientClosure
            if shouldLogGradientClosure {
                hasLoggedGradientClosure = true
                print("[LoRATrainer] gradClosure entered paramKeys=\(trainableParameterKeys)")
                print("[LoRATrainer] gradClosure updating model parameters")
            }
            let parameterValues = Array(allArgs.prefix(trainableArgumentCount))
            let parameterPairs = zip(trainableParameterKeys, parameterValues).map { ($0.0, $0.1) }
            trainingModel.update(parameters: ModuleParameters.unflattened(parameterPairs))
            if shouldLogGradientClosure {
                print("[LoRATrainer] gradClosure model parameters updated")
            }

            let argOffset = trainableArgumentCount
            let xt = allArgs[argOffset]
            let t = allArgs[argOffset + 1]
            let encoderStates = allArgs[argOffset + 2]
            let contextLatents = allArgs[argOffset + 3]
            let vTarget = allArgs[argOffset + 4]
            let encoderMask: MLXArray? = allArgs.count > argOffset + 5
                ? allArgs[argOffset + 5]
                : nil

            if shouldLogGradientClosure {
                print("[LoRATrainer] gradClosure calling DiT forward")
            }
            let (vPred, _) = trainingModel(
                hiddenStates: xt,
                timestep: t,
                timestepR: t,
                encoderHiddenStates: encoderStates,
                contextLatents: contextLatents,
                encoderAttentionMask: encoderMask,
                cache: nil,
                useCache: false
            )
            if shouldLogGradientClosure {
                print("[LoRATrainer] gradClosure DiT forward returned")
            }

            let diff = (vPred - vTarget).asType(.float32)
            let loss = (diff * diff).mean()
            if shouldLogGradientClosure {
                print("[LoRATrainer] gradClosure loss graph built")
            }
            return [loss]
        }, argumentNumbers: gradientArgumentNumbers)
        print("[LoRATrainer] valueAndGrad built trainableArgumentCount=\(trainableArgumentCount)")

        let startEpoch = currentEpoch + 1
        print(
            "[LoRATrainer] loop startEpoch=\(startEpoch) maxEpochs=\(config.maxEpochs) globalStep=\(globalStep) lossHistory=\(lossHistory.count)"
        )
        var hasLoggedFirstProcessedSample = false

        for epoch in startEpoch...config.maxEpochs {
            guard !isCancelled else { break }
            currentEpoch = epoch
            print("[LoRATrainer] epoch \(epoch) begin")

            var epochLoss: Float = 0
            var epochSamples: Int = 0
            let shuffledIndices = Array(0..<dataset.count).shuffled()
            print("[LoRATrainer] epoch \(epoch) shuffled count=\(shuffledIndices.count)")

            // Per-window tracking for accurate loss reporting
            var windowLoss: Float = 0
            var windowSamples: Int = 0
            var accumulatedFlatGrads: [MLXArray]?

            for idx in shuffledIndices {
                guard !isCancelled else { break }
                try Task.checkCancellation()
                let sample = dataset[idx]
                let shouldLogSample = !hasLoggedFirstProcessedSample
                if shouldLogSample {
                    hasLoggedFirstProcessedSample = true
                    let trainableKeys = model.trainableParameters().flattened().map(\.0)
                    print(
                        "[LoRATrainer] firstSample selected idx=\(idx) latent=\(sample.latent.shape) context=\(sample.contextLatents.shape) encoder=\(sample.encoderStates.shape) trainableKeys=\(trainableKeys)"
                    )
                }

                // Sample random timestep t in [0, 1]
                if shouldLogSample {
                    print("[LoRATrainer] firstSample making timestep")
                }
                let t = MLXRandom.uniform(0..<1, [1])

                // Interpolate: x_t = (1 - t) * x_0 + t * noise
                if shouldLogSample {
                    print("[LoRATrainer] firstSample making noise")
                }
                let noise = MLXRandom.normal(sample.latent.shape)
                if shouldLogSample {
                    print("[LoRATrainer] firstSample making xt")
                }
                let xt = (1.0 - t) * sample.latent + t * noise

                // Target velocity: v_target = noise - x_0
                if shouldLogSample {
                    print("[LoRATrainer] firstSample making target")
                }
                let vTarget = noise - sample.latent

                // Build args array
                var args = [xt, t, sample.encoderStates, sample.contextLatents, vTarget]
                if let mask = sample.encoderMask {
                    args.append(mask)
                }

                if shouldLogSample {
                    print("[LoRATrainer] firstSample calling valueAndGrad args=\(args.map { $0.shape })")
                }

                // Forward + backward. The optimizer step happens once per
                // accumulation window below.
                let trainableParameters = model.trainableParameters().flattened()
                let trainableValues = trainableParameters.map(\.1)
                let (lossValues, flatGrads) = lossGrad(trainableValues + args)
                if shouldLogSample {
                    print("[LoRATrainer] firstSample valueAndGrad returned flatGrads=\(flatGrads.count)")
                }
                let loss = lossValues[0]
                let lossValue = loss.item(Float.self)
                if shouldLogSample {
                    print("[LoRATrainer] firstSample loss materialized \(lossValue)")
                }

                if let current = accumulatedFlatGrads {
                    accumulatedFlatGrads = Self.addFlatGradients(current, flatGrads)
                } else {
                    accumulatedFlatGrads = flatGrads
                }
                if shouldLogSample {
                    print("[LoRATrainer] firstSample gradients accumulated")
                }

                epochLoss += lossValue
                epochSamples += 1
                windowLoss += lossValue
                windowSamples += 1

                if windowSamples >= accumSteps,
                   let windowFlatGrads = accumulatedFlatGrads {
                    updateOptimizerForAccumulationWindow(
                        optimizer: optimizer,
                        trainableParameterKeys: trainableParameterKeys,
                        accumulatedFlatGrads: windowFlatGrads,
                        windowLoss: windowLoss,
                        windowSamples: windowSamples,
                        epoch: epoch,
                        startTime: startTime,
                        onProgress: onProgress
                    )
                    windowLoss = 0
                    windowSamples = 0
                    accumulatedFlatGrads = nil
                }
            }

            if windowSamples > 0,
               let windowFlatGrads = accumulatedFlatGrads {
                updateOptimizerForAccumulationWindow(
                    optimizer: optimizer,
                    trainableParameterKeys: trainableParameterKeys,
                    accumulatedFlatGrads: windowFlatGrads,
                    windowLoss: windowLoss,
                    windowSamples: windowSamples,
                    epoch: epoch,
                    startTime: startTime,
                    onProgress: onProgress
                )
            }

            // Track best weights (per epoch)
            let avgLoss = epochSamples > 0 ? epochLoss / Float(epochSamples) : .infinity
            if avgLoss < bestLoss {
                bestLoss = avgLoss
                bestWeights = LoRAInjector.extractWeights(model: model, paths: injectedPaths)
            }

            // Yield at epoch boundary so the enclosing actor can process
            // pending messages (e.g. cancelTraining()) between epochs.
            await Task.yield()
        }

        setTrainingMode(false)
        return bestWeights ?? LoRAInjector.extractWeights(model: model, paths: injectedPaths)
    }

    /// Request cancellation. Training stops after the current sample completes.
    public func cancel() {
        isCancelled = true
    }

    /// Whether at least one complete epoch has finished (enough for a usable checkpoint).
    public var hasUsableCheckpoint: Bool {
        bestWeights != nil
    }

    /// Lowest loss seen, or nil if no complete epoch.
    public var lowestLoss: Float? {
        bestWeights != nil ? bestLoss : nil
    }

    /// Create a checkpoint of the current training state.
    public func makeCheckpoint(
        weightsURL: URL,
        optimizerStateURL: URL,
        baseModelID: String
    ) -> TrainingCheckpoint {
        TrainingCheckpoint(
            epoch: currentEpoch,
            globalStep: globalStep,
            totalSteps: totalSteps,
            bestLoss: bestWeights != nil ? bestLoss : nil,
            lossHistory: lossHistory,
            weightsURL: weightsURL,
            optimizerStateURL: optimizerStateURL,
            config: config,
            baseModelID: baseModelID
        )
    }

    /// Current loss history (for checkpoint saving from outside).
    public var currentLossHistory: [Float] { lossHistory }

    /// Current global step.
    public var currentGlobalStep: Int { globalStep }

    // MARK: - Private

    static func optimizerStepsPerEpoch(
        datasetSize: Int,
        gradientAccumulationSteps: Int
    ) -> Int {
        let sampleCount = max(1, datasetSize)
        let requestedAccumulation = max(1, gradientAccumulationSteps)
        let accumulation = min(requestedAccumulation, sampleCount)
        return max(1, (sampleCount + accumulation - 1) / accumulation)
    }

    private static func addFlatGradients(
        _ lhs: [MLXArray],
        _ rhs: [MLXArray]
    ) -> [MLXArray] {
        zip(lhs, rhs).map { pair in pair.0 + pair.1 }
    }

    private func updateOptimizerForAccumulationWindow(
        optimizer: AdamW,
        trainableParameterKeys: [String],
        accumulatedFlatGrads: [MLXArray],
        windowLoss: Float,
        windowSamples: Int,
        epoch: Int,
        startTime: Date,
        onProgress: @escaping @Sendable (TrainingProgress) -> Void
    ) {
        let currentLR = scheduler.learningRate(atStep: globalStep)
        optimizer.learningRate = currentLR

        let scale = 1.0 / Float(max(1, windowSamples))
        let averagedFlatGrads = accumulatedFlatGrads.map { $0 * scale }
        let gradPairs = zip(trainableParameterKeys, averagedFlatGrads).map { ($0.0, $0.1) }
        optimizer.update(model: model, gradients: ModuleParameters.unflattened(gradPairs))
        eval(model.trainableParameters(), optimizer.innerState())

        globalStep += 1

        let avgWindowLoss = windowLoss / Float(max(1, windowSamples))
        lossHistory.append(avgWindowLoss)

        let elapsed = Date().timeIntervalSince(startTime)
        let stepsRemaining = max(0, totalSteps - globalStep)
        let eta: TimeInterval? = globalStep > 0
            ? elapsed / Double(globalStep) * Double(stepsRemaining)
            : nil

        onProgress(TrainingProgress(
            epoch: epoch,
            totalEpochs: config.maxEpochs,
            step: globalStep,
            loss: avgWindowLoss,
            learningRate: currentLR,
            estimatedTimeRemaining: eta,
            lossHistory: lossHistory,
            elapsedTime: elapsed
        ))
    }

    private func setTrainingMode(_ training: Bool) {
        for path in injectedPaths {
            guard let (layerIdx, subPath) = parsePath(path),
                  layerIdx < model.layers.count else { continue }

            let layer = model.layers[layerIdx]
            if let loraLinear = resolveLoRALinear(in: layer, path: subPath) {
                loraLinear.isTraining = training
            }
        }
    }

    private func parsePath(_ path: String) -> (Int, String)? {
        let components = path.split(separator: ".")
        guard components.count >= 3,
              components[0] == "layers",
              let layerIdx = Int(components[1]) else { return nil }
        let subPath = components[2...].joined(separator: ".")
        return (layerIdx, subPath)
    }

    private func resolveLoRALinear(in layer: DiTLayer, path: String) -> LoRALinear? {
        let parts = path.split(separator: ".")
        guard parts.count == 2 else { return nil }
        let moduleName = String(parts[0])
        let projName = String(parts[1])

        let linear: Linear?
        switch moduleName {
        case "selfAttn":
            linear = resolveAttentionProj(layer.selfAttn, proj: projName)
        case "crossAttn":
            linear = resolveAttentionProj(layer.crossAttn, proj: projName)
        case "mlp":
            linear = resolveMLPProj(layer.mlp, proj: projName)
        default:
            linear = nil
        }

        return linear as? LoRALinear
    }

    private func resolveAttentionProj(_ attn: DiTAttention, proj: String) -> Linear? {
        switch proj {
        case "qProj": return attn.qProj
        case "kProj": return attn.kProj
        case "vProj": return attn.vProj
        case "oProj": return attn.oProj
        default: return nil
        }
    }

    private func resolveMLPProj(_ mlp: SwiGLUMLP, proj: String) -> Linear? {
        switch proj {
        case "gateProj": return mlp.gateProj
        case "upProj": return mlp.upProj
        case "downProj": return mlp.downProj
        default: return nil
        }
    }
}
