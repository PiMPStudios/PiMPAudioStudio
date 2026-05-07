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

        // Each epoch processes all samples; optimizer steps once per sample.
        // LR updates happen every gradientAccumulationSteps samples.
        // totalSteps = total LR update points across training.
        let updatesPerEpoch = max(1, datasetSize / max(1, config.gradientAccumulationSteps))
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
    public func train(
        dataset: [TrainingSample],
        onProgress: @escaping (TrainingProgress) -> Void
    ) -> [(String, MLXArray)] {
        guard !dataset.isEmpty else { return [] }

        let accumSteps = max(1, min(config.gradientAccumulationSteps, dataset.count))
        let optimizer = AdamW(learningRate: scheduler.learningRate(atStep: globalStep))
        let startTime = Date()

        setTrainingMode(true)

        // Build the valueAndGrad closure
        let lossGrad = valueAndGrad(model: model) {
            (model: DiTDecoder, args: [MLXArray]) -> [MLXArray] in
            let xt = args[0]
            let t = args[1]
            let encoderStates = args[2]
            let contextLatents = args[3]
            let vTarget = args[4]
            let encoderMask: MLXArray? = args.count > 5 ? args[5] : nil

            let (vPred, _) = model(
                hiddenStates: xt,
                timestep: t,
                timestepR: t,
                encoderHiddenStates: encoderStates,
                contextLatents: contextLatents,
                encoderAttentionMask: encoderMask,
                cache: nil,
                useCache: false
            )

            let diff = vPred - vTarget
            let loss = (diff * diff).mean()
            return [loss]
        }

        let startEpoch = currentEpoch + 1

        for epoch in startEpoch...config.maxEpochs {
            guard !isCancelled else { break }
            currentEpoch = epoch

            var epochLoss: Float = 0
            var epochSamples: Int = 0
            let shuffledIndices = Array(0..<dataset.count).shuffled()

            // Per-window tracking for accurate loss reporting
            var windowLoss: Float = 0
            var windowSamples: Int = 0

            for idx in shuffledIndices {
                guard !isCancelled else { break }
                let sample = dataset[idx]

                // Sample random timestep t in [0, 1]
                let t = MLXRandom.uniform(0..<1, [1])

                // Interpolate: x_t = (1 - t) * x_0 + t * noise
                let noise = MLXRandom.normal(sample.latent.shape)
                let xt = (1.0 - t) * sample.latent + t * noise

                // Target velocity: v_target = noise - x_0
                let vTarget = noise - sample.latent

                // Build args array
                var args = [xt, t, sample.encoderStates, sample.contextLatents, vTarget]
                if let mask = sample.encoderMask {
                    args.append(mask)
                }

                // Forward + backward + step (per-sample optimizer update)
                let (lossValues, grads) = lossGrad(model, args)
                let loss = lossValues[0]
                let lossValue = loss.item(Float.self)

                optimizer.update(model: model, gradients: grads)
                eval(model.parameters())

                epochLoss += lossValue
                epochSamples += 1
                windowLoss += lossValue
                windowSamples += 1

                // Update LR and report progress every accumSteps samples
                if windowSamples >= accumSteps {
                    let currentLR = scheduler.learningRate(atStep: globalStep)
                    optimizer.learningRate = currentLR
                    globalStep += 1

                    // Record per-window average loss (not running epoch average)
                    let avgWindowLoss = windowLoss / Float(windowSamples)
                    lossHistory.append(avgWindowLoss)

                    // Report progress
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

                    // Reset window for next accumulation period
                    windowLoss = 0
                    windowSamples = 0
                }
            }

            // Partial window at epoch end: discard remaining samples without
            // incrementing globalStep (no optimizer step was taken for this window).

            // Track best weights (per epoch)
            let avgLoss = epochSamples > 0 ? epochLoss / Float(epochSamples) : .infinity
            if avgLoss < bestLoss {
                bestLoss = avgLoss
                bestWeights = LoRAInjector.extractWeights(model: model, paths: injectedPaths)
            }
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
