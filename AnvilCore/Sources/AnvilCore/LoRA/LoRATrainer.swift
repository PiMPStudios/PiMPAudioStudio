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
    public let loss: Float
    public let elapsedTime: TimeInterval
    public let estimatedRemaining: TimeInterval
}

/// Flow-matching training loop for LoRA adapters.
public final class LoRATrainer {
    private let model: DiTDecoder
    private let config: LoRAConfig
    private let injectedPaths: [String]
    private var bestLoss: Float = .infinity
    private var bestWeights: [(String, MLXArray)]?
    private var isCancelled: Bool = false

    public init(model: DiTDecoder, config: LoRAConfig, injectedPaths: [String]) {
        self.model = model
        self.config = config
        self.injectedPaths = injectedPaths
    }

    /// Run the training loop. Returns the best LoRA weights (lowest loss across epochs).
    ///
    /// - Parameters:
    ///   - dataset: Training samples (latents + conditioning).
    ///   - onProgress: Callback invoked after each epoch with training metrics.
    /// - Returns: Array of (key, weight) tuples for the best checkpoint.
    public func train(
        dataset: [TrainingSample],
        onProgress: @escaping (TrainingProgress) -> Void
    ) -> [(String, MLXArray)] {
        guard !dataset.isEmpty else { return [] }

        let optimizer = AdamW(learningRate: config.learningRate)
        let startTime = Date()

        setTrainingMode(true)

        // Build the valueAndGrad closure. Using the [MLXArray] variant:
        // args layout: [xt, t, encoderStates, contextLatents, vTarget, encoderMask?]
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

        for epoch in 1...config.epochs {
            guard !isCancelled else { break }

            var epochLoss: Float = 0
            let shuffledIndices = Array(0..<dataset.count).shuffled()

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

                // Forward + backward
                let (lossValues, grads) = lossGrad(model, args)
                let loss = lossValues[0]

                // Optimizer step
                optimizer.update(model: model, gradients: grads)
                eval(model.parameters())

                epochLoss += loss.item(Float.self)
            }

            let avgLoss = epochLoss / Float(dataset.count)

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
