// AnvilCore/Sources/AnvilCore/Inference/StableAudio/SFXVAEActor.swift
import Foundation
import MLX
import MLXNN

public enum SFXVAEActorError: Error, LocalizedError, Sendable {
    case weightsNotLoaded

    public var errorDescription: String? {
        "SFX VAE model weights are not loaded."
    }
}

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

    public func decode(latents: MLXArray) throws -> MLXArray {
        guard loaded else { throw SFXVAEActorError.weightsNotLoaded }
        AudioDiagnostics.tensor("SFX_VAE_IN", latents)
        let result = model.decode(latents: latents)
        eval(result)
        AudioDiagnostics.tensor("SFX_VAE_OUT", result)
        // Log per-channel mean (left / right) to catch stereo imbalance or DC offset.
        if result.ndim >= 3 {
            let ch0 = result[0..., 0..., 0]
            let ch1 = result[0..., 0..., 1]
            let dc0 = ch0.mean().item(Float.self)
            let dc1 = ch1.mean().item(Float.self)
            AudioDiagnostics.note("SFX_VAE_DC", "ch0_mean=\(String(format:"%.4f",dc0)) ch1_mean=\(String(format:"%.4f",dc1))")
        }
        return result
    }

    /// See SFXDiTActor.unloadWeights() for the same note about memory.
    public func unloadWeights() {
        loaded = false
    }
}
