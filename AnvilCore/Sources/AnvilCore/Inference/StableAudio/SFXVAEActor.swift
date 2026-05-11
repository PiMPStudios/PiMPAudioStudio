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
        let suppliedKeys = Set(weights.map { $0.0 })
        model.update(parameters: ModuleParameters.unflattened(weights))
        eval(model.parameters())

        let modelKeys = Set(model.parameters().flattened().map { $0.0 })
        let matched   = suppliedKeys.intersection(modelKeys).count
        let supplied  = suppliedKeys.count
        let inModel   = modelKeys.count
        print("[SFXVAE] loadWeights: supplied=\(supplied) model=\(inModel) matched=\(matched)")
        if matched < inModel {
            print("[SFXVAE] WARNING: \(inModel - matched) model parameter(s) not matched — VAE may be partially unloaded")
        }

        loaded = true
    }

    /// Encode audio `[B, T_audio, audioChannels]` → latent mean `[B, T_latent, latentDim]`.
    ///
    /// Used by `InpaintCoordinator` to produce SAO latents for inpainting.
    /// Requires VAE weights to be loaded (`isLoaded == true`).
    public func encodeMean(audio: MLXArray) throws -> MLXArray {
        guard loaded else { throw SFXVAEActorError.weightsNotLoaded }
        let result = model.encodeMean(audio: audio)
        eval(result)
        return result
    }

    public func decode(latents: MLXArray) throws -> MLXArray {
        guard loaded else { throw SFXVAEActorError.weightsNotLoaded }
        AudioDiagnostics.tensor("SFX_VAE_IN", latents)

        // Unconditional diagnostics — always printed.
        let inAbsMean = abs(latents).mean().item(Float.self)
        let inAbsMax  = abs(latents).max().item(Float.self)
        print("[SFXVAE] decode_in  abs_mean=\(String(format:"%.4f",inAbsMean)) abs_max=\(String(format:"%.4f",inAbsMax)) shape=\(latents.shape)")

        let result = model.decode(latents: latents)
        eval(result)
        AudioDiagnostics.tensor("SFX_VAE_OUT", result)

        let outAbsMean = abs(result).mean().item(Float.self)
        let outAbsMax  = abs(result).max().item(Float.self)
        print("[SFXVAE] decode_out abs_mean=\(String(format:"%.4f",outAbsMean)) abs_max=\(String(format:"%.4f",outAbsMax)) shape=\(result.shape)")
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
