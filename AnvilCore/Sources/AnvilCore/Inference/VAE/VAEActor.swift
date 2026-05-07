import Foundation
import MLX
import MLXFast
import MLXNN
import OSLog

private let vaeLogger = Logger(subsystem: "com.pimp.AnvilApp", category: "VAEActor")

public enum VAEActorError: Error, LocalizedError, Sendable {
    case weightsNotLoaded

    public var errorDescription: String? {
        switch self {
        case .weightsNotLoaded: return "VAE weights have not been loaded."
        }
    }
}

public actor VAEActor {
    private var vae: AutoEncoderOobleck?
    public private(set) var loadedModelID: String?

    public init(vae: AutoEncoderOobleck = AutoEncoderOobleck()) {
        self.vae = vae
    }

    // Latent normalization constants from upstream
    // (acestep/music_dcae/music_dcae_pipeline.py:59-60). The diffusion model
    // operates in a normalized latent space; the trained VAE decoder expects
    // the original (un-normalized) latents. Encode applies the forward
    // transform; decode reverses it.
    //
    //   forward:  z_norm = (z_raw - shift) * scale
    //   inverse:  z_raw  = z_norm / scale + shift
    private static let vaeScaleFactor: Float = 0.1786
    private static let vaeShiftFactor: Float = -1.9091

    public func decode(latents: MLXArray, precision: Precision) throws -> MLXArray {
        guard let vae else { throw VAEActorError.weightsNotLoaded }
        AudioDiagnostics.tensor("vae/in_latents(normalized)", latents)
        // Reverse the diffusion-space normalization before feeding the
        // trained VAE decoder.
        let denormalized = latents / Self.vaeScaleFactor + Self.vaeShiftFactor
        AudioDiagnostics.tensor("vae/in_latents(denormalized)", denormalized)
        let input = precision == .float16 ? denormalized.asType(.float16) : denormalized
        let audio = VAETiledDecode.decode(input, using: vae)
        let resolved = precision == .float16 ? audio.asType(.float32) : audio
        AudioDiagnostics.tensor("vae/out_audio", resolved)

        let absA = MLX.abs(resolved)
        let absMean = absA.mean().item(Float.self)
        let absMax = absA.max().item(Float.self)
        vaeLogger.info("[VAEActor.decode] in shape=\(latents.shape) → out shape=\(resolved.shape) abs_mean=\(absMean) abs_max=\(absMax)")

        return resolved
    }

    public func encodeSample(audio: MLXArray, precision: Precision) throws -> MLXArray {
        guard let vae else { throw VAEActorError.weightsNotLoaded }
        let input = precision == .float16 ? audio.asType(.float16) : audio
        let rawLatents = vae.encodeSample(input)
        // Apply diffusion-space normalization so downstream code sees the
        // same distribution the DiT was trained on.
        let normalized = (rawLatents - Self.vaeShiftFactor) * Self.vaeScaleFactor
        return precision == .float16 ? normalized.asType(.float32) : normalized
    }

    public func encodeMean(audio: MLXArray, precision: Precision) throws -> MLXArray {
        guard let vae else { throw VAEActorError.weightsNotLoaded }
        let input = precision == .float16 ? audio.asType(.float16) : audio
        let rawLatents = vae.encodeMean(input)
        let normalized = (rawLatents - Self.vaeShiftFactor) * Self.vaeScaleFactor
        return precision == .float16 ? normalized.asType(.float32) : normalized
    }

    public func loadWeights(_ weights: [(String, MLXArray)], modelID: String) {
        if vae == nil {
            vae = AutoEncoderOobleck()
        }

        // Diagnostic: same matched/unmatched key check as DiTActor.
        let modelKeys = Set(vae!.parameters().flattened().map { $0.0 })
        let suppliedKeys = Set(weights.map { $0.0 })
        let matched = suppliedKeys.intersection(modelKeys).count
        let supExtra = suppliedKeys.subtracting(modelKeys)
        let modExtra = modelKeys.subtracting(suppliedKeys)
        vaeLogger.info("[VAEActor.loadWeights] supplied=\(suppliedKeys.count) model=\(modelKeys.count) matched=\(matched)")
        vaeLogger.info("[VAEActor.loadWeights] supplied-but-no-model-param (first 12): \(Array(supExtra.sorted().prefix(12)))")
        vaeLogger.info("[VAEActor.loadWeights] model-param-but-no-supplied (first 12): \(Array(modExtra.sorted().prefix(12)))")

        vae!.update(parameters: ModuleParameters.unflattened(weights))
        eval(vae!.parameters())
        loadedModelID = modelID
    }

    /// Legacy overload for callers that don't track model ID yet.
    public func loadWeights(_ weights: [(String, MLXArray)]) {
        loadWeights(weights, modelID: "")
    }

    public func unloadWeights() {
        vae = nil
        loadedModelID = nil
    }
}
