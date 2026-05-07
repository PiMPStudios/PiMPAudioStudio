import Foundation
import MLX
import MLXFast
import MLXNN

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

    public func decode(latents: MLXArray, precision: Precision) throws -> MLXArray {
        guard let vae else { throw VAEActorError.weightsNotLoaded }
        let input = precision == .float16 ? latents.asType(.float16) : latents
        let audio = VAETiledDecode.decode(input, using: vae)
        return precision == .float16 ? audio.asType(.float32) : audio
    }

    public func encodeSample(audio: MLXArray, precision: Precision) throws -> MLXArray {
        guard let vae else { throw VAEActorError.weightsNotLoaded }
        let input = precision == .float16 ? audio.asType(.float16) : audio
        let latents = vae.encodeSample(input)
        return precision == .float16 ? latents.asType(.float32) : latents
    }

    public func encodeMean(audio: MLXArray, precision: Precision) throws -> MLXArray {
        guard let vae else { throw VAEActorError.weightsNotLoaded }
        let input = precision == .float16 ? audio.asType(.float16) : audio
        let latents = vae.encodeMean(input)
        return precision == .float16 ? latents.asType(.float32) : latents
    }

    public func loadWeights(_ weights: [(String, MLXArray)], modelID: String) {
        if vae == nil {
            vae = AutoEncoderOobleck()
        }
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
