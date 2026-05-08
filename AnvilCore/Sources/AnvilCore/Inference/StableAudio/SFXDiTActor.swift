// AnvilCore/Sources/AnvilCore/Inference/StableAudio/SFXDiTActor.swift
import Foundation
import MLX
import MLXNN

public enum SFXDiTActorError: Error, LocalizedError, Sendable {
    case weightsNotLoaded

    public var errorDescription: String? {
        "SFX DiT model weights are not loaded."
    }
}

public actor SFXDiTActor {
    private let model: SFXDiTModel
    private var loaded = false

    public init(config: SFXDiTConfig) {
        self.model = SFXDiTModel(config: config)
    }

    public var isLoaded: Bool { loaded }

    public func loadWeights(_ weights: [(String, MLXArray)]) throws {
        model.update(parameters: ModuleParameters.unflattened(weights))
        eval(model.parameters())
        loaded = true
    }

    public func predict(
        latents: MLXArray,
        timestep: MLXArray,
        crossCond: MLXArray,
        globalCond: MLXArray
    ) throws -> MLXArray {
        guard loaded else { throw SFXDiTActorError.weightsNotLoaded }
        let result = model.forward(
            latents: latents,
            timestep: timestep,
            crossCond: crossCond,
            globalCond: globalCond
        )
        eval(result)
        return result
    }

    /// Marks this actor as unloaded. Callers will receive SFXDiTActorError.weightsNotLoaded
    /// until loadWeights() is called again.
    /// Note: the underlying MLX model object is retained (it is a `let` constant); actual
    /// GPU memory is freed when SFXModelManager.unloadPipeline() releases the pipeline.
    public func unloadWeights() {
        loaded = false
    }
}
