// AnvilCore/Sources/AnvilCore/Inference/StableAudio/SFXDiTActor.swift
import Foundation
import MLX
import MLXNN

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

    public func predict(latents: MLXArray, timestep: MLXArray, conditioning: MLXArray) -> MLXArray {
        let result = model.forward(latents: latents, timestep: timestep, conditioning: conditioning)
        eval(result)
        return result
    }
}
