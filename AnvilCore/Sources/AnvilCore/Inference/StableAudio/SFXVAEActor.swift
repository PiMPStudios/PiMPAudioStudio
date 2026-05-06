// AnvilCore/Sources/AnvilCore/Inference/StableAudio/SFXVAEActor.swift
import Foundation
import MLX
import MLXNN

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

    public func decode(latents: MLXArray) -> MLXArray {
        let result = model.decode(latents: latents)
        eval(result)
        return result
    }
}
