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
    /// Counts `predict` calls within a generation; reset on each new generation
    /// via `resetDiagCounter()`.  DiT internal probes fire only for the first
    /// two calls (step 0 cond + step 0 uncond) to avoid flooding the console.
    private var diagCallCount = 0

    public init(config: SFXDiTConfig) {
        self.model = SFXDiTModel(config: config)
    }

    public var isLoaded: Bool { loaded }

    public func loadWeights(_ weights: [(String, MLXArray)]) throws {
        let suppliedKeys = Set(weights.map { $0.0 })
        model.update(parameters: ModuleParameters.unflattened(weights))
        eval(model.parameters())

        // Coverage diagnostic: count how many supplied keys were actually absorbed
        // by the model (i.e. their paths exist in the flattened parameter tree).
        let modelKeys = Set(model.parameters().flattened().map { $0.key })
        let matched   = suppliedKeys.intersection(modelKeys).count
        let supplied  = suppliedKeys.count
        let inModel   = modelKeys.count
        print("[SFXDiT] loadWeights: supplied=\(supplied) model=\(inModel) matched=\(matched)")
        if matched < supplied {
            print("[SFXDiT] WARNING: \(supplied - matched) supplied key(s) not matched in model")
        }

        loaded = true
    }

    /// Call once before each generation to reset the internal-diag call counter.
    public func resetDiagCounter() {
        diagCallCount = 0
    }

    public func predict(
        latents: MLXArray,
        timestep: MLXArray,
        crossCond: MLXArray,
        globalCond: MLXArray
    ) throws -> MLXArray {
        guard loaded else { throw SFXDiTActorError.weightsNotLoaded }
        // Emit DiT-internal probes only on the first two calls per generation
        // (cond + uncond for step 0).  After that diagEnabled=false keeps it quiet.
        let shouldDiag = diagCallCount < 2
        diagCallCount += 1
        let result = model.forward(
            latents: latents,
            timestep: timestep,
            crossCond: crossCond,
            globalCond: globalCond,
            diagEnabled: shouldDiag
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
