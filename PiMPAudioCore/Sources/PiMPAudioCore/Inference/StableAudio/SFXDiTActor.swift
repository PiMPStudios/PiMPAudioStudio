// PiMPAudioCore/Sources/PiMPAudioCore/Inference/StableAudio/SFXDiTActor.swift
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
        model = SFXDiTModel(config: config)
    }

    public var isLoaded: Bool {
        loaded
    }

    public func loadWeights(_ weights: [(String, MLXArray)]) throws {
        let suppliedKeys = Set(weights.map(\.0))
        model.update(parameters: ModuleParameters.unflattened(weights))
        eval(model.parameters())

        // Coverage diagnostic: count how many supplied keys were actually absorbed
        // by the model (i.e. their paths exist in the flattened parameter tree).
        let modelKeys = Set(model.parameters().flattened().map(\.0))
        let matched = suppliedKeys.intersection(modelKeys).count
        let supplied = suppliedKeys.count
        let inModel = modelKeys.count
        AudioDiagnostics.note(
            "SFX_DIT_LOAD",
            "supplied=\(supplied) model=\(inModel) matched=\(matched)"
        )
        if matched < supplied {
            AudioDiagnostics.note(
                "SFX_DIT_LOAD",
                "WARNING: \(supplied - matched) supplied key(s) not matched in model"
            )
        }
        let unloadedModelKeys = modelKeys.subtracting(suppliedKeys).sorted()
        if !unloadedModelKeys.isEmpty {
            AudioDiagnostics.note(
                "SFX_DIT_LOAD",
                "WARNING: \(unloadedModelKeys.count) model param(s) got no weight — staying at init"
            )
            for k in unloadedModelKeys.prefix(20) {
                AudioDiagnostics.note("SFX_DIT_LOAD", "MISSING: \(k)")
            }
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
        let shouldDiag = AudioDiagnostics.isEnabled && diagCallCount < 2
        diagCallCount += 1
        let result = model.forward(
            latents: latents,
            timestep: timestep,
            crossCond: crossCond,
            globalCond: globalCond,
            diagEnabled: shouldDiag
        )
        eval(result)
        if shouldDiag {
            AudioDiagnostics.tensor("sfx/dit/predict\(diagCallCount - 1)/latents", latents)
            AudioDiagnostics.tensor("sfx/dit/predict\(diagCallCount - 1)/vt", result)
        }
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
