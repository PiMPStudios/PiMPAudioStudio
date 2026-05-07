import Foundation
import MLX
import MLXNN
import OSLog

private let logger = Logger(subsystem: "com.pimp.AnvilApp", category: "DiTActor")

public enum DiTActorError: Error, LocalizedError, Sendable {
    case weightsNotLoaded

    public var errorDescription: String? {
        "DiT model weights are not loaded. Please wait for model loading to complete."
    }
}

public actor DiTActor {
    private var decoder: DiTDecoder?
    private let decoderFactory: () -> DiTDecoder
    public private(set) var loadedModelID: String?

    public init(decoder: DiTDecoder) {
        self.decoder = decoder
        self.decoderFactory = { DiTDecoder() }
    }

    public func step(
        noisyLatents: MLXArray,
        timestep: Float,
        timestepR: Float,
        encoderStates: MLXArray,
        contextLatents: MLXArray,
        encoderAttentionMask: MLXArray?,
        cache: CrossAttentionCache?,
        useCache: Bool
    ) throws -> (output: MLXArray, cache: CrossAttentionCache?) {
        guard let decoder else {
            throw DiTActorError.weightsNotLoaded
        }
        let batchSize = noisyLatents.shape[0]
        let tArray = MLXArray.full([batchSize], values: MLXArray(timestep))
        let trArray = MLXArray.full([batchSize], values: MLXArray(timestepR))

        let (output, returnedCache) = decoder(
            hiddenStates: noisyLatents,
            timestep: tArray,
            timestepR: trArray,
            encoderHiddenStates: encoderStates,
            contextLatents: contextLatents,
            encoderAttentionMask: encoderAttentionMask,
            cache: cache,
            useCache: useCache
        )
        eval(output)
        return (output, returnedCache)
    }

    public func loadWeights(_ weights: [(String, MLXArray)], modelID: String) {
        if decoder == nil {
            decoder = decoderFactory()
        }

        // Diagnostic: build the set of model parameter paths and the set
        // of supplied weight paths, then report how many supplied keys
        // actually correspond to a real parameter path.
        let modelParams = decoder!.parameters().flattened()
        let modelKeys = Set(modelParams.map { $0.0 })
        let suppliedKeys = Set(weights.map { $0.0 })
        let matched = suppliedKeys.intersection(modelKeys).count
        let onlyInSupplied = suppliedKeys.subtracting(modelKeys)
        let onlyInModel = modelKeys.subtracting(suppliedKeys)
        logger.info("[DiTActor] supplied=\(suppliedKeys.count) model=\(modelKeys.count) matched=\(matched)")
        logger.info("[DiTActor] supplied-but-no-model-param (first 12): \(Array(onlyInSupplied.sorted().prefix(12)))")
        logger.info("[DiTActor] model-param-but-no-supplied (first 12): \(Array(onlyInModel.sorted().prefix(12)))")

        decoder!.update(parameters: ModuleParameters.unflattened(weights))
        eval(decoder!.parameters())
        loadedModelID = modelID
    }

    /// Legacy overload for callers that don't track model ID yet.
    public func loadWeights(_ weights: [(String, MLXArray)]) {
        loadWeights(weights, modelID: "")
    }

    public func unloadWeights() {
        decoder = nil
        loadedModelID = nil
    }

    /// Returns the learned `null_condition_emb` (shape [1, 1, hiddenSize])
    /// after weights are loaded. Used by DiffusionActor for the CFG
    /// unconditional branch — broadcasting this token across the encoder
    /// length matches what the model saw during training. Returns nil when
    /// no DiT weights are loaded yet.
    public func nullConditionEmbedding() -> MLXArray? {
        return decoder?.nullConditionEmb
    }

    // MARK: - LoRA

    private var injectedPaths: [String] = []
    public private(set) var injectedAdapterID: String?
    private var activeTrainer: LoRATrainer?

    /// Inject LoRA adapter layers into the DiT decoder.
    public func injectLoRA(config: LoRAConfig) throws -> [String] {
        guard let decoder else {
            throw DiTActorError.weightsNotLoaded
        }
        let paths = LoRAInjector.inject(model: decoder, config: config)
        injectedPaths = paths
        return paths
    }

    /// Remove LoRA adapter, restoring plain Linear layers.
    public func removeLoRA() {
        guard let decoder else { return }
        LoRAInjector.remove(model: decoder, paths: injectedPaths)
        injectedPaths = []
        injectedAdapterID = nil
    }

    /// Load LoRA weights into already-injected layers.
    public func loadLoRAWeights(_ weights: [(String, MLXArray)], adapterID: String) throws {
        guard let decoder else {
            throw DiTActorError.weightsNotLoaded
        }
        LoRAInjector.loadWeights(model: decoder, paths: injectedPaths, weights: weights)
        eval(decoder.parameters())
        injectedAdapterID = adapterID
    }

    /// Run LoRA training on the decoder. Must be called ON the actor's executor.
    public func trainLoRA(
        samples: [TrainingSample],
        config: LoRAConfig,
        onProgress: @escaping (TrainingProgress) -> Void
    ) throws -> (weights: [(String, MLXArray)], finalLoss: Float?) {
        guard let decoder else {
            throw DiTActorError.weightsNotLoaded
        }
        // Inject, train, extract, remove
        let paths = LoRAInjector.inject(model: decoder, config: config)
        let trainer = LoRATrainer(model: decoder, config: config, injectedPaths: paths, datasetSize: samples.count)
        activeTrainer = trainer
        let weights = trainer.train(dataset: samples, onProgress: onProgress)
        let finalLoss = trainer.lowestLoss
        activeTrainer = nil
        LoRAInjector.remove(model: decoder, paths: paths)
        return (weights, finalLoss)
    }

    /// Cancel ongoing training.
    public func cancelTraining() {
        activeTrainer?.cancel()
    }
}
