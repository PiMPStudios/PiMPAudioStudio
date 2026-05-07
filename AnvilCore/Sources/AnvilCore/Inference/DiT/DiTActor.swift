import Foundation
import MLX
import MLXNN

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
