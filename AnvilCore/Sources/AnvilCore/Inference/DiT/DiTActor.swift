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
    /// Keep on-device LoRA training inside a memory envelope that can survive
    /// local backprop. Late DiT blocks carry style/timbre detail and are the
    /// safest first target for local adapter training.
    private static let maxLoRATrainingLayers = 4

    private var decoder: DiTDecoder?
    private var decoderConfig: DiTConfig
    public private(set) var loadedModelID: String?

    public init(decoder: DiTDecoder) {
        self.decoder = decoder
        self.decoderConfig = .v15Base
    }

    public init(config: DiTConfig = .v15Base) {
        self.decoder = DiTDecoder(config: config)
        self.decoderConfig = config
    }

    public func configure(config: DiTConfig) {
        guard config != decoderConfig else { return }
        decoderConfig = config
        decoder = nil
        loadedModelID = nil
        injectedPaths = []
        injectedAdapterID = nil
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
            decoder = DiTDecoder(config: decoderConfig)
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

    /// Add a single adapter's weights as a `LoRAContribution` on every injected
    /// LoRALinear layer.  Call after `injectLoRA` — each call appends one entry
    /// to `loraLinear.contributions`; the forward pass sums all entries with
    /// their individual `strength` multipliers.
    public func addLoRAContribution(
        adapterID: String,
        displayName: String?,
        source: AdapterSource,
        weights: [(String, MLXArray)],
        strength: Float
    ) {
        guard let decoder else { return }
        LoRAInjector.addContribution(
            model: decoder,
            adapterID: adapterID,
            displayName: displayName,
            source: source,
            weights: weights,
            strength: strength,
            paths: injectedPaths
        )
        eval(decoder.parameters())
    }

    /// Remove all `LoRAContribution` entries from every injected LoRALinear layer,
    /// reverting the model to pure base-weight behaviour without removing the
    /// LoRALinear wrappers themselves.
    public func clearLoRAContributions() {
        guard let decoder else { return }
        LoRAInjector.clearContributions(model: decoder, paths: injectedPaths)
    }

    /// Run LoRA training on the decoder.
    ///
    /// Making this `async` allows `await Task.yield()` calls inside `LoRATrainer.train`
    /// to release the actor between epochs, letting `cancelTraining()` messages be
    /// processed cooperatively rather than queued until training completes.
    public func trainLoRA(
        samples: [TrainingSample],
        config: LoRAConfig,
        resumeFrom checkpoint: TrainingCheckpoint? = nil,
        initialWeights: [(String, MLXArray)]? = nil,
        onProgress: @escaping @Sendable (TrainingProgress) -> Void
    ) async throws -> (weights: [(String, MLXArray)], finalLoss: Float?) {
        guard let decoder else {
            throw DiTActorError.weightsNotLoaded
        }
        // Inject first so the module tree contains the LoRA wrappers, then
        // freeze the full DiT and unfreeze only the tiny adapter matrices.
        let trainableLayerIndices = Self.trainingLayerIndices(for: decoder.layers.count)
        let paths = LoRAInjector.inject(
            model: decoder,
            config: config,
            layerIndices: trainableLayerIndices
        )
        decoder.freeze()
        LoRAInjector.unfreezeAdapters(model: decoder, paths: paths)
        let trainableCount = decoder.trainableParameters().flattened().count
        print(
            "[DiTActor.trainLoRA] injectedLoRALayers=\(paths.count) trainableParams=\(trainableCount) layers=\(trainableLayerIndices.sorted())"
        )
        if let initialWeights {
            print("[DiTActor.trainLoRA] loading initialWeights count=\(initialWeights.count)")
            LoRAInjector.loadWeights(model: decoder, paths: paths, weights: initialWeights)
            print("[DiTActor.trainLoRA] loaded initialWeights")
        } else {
            print("[DiTActor.trainLoRA] no initialWeights")
        }
        print("[DiTActor.trainLoRA] creating trainer")
        let trainer = LoRATrainer(
            model: decoder,
            config: config,
            injectedPaths: paths,
            datasetSize: samples.count,
            resumeFrom: checkpoint
        )
        print("[DiTActor.trainLoRA] trainer created")
        activeTrainer = trainer
        print("[DiTActor.trainLoRA] starting trainer.train samples=\(samples.count)")
        defer {
            activeTrainer = nil
            LoRAInjector.remove(model: decoder, paths: paths)
            decoder.unfreeze()
        }

        let weights = try await trainer.train(dataset: samples, onProgress: onProgress)
        return (weights, trainer.lowestLoss)
    }

    private static func trainingLayerIndices(for layerCount: Int) -> Set<Int> {
        guard layerCount > maxLoRATrainingLayers else {
            return Set(0..<layerCount)
        }

        return Set((layerCount - maxLoRATrainingLayers)..<layerCount)
    }

    /// Cancel ongoing training.
    public func cancelTraining() {
        activeTrainer?.cancel()
    }
}
