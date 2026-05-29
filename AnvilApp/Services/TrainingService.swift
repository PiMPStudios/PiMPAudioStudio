import Foundation
import AnvilCore
import MLX

/// Training state for UI observation.
enum TrainingState: Equatable {
    case idle
    case loadingModels
    case preparingDataset(current: Int, total: Int)
    case training(TrainingUIState)
    case saving
    case complete(LoRAAdapter)
    case failed(String)
    case cancelled
    case paused(TrainingCheckpoint)
}

struct TrainingUIState: Equatable {
    let epoch: Int
    let totalEpochs: Int
    let step: Int
    let loss: Float
    let learningRate: Float
    let elapsed: TimeInterval
    let remaining: TimeInterval?
    let lossHistory: [Float]
}

/// Result returned from the pipeline's training method (added in Task 10).
struct LoRATrainingResult: Sendable {
    let weights: [(String, MLXArray)]
    let finalLoss: Float?
}

/// Orchestrates LoRA training: dataset preparation, training execution, and adapter saving.
@MainActor
final class TrainingService: ObservableObject {
    @Published var state: TrainingState = .idle

    /// ACE-Step LoRA backprop stores DiT activations for every latent frame.
    /// Training on full 30-200s clips can exhaust unified memory, so each
    /// sample contributes a short random window while preserving the clip's
    /// caption/conditioning signal.
    private static let maxTrainingLatentFrames = 64

    private let pipeline: MusicPipeline
    private let vae: VAEActor
    private var conditioning: ConditioningActor
    /// `false` until ACEStepGenerationService swaps in the real ConditioningActor
    /// (real Qwen3 tokenizer + loaded weights). Training crashes on the stub because
    /// the SFT prompt template uses newlines outside the stub's ASCII-only vocab.
    private var isConditioningReady = false
    private let registry: ModelRegistry
    private var ensureModelsLoaded: (() async throws -> Void)?
    private var onTrainingFinished: (() async -> Void)?
    private var trainingTask: Task<Void, Never>?
    private var isStopping = false
    private var shouldSaveCheckpointOnStop = false
    private let stackManager: AdapterStackManager?
    private var savedStack: [(adapter: LoRAAdapter, strength: Float)] = []

    init(
        pipeline: MusicPipeline,
        vae: VAEActor,
        conditioning: ConditioningActor,
        registry: ModelRegistry,
        stackManager: AdapterStackManager? = nil
    ) {
        self.pipeline = pipeline
        self.vae = vae
        self.conditioning = conditioning
        self.registry = registry
        self.stackManager = stackManager
    }

    /// Start training with selected history entries and/or imported audio files.
    ///
    /// - Parameters:
    ///   - entries: History entries to use as training data.
    ///   - audioURLs: File URLs corresponding 1:1 to `entries`.
    ///   - importedFiles: Additional audio files with text descriptions and optional lyrics.
    ///   - adapterName: User-facing name for the resulting adapter.
    ///   - config: LoRA training configuration.
    func startTraining(
        entries: [HistoryEntry],
        audioURLs: [URL],
        importedFiles: [TrainingImportedFile],
        adapterName: String,
        config: LoRAConfig,
        resumeAdapterID: String? = nil
    ) {
        let effectiveResumeAdapterID: String?
        if let resumeAdapterID {
            effectiveResumeAdapterID = resumeAdapterID
        } else if case .paused = state {
            effectiveResumeAdapterID = Self.adapterID(from: adapterName)
        } else {
            effectiveResumeAdapterID = nil
        }

        trainingTask?.cancel()
        isStopping = false
        shouldSaveCheckpointOnStop = false

        trainingTask = Task { [weak self] in
            guard let self else { return }
            defer {
                if let sm = self.stackManager, !self.savedStack.isEmpty {
                    for entry in self.savedStack {
                        sm.addToStack(entry.adapter, strength: entry.strength)
                    }
                    self.savedStack = []
                }
                self.trainingTask = nil
                if let onTrainingFinished {
                    Task {
                        await onTrainingFinished()
                    }
                }
            }

            do {
                self.state = .loadingModels
                if let ensureModelsLoaded {
                    try await ensureModelsLoaded()
                }
                guard self.isConditioningReady else {
                    self.state = .failed("Could not load the ACE-Step models needed for training.")
                    return
                }

                // Clear stack during training (restore after)
                if let sm = self.stackManager, sm.isStacking {
                    self.savedStack = sm.additionalAdapters
                    sm.clearStack()
                }

                let activeModelID = await self.registry.activeModelID(
                    for: .aceStepDiT
                ) ?? "unknown"
                let sessionConfig: LoRAConfig
                let resumeCheckpoint: TrainingCheckpoint?
                let initialWeights: [(String, MLXArray)]?

                if let resumeAdapterID = effectiveResumeAdapterID {
                    guard let checkpoint = AdapterRegistry.loadCheckpoint(adapterID: resumeAdapterID) else {
                        self.state = .failed("No paused training checkpoint found for this adapter.")
                        return
                    }
                    guard checkpoint.baseModelID == activeModelID else {
                        self.state = .failed(
                            CheckpointError.modelMismatch(
                                expected: checkpoint.baseModelID,
                                actual: activeModelID
                            ).localizedDescription
                        )
                        return
                    }
                    resumeCheckpoint = checkpoint
                    sessionConfig = checkpoint.config
                    initialWeights = try await AdapterRegistry.loadWeights(adapterID: resumeAdapterID)
                } else {
                    resumeCheckpoint = nil
                    sessionConfig = config
                    initialWeights = nil
                }

                // 1. Prepare dataset
                let totalSamples = entries.count + importedFiles.count
                self.state = .preparingDataset(current: 0, total: totalSamples)

                var samples: [TrainingSample] = []

                for (idx, entry) in entries.enumerated() {
                    try Task.checkCancellation()
                    let url = audioURLs[idx]
                    let sample = try await self.preprocessSample(
                        audioURL: url,
                        prompt: entry.prompt,
                        lyrics: entry.lyrics,
                        duration: entry.duration
                    )
                    samples.append(sample)
                    self.state = .preparingDataset(current: idx + 1, total: totalSamples)
                }

                for (idx, imported) in importedFiles.enumerated() {
                    try Task.checkCancellation()
                    let sample = try await self.preprocessSample(
                        audioURL: imported.url,
                        prompt: imported.description,
                        lyrics: imported.lyrics,
                        duration: 0,
                        cache: imported.preprocessCache
                    )
                    samples.append(sample)
                    self.state = .preparingDataset(
                        current: entries.count + idx + 1,
                        total: totalSamples
                    )
                }

                guard !samples.isEmpty else {
                    self.state = .failed("No valid training samples")
                    return
                }

                // 2. Run training through pipeline (trainLoRA added in Task 10)
                try Task.checkCancellation()

                let result = try await self.pipeline.trainLoRA(
                    samples: samples,
                    config: sessionConfig,
                    resumeFrom: resumeCheckpoint,
                    initialWeights: initialWeights,
                    onProgress: { [weak self] progress in
                        Task { @MainActor in
                            self?.state = .training(TrainingUIState(
                                epoch: progress.epoch,
                                totalEpochs: progress.totalEpochs,
                                step: progress.step,
                                loss: progress.loss,
                                learningRate: progress.learningRate,
                                elapsed: progress.elapsedTime,
                                remaining: progress.estimatedTimeRemaining,
                                lossHistory: progress.lossHistory
                            ))
                        }
                    }
                )

                try Task.checkCancellation()

                let adapterID = effectiveResumeAdapterID ?? Self.adapterID(from: adapterName)
                let manifest = TrainingManifest(
                    historyEntryIDs: entries.map(\.id),
                    importedFiles: importedFiles
                )
                let adapter = LoRAAdapter(
                    id: adapterID,
                    name: adapterName,
                    config: sessionConfig,
                    baseModelID: activeModelID,
                    baseModelCategory: .aceStepDiT,
                    createdAt: Date(),
                    sampleCount: samples.count,
                    finalLoss: result.finalLoss
                )

                if self.isStopping {
                    if self.shouldSaveCheckpointOnStop {
                        let checkpoint = self.makeCheckpoint(
                            adapterID: adapterID,
                            config: sessionConfig,
                            activeModelID: activeModelID,
                            sampleCount: samples.count,
                            fallback: resumeCheckpoint
                        )
                        try AdapterRegistry.save(adapter: adapter, weights: result.weights)
                        AdapterRegistry.saveManifest(manifest, adapterID: adapterID)
                        try AdapterRegistry.saveCheckpoint(checkpoint, adapterID: adapterID)
                        self.state = .paused(checkpoint)
                    } else {
                        self.state = .cancelled
                    }
                    return
                }

                // 3. Save adapter
                self.state = .saving

                try AdapterRegistry.save(adapter: adapter, weights: result.weights)

                // Persist the training manifest so the session can be reloaded
                // later for retraining with the same (or expanded) dataset.
                AdapterRegistry.saveManifest(manifest, adapterID: adapterID)
                AdapterRegistry.deleteCheckpoint(adapterID: adapterID)

                self.state = .complete(adapter)

            } catch is CancellationError {
                self.state = .cancelled
            } catch {
                self.state = .failed(error.localizedDescription)
            }
        }
    }

    /// Swap in a fully-loaded ConditioningActor (real tokenizer + model weights).
    /// Called by AppServices when ACEStepGenerationService upgrades from the stub.
    func updateConditioning(_ actor: ConditioningActor) {
        conditioning = actor
        isConditioningReady = true
    }

    /// Provides the same ACE-Step lazy-loading path used by generation, so
    /// training can start cold from launch without a dummy generation first.
    func setModelLoader(_ loader: @escaping () async throws -> Void) {
        ensureModelsLoaded = loader
    }

    /// Called after any training attempt finishes so app-level services can
    /// restore lightweight helpers that were evicted for memory headroom.
    func setTrainingFinishedHandler(_ handler: @escaping () async -> Void) {
        onTrainingFinished = handler
    }

    /// Reset to idle so the training sheet can reconfigure and re-run with the same dataset.
    func resetToIdle() {
        guard case .complete = state else { return }
        state = .idle
    }

    /// Stop training. When requested, persists the current LoRA weights and checkpoint.
    func stopTraining(saveCheckpoint: Bool = false) {
        isStopping = true
        shouldSaveCheckpointOnStop = saveCheckpoint
        Task {
            await pipeline.cancelTraining()
        }
    }

    /// Whether training can produce a persisted pause checkpoint on stop.
    var canSaveOnStop: Bool {
        if case .training(let uiState) = state {
            return uiState.step > 0
        }
        return false
    }

    func activeBaseModelDisplayName() async -> String {
        guard let spec = await registry.activeSpec(for: .aceStepDiT) else {
            return "ACE-Step DiT (not selected)"
        }
        return Self.baseModelDisplayName(for: spec)
    }

    nonisolated static func baseModelDisplayName(for spec: ModelSpec) -> String {
        let fingerprint = [
            spec.id,
            spec.displayName,
            spec.huggingFaceRepo,
            spec.variantFamily ?? "",
        ]
        .joined(separator: " ")
        .lowercased()

        if fingerprint.contains("xl") {
            return fingerprint.contains("sft") ? "ACE-Step XL SFT" : "ACE-Step XL"
        }
        if fingerprint.contains("sft") {
            return "ACE-Step SFT"
        }
        if fingerprint.contains("turbo") {
            return "ACE-Step Turbo"
        }
        if fingerprint.contains("base") || fingerprint.contains("full") {
            return "ACE-Step Full"
        }
        return spec.displayName
    }

    // MARK: - Private

    private static func adapterID(from adapterName: String) -> String {
        adapterName
            .lowercased()
            .replacingOccurrences(of: " ", with: "-")
            .replacingOccurrences(
                of: "[^a-z0-9\\-]",
                with: "",
                options: .regularExpression
            )
    }

    private func makeCheckpoint(
        adapterID: String,
        config: LoRAConfig,
        activeModelID: String,
        sampleCount: Int,
        fallback: TrainingCheckpoint?
    ) -> TrainingCheckpoint {
        let uiState: TrainingUIState?
        if case .training(let current) = state {
            uiState = current
        } else {
            uiState = nil
        }

        let effectiveSampleCount = max(1, sampleCount)
        let accumulation = min(max(1, config.gradientAccumulationSteps), effectiveSampleCount)
        let updatesPerEpoch = max(1, (effectiveSampleCount + accumulation - 1) / accumulation)
        let totalSteps = updatesPerEpoch * config.maxEpochs
        let lossHistory = uiState?.lossHistory ?? fallback?.lossHistory ?? []

        return TrainingCheckpoint(
            epoch: uiState?.epoch ?? fallback?.epoch ?? 0,
            globalStep: uiState?.step ?? fallback?.globalStep ?? 0,
            totalSteps: totalSteps,
            bestLoss: lossHistory.min() ?? fallback?.bestLoss,
            lossHistory: lossHistory,
            weightsURL: AdapterRegistry.weightsURL(adapterID: adapterID),
            optimizerStateURL: AdapterRegistry.optimizerStateURL(adapterID: adapterID),
            config: config,
            baseModelID: activeModelID
        )
    }

    private func preprocessSample(
        audioURL: URL,
        prompt: String,
        lyrics: String,
        duration: TimeInterval,
        cache: TrainingPreprocessCacheReference? = nil
    ) async throws -> TrainingSample {
        if let cache,
           let cachedSample = try? TrainingPreprocessCacheStore.load(reference: cache) {
            return cachedSample
        }

        // Load + resample to 48 kHz stereo → [1, pcmFrames, 2] NLC layout.
        // AudioFileLoader replaces the previous hand-rolled WAV parser which:
        //   (a) produced wrong shape [1, 1, N] instead of [1, frames, 2], and
        //   (b) didn't resample imported files to the VAE's 48 kHz target rate.
        let audioNLC = try AudioFileLoader.load(from: audioURL)

        // VAE-encode to latent space [1, latentFrames, 64]
        let latent = try await vae.encodeMean(audio: audioNLC, precision: .float16)

        // Derive duration from the actual PCM frame count when not provided
        // (imported files pass duration = 0).  audioNLC.shape[1] is PCM frames
        // at 48 kHz after resampling, so duration = frames / 48000.
        let actualDuration: TimeInterval = duration > 0
            ? duration
            : Double(audioNLC.shape[1]) / Double(MusicPipeline.defaultSampleRate)

        let cond = try await conditioning.encode(
            text: prompt,
            lyrics: lyrics,
            referenceAudio: nil,
            duration: actualDuration,
            precision: .float16
        )

        let (windowedLatent, windowedContextLatents) = Self.trainingWindow(
            latent: latent,
            contextLatents: cond.contextLatents
        )

        let frozenLatent = stopGradient(windowedLatent)
        let frozenEncoderStates = stopGradient(cond.encoderStates)
        let frozenContextLatents = stopGradient(windowedContextLatents)
        let frozenEncoderMask = stopGradient(cond.encoderMask)

        var arraysToMaterialize = [
            frozenLatent,
            frozenEncoderStates,
            frozenContextLatents
        ]
        arraysToMaterialize.append(frozenEncoderMask)
        eval(arraysToMaterialize)

        let sample = TrainingSample(
            latent: frozenLatent,
            encoderStates: frozenEncoderStates,
            contextLatents: frozenContextLatents,
            encoderMask: frozenEncoderMask
        )

        if let cache {
            do {
                try TrainingPreprocessCacheStore.save(sample: sample, reference: cache)
                try TrainingDatasetPackageWriter.refreshPreprocessCacheManifest(at: cache.datasetDirectory)
            } catch {
                print("[TrainingService] failed to save preprocess cache: \(error.localizedDescription)")
            }
        }

        return sample
    }

    private static func trainingWindow(
        latent: MLXArray,
        contextLatents: MLXArray
    ) -> (latent: MLXArray, contextLatents: MLXArray) {
        let availableFrames = min(latent.shape[1], contextLatents.shape[1])
        guard availableFrames > 0 else {
            return (latent, contextLatents)
        }

        let frameCount = min(availableFrames, maxTrainingLatentFrames)
        let maxStart = max(0, availableFrames - frameCount)
        let start = maxStart > 0 ? Int.random(in: 0...maxStart) : 0
        let end = start + frameCount

        return (
            latent[0..., start..<end, 0...],
            contextLatents[0..., start..<end, 0...]
        )
    }
}

enum TrainingPreprocessCacheStore {
    static let version = "1.0"

    static func load(reference: TrainingPreprocessCacheReference) throws -> TrainingSample? {
        let didStartAccessing = reference.datasetDirectory.startAccessingSecurityScopedResource()
        defer {
            if didStartAccessing {
                reference.datasetDirectory.stopAccessingSecurityScopedResource()
            }
        }

        let sampleURL = reference.datasetDirectory.appendingPathComponent(reference.sampleFile)
        guard FileManager.default.fileExists(atPath: sampleURL.path(percentEncoded: false)) else {
            return nil
        }
        let arrays = try MLX.loadArrays(url: sampleURL)
        guard let latent = arrays["latent"],
              let encoderStates = arrays["encoder_states"],
              let contextLatents = arrays["context_latents"] else {
            return nil
        }
        return TrainingSample(
            latent: latent,
            encoderStates: encoderStates,
            contextLatents: contextLatents,
            encoderMask: arrays["encoder_mask"]
        )
    }

    static func save(sample: TrainingSample, reference: TrainingPreprocessCacheReference) throws {
        let didStartAccessing = reference.datasetDirectory.startAccessingSecurityScopedResource()
        defer {
            if didStartAccessing {
                reference.datasetDirectory.stopAccessingSecurityScopedResource()
            }
        }

        let sampleURL = reference.datasetDirectory.appendingPathComponent(reference.sampleFile)
        try FileManager.default.createDirectory(
            at: sampleURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        var arrays: [String: MLXArray] = [
            "latent": sample.latent,
            "encoder_states": sample.encoderStates,
            "context_latents": sample.contextLatents,
        ]
        if let encoderMask = sample.encoderMask {
            arrays["encoder_mask"] = encoderMask
        }

        try MLX.save(
            arrays: arrays,
            metadata: [
                "anvil_preprocess_sample_version": version,
                "bundle": reference.bundleFileName,
                "bundle_name": reference.bundleName,
                "clip_id": reference.clipID,
                "source_audio": reference.sourceAudioPath,
                "fingerprint": reference.fingerprint,
            ],
            url: sampleURL
        )
        try updateIndex(reference: reference, sample: sample)
    }

    private static func updateIndex(
        reference: TrainingPreprocessCacheReference,
        sample: TrainingSample
    ) throws {
        let indexURL = reference.datasetDirectory.appendingPathComponent(reference.indexFile)
        try FileManager.default.createDirectory(
            at: indexURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        var index = readJSONObject(from: indexURL) ?? [
            "anvil_preprocess_index_version": version,
            "bundle": reference.bundleFileName,
            "bundle_name": reference.bundleName,
            "cache_dir": reference.cacheDirectory,
            "entries": [],
        ]
        var entries = index["entries"] as? [[String: Any]] ?? []
        entries.removeAll {
            ($0["clip_id"] as? String) == reference.clipID ||
                ($0["fingerprint"] as? String) == reference.fingerprint
        }
        entries.append([
            "clip_id": reference.clipID,
            "source_audio": reference.sourceAudioPath,
            "file": reference.sampleFile,
            "fingerprint": reference.fingerprint,
            "created_at": ISO8601DateFormatter().string(from: Date()),
            "shapes": [
                "latent": sample.latent.shape,
                "encoder_states": sample.encoderStates.shape,
                "context_latents": sample.contextLatents.shape,
                "encoder_mask": sample.encoderMask?.shape ?? [],
            ],
        ])
        index["anvil_preprocess_index_version"] = version
        index["updated_at"] = ISO8601DateFormatter().string(from: Date())
        index["entries"] = entries.sorted {
            ($0["clip_id"] as? String ?? "") < ($1["clip_id"] as? String ?? "")
        }
        try writeJSONObject(index, to: indexURL)
    }

    private static func readJSONObject(from url: URL) -> [String: Any]? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    }

    private static func writeJSONObject(_ object: [String: Any], to url: URL) throws {
        let data = try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: url, options: .atomic)
    }
}

enum TrainingError: Error, LocalizedError {
    case noSamples

    var errorDescription: String? {
        switch self {
        case .noSamples:
            return "No valid training samples"
        }
    }
}
