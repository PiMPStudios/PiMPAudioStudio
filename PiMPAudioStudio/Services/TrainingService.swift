import Foundation
import PiMPAudioCore
import MLX
import OSLog
import Darwin

private let trainingLogger = Logger(
    subsystem: "com.pimpstudios.PiMPAudioStudio",
    category: "TrainingService"
)

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
    let lossMovingAverage: Float?
    let gradientNorm: Float?
    let parameterNorm: Float?
    let updateNorm: Float?
    let validationLoss: Float?
    let elapsed: TimeInterval
    let remaining: TimeInterval?
    let lossHistory: [Float]
    let validationHistory: [TrainingValidationPoint]
}

/// Result returned from the pipeline's training method (added in Task 10).
struct LoRATrainingResult {
    let weights: [(String, MLXArray)]
    let finalLoss: Float?
    let trainedPaths: [String]
    let optimizerState: [(String, MLXArray)]
    let wasCancelled: Bool
}

private struct TrainingPreprocessResult {
    let sample: TrainingSample
    let cacheHit: Bool
}

private struct TrainingMemoryGuardError: LocalizedError {
    let message: String

    var errorDescription: String? { message }
}

struct TrainingLocalPreflightReport {
    let phase: String
    let kind: AdapterKind
    let activeModelID: String
    let sampleCount: Int
    let dimension: Int
    let layerCount: Int
    let projectionCount: Int
    let injectionPatterns: [String]
    let estimatedTrainablePathCount: Int
    let estimatedTrainableTensorCount: Int
    let cacheBeforeClearBytes: UInt64
    let activeMemoryBytes: UInt64
    let cacheMemoryBytes: UInt64
    let peakMemoryBytes: UInt64
    let physicalMemoryBytes: UInt64
    let systemAvailableMemoryBytes: UInt64?
    let memoryLimitBytes: UInt64
    let processHeadroomBytes: UInt64
    let effectiveHeadroomBytes: UInt64
    let requiredHeadroomBytes: UInt64
    let unsupportedConfigurationReason: String?
    let memoryHeadroomReason: String?
    let warnings: [String]

    var blockedReason: String? {
        unsupportedConfigurationReason ?? memoryHeadroomReason
    }

    var allowed: Bool {
        blockedReason == nil
    }
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
    private static let bytesPerGiB: UInt64 = 1_073_741_824
    private static let minimumTrainingHeadroomBytes = 6 * bytesPerGiB
    private static let maximumTrainingHeadroomBytes = 12 * bytesPerGiB

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
    private var activeAdapterKind: AdapterKind?
    private let stackManager: AdapterStackManager?
    private var savedStack: [(adapter: LoRAAdapter, strength: Float)] = []
    private var lastTrainingProgressLogStep = 0
    private var activeRunLog: TrainingRunLog?

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
        startAdapterTraining(
            kind: .lora,
            entries: entries,
            audioURLs: audioURLs,
            importedFiles: importedFiles,
            adapterName: adapterName,
            config: config,
            lokrConfig: nil,
            resumeAdapterID: resumeAdapterID
        )
    }

    func startLoKRTraining(
        entries: [HistoryEntry],
        audioURLs: [URL],
        importedFiles: [TrainingImportedFile],
        adapterName: String,
        config: LoKRConfig,
        resumeAdapterID: String? = nil
    ) {
        startAdapterTraining(
            kind: .lokr,
            entries: entries,
            audioURLs: audioURLs,
            importedFiles: importedFiles,
            adapterName: adapterName,
            config: config.loraCompatibleConfig,
            lokrConfig: config,
            resumeAdapterID: resumeAdapterID
        )
    }

    private func startAdapterTraining(
        kind: AdapterKind,
        entries: [HistoryEntry],
        audioURLs: [URL],
        importedFiles: [TrainingImportedFile],
        adapterName: String,
        config: LoRAConfig,
        lokrConfig: LoKRConfig?,
        resumeAdapterID: String?
    ) {
        let effectiveResumeAdapterID: String?
        if let resumeAdapterID {
            effectiveResumeAdapterID = resumeAdapterID
        } else if case .paused = state {
            effectiveResumeAdapterID = Self.adapterID(from: adapterName)
        } else {
            effectiveResumeAdapterID = nil
        }

        guard trainingTask == nil else {
            AudioDiagnostics.note(
                "training/startIgnored",
                "kind=\(kind.rawValue) reason=trainingTaskAlreadyActive"
            )
            return
        }
        isStopping = false
        shouldSaveCheckpointOnStop = false
        activeAdapterKind = kind
        activeRunLog = nil
        state = .loadingModels

        trainingTask = Task { [weak self] in
            guard let self else { return }
            let adapterID = effectiveResumeAdapterID ?? Self.adapterID(from: adapterName)
            defer {
                AudioDiagnostics.memory("training/defer/begin")
                if let sm = self.stackManager, !self.savedStack.isEmpty {
                    AudioDiagnostics.note(
                        "training/restoreStack",
                        "count=\(self.savedStack.count)"
                    )
                    for entry in self.savedStack {
                        sm.addToStack(entry.adapter, strength: entry.strength)
                    }
                    self.savedStack = []
                }
                self.trainingTask = nil
                self.activeAdapterKind = nil
                self.activeRunLog = nil
                if let onTrainingFinished {
                    Task {
                        await onTrainingFinished()
                        AudioDiagnostics.memory("training/defer/afterFinishedHandler")
                    }
                }
                AudioDiagnostics.memory("training/defer/end")
            }

            do {
                let totalSamples = entries.count + importedFiles.count
                self.state = .loadingModels
                self.logTrainingPhase(
                    "loadingModels/begin",
                    kind: kind,
                    adapterID: adapterID,
                    adapterName: adapterName
                )
                if let ensureModelsLoaded {
                    try await ensureModelsLoaded()
                    AudioDiagnostics.memory("training/loadingModels/afterEnsureLoaded")
                }
                guard self.isConditioningReady else {
                    AudioDiagnostics.note(
                        "training/loadingModels/notReady",
                        "kind=\(kind.rawValue) adapterID=\(adapterID)"
                    )
                    self.state = .failed("Could not load the ACE-Step models needed for training.")
                    return
                }
                let activeModelID = await self.registry.activeModelID(
                    for: .aceStepDiT
                ) ?? "unknown"
                AudioDiagnostics.note(
                    "training/baseModel",
                    "kind=\(kind.rawValue) adapterID=\(adapterID) activeModel=\(activeModelID)"
                )
                try self.validateLocalTrainingMemory(
                    phase: "afterEnsureLoaded",
                    kind: kind,
                    activeModelID: activeModelID,
                    sampleCount: totalSamples,
                    config: config,
                    lokrConfig: lokrConfig
                )
                self.logTrainingPhase(
                    "loadingModels/end",
                    kind: kind,
                    adapterID: adapterID,
                    adapterName: adapterName
                )

                // Clear stack during training (restore after)
                if let sm = self.stackManager, sm.isStacking {
                    self.savedStack = sm.additionalAdapters
                    sm.clearStack()
                    AudioDiagnostics.note(
                        "training/clearStack",
                        "kind=\(kind.rawValue) adapterID=\(adapterID) savedCount=\(self.savedStack.count)"
                    )
                }

                var sessionConfig: LoRAConfig
                var sessionLoKRConfig: LoKRConfig?
                let resumeCheckpoint: TrainingCheckpoint?
                let initialWeights: [(String, MLXArray)]?

                if let resumeAdapterID = effectiveResumeAdapterID {
                    AudioDiagnostics.note(
                        "training/resume/begin",
                        "kind=\(kind.rawValue) adapterID=\(resumeAdapterID)"
                    )
                    guard let checkpoint = AdapterRegistry.loadCheckpoint(adapterID: resumeAdapterID) else {
                        AudioDiagnostics.note(
                            "training/resume/missingCheckpoint",
                            "kind=\(kind.rawValue) adapterID=\(resumeAdapterID)"
                        )
                        self.state = .failed("No paused training checkpoint found for this adapter.")
                        return
                    }
                    guard checkpoint.baseModelID == activeModelID else {
                        AudioDiagnostics.note(
                            "training/resume/modelMismatch",
                            "kind=\(kind.rawValue) adapterID=\(resumeAdapterID) expected=\(checkpoint.baseModelID) actual=\(activeModelID)"
                        )
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
                    if kind == .lokr {
                        let savedAdapter = AdapterRegistry.listAdapters().first { $0.id == resumeAdapterID }
                        sessionLoKRConfig = savedAdapter?.lokrConfig
                            ?? LoKRConfig(loraCompatibleConfig: checkpoint.config)
                    } else {
                        sessionLoKRConfig = nil
                    }
                    initialWeights = try await AdapterRegistry.loadWeights(adapterID: resumeAdapterID)
                    AudioDiagnostics.note(
                        "training/resume/end",
                        "kind=\(kind.rawValue) adapterID=\(resumeAdapterID) checkpointStep=\(checkpoint.globalStep) checkpointEpoch=\(checkpoint.epoch) initialWeights=\(initialWeights?.count ?? 0) lokrConfig=\(sessionLoKRConfig != nil)"
                    )
                    try self.validateLocalTrainingMemory(
                        phase: "afterResumeConfig",
                        kind: kind,
                        activeModelID: activeModelID,
                        sampleCount: totalSamples,
                        config: sessionConfig,
                        lokrConfig: sessionLoKRConfig
                    )
                } else {
                    resumeCheckpoint = nil
                    sessionConfig = config
                    sessionLoKRConfig = lokrConfig
                    initialWeights = nil
                }
                let requestedDropout = sessionLoKRConfig?.dropout ?? sessionConfig.dropout
                if requestedDropout > 0 {
                    AudioDiagnostics.note(
                        "training/config/dropoutRNG",
                        "kind=\(kind.rawValue) adapterID=\(adapterID) dropout=\(Self.formatFloat(requestedDropout)) rng=explicitKey"
                    )
                }

                // 1. Prepare dataset
                self.state = .preparingDataset(current: 0, total: totalSamples)
                AudioDiagnostics.note(
                    "training/dataset/begin",
                    "kind=\(kind.rawValue) adapterID=\(adapterID) historyEntries=\(entries.count) importedFiles=\(importedFiles.count) total=\(totalSamples)"
                )
                AudioDiagnostics.memory("training/dataset/begin")

                var samples: [TrainingSample] = []

                let runTrainingSeed = sessionLoKRConfig?.effectiveTrainingSeed
                    ?? sessionConfig.effectiveTrainingSeed

                for (idx, entry) in entries.enumerated() {
                    try Task.checkCancellation()
                    let url = audioURLs[idx]
                    let preprocessResult = try await self.preprocessSample(
                        audioURL: url,
                        prompt: entry.prompt,
                        lyrics: entry.lyrics,
                        duration: entry.duration,
                        windowSeed: Self.windowSeed(
                            trainingSeed: runTrainingSeed,
                            clipIdentifier: "history|\(idx)|\(url.lastPathComponent)"
                        )
                    )
                    let sample = preprocessResult.sample
                    samples.append(sample)
                    self.logPreparedSample(
                        sample,
                        kind: kind,
                        adapterID: adapterID,
                        source: "history",
                        index: idx + 1,
                        total: totalSamples,
                        cacheHit: preprocessResult.cacheHit,
                        audioURL: url
                    )
                    self.state = .preparingDataset(current: idx + 1, total: totalSamples)
                }

                for (idx, imported) in importedFiles.enumerated() {
                    try Task.checkCancellation()
                    let preprocessResult = try await self.preprocessSample(
                        audioURL: imported.url,
                        prompt: imported.description,
                        lyrics: imported.lyrics,
                        duration: 0,
                        windowSeed: Self.windowSeed(
                            trainingSeed: runTrainingSeed,
                            clipIdentifier: imported.preprocessCache?.clipID
                                ?? "imported|\(idx)|\(imported.url.lastPathComponent)"
                        ),
                        cache: imported.preprocessCache
                    )
                    let sample = preprocessResult.sample
                    samples.append(sample)
                    self.logPreparedSample(
                        sample,
                        kind: kind,
                        adapterID: adapterID,
                        source: "imported",
                        index: entries.count + idx + 1,
                        total: totalSamples,
                        cacheHit: preprocessResult.cacheHit,
                        audioURL: imported.url
                    )
                    self.state = .preparingDataset(
                        current: entries.count + idx + 1,
                        total: totalSamples
                    )
                }

                guard !samples.isEmpty else {
                    AudioDiagnostics.note(
                        "training/dataset/empty",
                        "kind=\(kind.rawValue) adapterID=\(adapterID)"
                    )
                    self.state = .failed("No valid training samples")
                    return
                }
                AudioDiagnostics.note(
                    "training/dataset/end",
                    "kind=\(kind.rawValue) adapterID=\(adapterID) samples=\(samples.count) \(Self.trainingDatasetShapeSummary(samples))"
                )
                AudioDiagnostics.memory("training/dataset/end")
                try self.validateLocalTrainingMemory(
                    phase: "beforeTraining",
                    kind: kind,
                    activeModelID: activeModelID,
                    sampleCount: samples.count,
                    config: sessionConfig,
                    lokrConfig: sessionLoKRConfig
                )
                let manifest = TrainingManifest(
                    historyEntryIDs: entries.map(\.id),
                    importedFiles: importedFiles
                )
                self.activeRunLog = TrainingRunLog(
                    adapterID: adapterID,
                    adapterName: adapterName,
                    adapterKind: kind,
                    baseModelID: activeModelID,
                    sampleCount: samples.count,
                    config: sessionConfig,
                    lokrConfig: sessionLoKRConfig,
                    resumedFromCheckpoint: resumeCheckpoint != nil
                )
                self.logTrainingStart(
                    kind: kind,
                    adapterName: adapterName,
                    activeModelID: activeModelID,
                    sampleCount: samples.count,
                    config: sessionConfig,
                    lokrConfig: sessionLoKRConfig
                )

                // 2. Run training through pipeline
                try Task.checkCancellation()
                self.lastTrainingProgressLogStep = 0
                self.logTrainingPhase(
                    "run/begin",
                    kind: kind,
                    adapterID: adapterID,
                    adapterName: adapterName
                )

                // Drop any inference-time LoRA/LoKR wrappers left on the DiT from
                // a previous generate. trainLoRA/trainLoKR inject directly and
                // skip already-wrapped linears, which silently trains the wrong
                // surface (or nothing) if generation adapters are still attached.
                await self.pipeline.clearLoRAContributions()
                await self.pipeline.clearLoKRContributions()
                await self.pipeline.removeLoRA()
                AudioDiagnostics.note(
                    "training/run/clearedInferenceAdapters",
                    "kind=\(kind.rawValue) adapterID=\(adapterID)"
                )

                let progressHandler: @Sendable (TrainingProgress) -> Void = { [weak self] progress in
                    Task { @MainActor in
                        self?.handleTrainingProgress(
                            progress,
                            kind: kind,
                            adapterName: adapterName
                        )
                    }
                }

                let result: LoRATrainingResult
                switch kind {
                case .lora:
                    let trained = try await self.pipeline.trainLoRA(
                        samples: samples,
                        config: sessionConfig,
                        resumeFrom: resumeCheckpoint,
                        initialWeights: initialWeights,
                        onProgress: progressHandler
                    )
                    result = LoRATrainingResult(
                        weights: trained.weights,
                        finalLoss: trained.finalLoss,
                        trainedPaths: trained.trainedPaths,
                        optimizerState: trained.optimizerState,
                        wasCancelled: trained.wasCancelled
                    )
                case .lokr:
                    guard let sessionLoKRConfig else {
                        self.state = .failed("LoKR training configuration was missing.")
                        return
                    }
                    let trained = try await self.pipeline.trainLoKR(
                        samples: samples,
                        config: sessionLoKRConfig,
                        resumeFrom: resumeCheckpoint,
                        initialWeights: initialWeights,
                        onProgress: progressHandler
                    )
                    result = LoRATrainingResult(
                        weights: trained.weights,
                        finalLoss: trained.finalLoss,
                        trainedPaths: trained.trainedPaths,
                        optimizerState: trained.optimizerState,
                        wasCancelled: trained.wasCancelled
                    )
                }
                self.logTrainingPhase(
                    "run/end",
                    kind: kind,
                    adapterID: adapterID,
                    adapterName: adapterName
                )
                try Task.checkCancellation()

                let adapter = LoRAAdapter(
                    id: adapterID,
                    name: adapterName,
                    config: sessionConfig,
                    kind: kind,
                    lokrConfig: sessionLoKRConfig,
                    baseModelID: activeModelID,
                    baseModelCategory: .aceStepDiT,
                    createdAt: Date(),
                    sampleCount: samples.count,
                    finalLoss: result.finalLoss,
                    trainedPaths: result.trainedPaths,
                    trainedLayerIndices: LoRAAdapter.layerIndices(from: result.trainedPaths)
                )

                if self.isStopping {
                    if self.shouldSaveCheckpointOnStop {
                        AudioDiagnostics.note(
                            "training/saveCheckpoint/begin",
                            "kind=\(kind.rawValue) adapterID=\(adapterID)"
                        )
                        trainingLogger.info("Training paused and checkpoint saved for '\(adapterName)'")
                        let checkpoint = self.makeCheckpoint(
                            adapterID: adapterID,
                            config: sessionConfig,
                            activeModelID: activeModelID,
                            sampleCount: samples.count,
                            bestLoss: result.finalLoss,
                            fallback: resumeCheckpoint
                        )
                        try AdapterRegistry.save(adapter: adapter, weights: result.weights)
                        try AdapterRegistry.saveOptimizerState(result.optimizerState, adapterID: adapterID)
                        AdapterRegistry.saveManifest(manifest, adapterID: adapterID)
                        try AdapterRegistry.saveCheckpoint(checkpoint, adapterID: adapterID)
                        self.finishActiveRunLog(
                            status: .paused,
                            result: result,
                            message: "Training paused and checkpoint saved."
                        )
                        self.saveActiveRunLog(adapterID: adapterID)
                        AudioDiagnostics.note(
                            "training/saveCheckpoint/end",
                            "kind=\(kind.rawValue) adapterID=\(adapterID) weights=\(result.weights.count) paths=\(result.trainedPaths.count)"
                        )
                        self.state = .paused(checkpoint)
                    } else {
                        trainingLogger.info("Training cancelled for '\(adapterName)'")
                        AudioDiagnostics.note(
                            "training/cancelled",
                            "kind=\(kind.rawValue) adapterID=\(adapterID) saveCheckpoint=false"
                        )
                        self.finishActiveRunLog(
                            status: .cancelled,
                            result: result,
                            message: "Training stopped without saving a checkpoint."
                        )
                        self.state = .cancelled
                    }
                    return
                }

                // 3. Save adapter
                self.state = .saving
                AudioDiagnostics.note(
                    "training/save/begin",
                    "kind=\(kind.rawValue) adapterID=\(adapterID) weights=\(result.weights.count) paths=\(result.trainedPaths.count)"
                )
                AudioDiagnostics.memory("training/save/begin")

                try AdapterRegistry.save(adapter: adapter, weights: result.weights)
                trainingLogger.info("Saved \(kind.rawValue.uppercased()) adapter '\(adapterName)' id=\(adapterID)")

                // Persist the training manifest so the session can be reloaded
                // later for retraining with the same (or expanded) dataset.
                AdapterRegistry.saveManifest(manifest, adapterID: adapterID)
                AdapterRegistry.deleteCheckpoint(adapterID: adapterID)
                self.logTrainingComplete(
                    kind: kind,
                    adapterName: adapterName,
                    result: result
                )
                self.finishActiveRunLog(
                    status: .completed,
                    result: result,
                    message: "Training completed and adapter was saved."
                )
                self.saveActiveRunLog(adapterID: adapterID)
                AudioDiagnostics.note(
                    "training/save/end",
                    "kind=\(kind.rawValue) adapterID=\(adapterID) directory=\(AdapterRegistry.adapterDirectory(id: adapterID).path(percentEncoded: false))"
                )
                AudioDiagnostics.memory("training/save/end")

                self.state = .complete(adapter)

            } catch is CancellationError {
                trainingLogger.info("Training task cancelled")
                AudioDiagnostics.note(
                    "training/cancelled",
                    "kind=\(kind.rawValue) adapterID=\(adapterID) taskCancellation=true"
                )
                self.finishActiveRunLog(
                    status: .cancelled,
                    message: "Training task was cancelled."
                )
                self.saveActiveRunLogIfAdapterDirectoryExists(adapterID: adapterID)
                self.state = .cancelled
            } catch {
                trainingLogger.error("Training failed: \(error.localizedDescription)")
                AudioDiagnostics.note("training/error", "error=\(error.localizedDescription)")
                self.finishActiveRunLog(
                    status: .failed,
                    message: "Training failed.",
                    errorMessage: error.localizedDescription
                )
                self.saveActiveRunLogIfAdapterDirectoryExists(adapterID: adapterID)
                self.state = .failed(error.localizedDescription)
            }
        }
    }

    private static func formatFloat(_ value: Float) -> String {
        String(format: "%.3f", value)
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

    private func handleTrainingProgress(
        _ progress: TrainingProgress,
        kind: AdapterKind,
        adapterName: String
    ) {
        state = .training(TrainingUIState(
            epoch: progress.epoch,
            totalEpochs: progress.totalEpochs,
            step: progress.step,
            loss: progress.loss,
            learningRate: progress.learningRate,
            lossMovingAverage: progress.lossMovingAverage,
            gradientNorm: progress.gradientNorm,
            parameterNorm: progress.parameterNorm,
            updateNorm: progress.updateNorm,
            validationLoss: progress.validationLoss,
            elapsed: progress.elapsedTime,
            remaining: progress.estimatedTimeRemaining,
            lossHistory: progress.lossHistory,
            validationHistory: progress.validationHistory
        ))

        let now = Date()
        activeRunLog?.lossHistory = progress.lossHistory
        activeRunLog?.validationHistory = progress.validationHistory
        activeRunLog?.updatedAt = now

        guard progress.step == 1 || progress.step - lastTrainingProgressLogStep >= 25 else {
            return
        }
        lastTrainingProgressLogStep = progress.step
        activeRunLog?.recordProgress(
            TrainingRunProgressEntry(
                timestamp: now,
                epoch: progress.epoch,
                totalEpochs: progress.totalEpochs,
                step: progress.step,
                loss: progress.loss,
                learningRate: progress.learningRate,
                lossMovingAverage: progress.lossMovingAverage,
                gradientNorm: progress.gradientNorm,
                parameterNorm: progress.parameterNorm,
                updateNorm: progress.updateNorm,
                validationLoss: progress.validationLoss,
                elapsed: progress.elapsedTime,
                estimatedRemaining: progress.estimatedTimeRemaining
            ),
            lossHistory: progress.lossHistory,
            validationHistory: progress.validationHistory
        )
        let lossText = String(format: "%.5f", progress.loss)
        let lrText = String(format: "%.2e", progress.learningRate)
        let movingLossText = progress.lossMovingAverage.map { String(format: "%.5f", $0) } ?? "nil"
        let gradNormText = progress.gradientNorm.map { String(format: "%.3e", $0) } ?? "nil"
        let paramNormText = progress.parameterNorm.map { String(format: "%.3e", $0) } ?? "nil"
        let updateNormText = progress.updateNorm.map { String(format: "%.3e", $0) } ?? "nil"
        let validationLossText = progress.validationLoss.map { String(format: "%.5f", $0) } ?? "nil"
        trainingLogger.info(
            "\(kind.rawValue.uppercased()) training progress name='\(adapterName)' epoch=\(progress.epoch)/\(progress.totalEpochs) step=\(progress.step) loss=\(lossText) valLoss=\(validationLossText) avgLoss=\(movingLossText) lr=\(lrText) gradNorm=\(gradNormText) paramNorm=\(paramNormText) updateNorm=\(updateNormText)"
        )
        AudioDiagnostics.note(
            "training/progress",
            "kind=\(kind.rawValue) name=\(adapterName) epoch=\(progress.epoch)/\(progress.totalEpochs) step=\(progress.step) loss=\(lossText) valLoss=\(validationLossText) avgLoss=\(movingLossText) lr=\(lrText) gradNorm=\(gradNormText) paramNorm=\(paramNormText) updateNorm=\(updateNormText)"
        )
    }

    private func logTrainingStart(
        kind: AdapterKind,
        adapterName: String,
        activeModelID: String,
        sampleCount: Int,
        config: LoRAConfig,
        lokrConfig: LoKRConfig?
    ) {
        let rankLabel = kind == .lokr
            ? "linearDim=\(lokrConfig?.linearDim ?? config.rank)"
            : "rank=\(config.rank)"
        let layerCount = lokrConfig?.trainingLayerCount ?? config.trainingLayerCount
        let alpha = lokrConfig?.alpha ?? config.alpha
        let dropout = lokrConfig?.dropout ?? config.dropout
        let learningRate = lokrConfig?.learningRate ?? config.learningRate
        let gradientAccumulationSteps = lokrConfig?.gradientAccumulationSteps ?? config.gradientAccumulationSteps
        let trainingSeed = lokrConfig?.effectiveTrainingSeed ?? config.effectiveTrainingSeed
        let weightDecomposeText = kind == .lokr
            ? " weightDecompose=\(lokrConfig?.weightDecompose ?? false)"
            : ""
        let targets = config.injectionPatterns.joined(separator: ",")
        let numericSummary = "alpha=\(String(format: "%.3f", alpha)) dropout=\(String(format: "%.3f", dropout)) lr=\(String(format: "%.2e", learningRate)) gradAccum=\(gradientAccumulationSteps) seed=\(trainingSeed)\(weightDecomposeText)"
        trainingLogger.info(
            "Starting \(kind.rawValue.uppercased()) training name='\(adapterName)' base=\(activeModelID) samples=\(sampleCount) \(rankLabel) \(numericSummary) epochs=\(config.maxEpochs) layerCoverage=\(layerCount ?? 0) targets=\(targets)"
        )
        AudioDiagnostics.note(
            "training/start",
            "kind=\(kind.rawValue) name=\(adapterName) base=\(activeModelID) samples=\(sampleCount) \(rankLabel) \(numericSummary) epochs=\(config.maxEpochs) layerCoverage=\(layerCount ?? 0) targets=\(config.injectionPatterns)"
        )
    }

    private func logTrainingComplete(
        kind: AdapterKind,
        adapterName: String,
        result: LoRATrainingResult
    ) {
        let lossText = result.finalLoss.map { String(format: "%.5f", $0) } ?? "nil"
        trainingLogger.info(
            "\(kind.rawValue.uppercased()) training complete name='\(adapterName)' tensors=\(result.weights.count) trainedPaths=\(result.trainedPaths.count) finalLoss=\(lossText)"
        )
        AudioDiagnostics.note(
            "training/complete",
            "kind=\(kind.rawValue) name=\(adapterName) tensors=\(result.weights.count) trainedPaths=\(result.trainedPaths.count) finalLoss=\(lossText)"
        )
    }

    private func logTrainingPhase(
        _ phase: String,
        kind: AdapterKind,
        adapterID: String,
        adapterName: String
    ) {
        AudioDiagnostics.note(
            "training/\(phase)",
            "kind=\(kind.rawValue) adapterID=\(adapterID) name=\(adapterName)"
        )
        AudioDiagnostics.memory("training/\(phase)")
    }

    private func logPreparedSample(
        _ sample: TrainingSample,
        kind: AdapterKind,
        adapterID: String,
        source: String,
        index: Int,
        total: Int,
        cacheHit: Bool,
        audioURL: URL
    ) {
        AudioDiagnostics.note(
            "training/dataset/sample",
            "kind=\(kind.rawValue) adapterID=\(adapterID) index=\(index)/\(total) source=\(source) cacheHit=\(cacheHit) file=\(audioURL.lastPathComponent) latent=\(sample.latent.shape) encoder=\(sample.encoderStates.shape) context=\(sample.contextLatents.shape) mask=\(sample.encoderMask?.shape ?? [])"
        )
    }

    private static func trainingDatasetShapeSummary(_ samples: [TrainingSample]) -> String {
        let latentFrameCounts = samples.map { $0.latent.shape.dropFirst().first ?? 0 }
        let minFrames = latentFrameCounts.min() ?? 0
        let maxFrames = latentFrameCounts.max() ?? 0

        return "latentFrames=\(minFrames)...\(maxFrames) "
            + "latentShapes=\(shapeSummary(samples.map(\.latent.shape))) "
            + "contextShapes=\(shapeSummary(samples.map(\.contextLatents.shape))) "
            + "encoderShapes=\(shapeSummary(samples.map(\.encoderStates.shape)))"
    }

    private static func shapeSummary(_ shapes: [[Int]]) -> String {
        let uniqueShapes = Set(shapes.map { "\($0)" }).sorted()
        let shownShapes = uniqueShapes.prefix(5).joined(separator: ",")
        guard uniqueShapes.count > 5 else {
            return "[\(shownShapes)]"
        }
        return "[\(shownShapes),...+\(uniqueShapes.count - 5)]"
    }

    private func finishActiveRunLog(
        status: TrainingRunStatus,
        result: LoRATrainingResult? = nil,
        message: String,
        errorMessage: String? = nil
    ) {
        guard var log = activeRunLog else { return }
        let trainedPaths = result?.trainedPaths ?? []
        log.finish(
            status: status,
            finalLoss: result?.finalLoss,
            trainedPaths: trainedPaths,
            trainedLayerIndices: LoRAAdapter.layerIndices(from: trainedPaths),
            weightsTensorCount: result?.weights.count,
            message: message,
            errorMessage: errorMessage
        )
        activeRunLog = log
    }

    private func saveActiveRunLog(adapterID: String) {
        guard let activeRunLog else { return }
        do {
            try AdapterRegistry.saveTrainingRunLog(activeRunLog, adapterID: adapterID)
        } catch {
            trainingLogger.warning("Could not save training log for '\(adapterID)': \(error.localizedDescription)")
        }
    }

    private func saveActiveRunLogIfAdapterDirectoryExists(adapterID: String) {
        let directory = AdapterRegistry.adapterDirectory(id: adapterID)
        guard FileManager.default.fileExists(atPath: directory.path(percentEncoded: false)) else {
            return
        }
        saveActiveRunLog(adapterID: adapterID)
    }

    /// Stop training. When requested, persists the current adapter weights and checkpoint.
    func stopTraining(saveCheckpoint: Bool = false) {
        isStopping = true
        shouldSaveCheckpointOnStop = saveCheckpoint
        AudioDiagnostics.note(
            "training/stopRequested",
            "saveCheckpoint=\(saveCheckpoint) activeKind=\(activeAdapterKind?.rawValue ?? "nil")"
        )
        Task {
            await pipeline.cancelTraining()
        }
    }

    /// Whether training can produce a persisted pause checkpoint on stop.
    var canSaveOnStop: Bool {
        guard activeAdapterKind == .lora || activeAdapterKind == .lokr else { return false }
        if case .training(let uiState) = state {
            return uiState.step > 0
        }
        return false
    }

    /// Snapshot of the in-memory run log for MCP/read-only diagnostics.
    var activeRunLogSnapshot: TrainingRunLog? {
        activeRunLog
    }

    func localTrainingPreflight(
        kind: AdapterKind,
        sampleCount: Int,
        config: LoRAConfig,
        lokrConfig: LoKRConfig?,
        clearCache: Bool = true
    ) async -> TrainingLocalPreflightReport {
        let activeModelID = await registry.activeModelID(for: .aceStepDiT) ?? "unknown"
        return Self.localTrainingPreflightReport(
            phase: "mcpPreflight",
            kind: kind,
            activeModelID: activeModelID,
            sampleCount: sampleCount,
            config: config,
            lokrConfig: lokrConfig,
            clearCache: clearCache
        )
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
            if fingerprint.contains("turbo") {
                return "ACE-Step XL Turbo"
            }
            if fingerprint.contains("sft") {
                return "ACE-Step XL SFT"
            }
            if fingerprint.contains("base") || fingerprint.contains("full") {
                return "ACE-Step XL Base"
            }
            return "ACE-Step XL"
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
        bestLoss: Float?,
        fallback: TrainingCheckpoint?
    ) -> TrainingCheckpoint {
        let uiState: TrainingUIState?
        if case .training(let current) = state {
            uiState = current
        } else {
            uiState = nil
        }

        let effectiveSampleCount = max(1, AdapterTrainingSplit.trainingSampleCount(total: sampleCount))
        let accumulation = min(max(1, config.gradientAccumulationSteps), effectiveSampleCount)
        let updatesPerEpoch = max(1, (effectiveSampleCount + accumulation - 1) / accumulation)
        let totalSteps = updatesPerEpoch * config.maxEpochs
        let lossHistory = uiState?.lossHistory ?? fallback?.lossHistory ?? []
        let validationHistory = uiState?.validationHistory ?? fallback?.validationHistory ?? []

        return TrainingCheckpoint(
            epoch: uiState?.epoch ?? fallback?.epoch ?? 0,
            globalStep: uiState?.step ?? fallback?.globalStep ?? 0,
            totalSteps: totalSteps,
            bestLoss: bestLoss ?? fallback?.bestLoss ?? validationHistory.map(\.loss).min() ?? lossHistory.min(),
            lossHistory: lossHistory,
            validationHistory: validationHistory,
            weightsURL: AdapterRegistry.weightsURL(adapterID: adapterID),
            optimizerStateURL: AdapterRegistry.optimizerStateURL(adapterID: adapterID),
            config: config,
            baseModelID: activeModelID
        )
    }

    private func validateLocalTrainingMemory(
        phase: String,
        kind: AdapterKind,
        activeModelID: String,
        sampleCount: Int,
        config: LoRAConfig,
        lokrConfig: LoKRConfig?
    ) throws {
        let report = Self.localTrainingPreflightReport(
            phase: phase,
            kind: kind,
            activeModelID: activeModelID,
            sampleCount: sampleCount,
            config: config,
            lokrConfig: lokrConfig,
            clearCache: true
        )

        AudioDiagnostics.note(
            "training/memoryGuard",
            "phase=\(phase) kind=\(kind.rawValue) base=\(activeModelID) samples=\(sampleCount) dimension=\(report.dimension) layerCoverage=\(report.layerCount) projections=\(report.projectionCount) active=\(report.activeMemoryBytes) cacheBefore=\(report.cacheBeforeClearBytes) cacheAfter=\(report.cacheMemoryBytes) peak=\(report.peakMemoryBytes) physical=\(report.physicalMemoryBytes) systemAvailable=\(report.systemAvailableMemoryBytes.map { String($0) } ?? "unknown") memoryLimit=\(report.memoryLimitBytes) processHeadroom=\(report.processHeadroomBytes) headroom=\(report.effectiveHeadroomBytes) requiredHeadroom=\(report.requiredHeadroomBytes)"
        )
        AudioDiagnostics.memory("training/memoryGuard/\(phase)")

        if let unsupportedMessage = report.unsupportedConfigurationReason {
            AudioDiagnostics.note(
                "training/configGuard/fail",
                "phase=\(phase) kind=\(kind.rawValue) layerCoverage=\(report.layerCount) dimension=\(report.dimension) projections=\(report.projectionCount) message=\(unsupportedMessage)"
            )
            throw TrainingMemoryGuardError(message: unsupportedMessage)
        }

        guard let memoryMessage = report.memoryHeadroomReason else { return }

        AudioDiagnostics.note(
            "training/memoryGuard/fail",
            "phase=\(phase) kind=\(kind.rawValue) active=\(report.activeMemoryBytes) headroom=\(report.effectiveHeadroomBytes) requiredHeadroom=\(report.requiredHeadroomBytes) message=\(memoryMessage)"
        )
        throw TrainingMemoryGuardError(message: memoryMessage)
    }

    private static func localTrainingPreflightReport(
        phase: String,
        kind: AdapterKind,
        activeModelID: String,
        sampleCount: Int,
        config: LoRAConfig,
        lokrConfig: LoKRConfig?,
        clearCache: Bool
    ) -> TrainingLocalPreflightReport {
        let beforeClear = Memory.snapshot()
        if clearCache {
            Memory.clearCache()
        }
        let snapshot = Memory.snapshot()
        let physicalMemory = ProcessInfo.processInfo.physicalMemory
        let systemAvailableMemory = availableSystemMemoryBytes()
        let memoryLimit = UInt64(max(0, Memory.memoryLimit))
        let activeMemory = UInt64(max(0, snapshot.activeMemory))
        let cacheMemory = UInt64(max(0, snapshot.cacheMemory))
        let peakMemory = UInt64(max(0, snapshot.peakMemory))
        let injectionPatterns = lokrConfig?.injectionPatterns ?? config.injectionPatterns
        let requiredHeadroom = trainingHeadroomRequirement(
            kind: kind,
            sampleCount: sampleCount,
            config: config,
            lokrConfig: lokrConfig
        )
        let physicalHeadroom = physicalMemory > activeMemory ? physicalMemory - activeMemory : 0
        let limitHeadroom = memoryLimit > activeMemory ? memoryLimit - activeMemory : 0
        let processHeadroom = min(physicalHeadroom, limitHeadroom)
        let effectiveHeadroom = min(processHeadroom, systemAvailableMemory ?? processHeadroom)
        let layerCount = max(1, min(lokrConfig?.trainingLayerCount ?? config.trainingLayerCount ?? 4, 32))
        let projectionCount = injectionPatterns.count
        let dimension: Int
        if kind == .lokr {
            dimension = max(1, lokrConfig?.linearDim ?? config.rank)
        } else {
            dimension = max(1, config.rank)
        }

        let unsupportedMessage: String? = nil
        let hasFullAttention = injectionPatterns.contains { pattern in
            pattern.hasSuffix(".kProj") || pattern.hasSuffix(".oProj")
        }
        let memoryMessage: String?
        if effectiveHeadroom < requiredHeadroom {
            let recipeHint = kind == .lokr
                ? "Try Narrow LoKR with Q/V and last 4-8 layers, or switch to a smaller ACE-Step DiT model."
                : "Try Narrow LoRA with Q/V and last 4-8 layers, or switch to a smaller ACE-Step DiT model."
            memoryMessage = "Local training needs about \(formatGiB(requiredHeadroom)) of unified-memory headroom before the first backward pass, but this run only has about \(formatGiB(effectiveHeadroom)) available after loading ACE-Step (\(formatGiB(activeMemory)) MLX active on a \(formatGiB(physicalMemory)) Mac). \(recipeHint) Use Remote GPU for wide XL recipes."
        } else {
            memoryMessage = nil
        }

        var warnings: [String] = []
        if clearCache, UInt64(max(0, beforeClear.cacheMemory)) > cacheMemory {
            warnings.append("MLX cache was cleared before measuring local training headroom.")
        }
        if projectionCount >= 8 {
            warnings.append("Full Q/K/V/O attention has a much larger local memory and first-backward graph footprint than Q/V.")
        }
        if kind == .lora, hasFullAttention, layerCount >= 12 {
            warnings.append("Full Q/K/V/O LoRA across 12+ local XL layers is a large local shape; Q/V projections or Last 8 remain the safer starting point.")
        }
        if kind == .lokr, hasFullAttention, layerCount >= 16 {
            warnings.append("LoKR across 16+ local layers is sensitive on XL models; Q/V projection coverage is recommended.")
        }
        if kind == .lokr, hasFullAttention, dimension > 32 {
            warnings.append("Full Q/K/V/O LoKR above Linear Dim 32 is a large local shape; Linear Dim 32 or Q/V projections are recommended.")
        }
        let requestedDropout = lokrConfig?.dropout ?? config.dropout
        if requestedDropout > 0 {
            warnings.append("Local Swift/MLX adapter dropout uses explicit per-step MLX keys to avoid the confirmed implicit-RNG first-backward VJP stall.")
        }
        if sampleCount >= 500 {
            warnings.append("Large datasets raise the preflight reserve because preprocessing and optimizer state stay hot during training.")
        }

        return TrainingLocalPreflightReport(
            phase: phase,
            kind: kind,
            activeModelID: activeModelID,
            sampleCount: sampleCount,
            dimension: dimension,
            layerCount: layerCount,
            projectionCount: projectionCount,
            injectionPatterns: injectionPatterns,
            estimatedTrainablePathCount: layerCount * projectionCount,
            estimatedTrainableTensorCount: layerCount * projectionCount * (kind == .lokr && lokrConfig?.weightDecompose == true ? 3 : 2),
            cacheBeforeClearBytes: UInt64(max(0, beforeClear.cacheMemory)),
            activeMemoryBytes: activeMemory,
            cacheMemoryBytes: cacheMemory,
            peakMemoryBytes: peakMemory,
            physicalMemoryBytes: physicalMemory,
            systemAvailableMemoryBytes: systemAvailableMemory,
            memoryLimitBytes: memoryLimit,
            processHeadroomBytes: processHeadroom,
            effectiveHeadroomBytes: effectiveHeadroom,
            requiredHeadroomBytes: requiredHeadroom,
            unsupportedConfigurationReason: unsupportedMessage,
            memoryHeadroomReason: memoryMessage,
            warnings: warnings
        )
    }

    private static func trainingHeadroomRequirement(
        kind: AdapterKind,
        sampleCount: Int,
        config: LoRAConfig,
        lokrConfig: LoKRConfig?
    ) -> UInt64 {
        let layerCount = max(1, min(lokrConfig?.trainingLayerCount ?? config.trainingLayerCount ?? 4, 32))
        let projectionCount = max(1, (lokrConfig?.injectionPatterns ?? config.injectionPatterns).count)
        let dimension: Int
        if kind == .lokr {
            dimension = max(1, lokrConfig?.linearDim ?? config.rank)
        } else {
            dimension = max(1, config.rank)
        }

        var requirement = minimumTrainingHeadroomBytes
        if layerCount >= 16 {
            requirement += 2 * bytesPerGiB
        }
        if layerCount >= 32 {
            requirement += 2 * bytesPerGiB
        }
        if projectionCount >= 8 {
            requirement += 2 * bytesPerGiB
        }
        if dimension >= 96 {
            requirement += bytesPerGiB
        }
        if kind == .lokr && lokrConfig?.weightDecompose == true && layerCount >= 16 {
            requirement += bytesPerGiB
        }
        if sampleCount >= 500 {
            requirement += bytesPerGiB
        }
        return min(requirement, maximumTrainingHeadroomBytes)
    }

    private static func formatGiB(_ bytes: UInt64) -> String {
        String(format: "%.1f GiB", Double(bytes) / Double(bytesPerGiB))
    }

    private static func availableSystemMemoryBytes() -> UInt64? {
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(
            MemoryLayout<vm_statistics64_data_t>.stride / MemoryLayout<integer_t>.stride
        )
        let result = withUnsafeMutablePointer(to: &stats) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
                host_statistics64(mach_host_self(), HOST_VM_INFO64, rebound, &count)
            }
        }
        guard result == KERN_SUCCESS else { return nil }

        let reclaimablePages = UInt64(stats.free_count)
            + UInt64(stats.inactive_count)
            + UInt64(stats.speculative_count)
        return reclaimablePages * UInt64(vm_kernel_page_size)
    }

    private func preprocessSample(
        audioURL: URL,
        prompt: String,
        lyrics: String,
        duration: TimeInterval,
        windowSeed: UInt64,
        cache: TrainingPreprocessCacheReference? = nil
    ) async throws -> TrainingPreprocessResult {
        if let cache {
            do {
                if let cachedSample = try TrainingPreprocessCacheStore.load(reference: cache) {
                    AudioDiagnostics.note(
                        "training/preprocess/cacheHit",
                        "clipID=\(cache.clipID) file=\(cache.sampleFile)"
                    )
                    return TrainingPreprocessResult(sample: cachedSample, cacheHit: true)
                }
                AudioDiagnostics.note(
                    "training/preprocess/cacheMiss",
                    "clipID=\(cache.clipID) file=\(cache.sampleFile)"
                )
            } catch {
                AudioDiagnostics.note(
                    "training/preprocess/cacheLoadError",
                    "clipID=\(cache.clipID) error=\(error.localizedDescription)"
                )
            }
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
            contextLatents: cond.contextLatents,
            seed: windowSeed
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
                AudioDiagnostics.note(
                    "training/preprocess/cacheSaveError",
                    "clipID=\(cache.clipID) error=\(error.localizedDescription)"
                )
                trainingLogger.warning("Failed to save preprocess cache: \(error.localizedDescription)")
            }
        }

        return TrainingPreprocessResult(sample: sample, cacheHit: false)
    }

    private static func trainingWindow(
        latent: MLXArray,
        contextLatents: MLXArray,
        seed: UInt64
    ) -> (latent: MLXArray, contextLatents: MLXArray) {
        let availableFrames = min(latent.shape[1], contextLatents.shape[1])
        guard availableFrames > 0 else {
            return (latent, contextLatents)
        }

        let frameCount = min(availableFrames, maxTrainingLatentFrames)
        let maxStart = max(0, availableFrames - frameCount)
        // Deterministic window selection: derived from the run's training
        // seed and the clip identity (see windowSeed(...)). Previously this
        // used `Int.random`, so two runs with identical training seeds could
        // still train on different latent windows.
        let start: Int
        if maxStart > 0 {
            var rng = SeededWindowRandomNumberGenerator(seed: seed)
            start = Int.random(in: 0...maxStart, using: &rng)
        } else {
            start = 0
        }
        let end = start + frameCount
        AudioDiagnostics.note(
            "training/preprocess/window",
            "seed=\(seed) start=\(start) frames=\(frameCount) available=\(availableFrames)"
        )

        return (
            latent[0..., start..<end, 0...],
            contextLatents[0..., start..<end, 0...]
        )
    }

    /// Derive a per-clip window seed from the run's training seed and the
    /// clip's identity, so preprocessing is reproducible for a fixed seed
    /// but still selects different windows for different clips.
    static func windowSeed(trainingSeed: UInt64, clipIdentifier: String) -> UInt64 {
        // FNV-1a over "seed|clipIdentifier"
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for byte in "\(trainingSeed)|\(clipIdentifier)".utf8 {
            hash ^= UInt64(byte)
            hash = hash &* 0x0000_0100_0000_01B3
        }
        return hash == 0 ? 0x9e37_79b9_7f4a_7c15 : hash
    }

    /// SplitMix64 — deterministic RNG for window selection.
    private struct SeededWindowRandomNumberGenerator: RandomNumberGenerator {
        private var state: UInt64

        init(seed: UInt64) {
            self.state = seed == 0 ? 0x9e37_79b9_7f4a_7c15 : seed
        }

        mutating func next() -> UInt64 {
            state &+= 0x9e37_79b9_7f4a_7c15
            var z = state
            z = (z ^ (z >> 30)) &* 0xbf58_476d_1ce4_e5b9
            z = (z ^ (z >> 27)) &* 0x94d0_49bb_1331_11eb
            return z ^ (z >> 31)
        }
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
                "pimp_audio_preprocess_sample_version": version,
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
            "pimp_audio_preprocess_index_version": version,
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
        index.removeValue(forKey: "anvil_preprocess_index_version")
        index["pimp_audio_preprocess_index_version"] = version
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
