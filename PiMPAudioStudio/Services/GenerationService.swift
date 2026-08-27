import Foundation
import MLX
import OSLog
import PiMPAudioCore

private let memLogger = Logger(subsystem: "com.pimpstudios.PiMPAudioStudio", category: "GenerationService.unload")
private let traceLogger = Logger(subsystem: "com.pimpstudios.PiMPAudioStudio", category: "GenerationTrace")

struct GenerationResult: Sendable {
    let fileURL: URL
    let entry: HistoryEntry
    /// Pre-computed normalised peak amplitudes (400 buckets, values 0–1).
    /// Computed from the decoded MLXArray at generation time so the player
    /// can display the waveform immediately without re-reading the WAV file.
    let waveformPeaks: [Float]?
}

struct MemoryReleaseSummary: Sendable {
    let beforeActiveMemory: Int
    let beforeCacheMemory: Int
    let beforePeakMemory: Int
    let afterActiveMemory: Int
    let afterCacheMemory: Int
    let afterPeakMemory: Int

    var releasedActiveMemory: Int {
        max(0, beforeActiveMemory - afterActiveMemory)
    }

    var releasedCacheMemory: Int {
        max(0, beforeCacheMemory - afterCacheMemory)
    }
}

struct AdapterAuditionVariant: Identifiable, Sendable, Equatable {
    let id: String
    let label: String
    let adapterID: String?
    let adapterName: String?
    let strength: Float?

    var adapterStack: [(id: String, weight: Float)] {
        guard let adapterID, let strength else { return [] }
        return [(id: adapterID, weight: strength)]
    }

    var isBase: Bool {
        adapterID == nil
    }

    static func variants(
        for adapter: LoRAAdapter,
        strengths: [Float],
        includeBase: Bool
    ) -> [AdapterAuditionVariant] {
        var variants: [AdapterAuditionVariant] = []
        if includeBase {
            variants.append(AdapterAuditionVariant(
                id: "base",
                label: "Base",
                adapterID: nil,
                adapterName: nil,
                strength: nil
            ))
        }

        variants.append(contentsOf: normalizedStrengths(strengths).map { strength in
            AdapterAuditionVariant(
                id: "\(adapter.id)-\(String(format: "%.2f", strength))",
                label: "\(String(format: "%.2fx", strength))",
                adapterID: adapter.id,
                adapterName: adapter.name,
                strength: strength
            )
        })
        return variants
    }

    static func normalizedStrengths(_ strengths: [Float]) -> [Float] {
        var seen = Set<Int>()
        return strengths.compactMap { raw in
            guard raw > 0 else { return nil }
            let clamped = AdapterStrengthLimits.clamped(raw)
            let key = Int((clamped * 100).rounded())
            guard seen.insert(key).inserted else { return nil }
            return Float(key) / 100.0
        }
    }

    static func parseStrengths(_ text: String) -> [Float] {
        let separators = CharacterSet(charactersIn: ",; \n\t")
        return normalizedStrengths(
            text.components(separatedBy: separators)
                .compactMap { Float($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
        )
    }

    func listeningNote(index: Int, total: Int, seed: Int, duration: TimeInterval) -> String {
        let target = adapterName.map { "\($0) \(label)" } ?? "Base"
        return "Adapter audition \(index)/\(total): \(target), seed \(seed), \(Int(duration.rounded()))s fixed-seed snapshot."
    }
}

enum GenerationBatchError: LocalizedError {
    case failed(String)
    case cancelled
    case noResult

    var errorDescription: String? {
        switch self {
        case let .failed(message):
            message
        case .cancelled:
            "Generation was cancelled."
        case .noResult:
            "Generation finished without producing a history result."
        }
    }
}

/// Immutable inputs for a single generation submission.
///
/// Captured at `generate()` entry so adapter/UI mutations cannot alter a run
/// after it has been queued — especially across `await` model-load points.
private struct GenerationSubmissionContext: Sendable {
    let id: UUID
    let request: GenerationRequest
    let adapterStack: [(id: String, weight: Float)]
    let selectedAdapterID: String?

    /// Prefers the full stack; falls back to a single selected ID at 1.0.
    var resolvedAdapterStack: [(id: String, weight: Float)] {
        if !adapterStack.isEmpty {
            return adapterStack
        }
        if let selectedAdapterID {
            return [(id: selectedAdapterID, weight: 1.0)]
        }
        return []
    }
}

@MainActor
final class GenerationService: ObservableObject {
    @Published var state: GenerationState = .idle
    @Published var currentResult: GenerationResult?
    var selectedAdapterID: String?
    /// Full adapter stack for multi-adapter generation. When non-empty, this
    /// takes priority over `selectedAdapterID` — the stack is applied via
    /// `LoRACoordinator.ensureStack()` which handles per-adapter strengths.
    var adapterStack: [(id: String, weight: Float)] = []

    /// When set, generation is blocked while training is active (and vice versa).
    weak var trainingService: TrainingService?

    private var generationTask: Task<Void, Never>?
    private var activeGenerationID: UUID?
    /// Set while `unloadAllWeights` owns the lifecycle; generation completion
    /// must not flip state back to a terminal submit-allowed value mid-release.
    private var activeReleaseID: UUID?
    private let aceStepService: ACEStepGenerationService
    private let sfxService: SFXGenerationService
    private let sfxModelManager: SFXModelManager
    private let loraCoordinator: LoRACoordinator
    private let coverSongCoordinator: CoverSongCoordinator
    private let historyStore: HistoryStore
    private let outputDirectoryManager: OutputDirectoryManager
    private let llmManager: LLMModelManager
    private let registry: ModelRegistry
    private let traceMonitor: GenerationTraceMonitor

    init(
        aceStepService: ACEStepGenerationService,
        sfxService: SFXGenerationService,
        sfxModelManager: SFXModelManager,
        loraCoordinator: LoRACoordinator,
        coverSongCoordinator: CoverSongCoordinator,
        historyStore: HistoryStore,
        outputDirectoryManager: OutputDirectoryManager,
        llmManager: LLMModelManager,
        registry: ModelRegistry,
        traceMonitor: GenerationTraceMonitor
    ) {
        self.aceStepService = aceStepService
        self.sfxService = sfxService
        self.sfxModelManager = sfxModelManager
        self.loraCoordinator = loraCoordinator
        self.coverSongCoordinator = coverSongCoordinator
        self.historyStore = historyStore
        self.outputDirectoryManager = outputDirectoryManager
        self.llmManager = llmManager
        self.registry = registry
        self.traceMonitor = traceMonitor
    }

    /// True while a generation task is actively running or models are releasing.
    var isGenerating: Bool {
        switch state {
        case .idle, .complete, .cancelled, .error:
            false
        case .releasing:
            true
        default:
            true
        }
    }

    var canSubmitGeneration: Bool {
        if trainingService?.isTrainingActive == true {
            return false
        }
        switch state {
        case .idle, .complete, .cancelled, .error:
            return true
        case .releasing:
            return false
        default:
            return false
        }
    }

    var outputDirectoryPath: String? {
        outputDirectoryManager.directory?.path(percentEncoded: false)
    }

    func generate(request: GenerationRequest) {
        if trainingService?.isTrainingActive == true {
            state = .error("Cannot generate while training is running. Stop training first.")
            return
        }
        if case .releasing = state {
            state = .error("Cannot generate while models are still releasing. Wait a moment and try again.")
            return
        }

        // Capture submission inputs immutably at submit time so later UI
        // mutations (adapter stack, selection) cannot alter this run.
        let submission = GenerationSubmissionContext(
            id: UUID(),
            request: request,
            adapterStack: adapterStack,
            selectedAdapterID: selectedAdapterID
        )

        let previousTask = generationTask
        previousTask?.cancel()
        activeGenerationID = submission.id
        state = .conditioning

        // [weak self] breaks the retain cycle where generationTask retains self, which retains generationTask.
        // SwiftUI holds GenerationService strongly for the view's lifetime, so self being nil is rare
        // (view dismissed mid-generation), and returning silently is correct — no observer remains.
        generationTask = Task { [weak self] in
            await previousTask?.value
            guard let self else { return }
            guard isCurrentGeneration(submission.id) else { return }
            if Task.isCancelled {
                markGenerationCancelled(submission.id)
                finishGenerationTask(submission.id, reloadLLM: true)
                return
            }

            guard !submission.request.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                state = .error("Prompt cannot be empty")
                finishGenerationTask(submission.id, reloadLLM: true)
                return
            }

            guard let outputDir = outputDirectoryManager.directory else {
                state = .error("No output directory selected")
                finishGenerationTask(submission.id, reloadLLM: true)
                return
            }

            let resolvedRequest = resolveSeedIfNeeded(for: submission.request)
            traceMonitor.start(request: resolvedRequest)
            AudioDiagnostics.memory("generation/begin")
            AudioDiagnostics.note(
                "generation/request",
                Self.diagnosticsRequestSummary(resolvedRequest)
            )
            if submission.request.seed == nil, let seed = resolvedRequest.seed {
                traceMonitor.recordStage("Seed", message: "Resolved random seed", detail: "\(seed)")
            }
            await noteXLSoftLateGuidanceIfNeeded(
                request: resolvedRequest,
                adapterStack: submission.resolvedAdapterStack
            )

            // Free the LLM (~2 GB GPU) before loading DiT + VAE (~5+ GB).
            // The $llmActor publisher propagates nil to PromptEnhancer /
            // IntentRouter / LyricWriterManager automatically, disabling
            // inference for the duration.
            traceMonitor.recordStage("Memory", message: "Evicting lyric model before generation")
            AudioDiagnostics.memory("generation/beforeLLMEvict")
            llmManager.evict()
            AudioDiagnostics.memory("generation/afterLLMEvict")

            do {
                let usesStableAudio = shouldUseStableAudio(request: resolvedRequest)
                let usesCoverSong = !usesStableAudio && shouldUseCoverSong(request: resolvedRequest)
                let pipelineLabel = if usesStableAudio {
                    "stableAudio"
                } else if usesCoverSong {
                    "coverSong"
                } else {
                    "aceStep"
                }
                let mainModelCategory: ModelCategory = usesStableAudio ? .stableAudio : .aceStepDiT
                let mainModel = await historyModelSnapshot(for: mainModelCategory)
                let supportingModels = await supportingModelSnapshots(
                    usesStableAudio: usesStableAudio,
                    usesCoverSong: usesCoverSong
                )
                // Adapter snapshots from the immutable submission — not live UI state.
                let injectionStack = submission.resolvedAdapterStack
                let adapters = usesStableAudio
                    ? []
                    : historyAdapterSnapshots(for: injectionStack)
                traceMonitor.configure(
                    pipeline: pipelineLabel,
                    primaryModel: mainModel,
                    supportingModels: supportingModels,
                    adapters: adapters
                )
                AudioDiagnostics.note(
                    "generation/route",
                    "pipeline=\(pipelineLabel) primary=\(Self.diagnosticsModelDescription(mainModel)) supporting=[\(supportingModels.map { Self.diagnosticsModelDescription($0) }.joined(separator: ","))] adapters=[\(adapters.map { Self.diagnosticsAdapterDescription($0) }.joined(separator: ","))]"
                )
                AudioDiagnostics.memory("generation/afterRoute")

                let audioResult: AudioResult
                if usesStableAudio {
                    AudioDiagnostics.memory("generation/stableAudio/beforeRun")
                    audioResult = try await runSFX(request: resolvedRequest)
                    AudioDiagnostics.memory("generation/stableAudio/afterRun")
                } else if usesCoverSong {
                    if await coverSongCoordinator.needsModelLoad() {
                        setRunningState(.loadingWeights(progress: 0))
                        traceMonitor.recordStage("Models", message: "Loading active cover-song models")
                        traceMonitor.updateLoadingWeights(progress: 0)
                    }
                    AudioDiagnostics.memory("generation/cover/beforeEnsureLoaded")
                    try await coverSongCoordinator.ensureLoaded()
                    AudioDiagnostics.memory("generation/cover/afterEnsureLoaded")

                    AudioDiagnostics.memory("generation/cover/beforeLoRA")
                    AdapterEffectDiagnostics.beginGeneration()
                    await loraCoordinator.clearAll()
                    // Always use ensureStack so strength is applied via contributions.
                    // ensureAdapter ignored strength and used a different load path.
                    if !injectionStack.isEmpty {
                        traceMonitor.recordStage("Adapters", message: "Applying adapter stack to cover diffusion")
                        try await loraCoordinator.ensureStack(stack: injectionStack)
                    }
                    AudioDiagnostics.memory("generation/cover/afterLoRA")

                    AudioDiagnostics.memory("generation/cover/beforeRun")
                    audioResult = try await runCoverSong(request: resolvedRequest)
                    AudioDiagnostics.memory("generation/cover/afterRun")
                } else {
                    if await aceStepService.needsModelLoad() {
                        setRunningState(.loadingWeights(progress: 0))
                        traceMonitor.recordStage("Models", message: "Loading active ACE-Step models")
                        traceMonitor.updateLoadingWeights(progress: 0)
                    }
                    AudioDiagnostics.memory("generation/aceStep/beforeEnsureLoaded")
                    try await aceStepService.ensureLoaded()
                    AudioDiagnostics.memory("generation/aceStep/afterEnsureLoaded")

                    // Model loading may unload/recreate the DiT actor. Reset
                    // LoRA coordinator state so the next injection attaches to
                    // the freshly active architecture (XL = 32 layers, SFT =
                    // 24 layers) instead of stale wrappers from a previous run.
                    AudioDiagnostics.memory("generation/aceStep/beforeLoRA")
                    AdapterEffectDiagnostics.beginGeneration()
                    await loraCoordinator.clearAll()

                    // Always use ensureStack so strength is applied via contributions.
                    // ensureAdapter ignored strength and used a different load path.
                    if !injectionStack.isEmpty {
                        traceMonitor.recordStage("Adapters", message: "Applying adapter stack")
                        try await loraCoordinator.ensureStack(stack: injectionStack)
                    }
                    AudioDiagnostics.memory("generation/aceStep/afterLoRA")
                    AudioDiagnostics.memory("generation/aceStep/beforeRun")
                    audioResult = try await runACEStep(request: resolvedRequest)
                    AudioDiagnostics.memory("generation/aceStep/afterRun")
                }

                try Task.checkCancellation()

                let fileName = HistoryEntry.fileName(prompt: resolvedRequest.text)
                let fileURL = outputDir.appending(path: fileName)

                traceMonitor.recordStage("File", message: "Writing WAV and analyzing waveform", detail: fileName)
                AudioDiagnostics.memory("generation/beforeWriteAnalyze")
                let audioSnapshot = try await Self.writeAudioAndAnalyze(
                    audioResult: audioResult,
                    fileURL: fileURL
                )
                AudioDiagnostics.note(
                    "generation/audioSnapshot",
                    "sampleRate=\(audioSnapshot.sampleRate) duration=\(audioSnapshot.duration) shape=\(audioSnapshot.sampleShape) absMean=\(String(format: "%.4f", audioSnapshot.absMean)) absMax=\(String(format: "%.4f", audioSnapshot.absMax))"
                )
                AudioDiagnostics.memory("generation/afterWriteAnalyze")
                traceMonitor.recordStage(
                    "Audio",
                    message: "Waveform analysis complete",
                    detail: String(format: "absMean %.4f, absMax %.3f", audioSnapshot.absMean, audioSnapshot.absMax),
                    level: .success
                )

                let recordedGuidanceScale = usesStableAudio
                    ? resolvedRequest.guidanceScale
                    : resolvedRequest.effectiveGuidanceScale
                let entryID = UUID()
                let createdAt = Date()
                let traceFileName = HistoryEntry.traceFileName(forAudioFileName: fileName)
                let entry = HistoryEntry(
                    id: entryID,
                    createdAt: createdAt,
                    prompt: resolvedRequest.text,
                    lyrics: resolvedRequest.lyrics,
                    duration: resolvedRequest.duration,
                    seed: resolvedRequest.seed ?? 0,
                    method: resolvedRequest.method.rawValue,
                    guidanceScale: recordedGuidanceScale,
                    precision: resolvedRequest.precision.rawValue,
                    fileName: fileName,
                    traceFileName: traceFileName,
                    pipeline: pipelineLabel,
                    originalPrompt: nil,
                    modelCategory: mainModel?.category,
                    modelID: mainModel?.id,
                    modelName: mainModel?.displayName,
                    supportingModels: supportingModels.isEmpty ? nil : supportingModels,
                    adapters: adapters.isEmpty ? nil : adapters,
                    listeningNote: nil,
                    isPreferred: nil
                )
                let trace = makeGenerationTrace(
                    entryID: entryID,
                    createdAt: createdAt,
                    audioFileName: fileName,
                    traceFileName: traceFileName,
                    pipeline: pipelineLabel,
                    request: resolvedRequest,
                    primaryModel: mainModel,
                    supportingModels: supportingModels,
                    adapters: adapters,
                    timing: audioResult.timeCosts,
                    audioSnapshot: audioSnapshot
                )
                do {
                    try await Self.writeTrace(trace, to: outputDir.appending(path: traceFileName))
                } catch {
                    traceLogger
                        .warning(
                            "Failed to write trace sidecar \(traceFileName, privacy: .public): \(error.localizedDescription, privacy: .public)"
                        )
                }
                traceMonitor.finish(trace: trace)

                if isCurrentGeneration(submission.id), activeReleaseID == nil {
                    historyStore.add(entry)
                    currentResult = GenerationResult(
                        fileURL: fileURL,
                        entry: entry,
                        waveformPeaks: audioSnapshot.waveformPeaks
                    )
                    state = .complete
                }
            } catch is CancellationError {
                AudioDiagnostics.note("generation/cancelled", Self.diagnosticsRequestSummary(resolvedRequest))
                AudioDiagnostics.memory("generation/cancelled")
                markGenerationCancelled(submission.id)
            } catch {
                AudioDiagnostics.note(
                    "generation/error",
                    "\(error.localizedDescription) request=\(Self.diagnosticsRequestSummary(resolvedRequest))"
                )
                AudioDiagnostics.memory("generation/error")
                if isCurrentGeneration(submission.id), activeReleaseID == nil {
                    state = .error(error.localizedDescription)
                    traceMonitor.fail(error.localizedDescription)
                    writeFailureTraceIfPossible(
                        request: resolvedRequest,
                        outputDir: outputDir,
                        errorMessage: error.localizedDescription
                    )
                }
            }

            // Recreate the LLMActor (lazy — no weights loaded yet) so prompt
            // enhancement is available again after generation finishes.
            finishGenerationTask(submission.id, reloadLLM: true)
        }
    }

    func generateAndWait(
        request: GenerationRequest,
        adapterStack requestedAdapterStack: [(id: String, weight: Float)],
        selectedAdapterID requestedSelectedAdapterID: String?
    ) async throws -> GenerationResult {
        let previousSelectedAdapterID = selectedAdapterID
        let previousAdapterStack = adapterStack
        let previousResultID = currentResult?.entry.id

        selectedAdapterID = requestedSelectedAdapterID
        adapterStack = requestedAdapterStack
        defer {
            selectedAdapterID = previousSelectedAdapterID
            adapterStack = previousAdapterStack
        }

        generate(request: request)
        guard let task = generationTask else {
            throw GenerationBatchError.noResult
        }

        await task.value
        try Task.checkCancellation()

        switch state {
        case .complete:
            guard let result = currentResult,
                  result.entry.id != previousResultID else {
                throw GenerationBatchError.noResult
            }
            return result
        case let .error(message):
            throw GenerationBatchError.failed(message)
        case .cancelled:
            throw GenerationBatchError.cancelled
        default:
            throw GenerationBatchError.noResult
        }
    }

    func cancel() {
        generationTask?.cancel()
        // During release, unload owns lifecycle state.
        if activeReleaseID != nil || state == .releasing {
            traceMonitor.cancel()
            return
        }
        state = .cancelled
        traceMonitor.cancel()
    }

    private func isCurrentGeneration(_ generationID: UUID) -> Bool {
        activeGenerationID == generationID
    }

    /// Publish a running-generation state only when no release operation owns lifecycle.
    private func setRunningState(_ newState: GenerationState) {
        guard activeReleaseID == nil else { return }
        if case .releasing = state {
            return
        }
        state = newState
    }

    private func markGenerationCancelled(_ generationID: UUID) {
        guard isCurrentGeneration(generationID) else { return }
        // Unload/release owns the lifecycle while models are being torn down.
        if activeReleaseID != nil || state == .releasing {
            return
        }
        state = .cancelled
        traceMonitor.cancel()
    }

    private func finishGenerationTask(_ generationID: UUID, reloadLLM: Bool) {
        guard isCurrentGeneration(generationID) else { return }
        if reloadLLM, activeReleaseID == nil {
            llmManager.reload()
            AudioDiagnostics.memory("generation/end")
        }
        // Keep the generation ID while a release is in progress so the release
        // operation can still identify the run it cancelled.
        if activeReleaseID == nil {
            activeGenerationID = nil
        }
    }

    private func resolveSeedIfNeeded(for request: GenerationRequest) -> GenerationRequest {
        guard request.seed == nil else { return request }

        return GenerationRequest(
            text: request.text,
            lyrics: request.lyrics,
            referenceAudioURL: request.referenceAudioURL,
            duration: request.duration,
            seed: Int.random(in: 1 ... Int(Int32.max)),
            method: request.method,
            guidanceScale: request.guidanceScale,
            guidancePreset: request.guidancePreset,
            precision: request.precision,
            stepsOverride: request.stepsOverride,
            shiftOverride: request.shiftOverride,
            resolvedMode: request.resolvedMode,
            pipelineOverride: request.pipelineOverride,
            coverStrength: request.coverStrength
        )
    }

    /// Soft Late is a late-only CFG recipe that often degrades XL quality. Leave the
    /// request alone so deliberate A/Bs still work, but log when the risky combo runs.
    private func noteXLSoftLateGuidanceIfNeeded(
        request: GenerationRequest,
        adapterStack: [(id: String, weight: Float)]
    ) async {
        guard request.guidancePreset == .softLate else { return }
        guard !shouldUseStableAudio(request: request) else { return }

        let activeID = await registry.activeModelID(for: .aceStepDiT)
        let activeSpec = await registry.activeSpec(for: .aceStepDiT)
        let isXLModel = activeID.map { LoRAAdapter.isACEBaseModelXL($0, spec: activeSpec) } ?? false
        let isXLAdapter = adapterStack.contains { entry in
            guard let adapter = AdapterRegistry.listAdapters().first(where: { $0.id == entry.id }) else {
                return false
            }
            return adapter.baseModelCategory == .aceStepDiT
                && LoRAAdapter.isACEBaseModelXL(adapter.baseModelID)
        }
        guard isXLModel || isXLAdapter else { return }

        AudioDiagnostics.note(
            "generation/guidance",
            "Soft Late on XL often degrades quality (late-only CFG + forced scale 6.0). Prefer Constant for full-schedule CFG."
        )
        traceMonitor.recordStage(
            "Guidance",
            message: "Soft Late on XL",
            detail: "Late-only CFG can degrade XL quality; Constant is recommended",
            level: .warning
        )
    }

    private nonisolated static func diagnosticsRequestSummary(_ request: GenerationRequest) -> String {
        let seed = request.seed.map { String($0) } ?? "random"
        let stepsOverride = request.stepsOverride.map { String($0) } ?? "nil"
        let shiftOverride = request.shiftOverride.map { String($0) } ?? "nil"
        let referenceAudio = request.referenceAudioURL?.lastPathComponent ?? "nil"

        var parts: [String] = []
        parts.append("duration=\(request.duration)")
        parts.append("seed=\(seed)")
        parts.append("method=\(request.method)")
        parts.append("pipelineOverride=\(request.pipelineOverride.rawValue)")
        parts.append("resolvedMode=\(request.resolvedMode.rawValue)")
        parts.append("guidancePreset=\(request.guidancePreset.rawValue)")
        parts.append("guidancePresetName=\(request.guidancePreset.displayName)")
        parts.append("userGuidance=\(request.guidanceScale)")
        parts.append("effectiveGuidance=\(request.effectiveGuidanceScale)")
        parts.append("cfgInterval=\(request.cfgIntervalStart)...\(request.cfgIntervalEnd)")
        parts.append("precision=\(request.precision.rawValue)")
        parts.append("stepsOverride=\(stepsOverride)")
        parts.append("shiftOverride=\(shiftOverride)")
        parts.append("coverStrength=\(request.coverStrength)")
        parts.append("referenceAudio=\(referenceAudio)")
        parts.append("promptChars=\(request.text.count)")
        parts.append("lyricChars=\(request.lyrics.count)")
        return parts.joined(separator: " ")
    }

    private nonisolated static func diagnosticsModelDescription(_ snapshot: HistoryModelSnapshot?) -> String {
        guard let snapshot else { return "nil" }
        return "\(snapshot.category):\(snapshot.id):\(snapshot.displayName ?? "unnamed")"
    }

    private nonisolated static func diagnosticsAdapterDescription(_ snapshot: HistoryAdapterSnapshot) -> String {
        "id=\(snapshot.id):name=\(snapshot.name ?? "unnamed"):weight=\(String(format: "%.3f", snapshot.weight)):source=\(snapshot.source ?? "unknown"):rank=\(snapshot.rank.map(String.init) ?? "nil"):base=\(snapshot.baseModelID ?? "nil")"
    }

    func unloadAllWeights() async -> MemoryReleaseSummary {
        let releaseID = UUID()
        let cancelledGenerationID = activeGenerationID
        let taskToAwait = generationTask

        // Claim the lifecycle before awaiting cancel so completion handlers
        // cannot flip state back to a submit-allowed terminal value, and so
        // a new generation cannot start mid-unload.
        activeReleaseID = releaseID
        state = .releasing
        taskToAwait?.cancel()

        if let taskToAwait {
            await taskToAwait.value
            // Only clear the task slot if no newer generation replaced it
            // (should not happen while .releasing blocks submit).
            if activeReleaseID == releaseID {
                generationTask = nil
            }
        }

        // If another release superseded this one, do not unload under its feet.
        guard activeReleaseID == releaseID else {
            return MemoryReleaseSummary(
                beforeActiveMemory: 0,
                beforeCacheMemory: 0,
                beforePeakMemory: 0,
                afterActiveMemory: 0,
                afterCacheMemory: 0,
                afterPeakMemory: 0
            )
        }

        // Snapshot MLX memory before/after so the user can see the
        // actual delta in the Xcode console. MLX-Swift maintains an internal
        // allocator pool that is NOT returned to the OS just by setting
        // model references to nil — `Memory.clearCache()` is required.
        let before = Memory.snapshot()
        memLogger
            .info(
                "[unloadAllWeights] before active=\(before.activeMemory) cache=\(before.cacheMemory) peak=\(before.peakMemory)"
            )

        await aceStepService.unloadWeights()
        await coverSongCoordinator.unloadWeights()
        await loraCoordinator.clearAll()
        sfxModelManager.unloadPipeline()
        llmManager.evict(clearCache: false)

        // Return MLX's GPU allocator pool to the OS. Without this, even
        // after DiT/VAE weights are released, multi-GB stays "held" by MLX
        // and Activity Monitor doesn't drop. This is the actual fix for
        // "Release didn't actually free memory."
        await Self.flushMLXGPUAllocator()

        let after = Memory.snapshot()
        memLogger
            .info(
                "[unloadAllWeights] after  active=\(after.activeMemory) cache=\(after.cacheMemory) peak=\(after.peakMemory)"
            )
        memLogger
            .info(
                "[unloadAllWeights] delta active=-\(before.activeMemory - after.activeMemory) cache=-\(before.cacheMemory - after.cacheMemory)"
            )

        if activeReleaseID == releaseID {
            if cancelledGenerationID != nil, activeGenerationID == cancelledGenerationID {
                activeGenerationID = nil
            }
            activeReleaseID = nil
            state = cancelledGenerationID != nil ? .cancelled : .idle
            llmManager.reload()
            AudioDiagnostics.memory("generation/release/end")
        }

        return MemoryReleaseSummary(
            beforeActiveMemory: before.activeMemory,
            beforeCacheMemory: before.cacheMemory,
            beforePeakMemory: before.peakMemory,
            afterActiveMemory: after.activeMemory,
            afterCacheMemory: after.cacheMemory,
            afterPeakMemory: after.peakMemory
        )
    }

    private nonisolated static func flushMLXGPUAllocator() async {
        await Task.detached(priority: .utility) {
            Memory.clearCache()
            GPU.resetPeakMemory()
        }.value
    }

    private nonisolated static func analyzeAudio(
        samples: MLXArray,
        sampleRate: Int,
        duration: TimeInterval
    ) -> GenerationTraceAudioSnapshot {
        let absSamples = MLX.abs(samples)
        let absMeanArray = absSamples.mean()
        let absMaxArray = absSamples.max()
        eval(absMeanArray, absMaxArray)

        let perFrame = perFramePeakAmplitudes(from: samples)
        let waveformPeaks = computeWaveformPeaks(fromPerFramePeaks: perFrame)
        let perSecond = computePerSecondStats(fromPerFramePeaks: perFrame, sampleRate: sampleRate)

        return GenerationTraceAudioSnapshot(
            sampleRate: sampleRate,
            duration: duration,
            sampleShape: samples.shape,
            absMean: absMeanArray.item(Float.self),
            absMax: absMaxArray.item(Float.self),
            waveformPeaks: waveformPeaks,
            perSecondAbsMean: perSecond.means,
            perSecondAbsMax: perSecond.maxes
        )
    }

    private nonisolated static func writeAudioAndAnalyze(
        audioResult: AudioResult,
        fileURL: URL
    ) async throws -> GenerationTraceAudioSnapshot {
        try await Task.detached(priority: .utility) {
            try WAVWriter.write(
                samples: audioResult.samples,
                sampleRate: audioResult.sampleRate,
                to: fileURL
            )
            return analyzeAudio(
                samples: audioResult.samples,
                sampleRate: audioResult.sampleRate,
                duration: audioResult.duration
            )
        }.value
    }

    private nonisolated static func writeTrace(_ trace: GenerationTrace, to url: URL) async throws {
        try await Task.detached(priority: .utility) {
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(trace)
            try data.write(to: url, options: .atomic)
        }.value
    }

    private func writeFailureTraceIfPossible(
        request: GenerationRequest,
        outputDir: URL,
        errorMessage: String
    ) {
        let traceFileName = Self.failureTraceFileName(prompt: request.text)
        guard let failureTrace = traceMonitor.failureTrace(
            fileName: traceFileName,
            errorMessage: errorMessage
        ) else {
            return
        }

        do {
            try historyStore.saveFailureTrace(failureTrace)
            traceMonitor.recordStage(
                "Trace",
                message: "Failure trace written",
                detail: outputDir.appending(path: traceFileName).lastPathComponent,
                level: .warning
            )
        } catch {
            traceLogger
                .warning(
                    "Failed to write failure trace \(traceFileName, privacy: .public): \(error.localizedDescription, privacy: .public)"
                )
        }
    }

    private nonisolated static func failureTraceFileName(prompt: String, date: Date = Date()) -> String {
        let audioName = HistoryEntry.fileName(prompt: prompt, date: date)
        let baseName = (audioName as NSString).deletingPathExtension
        return "\(baseName).failed.trace.json"
    }

    private func historyModelSnapshot(for category: ModelCategory) async -> HistoryModelSnapshot? {
        if let spec = await registry.activeSpec(for: category) {
            return HistoryModelSnapshot(
                category: spec.category.rawValue,
                id: spec.id,
                displayName: spec.displayName
            )
        }

        guard let id = await registry.activeModelID(for: category) else {
            return nil
        }
        return HistoryModelSnapshot(
            category: category.rawValue,
            id: id,
            displayName: nil
        )
    }

    private func supportingModelSnapshots(
        usesStableAudio: Bool,
        usesCoverSong _: Bool
    ) async -> [HistoryModelSnapshot] {
        guard !usesStableAudio else { return [] }

        // Source-audio covers use tokenizer/detokenizer hints from the DiT
        // bundle, not the experimental ACE-Step LM path.
        let categories: [ModelCategory] = [.aceStepVAE]

        var snapshots: [HistoryModelSnapshot] = []
        for category in categories {
            if let snapshot = await historyModelSnapshot(for: category) {
                snapshots.append(snapshot)
            }
        }
        return snapshots
    }

    private func historyAdapterSnapshots(
        for requestedStack: [(id: String, weight: Float)]
    ) -> [HistoryAdapterSnapshot] {
        guard !requestedStack.isEmpty else { return [] }

        let adaptersByID = Dictionary(
            uniqueKeysWithValues: AdapterRegistry.listAdapters().map { ($0.id, $0) }
        )
        return requestedStack.map { requested in
            let adapter = adaptersByID[requested.id]
            return HistoryAdapterSnapshot(
                id: requested.id,
                name: adapter?.name,
                weight: requested.weight,
                source: adapter?.source.rawValue,
                rank: adapter?.config.rank,
                baseModelID: adapter?.baseModelID
            )
        }
    }

    private func makeGenerationTrace(
        entryID: UUID,
        createdAt: Date,
        audioFileName: String,
        traceFileName: String,
        pipeline: String,
        request: GenerationRequest,
        primaryModel: HistoryModelSnapshot?,
        supportingModels: [HistoryModelSnapshot],
        adapters: [HistoryAdapterSnapshot],
        timing: TimeCosts,
        audioSnapshot: GenerationTraceAudioSnapshot
    ) -> GenerationTrace {
        GenerationTrace(
            schemaVersion: GenerationTrace.currentSchemaVersion,
            historyEntryID: entryID,
            createdAt: createdAt,
            audioFileName: audioFileName,
            traceFileName: traceFileName,
            pipeline: pipeline,
            prompt: request.text,
            originalPrompt: nil,
            lyrics: request.lyrics,
            request: GenerationTraceRequestSnapshot(
                duration: request.duration,
                seed: request.seed,
                method: request.method.rawValue,
                guidanceScale: request.effectiveGuidanceScale,
                guidancePreset: request.guidancePreset.rawValue,
                guidancePresetName: request.guidancePreset.displayName,
                cfgIntervalStart: request.cfgIntervalStart,
                cfgIntervalEnd: request.cfgIntervalEnd,
                precision: request.precision.rawValue,
                stepsOverride: request.stepsOverride,
                shiftOverride: request.shiftOverride,
                resolvedMode: request.resolvedMode.rawValue,
                pipelineOverride: request.pipelineOverride.rawValue,
                coverStrength: request.coverStrength,
                referenceAudioFileName: request.referenceAudioURL?.lastPathComponent
            ),
            primaryModel: primaryModel,
            supportingModels: supportingModels,
            adapters: adapters,
            featureFlags: GenerationTraceFeatureFlags(
                lmThinkingEnabled: false,
                cotMetadataEnabled: false,
                cotCaptionEnabled: false,
                dcwEnabled: false
            ),
            timing: GenerationTraceTimingSnapshot(
                diffusionTime: timing.diffusionTime,
                perStepTime: timing.perStepTime,
                totalTime: timing.totalTime
            ),
            audio: audioSnapshot,
            listeningNote: nil,
            isPreferred: nil,
            status: "complete",
            errorMessage: nil,
            events: traceMonitor.currentEventSnapshots()
        )
    }

    private func shouldUseStableAudio(request: GenerationRequest) -> Bool {
        switch request.pipelineOverride {
        case .aceStep: return false
        case .stableAudio: return true
        case .auto:
            // Auto-routing only flips to Stable Audio when the prompt
            // genuinely classifies as SFX/ambient AND the SFX pipeline is
            // actually loadable. Without the runtime guard, an enhanced
            // music prompt that mentions "soundscape / atmospheric / etc."
            // can flip Auto to Ambient → SFX path → crash if the SFX VAE
            // file isn't present in the expected format.
            let wantsSFX = request.resolvedMode == .soundFX || request.resolvedMode == .ambient
            guard wantsSFX else { return false }

            if sfxModelManager.isPipelineLoadable() {
                return true
            }
            // Surface in the live generation trace so Auto→ACE fallback is
            // visible without digging through console logs.
            memLogger
                .info(
                    "[shouldUseStableAudio] auto detected \(String(describing: request.resolvedMode)), but SFX pipeline not loadable — falling back to ACE-Step"
                )
            traceMonitor.recordStage(
                "Routing",
                message: "SFX/ambient requested but Stable Audio is not loadable — using ACE-Step",
                detail: "Install/load Stable Audio models, or set pipeline to ACE-Step explicitly",
                level: .warning
            )
            return false
        }
    }

    private func shouldUseCoverSong(request: GenerationRequest) -> Bool {
        // Cover song mode: ACE-Step pipeline + user-supplied source audio.
        // SFX/ambient requests are handled by shouldUseStableAudio and must
        // NOT reach the cover song path.
        request.referenceAudioURL != nil
    }

    private func runCoverSong(request: GenerationRequest) async throws -> AudioResult {
        for try await event in await coverSongCoordinator.generate(request: request) {
            switch event {
            case let .loadingWeights(progress):
                setRunningState(.loadingWeights(progress: progress))
                traceMonitor.updateLoadingWeights(progress: progress)
            case .conditioning:
                setRunningState(.conditioning)
                traceMonitor.recordStage("Conditioning", message: "Encoding prompt, lyrics, and source audio")
            case .tokenizingSource:
                setRunningState(.tokenizingSource)
                traceMonitor.recordStage("Source Audio", message: "Tokenizing source audio")
            case .lmAudioCodeConditioning:
                setRunningState(.lmAudioCodeConditioning)
                traceMonitor.recordStage("Audio Codes", message: "Running 5 Hz LM audio-code conditioning")
            case let .generating(step, totalSteps, eta):
                setRunningState(.generating(step: step, totalSteps: totalSteps, eta: eta))
                traceMonitor.updateGenerationStep(step: step, totalSteps: totalSteps, eta: eta)
            case .decoding:
                setRunningState(.decoding)
                traceMonitor.recordStage("Decode", message: "Decoding latents to audio")
            case let .result(audioResult):
                traceMonitor.recordStage("Result", message: "Audio returned from cover pipeline", level: .success)
                return audioResult
            }
        }
        throw GenerationError.pipelineNotReady
    }

    private func runSFX(request: GenerationRequest) async throws -> AudioResult {
        for try await event in sfxService.generate(request: request) {
            switch event {
            case let .downloadingModel(progress):
                setRunningState(.downloadingModel(progress: progress))
                traceMonitor.updateLoadingWeights(progress: progress)
            case let .generating(step, totalSteps, eta):
                setRunningState(.generating(step: step, totalSteps: totalSteps, eta: eta))
                traceMonitor.updateGenerationStep(step: step, totalSteps: totalSteps, eta: eta)
            case .decoding:
                setRunningState(.decoding)
                traceMonitor.recordStage("Decode", message: "Decoding latents to audio")
            case let .result(audioResult):
                traceMonitor.recordStage(
                    "Result",
                    message: "Audio returned from Stable Audio pipeline",
                    level: .success
                )
                return audioResult
            }
        }
        throw GenerationError.pipelineNotReady
    }

    private func runACEStep(request: GenerationRequest) async throws -> AudioResult {
        for try await event in await aceStepService.generate(request: request) {
            switch event {
            case let .loadingWeights(progress):
                setRunningState(.loadingWeights(progress: progress))
                traceMonitor.updateLoadingWeights(progress: progress)
            case .conditioning:
                setRunningState(.conditioning)
                traceMonitor.recordStage("Conditioning", message: "Encoding prompt and lyrics")
            case .tokenizingSource:
                // Cover-song-only stage — not emitted by the standard ACE-Step path.
                break
            case .lmAudioCodeConditioning:
                // Cover-song-only stage — not emitted by the standard ACE-Step path.
                break
            case let .generating(step, totalSteps, eta):
                setRunningState(.generating(step: step, totalSteps: totalSteps, eta: eta))
                traceMonitor.updateGenerationStep(step: step, totalSteps: totalSteps, eta: eta)
            case .decoding:
                setRunningState(.decoding)
                traceMonitor.recordStage("Decode", message: "Decoding latents to audio")
            case let .result(audioResult):
                traceMonitor.recordStage("Result", message: "Audio returned from ACE-Step pipeline", level: .success)
                return audioResult
            }
        }
        throw GenerationError.pipelineNotReady
    }

    /// Compute 400-bucket normalised peak amplitudes from a decoded audio MLXArray.
    ///
    /// Handles any rank the VAE decoder can produce:
    ///   1D  [frames]                — mono flat buffer
    ///   2D  [channels, frames]      — interleaved channel-first
    ///   3D  [batch, frames, channels] — standard VAE output (batch=1)
    ///
    /// Returns an empty array on unexpected input so callers get a graceful no-op.
    private nonisolated static func computeWaveformPeaks(
        from samples: MLXArray,
        bucketCount: Int = 400
    ) -> [Float] {
        computeWaveformPeaks(
            fromPerFramePeaks: perFramePeakAmplitudes(from: samples),
            bucketCount: bucketCount
        )
    }

    private nonisolated static func perFramePeakAmplitudes(from samples: MLXArray) -> MLXArray? {
        switch samples.ndim {
        case 1:
            MLX.abs(samples)
        case 2:
            // [channels, frames] → max across channel axis
            MLX.abs(samples).max(axis: 0)
        case 3:
            // [batch=1, frames, channels] → drop batch, max across channels
            MLX.abs(samples[0]).max(axis: -1)
        default:
            nil
        }
    }

    private nonisolated static func computeWaveformPeaks(
        fromPerFramePeaks perFrame: MLXArray?,
        bucketCount: Int = 400
    ) -> [Float] {
        guard let perFrame else { return [] }

        let numFrames = perFrame.shape[0]
        guard numFrames >= bucketCount else { return [] }

        // Bucket into `bucketCount` equal-width windows; take the max per window.
        let trimLen = (numFrames / bucketCount) * bucketCount
        let bucketed = perFrame[0 ..< trimLen]
            .reshaped([bucketCount, numFrames / bucketCount])
            .max(axis: -1) // [bucketCount]
        eval(bucketed)

        // Normalise so the tallest bar reaches 1.0.
        let maxVal = bucketed.max().item(Float.self)
        let normalized = maxVal > 0 ? (bucketed / maxVal) : bucketed
        eval(normalized)

        // Convert MLXArray → Swift [Float].
        let rawData = normalized.asType(.float32).asData(access: .copy).data
        return rawData.withUnsafeBytes { buffer in
            Array(buffer.bindMemory(to: Float.self))
        }
    }

    private nonisolated static func computePerSecondStats(
        fromPerFramePeaks perFrame: MLXArray?,
        sampleRate: Int
    ) -> (means: [Float], maxes: [Float]) {
        guard let perFrame, sampleRate > 0 else { return ([], []) }

        let numFrames = perFrame.shape[0]
        guard numFrames > 0 else { return ([], []) }

        let seconds = Int(ceil(Double(numFrames) / Double(sampleRate)))
        var means: [Float] = []
        var maxes: [Float] = []
        means.reserveCapacity(seconds)
        maxes.reserveCapacity(seconds)

        for second in 0 ..< seconds {
            let start = second * sampleRate
            let end = min(start + sampleRate, numFrames)
            guard start < end else { continue }

            let segment = perFrame[start ..< end]
            let mean = segment.mean()
            let max = segment.max()
            eval(mean, max)
            means.append(mean.item(Float.self))
            maxes.append(max.item(Float.self))
        }

        return (means, maxes)
    }
}

enum GenerationError: Error, LocalizedError {
    case pipelineNotReady
    case modelNotConfigured(String)
    case conditioningIncomplete(String)

    var errorDescription: String? {
        switch self {
        case .pipelineNotReady: "Generation pipeline did not return a result"
        case let .modelNotConfigured(name): "\(name) model is not configured"
        case let .conditioningIncomplete(detail):
            "Text conditioning is incomplete: \(detail)"
        }
    }
}
