import Foundation
import AnvilCore
import MLX
import OSLog

private let memLogger = Logger(subsystem: "com.pimp.AnvilApp", category: "GenerationService.unload")
private let traceLogger = Logger(subsystem: "com.pimp.AnvilApp", category: "GenerationTrace")

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

@MainActor
final class GenerationService: ObservableObject {
    @Published var state: GenerationState = .idle
    @Published var currentResult: GenerationResult?
    var selectedAdapterID: String?
    /// Full adapter stack for multi-adapter generation. When non-empty, this
    /// takes priority over `selectedAdapterID` — the stack is applied via
    /// `LoRACoordinator.ensureStack()` which handles per-adapter strengths.
    var adapterStack: [(id: String, weight: Float)] = []

    private var generationTask: Task<Void, Never>?
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

    var canSubmitGeneration: Bool {
        switch state {
        case .idle, .complete, .cancelled, .error:
            return true
        default:
            return false
        }
    }

    var outputDirectoryPath: String? {
        outputDirectoryManager.directory?.path(percentEncoded: false)
    }

    func generate(request: GenerationRequest) {
        generationTask?.cancel()
        state = .conditioning

        // [weak self] breaks the retain cycle where generationTask retains self, which retains generationTask.
        // SwiftUI holds GenerationService strongly for the view's lifetime, so self being nil is rare
        // (view dismissed mid-generation), and returning silently is correct — no observer remains.
        generationTask = Task { [weak self] in
            guard let self else { return }

            guard !request.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                self.state = .error("Prompt cannot be empty")
                return
            }

            guard let outputDir = self.outputDirectoryManager.directory else {
                self.state = .error("No output directory selected")
                return
            }

            let resolvedRequest = self.resolveSeedIfNeeded(for: request)
            self.traceMonitor.start(request: resolvedRequest)
            if request.seed == nil, let seed = resolvedRequest.seed {
                self.traceMonitor.recordStage("Seed", message: "Resolved random seed", detail: "\(seed)")
            }

            // Free the LLM (~2 GB GPU) before loading DiT + VAE (~5+ GB).
            // The $llmActor publisher propagates nil to PromptEnhancer /
            // IntentRouter / LyricWriterManager automatically, disabling
            // inference for the duration.
            self.traceMonitor.recordStage("Memory", message: "Evicting lyric model before generation")
            self.llmManager.evict()

            do {
                let usesStableAudio = self.shouldUseStableAudio(request: resolvedRequest)
                let usesCoverSong = !usesStableAudio && self.shouldUseCoverSong(request: resolvedRequest)
                let pipelineLabel: String
                if usesStableAudio {
                    pipelineLabel = "stableAudio"
                } else if usesCoverSong {
                    pipelineLabel = "coverSong"
                } else {
                    pipelineLabel = "aceStep"
                }
                let mainModelCategory: ModelCategory = usesStableAudio ? .stableAudio : .aceStepDiT
                let mainModel = await self.historyModelSnapshot(for: mainModelCategory)
                let supportingModels = await self.supportingModelSnapshots(
                    usesStableAudio: usesStableAudio,
                    usesCoverSong: usesCoverSong
                )
                let adapters = usesStableAudio || usesCoverSong
                    ? []
                    : self.historyAdapterSnapshots()
                self.traceMonitor.configure(
                    pipeline: pipelineLabel,
                    primaryModel: mainModel,
                    supportingModels: supportingModels,
                    adapters: adapters
                )

                let audioResult: AudioResult
                if usesStableAudio {
                    audioResult = try await self.runSFX(request: resolvedRequest)
                } else if usesCoverSong {
                    audioResult = try await self.runCoverSong(request: resolvedRequest)
                } else {
                    if await self.aceStepService.needsModelLoad() {
                        self.state = .loadingWeights(progress: 0)
                        self.traceMonitor.recordStage("Models", message: "Loading active ACE-Step models")
                        self.traceMonitor.updateLoadingWeights(progress: 0)
                    }
                    try await self.aceStepService.ensureLoaded()

                    // Model loading may unload/recreate the DiT actor. Reset
                    // LoRA coordinator state so the next injection attaches to
                    // the freshly active architecture (XL = 32 layers, SFT =
                    // 24 layers) instead of stale wrappers from a previous run.
                    await self.loraCoordinator.clearAll()

                    if !self.adapterStack.isEmpty {
                        self.traceMonitor.recordStage("Adapters", message: "Applying adapter stack")
                        await self.loraCoordinator.ensureStack(stack: self.adapterStack)
                    } else {
                        self.traceMonitor.recordStage("Adapters", message: "Applying selected adapter")
                        await self.loraCoordinator.ensureAdapter(selectedID: self.selectedAdapterID)
                    }
                    audioResult = try await self.runACEStep(request: resolvedRequest)
                }

                try Task.checkCancellation()

                let fileName = HistoryEntry.fileName(prompt: resolvedRequest.text)
                let fileURL = outputDir.appending(path: fileName)

                self.traceMonitor.recordStage("File", message: "Writing WAV and analyzing waveform", detail: fileName)
                let audioSnapshot = try await Self.writeAudioAndAnalyze(
                    audioResult: audioResult,
                    fileURL: fileURL
                )
                self.traceMonitor.recordStage(
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
                let trace = self.makeGenerationTrace(
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
                    traceLogger.warning("Failed to write trace sidecar \(traceFileName, privacy: .public): \(error.localizedDescription, privacy: .public)")
                }
                self.traceMonitor.finish(trace: trace)

                self.historyStore.add(entry)
                self.currentResult = GenerationResult(
                    fileURL: fileURL,
                    entry: entry,
                    waveformPeaks: audioSnapshot.waveformPeaks
                )
                self.state = .complete
            } catch is CancellationError {
                self.state = .cancelled
                self.traceMonitor.cancel()
            } catch {
                self.state = .error(error.localizedDescription)
                self.traceMonitor.fail(error.localizedDescription)
                self.writeFailureTraceIfPossible(
                    request: resolvedRequest,
                    outputDir: outputDir,
                    errorMessage: error.localizedDescription
                )
            }

            // Recreate the LLMActor (lazy — no weights loaded yet) so prompt
            // enhancement is available again after generation finishes.
            self.llmManager.reload()
        }
    }

    func cancel() {
        generationTask?.cancel()
        state = .cancelled
        traceMonitor.cancel()
    }

    private func resolveSeedIfNeeded(for request: GenerationRequest) -> GenerationRequest {
        guard request.seed == nil else { return request }

        return GenerationRequest(
            text: request.text,
            lyrics: request.lyrics,
            referenceAudioURL: request.referenceAudioURL,
            duration: request.duration,
            seed: Int.random(in: 1...Int(Int32.max)),
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

    func unloadAllWeights() async -> MemoryReleaseSummary {
        if let generationTask {
            generationTask.cancel()
            self.generationTask = nil
            state = .cancelled
            await generationTask.value
        }

        // Snapshot MLX GPU memory before/after so the user can see the
        // actual delta in the Xcode console. MLX-Swift maintains an internal
        // allocator pool that is NOT returned to the OS just by setting
        // model references to nil — `clearCache()` is required.
        let before = MLX.GPU.snapshot()
        memLogger.info("[unloadAllWeights] before active=\(before.activeMemory) cache=\(before.cacheMemory) peak=\(before.peakMemory)")

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

        let after = MLX.GPU.snapshot()
        memLogger.info("[unloadAllWeights] after  active=\(after.activeMemory) cache=\(after.cacheMemory) peak=\(after.peakMemory)")
        memLogger.info("[unloadAllWeights] delta active=-\(before.activeMemory - after.activeMemory) cache=-\(before.cacheMemory - after.cacheMemory)")

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
            MLX.GPU.clearCache()
            MLX.GPU.resetPeakMemory()
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
            traceLogger.warning("Failed to write failure trace \(traceFileName, privacy: .public): \(error.localizedDescription, privacy: .public)")
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
        usesCoverSong: Bool
    ) async -> [HistoryModelSnapshot] {
        guard !usesStableAudio else { return [] }

        var categories: [ModelCategory] = [.aceStepVAE]
        if usesCoverSong {
            categories.append(.aceStepLM)
        }

        var snapshots: [HistoryModelSnapshot] = []
        for category in categories {
            if let snapshot = await historyModelSnapshot(for: category) {
                snapshots.append(snapshot)
            }
        }
        return snapshots
    }

    private func historyAdapterSnapshots() -> [HistoryAdapterSnapshot] {
        let requestedStack: [(id: String, weight: Float)]
        if !adapterStack.isEmpty {
            requestedStack = adapterStack
        } else if let selectedAdapterID {
            requestedStack = [(id: selectedAdapterID, weight: 1.0)]
        } else {
            requestedStack = []
        }

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
            memLogger.info("[shouldUseStableAudio] auto detected \(String(describing: request.resolvedMode)), but SFX pipeline not loadable — falling back to ACE-Step")
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
            case .loadingWeights(let progress):
                self.state = .loadingWeights(progress: progress)
                self.traceMonitor.updateLoadingWeights(progress: progress)
            case .conditioning:
                self.state = .conditioning
                self.traceMonitor.recordStage("Conditioning", message: "Encoding prompt, lyrics, and source audio")
            case .tokenizingSource:
                self.state = .tokenizingSource
                self.traceMonitor.recordStage("Source Audio", message: "Tokenizing source audio")
            case .generating(let step, let totalSteps, let eta):
                self.state = .generating(step: step, totalSteps: totalSteps, eta: eta)
                self.traceMonitor.updateGenerationStep(step: step, totalSteps: totalSteps, eta: eta)
            case .decoding:
                self.state = .decoding
                self.traceMonitor.recordStage("Decode", message: "Decoding latents to audio")
            case .result(let audioResult):
                self.traceMonitor.recordStage("Result", message: "Audio returned from cover pipeline", level: .success)
                return audioResult
            }
        }
        throw GenerationError.pipelineNotReady
    }

    private func runSFX(request: GenerationRequest) async throws -> AudioResult {
        for try await event in sfxService.generate(request: request) {
            switch event {
            case .downloadingModel(let progress):
                self.state = .downloadingModel(progress: progress)
                self.traceMonitor.updateLoadingWeights(progress: progress)
            case .generating(let step, let totalSteps, let eta):
                self.state = .generating(step: step, totalSteps: totalSteps, eta: eta)
                self.traceMonitor.updateGenerationStep(step: step, totalSteps: totalSteps, eta: eta)
            case .decoding:
                self.state = .decoding
                self.traceMonitor.recordStage("Decode", message: "Decoding latents to audio")
            case .result(let audioResult):
                self.traceMonitor.recordStage("Result", message: "Audio returned from Stable Audio pipeline", level: .success)
                return audioResult
            }
        }
        throw GenerationError.pipelineNotReady
    }

    private func runACEStep(request: GenerationRequest) async throws -> AudioResult {
        for try await event in await aceStepService.generate(request: request) {
            switch event {
            case .loadingWeights(let progress):
                self.state = .loadingWeights(progress: progress)
                self.traceMonitor.updateLoadingWeights(progress: progress)
            case .conditioning:
                self.state = .conditioning
                self.traceMonitor.recordStage("Conditioning", message: "Encoding prompt and lyrics")
            case .tokenizingSource:
                // Cover-song-only stage — not emitted by the standard ACE-Step path.
                break
            case .generating(let step, let totalSteps, let eta):
                self.state = .generating(step: step, totalSteps: totalSteps, eta: eta)
                self.traceMonitor.updateGenerationStep(step: step, totalSteps: totalSteps, eta: eta)
            case .decoding:
                self.state = .decoding
                self.traceMonitor.recordStage("Decode", message: "Decoding latents to audio")
            case .result(let audioResult):
                self.traceMonitor.recordStage("Result", message: "Audio returned from ACE-Step pipeline", level: .success)
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
            return MLX.abs(samples)
        case 2:
            // [channels, frames] → max across channel axis
            return MLX.abs(samples).max(axis: 0)
        case 3:
            // [batch=1, frames, channels] → drop batch, max across channels
            return MLX.abs(samples[0]).max(axis: -1)
        default:
            return nil
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
        let bucketed = perFrame[0..<trimLen]
            .reshaped([bucketCount, numFrames / bucketCount])
            .max(axis: -1)                         // [bucketCount]
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

        for second in 0..<seconds {
            let start = second * sampleRate
            let end = min(start + sampleRate, numFrames)
            guard start < end else { continue }

            let segment = perFrame[start..<end]
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

    var errorDescription: String? {
        switch self {
        case .pipelineNotReady: return "Generation pipeline did not return a result"
        case .modelNotConfigured(let name): return "\(name) model is not configured"
        }
    }
}
