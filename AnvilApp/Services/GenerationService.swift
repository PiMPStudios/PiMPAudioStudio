import Foundation
import AnvilCore
import MLX
import OSLog

private let memLogger = Logger(subsystem: "com.pimp.AnvilApp", category: "GenerationService.unload")

struct GenerationResult: Sendable {
    let fileURL: URL
    let entry: HistoryEntry
    /// Pre-computed normalised peak amplitudes (400 buckets, values 0–1).
    /// Computed from the decoded MLXArray at generation time so the player
    /// can display the waveform immediately without re-reading the WAV file.
    let waveformPeaks: [Float]?
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

    init(
        aceStepService: ACEStepGenerationService,
        sfxService: SFXGenerationService,
        sfxModelManager: SFXModelManager,
        loraCoordinator: LoRACoordinator,
        coverSongCoordinator: CoverSongCoordinator,
        historyStore: HistoryStore,
        outputDirectoryManager: OutputDirectoryManager,
        llmManager: LLMModelManager
    ) {
        self.aceStepService = aceStepService
        self.sfxService = sfxService
        self.sfxModelManager = sfxModelManager
        self.loraCoordinator = loraCoordinator
        self.coverSongCoordinator = coverSongCoordinator
        self.historyStore = historyStore
        self.outputDirectoryManager = outputDirectoryManager
        self.llmManager = llmManager
    }

    func generate(request: GenerationRequest) {
        generationTask?.cancel()

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

            // Free the LLM (~2 GB GPU) before loading DiT + VAE (~5+ GB).
            // The $llmActor publisher propagates nil to PromptEnhancer /
            // IntentRouter / LyricWriterManager automatically, disabling
            // inference for the duration.
            self.llmManager.evict()

            do {
                let usesStableAudio = self.shouldUseStableAudio(request: request)
                let usesCoverSong = !usesStableAudio && self.shouldUseCoverSong(request: request)
                let audioResult: AudioResult
                if usesStableAudio {
                    audioResult = try await self.runSFX(request: request)
                } else if usesCoverSong {
                    audioResult = try await self.runCoverSong(request: request)
                } else {
                    if !self.adapterStack.isEmpty {
                        await self.loraCoordinator.ensureStack(stack: self.adapterStack)
                    } else {
                        await self.loraCoordinator.ensureAdapter(selectedID: self.selectedAdapterID)
                    }
                    audioResult = try await self.runACEStep(request: request)
                }

                try Task.checkCancellation()

                let fileName = HistoryEntry.fileName(prompt: request.text)
                let fileURL = outputDir.appending(path: fileName)

                let peaks = try await Self.writeAudioAndComputeWaveformPeaks(
                    audioResult: audioResult,
                    fileURL: fileURL
                )

                let pipelineLabel: String
                if usesStableAudio {
                    pipelineLabel = "stableAudio"
                } else if usesCoverSong {
                    pipelineLabel = "coverSong"
                } else {
                    pipelineLabel = "aceStep"
                }
                let recordedGuidanceScale = usesStableAudio
                    ? request.guidanceScale
                    : request.effectiveGuidanceScale
                let entry = HistoryEntry(
                    id: UUID(),
                    createdAt: Date(),
                    prompt: request.text,
                    lyrics: request.lyrics,
                    duration: request.duration,
                    seed: request.seed ?? 0,
                    method: request.method.rawValue,
                    guidanceScale: recordedGuidanceScale,
                    precision: request.precision.rawValue,
                    fileName: fileName,
                    pipeline: pipelineLabel,
                    originalPrompt: nil
                )

                self.historyStore.add(entry)
                self.currentResult = GenerationResult(fileURL: fileURL, entry: entry, waveformPeaks: peaks)
                self.state = .complete
            } catch is CancellationError {
                self.state = .cancelled
            } catch {
                self.state = .error(error.localizedDescription)
            }

            // Recreate the LLMActor (lazy — no weights loaded yet) so prompt
            // enhancement is available again after generation finishes.
            self.llmManager.reload()
        }
    }

    func cancel() {
        generationTask?.cancel()
        state = .cancelled
    }

    func unloadAllWeights() async {
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
    }

    private nonisolated static func flushMLXGPUAllocator() async {
        await Task.detached(priority: .utility) {
            MLX.GPU.clearCache()
            MLX.GPU.resetPeakMemory()
        }.value
    }

    private nonisolated static func writeAudioAndComputeWaveformPeaks(
        audioResult: AudioResult,
        fileURL: URL
    ) async throws -> [Float]? {
        try await Task.detached(priority: .utility) {
            try WAVWriter.write(
                samples: audioResult.samples,
                sampleRate: audioResult.sampleRate,
                to: fileURL
            )
            return computeWaveformPeaks(from: audioResult.samples)
        }.value
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
            case .conditioning:
                self.state = .conditioning
            case .tokenizingSource:
                self.state = .tokenizingSource
            case .generating(let step, let totalSteps, let eta):
                self.state = .generating(step: step, totalSteps: totalSteps, eta: eta)
            case .decoding:
                self.state = .decoding
            case .result(let audioResult):
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
            case .generating(let step, let totalSteps, let eta):
                self.state = .generating(step: step, totalSteps: totalSteps, eta: eta)
            case .decoding:
                self.state = .decoding
            case .result(let audioResult):
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
            case .conditioning:
                self.state = .conditioning
            case .tokenizingSource:
                // Cover-song-only stage — not emitted by the standard ACE-Step path.
                break
            case .generating(let step, let totalSteps, let eta):
                self.state = .generating(step: step, totalSteps: totalSteps, eta: eta)
            case .decoding:
                self.state = .decoding
            case .result(let audioResult):
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
        // Collapse channels → 1-D per-frame peak amplitude.
        let perFrame: MLXArray
        switch samples.ndim {
        case 1:
            perFrame = MLX.abs(samples)
        case 2:
            // [channels, frames] → max across channel axis
            perFrame = MLX.abs(samples).max(axis: 0)
        case 3:
            // [batch=1, frames, channels] → drop batch, max across channels
            perFrame = MLX.abs(samples[0]).max(axis: -1)
        default:
            return []
        }

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
