import Foundation
import AnvilCore
import MLX
import OSLog

private let memLogger = Logger(subsystem: "com.pimp.AnvilApp", category: "GenerationService.unload")

struct GenerationResult: Sendable {
    let fileURL: URL
    let entry: HistoryEntry
}

@MainActor
final class GenerationService: ObservableObject {
    @Published var state: GenerationState = .idle
    @Published var currentResult: GenerationResult?
    var selectedAdapterID: String?

    private var generationTask: Task<Void, Never>?
    private let aceStepService: ACEStepGenerationService
    private let sfxService: SFXGenerationService
    private let sfxModelManager: SFXModelManager
    private let loraCoordinator: LoRACoordinator
    private let historyStore: HistoryStore
    private let outputDirectoryManager: OutputDirectoryManager

    init(
        aceStepService: ACEStepGenerationService,
        sfxService: SFXGenerationService,
        sfxModelManager: SFXModelManager,
        loraCoordinator: LoRACoordinator,
        historyStore: HistoryStore,
        outputDirectoryManager: OutputDirectoryManager
    ) {
        self.aceStepService = aceStepService
        self.sfxService = sfxService
        self.sfxModelManager = sfxModelManager
        self.loraCoordinator = loraCoordinator
        self.historyStore = historyStore
        self.outputDirectoryManager = outputDirectoryManager
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

            do {
                let usesStableAudio = self.shouldUseStableAudio(request: request)
                let audioResult: AudioResult
                if usesStableAudio {
                    audioResult = try await self.runSFX(request: request)
                } else {
                    await self.loraCoordinator.ensureAdapter(selectedID: self.selectedAdapterID)
                    audioResult = try await self.runACEStep(request: request)
                }

                try Task.checkCancellation()

                let fileName = HistoryEntry.fileName(prompt: request.text)
                let fileURL = outputDir.appending(path: fileName)

                try WAVWriter.write(
                    samples: audioResult.samples,
                    sampleRate: audioResult.sampleRate,
                    to: fileURL
                )

                let entry = HistoryEntry(
                    id: UUID(),
                    createdAt: Date(),
                    prompt: request.text,
                    lyrics: request.lyrics,
                    duration: request.duration,
                    seed: request.seed ?? 0,
                    method: request.method.rawValue,
                    guidanceScale: request.guidanceScale,
                    precision: request.precision.rawValue,
                    fileName: fileName,
                    pipeline: usesStableAudio ? "stableAudio" : "aceStep",
                    originalPrompt: nil
                )

                self.historyStore.add(entry)
                self.currentResult = GenerationResult(fileURL: fileURL, entry: entry)
                self.state = .complete
            } catch is CancellationError {
                self.state = .cancelled
            } catch {
                self.state = .error(error.localizedDescription)
            }
        }
    }

    func cancel() {
        generationTask?.cancel()
        state = .cancelled
    }

    func unloadAllWeights() async {
        // Snapshot MLX GPU memory before/after so the user can see the
        // actual delta in the Xcode console. MLX-Swift maintains an internal
        // allocator pool that is NOT returned to the OS just by setting
        // model references to nil — `clearCache()` is required.
        let before = MLX.GPU.snapshot()
        memLogger.info("[unloadAllWeights] before active=\(before.activeMemory) cache=\(before.cacheMemory) peak=\(before.peakMemory)")

        await aceStepService.unloadWeights()
        await loraCoordinator.clearAll()

        // Return MLX's GPU allocator pool to the OS. Without this, even
        // after DiT/VAE weights are released, multi-GB stays "held" by MLX
        // and Activity Monitor doesn't drop. This is the actual fix for
        // "Release didn't actually free memory."
        MLX.GPU.clearCache()
        MLX.GPU.resetPeakMemory()

        let after = MLX.GPU.snapshot()
        memLogger.info("[unloadAllWeights] after  active=\(after.activeMemory) cache=\(after.cacheMemory) peak=\(after.peakMemory)")
        memLogger.info("[unloadAllWeights] delta active=-\(before.activeMemory - after.activeMemory) cache=-\(before.cacheMemory - after.cacheMemory)")

        // NOTE: Conditioning sub-modules (Qwen3 ~1.2 GB, lyric/timbre/
        // condition encoders, silence_latent) are NOT unloaded here yet —
        // their fields are `let` on ConditioningActor and need a refactor
        // to make replaceable. Tracked as a follow-up: most of the savings
        // already come from DiT (4.8 GB) + VAE (0.3 GB) + clearCache.
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
