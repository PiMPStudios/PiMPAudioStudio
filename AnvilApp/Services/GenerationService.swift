import Foundation
import AnvilCore

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
    private let loraCoordinator: LoRACoordinator
    private let historyStore: HistoryStore
    private let outputDirectoryManager: OutputDirectoryManager

    init(
        aceStepService: ACEStepGenerationService,
        sfxService: SFXGenerationService,
        loraCoordinator: LoRACoordinator,
        historyStore: HistoryStore,
        outputDirectoryManager: OutputDirectoryManager
    ) {
        self.aceStepService = aceStepService
        self.sfxService = sfxService
        self.loraCoordinator = loraCoordinator
        self.historyStore = historyStore
        self.outputDirectoryManager = outputDirectoryManager
    }

    func generate(request: GenerationRequest) {
        generationTask?.cancel()

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
                let audioResult: AudioResult
                if self.shouldUseStableAudio(request: request) {
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
                    pipeline: nil,
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
        await aceStepService.unloadWeights()
        await loraCoordinator.clearAll()
    }

    private func shouldUseStableAudio(request: GenerationRequest) -> Bool {
        switch request.pipelineOverride {
        case .aceStep: return false
        case .stableAudio: return true
        case .auto:
            return request.resolvedMode == .soundFX || request.resolvedMode == .ambient
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
        for try await event in aceStepService.generate(request: request) {
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
        case .pipelineNotReady: return "Sound FX pipeline is not ready"
        case .modelNotConfigured(let name): return "\(name) model is not configured"
        }
    }
}
