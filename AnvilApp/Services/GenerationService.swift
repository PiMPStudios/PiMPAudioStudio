import Foundation
import AnvilCore
import MLX

struct GenerationResult: Sendable {
    let fileURL: URL
    let entry: HistoryEntry
}

@MainActor
final class GenerationService: ObservableObject {
    @Published var state: GenerationState = .idle
    @Published var currentResult: GenerationResult?

    private var generationTask: Task<Void, Never>?
    private let pipeline: MusicPipeline
    private let conditioning: ConditioningActor
    private let weightConverter: WeightConverter
    private let historyStore: HistoryStore
    private let outputDirectoryManager: OutputDirectoryManager
    private var weightsLoaded = false

    init(
        pipeline: MusicPipeline,
        conditioning: ConditioningActor,
        weightConverter: WeightConverter,
        historyStore: HistoryStore,
        outputDirectoryManager: OutputDirectoryManager
    ) {
        self.pipeline = pipeline
        self.conditioning = conditioning
        self.weightConverter = weightConverter
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
                if !self.weightsLoaded {
                    self.state = .loadingWeights(progress: 0)
                    try await self.loadWeights()
                    self.weightsLoaded = true
                }

                try Task.checkCancellation()
                self.state = .conditioning

                let config = DiffusionConfig(
                    seed: request.seed,
                    method: request.method,
                    guidanceScale: request.guidanceScale,
                    precision: request.precision
                )

                let totalSteps = config.steps ?? 8
                self.state = .generating(step: 0, totalSteps: totalSteps, eta: 0)

                let progressStream = await self.pipeline.progress
                let progressTask = Task { [weak self] in
                    guard let self else { return }
                    for await progress in progressStream {
                        if case .diffusion(let dp) = progress {
                            let perStep = dp.currentStep > 0
                                ? dp.elapsedTime / Double(dp.currentStep)
                                : 0
                            let eta = perStep * Double(dp.totalSteps - dp.currentStep)
                            self.state = .generating(
                                step: dp.currentStep,
                                totalSteps: dp.totalSteps,
                                eta: eta
                            )
                        }
                    }
                }
                defer { progressTask.cancel() }

                let result = try await self.pipeline.generate(
                    text: request.text,
                    lyrics: request.lyrics,
                    referenceAudio: nil,
                    duration: request.duration,
                    config: config,
                    conditioning: self.conditioning
                )

                try Task.checkCancellation()
                self.state = .decoding

                let fileName = HistoryEntry.fileName(prompt: request.text)
                let fileURL = outputDir.appending(path: fileName)

                try WAVWriter.write(
                    samples: result.samples,
                    sampleRate: result.sampleRate,
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
                    fileName: fileName
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

    private func loadWeights() async throws {
        state = .loadingWeights(progress: 1.0)
    }
}
