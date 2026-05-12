import Foundation
import AnvilCore
import MLX

/// Training state for UI observation.
enum TrainingState: Equatable {
    case idle
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

    private let pipeline: MusicPipeline
    private let vae: VAEActor
    private var conditioning: ConditioningActor
    /// `false` until ACEStepGenerationService swaps in the real ConditioningActor
    /// (real Qwen3 tokenizer + loaded weights). Training crashes on the stub because
    /// the SFT prompt template uses newlines outside the stub's ASCII-only vocab.
    private var isConditioningReady = false
    private let registry: ModelRegistry
    private var trainingTask: Task<Void, Never>?
    private var isStopping = false
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
    ///   - importedFiles: Additional audio files with text descriptions.
    ///   - adapterName: User-facing name for the resulting adapter.
    ///   - config: LoRA training configuration.
    func startTraining(
        entries: [HistoryEntry],
        audioURLs: [URL],
        importedFiles: [(url: URL, description: String)],
        adapterName: String,
        config: LoRAConfig
    ) {
        guard isConditioningReady else {
            state = .failed(
                "The AI model hasn't been loaded yet. Generate a track first, then return here to train."
            )
            return
        }

        trainingTask?.cancel()
        isStopping = false

        trainingTask = Task { [weak self] in
            guard let self else { return }

            do {
                // Clear stack during training (restore after)
                if let sm = self.stackManager, sm.isStacking {
                    self.savedStack = sm.additionalAdapters
                    sm.clearStack()
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
                        lyrics: "",
                        duration: 0
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
                    config: config,
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

                // 3. Save adapter
                self.state = .saving

                let adapterID = adapterName
                    .lowercased()
                    .replacingOccurrences(of: " ", with: "-")
                    .replacingOccurrences(
                        of: "[^a-z0-9\\-]",
                        with: "",
                        options: .regularExpression
                    )

                let activeModelID = await self.registry.activeModelID(
                    for: .aceStepDiT
                ) ?? "unknown"

                let adapter = LoRAAdapter(
                    id: adapterID,
                    name: adapterName,
                    config: config,
                    baseModelID: activeModelID,
                    baseModelCategory: .aceStepDiT,
                    createdAt: Date(),
                    sampleCount: samples.count,
                    finalLoss: result.finalLoss
                )

                try AdapterRegistry.save(adapter: adapter, weights: result.weights)
                self.state = .complete(adapter)

            } catch is CancellationError {
                self.state = .cancelled
            } catch {
                self.state = .failed(error.localizedDescription)
            }

            // Restore stack if it was cleared
            if let sm = self.stackManager, !self.savedStack.isEmpty {
                for entry in self.savedStack {
                    sm.addToStack(entry.adapter, strength: entry.strength)
                }
                self.savedStack = []
            }

            self.trainingTask = nil
        }
    }

    /// Swap in a fully-loaded ConditioningActor (real tokenizer + model weights).
    /// Called by AppServices when ACEStepGenerationService upgrades from the stub.
    func updateConditioning(_ actor: ConditioningActor) {
        conditioning = actor
        isConditioningReady = true
    }

    /// Stop training. Pipeline will save best checkpoint if available.
    func stopTraining() {
        isStopping = true
        Task {
            await pipeline.cancelTraining()
        }
    }

    /// Whether training can produce a usable checkpoint on stop.
    var canSaveOnStop: Bool {
        if case .training(let uiState) = state {
            return uiState.epoch >= 2
        }
        return false
    }

    // MARK: - Private

    private func preprocessSample(
        audioURL: URL,
        prompt: String,
        lyrics: String,
        duration: TimeInterval
    ) async throws -> TrainingSample {
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

        return TrainingSample(
            latent: latent,
            encoderStates: cond.encoderStates,
            contextLatents: cond.contextLatents,
            encoderMask: cond.encoderMask
        )
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
