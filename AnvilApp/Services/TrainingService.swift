import Foundation
import AnvilCore
import MLX

/// Training state for UI observation.
enum TrainingState: Equatable {
    case idle
    case preparingDataset(current: Int, total: Int)
    case training(epoch: Int, totalEpochs: Int, loss: Float, elapsed: TimeInterval, remaining: TimeInterval)
    case saving
    case complete(LoRAAdapter)
    case failed(String)
    case cancelled
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
    private let conditioning: ConditioningActor
    private let registry: ModelRegistry
    private var trainingTask: Task<Void, Never>?
    private var isStopping = false

    init(
        pipeline: MusicPipeline,
        vae: VAEActor,
        conditioning: ConditioningActor,
        registry: ModelRegistry
    ) {
        self.pipeline = pipeline
        self.vae = vae
        self.conditioning = conditioning
        self.registry = registry
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
        trainingTask?.cancel()
        isStopping = false

        trainingTask = Task { [weak self] in
            guard let self else { return }

            do {
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
                            self?.state = .training(
                                epoch: progress.epoch,
                                totalEpochs: progress.totalEpochs,
                                loss: progress.loss,
                                elapsed: progress.elapsedTime,
                                remaining: progress.estimatedRemaining
                            )
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

            self.trainingTask = nil
        }
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
        if case .training(let epoch, _, _, _, _) = state {
            return epoch >= 2
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
        let audioData = try loadWAVFile(url: audioURL)
        let audioArray = MLXArray(audioData.samples)
            .reshaped([1, 1, audioData.samples.count])

        // Encode to latent via VAE
        let latent = await vae.encodeMean(audio: audioArray, precision: .float16)

        // Encode text conditioning
        let actualDuration: TimeInterval
        if duration > 0 {
            actualDuration = duration
        } else {
            actualDuration = Double(audioData.samples.count) / Double(audioData.sampleRate)
        }

        let cond = await conditioning.encode(
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

    /// Load a WAV file and return normalized float samples and sample rate.
    private func loadWAVFile(url: URL) throws -> (samples: [Float], sampleRate: Int) {
        let data = try Data(contentsOf: url)

        guard data.count > 44 else {
            throw TrainingError.invalidAudioFile(url.lastPathComponent)
        }

        let sampleRate = data.withUnsafeBytes { ptr -> Int in
            Int(ptr.load(fromByteOffset: 24, as: UInt32.self))
        }

        let bitsPerSample = data.withUnsafeBytes { ptr -> Int in
            Int(ptr.load(fromByteOffset: 34, as: UInt16.self))
        }

        let dataOffset = 44
        let dataSize = data.count - dataOffset

        var samples: [Float] = []

        if bitsPerSample == 16 {
            let numSamples = dataSize / 2
            samples.reserveCapacity(numSamples)
            data.withUnsafeBytes { ptr in
                for i in 0..<numSamples {
                    let sample = ptr.load(fromByteOffset: dataOffset + i * 2, as: Int16.self)
                    samples.append(Float(sample) / 32768.0)
                }
            }
        } else if bitsPerSample == 32 {
            let numSamples = dataSize / 4
            samples.reserveCapacity(numSamples)
            data.withUnsafeBytes { ptr in
                for i in 0..<numSamples {
                    let sample = ptr.load(fromByteOffset: dataOffset + i * 4, as: Float.self)
                    samples.append(sample)
                }
            }
        } else {
            throw TrainingError.unsupportedAudioFormat(bitsPerSample)
        }

        return (samples, sampleRate)
    }
}

enum TrainingError: Error, LocalizedError {
    case invalidAudioFile(String)
    case unsupportedAudioFormat(Int)
    case noSamples

    var errorDescription: String? {
        switch self {
        case .invalidAudioFile(let name):
            return "Invalid audio file: \(name)"
        case .unsupportedAudioFormat(let bits):
            return "Unsupported audio format: \(bits)-bit"
        case .noSamples:
            return "No valid training samples"
        }
    }
}
