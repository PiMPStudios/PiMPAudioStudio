// AnvilApp/Services/ACEStepGenerationService.swift
import Foundation
import AnvilCore
import OSLog

private let logger = Logger(subsystem: "com.pimp.AnvilApp", category: "ACEStepGenerationService")

/// Handles ACE-Step generation: weight loading (DiT + VAE), ConditioningActor lifecycle,
/// and diffusion. Internal to the GenerationService facade.
@MainActor
final class ACEStepGenerationService {
    enum Event {
        case loadingWeights(progress: Float)
        case conditioning
        case generating(step: Int, totalSteps: Int, eta: TimeInterval)
        case decoding
        case result(AudioResult)
    }

    private let pipeline: MusicPipeline
    private var conditioning: ConditioningActor   // var — upgraded after DiT download
    private let weightConverter: WeightConverter
    private let registry: ModelRegistry

    private var currentlyLoadedDiTModelID: String?
    private var currentlyLoadedVAEModelID: String?

    init(
        pipeline: MusicPipeline,
        conditioning: ConditioningActor,
        weightConverter: WeightConverter,
        registry: ModelRegistry
    ) {
        self.pipeline = pipeline
        self.conditioning = conditioning
        self.weightConverter = weightConverter
        self.registry = registry
    }

    /// On-demand: checks active model IDs vs loaded IDs, loads/swaps weights if stale, then generates.
    func generate(request: GenerationRequest) -> AsyncThrowingStream<Event, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    // Check DiT model
                    let activeDiTID = await registry.activeModelID(for: .aceStepDiT)
                    guard let activeDiTID else {
                        throw GenerationError.modelNotConfigured("ACE-Step DiT")
                    }
                    if activeDiTID != currentlyLoadedDiTModelID {
                        continuation.yield(.loadingWeights(progress: 0))
                        try await loadDiTWeights()
                    }

                    // Check VAE model
                    let activeVAEID = await registry.activeModelID(for: .aceStepVAE)
                    guard let activeVAEID else {
                        throw GenerationError.modelNotConfigured("ACE-Step VAE")
                    }
                    if activeVAEID != currentlyLoadedVAEModelID {
                        try await loadVAEWeights()
                    }

                    try Task.checkCancellation()
                    continuation.yield(.conditioning)

                    let config = DiffusionConfig(
                        seed: request.seed,
                        method: request.method,
                        guidanceScale: request.guidanceScale,
                        precision: request.precision
                    )

                    let totalSteps = config.steps ?? 8
                    continuation.yield(.generating(step: 0, totalSteps: totalSteps, eta: 0))

                    let progressStream = await pipeline.makeProgressStream()
                    let progressTask = Task {
                        for await progress in progressStream {
                            if case .diffusion(let dp) = progress {
                                let perStep = dp.currentStep > 0
                                    ? dp.elapsedTime / Double(dp.currentStep) : 0
                                let eta = perStep * Double(dp.totalSteps - dp.currentStep)
                                continuation.yield(.generating(
                                    step: dp.currentStep,
                                    totalSteps: dp.totalSteps,
                                    eta: eta
                                ))
                            }
                        }
                    }
                    defer { progressTask.cancel() }

                    let audioResult = try await pipeline.generate(
                        text: request.text,
                        lyrics: request.lyrics,
                        referenceAudio: nil,
                        duration: request.duration,
                        config: config,
                        conditioning: conditioning
                    )

                    continuation.yield(.decoding)
                    continuation.yield(.result(audioResult))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// Unloads all model weights. Call before model swap or on memory pressure.
    func unloadWeights() async {
        await pipeline.unloadDiTWeights()
        await pipeline.unloadVAEWeights()
        currentlyLoadedDiTModelID = nil
        currentlyLoadedVAEModelID = nil
    }

    // MARK: - Weight Loading

    private func loadDiTWeights() async throws {
        guard let spec = await registry.activeSpec(for: .aceStepDiT) else {
            throw GenerationError.modelNotConfigured("ACE-Step DiT")
        }
        let baseDir = await registry.baseDirectory
        let previousModelID = currentlyLoadedDiTModelID

        await pipeline.unloadDiTWeights()
        currentlyLoadedDiTModelID = nil

        do {
            let weightsPath = spec.localPath(for: "model.safetensors", base: baseDir)
            let rawWeights = try await weightConverter.loadSafetensors(from: weightsPath)
            let weights = try await weightConverter.convertDiTWeights(stateDict: rawWeights)
            await pipeline.loadDiTWeights(weights, modelID: spec.id)
            currentlyLoadedDiTModelID = spec.id

            // Upgrade stub ConditioningActor to one backed by the real downloaded tokenizer.
            let tokenizerPath = spec.localPath(for: "tokenizer.json", base: baseDir)
            if FileManager.default.fileExists(atPath: tokenizerPath.path()) {
                if let real = try? ConditioningActor.make(tokenizerURL: tokenizerPath) {
                    conditioning = real
                } else {
                    logger.warning("Could not upgrade ConditioningActor from tokenizer at \(tokenizerPath.path()) — using stub")
                }
            }
        } catch {
            if let previousModelID {
                try? await registry.setActiveModel(id: previousModelID, for: .aceStepDiT)
            }
            throw error
        }
    }

    private func loadVAEWeights() async throws {
        guard let spec = await registry.activeSpec(for: .aceStepVAE) else {
            throw GenerationError.modelNotConfigured("ACE-Step VAE")
        }
        let baseDir = await registry.baseDirectory
        let previousModelID = currentlyLoadedVAEModelID

        await pipeline.unloadVAEWeights()
        currentlyLoadedVAEModelID = nil

        do {
            let vaeFile = spec.files.first(where: { $0.contains("vae") }) ?? "vae.safetensors"
            let weightsPath = spec.localPath(for: vaeFile, base: baseDir)
            let rawWeights = try await weightConverter.loadSafetensors(from: weightsPath)
            let weights = try await weightConverter.convertVAEWeights(stateDict: rawWeights)
            await pipeline.loadVAEWeights(weights, modelID: spec.id)
            currentlyLoadedVAEModelID = spec.id
        } catch {
            if let previousModelID {
                try? await registry.setActiveModel(id: previousModelID, for: .aceStepVAE)
            }
            throw error
        }
    }
}
