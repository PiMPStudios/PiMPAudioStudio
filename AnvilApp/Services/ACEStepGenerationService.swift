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

                    // Guard against guidance values that disable CFG. ACE-Step is
                    // an instruction-tuned diffusion model — guidance ≤ 1.0 runs
                    // the unconditional path and produces near-silent noise.
                    // Mirror SFXGenerationService's defensive fallback here.
                    let effectiveGuidance: Float = request.guidanceScale > 1.0
                        ? request.guidanceScale
                        : 7.5

                    // Auto-pick diffusion step count based on the active DiT
                    // variant. Turbo is trained for 8 steps; Full (base) and
                    // SFT need ~50 steps to converge to good quality. Running
                    // Full or SFT at 8 steps produces unstable / noisy output.
                    // The user can override via Advanced settings.
                    let loadedDiTID = currentlyLoadedDiTModelID ?? ""
                    let autoSteps: Int = loadedDiTID.contains("turbo") ? 8 : 50
                    let inferSteps: Int = request.stepsOverride ?? autoSteps

                    // Shift defaults to the upstream MLX-handler value (3.0).
                    // Honor user override when present.
                    let shift: Float = request.shiftOverride ?? 3.0

                    let config = DiffusionConfig(
                        seed: request.seed,
                        method: request.method,
                        shift: shift,
                        steps: inferSteps,
                        guidanceScale: effectiveGuidance,
                        precision: request.precision
                    )

                    let totalSteps = inferSteps
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
            // tokenizer.json lives in the VAE spec's local directory (shared between DiT variants).
            // Use path(percentEncoded: false) so paths containing "Application Support" (with
            // its space) match the actual filesystem layout, not the percent-encoded URL form.
            if let vaeSpec = await registry.activeSpec(for: .aceStepVAE) {
                let tokenizerPath = vaeSpec.localPath(for: "tokenizer.json", base: baseDir)
                let tokenizerFSPath = tokenizerPath.path(percentEncoded: false)
                if FileManager.default.fileExists(atPath: tokenizerFSPath) {
                    if let real = try? ConditioningActor.make(tokenizerURL: tokenizerPath) {
                        conditioning = real
                        logger.info("Upgraded ConditioningActor with tokenizer at \(tokenizerFSPath)")
                    } else {
                        logger.warning("Could not upgrade ConditioningActor from tokenizer at \(tokenizerFSPath) — using stub")
                    }
                } else {
                    logger.info("tokenizer.json not yet downloaded at \(tokenizerFSPath) — keeping stub conditioning actor")
                }

                // Load conditioning weights:
                //   - Qwen3 text encoder weights from VAE spec's qwen3_model.safetensors
                //   - lyric/timbre/text-projector encoder weights from the DiT bundle's
                //     `encoder.*` subtree (already in `rawWeights` we just loaded)
                let qwen3Path = vaeSpec.localPath(for: "qwen3_model.safetensors", base: baseDir)
                let qwen3FSPath = qwen3Path.path(percentEncoded: false)
                if FileManager.default.fileExists(atPath: qwen3FSPath) {
                    do {
                        let qwen3Raw = try await weightConverter.loadSafetensors(from: qwen3Path)
                        let qwen3Weights = await weightConverter.convertQwen3Weights(from: qwen3Raw)
                        let condWeights = await weightConverter.convertConditionEncoderWeights(from: rawWeights)
                        await conditioning.loadWeights(qwen3Weights: qwen3Weights, conditionWeights: condWeights)
                        logger.info("Loaded conditioning weights: qwen3=\(qwen3Weights.count), condition=\(condWeights.count)")
                    } catch {
                        logger.warning("Failed to load conditioning weights — using default-init: \(error.localizedDescription)")
                    }
                } else {
                    logger.info("qwen3_model.safetensors not yet downloaded at \(qwen3FSPath) — text conditioning will use random embeddings")
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
            let weightsPath = spec.localPath(for: "vae.safetensors", base: baseDir)
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
