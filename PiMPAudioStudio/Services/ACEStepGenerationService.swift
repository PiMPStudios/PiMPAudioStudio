// PiMPAudioStudio/Services/ACEStepGenerationService.swift
import Foundation
import PiMPAudioCore
import MLX
import MLXNN
import OSLog

private let logger = Logger(subsystem: "com.pimpstudios.PiMPAudioStudio", category: "ACEStepGenerationService")

/// Handles ACE-Step generation: weight loading (DiT + VAE), ConditioningActor lifecycle,
/// and diffusion. Internal to the GenerationService facade.
actor ACEStepGenerationService {
    /// Text-to-music generations consistently show a near-silent VAE tail of
    /// about two seconds. Generate a small guard band, then crop back to the
    /// requested duration so the audible body reaches the UI/export boundary.
    private static let textToMusicTailGuardSeconds: TimeInterval = 2.0

    enum Event {
        case loadingWeights(progress: Float)
        case conditioning
        case tokenizingSource
        case lmAudioCodeConditioning
        case generating(step: Int, totalSteps: Int, eta: TimeInterval)
        case decoding
        case result(AudioResult)
    }

    private let pipeline: MusicPipeline
    private var conditioning: ConditioningActor   // var — upgraded after DiT download
    private let weightConverter: WeightConverter
    private let registry: ModelRegistry

    /// Called after the real ConditioningActor (with full tokenizer + weights) replaces the
    /// launch-time stub. TrainingService uses this to stay in sync with the generation path.
    var onConditioningUpgraded: (@Sendable (ConditioningActor) async -> Void)?

    private var currentlyLoadedDiTModelID: String?
    private var currentlyLoadedVAEModelID: String?
    private var currentlyLoadedDiTConfig: DiTConfig?

    init(
        pipeline: MusicPipeline,
        conditioning: ConditioningActor,
        weightConverter: WeightConverter,
        registry: ModelRegistry,
        onConditioningUpgraded: (@Sendable (ConditioningActor) async -> Void)? = nil
    ) {
        self.pipeline = pipeline
        self.conditioning = conditioning
        self.weightConverter = weightConverter
        self.registry = registry
        self.onConditioningUpgraded = onConditioningUpgraded
    }

    /// On-demand: checks active model IDs vs loaded IDs, loads/swaps weights if stale, then generates.
    func generate(request: GenerationRequest) -> AsyncThrowingStream<Event, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let activeDiTID = await registry.activeModelID(for: .aceStepDiT)
                    let activeVAEID = await registry.activeModelID(for: .aceStepVAE)
                    if activeDiTID != currentlyLoadedDiTModelID || activeVAEID != currentlyLoadedVAEModelID {
                        continuation.yield(.loadingWeights(progress: 0))
                    }
                    try await ensureLoaded()

                    try Task.checkCancellation()
                    continuation.yield(.conditioning)

                    // Guard against guidance values that disable CFG. ACE-Step is
                    // an instruction-tuned diffusion model — guidance ≤ 1.0 runs
                    // the unconditional path and produces near-silent noise.
                    // Mirror SFXGenerationService's defensive fallback here.
                    let effectiveGuidance = request.effectiveGuidanceScale

                    // Auto-pick diffusion step count based on the active DiT
                    // variant. Turbo is trained for 8 steps; Full (base) and
                    // SFT need ~50 steps to converge to good quality. Running
                    // Full or SFT at 8 steps produces unstable / noisy output.
                    // The user can override via Advanced settings.
                    let loadedDiTID = currentlyLoadedDiTModelID ?? ""
                    let autoSteps: Int = (currentlyLoadedDiTConfig?.isTurbo == true || loadedDiTID.contains("turbo"))
                        ? 8
                        : 50
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
                        cfgIntervalStart: request.cfgIntervalStart,
                        cfgIntervalEnd: request.cfgIntervalEnd,
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

                    let generationDuration = Self.durationWithTailGuard(request.duration)
                    if generationDuration > request.duration {
                        AudioDiagnostics.note(
                            "config/tailGuard",
                            "requested=\(request.duration) internal=\(generationDuration) crop=true"
                        )
                    }

                    let generatedAudio = try await pipeline.generate(
                        text: request.text,
                        lyrics: request.lyrics,
                        referenceAudio: nil,
                        duration: generationDuration,
                        config: config,
                        conditioning: conditioning
                    )
                    let audioResult = Self.crop(generatedAudio, toDuration: request.duration)

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

    private static func durationWithTailGuard(_ requested: TimeInterval) -> TimeInterval {
        guard requested >= 10 else { return requested }
        return requested + min(textToMusicTailGuardSeconds, requested * 0.10)
    }

    private static func crop(_ result: AudioResult, toDuration requestedDuration: TimeInterval) -> AudioResult {
        let requestedFrames = max(1, Int((requestedDuration * Double(result.sampleRate)).rounded()))
        guard result.samples.shape.count >= 3, result.samples.shape[1] > requestedFrames else {
            return AudioResult(
                samples: result.samples,
                sampleRate: result.sampleRate,
                duration: requestedDuration,
                timeCosts: result.timeCosts
            )
        }

        let cropped = result.samples[0..., 0..<requestedFrames, 0...]
        eval(cropped)
        AudioDiagnostics.note(
            "vae/out_audio_crop",
            "internalSamples=\(result.samples.shape[1]) requestedSamples=\(requestedFrames)"
        )
        return AudioResult(
            samples: cropped,
            sampleRate: result.sampleRate,
            duration: requestedDuration,
            timeCosts: result.timeCosts
        )
    }

    /// Unloads all model weights. Call before model swap or on memory pressure.
    func unloadWeights() async {
        await pipeline.unloadDiTWeights()
        await pipeline.unloadVAEWeights()
        await conditioning.unloadWeights()
        currentlyLoadedDiTModelID = nil
        currentlyLoadedDiTConfig = nil
        currentlyLoadedVAEModelID = nil
    }

    func needsModelLoad() async -> Bool {
        let activeDiTID = await registry.activeModelID(for: .aceStepDiT)
        let activeVAEID = await registry.activeModelID(for: .aceStepVAE)
        return activeDiTID != currentlyLoadedDiTModelID || activeVAEID != currentlyLoadedVAEModelID
    }

    /// Loads the ACE-Step stack without generating audio.
    /// Generation calls this before LoRA injection so adapters attach to the
    /// currently active DiT architecture, not whatever model happened to be
    /// resident from a previous run. Training also uses this path when launched
    /// cold from Settings.
    func ensureLoaded() async throws {
        let activeDiTID = await registry.activeModelID(for: .aceStepDiT)
        guard let activeDiTID else {
            throw GenerationError.modelNotConfigured("ACE-Step DiT")
        }
        if activeDiTID != currentlyLoadedDiTModelID {
            try await loadDiTWeights()
        }

        let activeVAEID = await registry.activeModelID(for: .aceStepVAE)
        guard let activeVAEID else {
            throw GenerationError.modelNotConfigured("ACE-Step VAE")
        }
        if activeVAEID != currentlyLoadedVAEModelID {
            try await loadVAEWeights()
        }
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
        currentlyLoadedDiTConfig = nil

        do {
            let configPath = spec.localPath(for: "config.json", base: baseDir)
            let ditConfig = try DiTConfig.load(from: configPath)
            guard ditConfig.isSupportedBySwiftRuntime else {
                throw GenerationError.modelNotConfigured(ditConfig.runtimeSupportNote ?? "Unsupported ACE-Step DiT architecture")
            }
            await pipeline.configureDiT(config: ditConfig)

            let weightPaths = ditWeightPaths(for: spec, baseDir: baseDir)
            let rawWeights = try await weightConverter.loadSafetensors(from: weightPaths)
            let weights = try await weightConverter.convertDiTWeights(stateDict: rawWeights)
            await pipeline.loadDiTWeights(weights, modelID: spec.id)
            currentlyLoadedDiTModelID = spec.id
            currentlyLoadedDiTConfig = ditConfig

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

            // Load silence_latent.pt (the trained "no source audio" prior).
            // Lives next to model.safetensors in each DiT checkpoint dir.
            // Required for proper text2music context_latents construction —
            // see ConditioningActor.encode for how it's used.
            let silencePath = spec.localPath(for: "silence_latent.pt", base: baseDir)
            let silenceFSPath = silencePath.path(percentEncoded: false)
            if FileManager.default.fileExists(atPath: silenceFSPath) {
                do {
                    let silence = try SilenceLatentLoader.load(from: silencePath)
                    await conditioning.setSilenceLatent(silence)
                    logger.info("Loaded silence_latent.pt shape=\(silence.shape)")
                } catch {
                    logger.warning("Failed to load silence_latent.pt — falling back to zeros: \(error.localizedDescription)")
                    await conditioning.setSilenceLatent(nil)
                }
            } else {
                logger.info("silence_latent.pt not present at \(silenceFSPath) — falling back to zeros for text2music src_latents")
                await conditioning.setSilenceLatent(nil)
            }

            // Notify dependents (e.g. TrainingService) that a fully-initialised
            // ConditioningActor is now available.
            await onConditioningUpgraded?(conditioning)

        } catch {
            if let previousModelID {
                try? await registry.setActiveModel(id: previousModelID, for: .aceStepDiT)
            }
            throw error
        }
    }

    private func ditWeightPaths(for spec: ModelSpec, baseDir: URL) -> [URL] {
        let paths = spec.files
            .filter { file in
                file.localName == "model.safetensors"
                    || (file.localName.hasPrefix("model-") && file.localName.hasSuffix(".safetensors"))
            }
            .map { spec.localPath(for: $0.localName, base: baseDir) }

        return paths.isEmpty
            ? [spec.localPath(for: "model.safetensors", base: baseDir)]
            : paths
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

#if DEBUG
    /// Phase 2A smoke test: load AudioTokenizer, AudioTokenDetokenizer, and
    /// AceStepLM weights, log per-module match counts. Intended to be called
    /// once from a debug menu or via lldb after a generation run.
    /// Will be removed when Phase 2B introduces CoverSongCoordinator.
    func testCoverSongFoundation() async {
        guard let ditSpec = await registry.activeSpec(for: .aceStepDiT) else {
            logger.warning("[cover-smoke] no active DiT spec")
            return
        }
        let baseDir = await registry.baseDirectory
        let modelSafetensors = ditSpec.files
            .filter { $0.localName.hasSuffix(".safetensors") && $0.localName != "vae.safetensors" }
            .map { ditSpec.localPath(for: $0.localName, base: baseDir) }
        let ditWeightsURLs = modelSafetensors.isEmpty
            ? [ditSpec.localPath(for: "model.safetensors", base: baseDir)]
            : modelSafetensors

        do {
            let raw = try await weightConverter.loadSafetensors(from: ditWeightsURLs)

            // Audio tokenizer
            let tokWeights = await weightConverter.convertAudioTokenizerWeights(from: raw)
            let tokenizer = AudioTokenizer()
            let tokModelKeys = Set(tokenizer.parameters().flattened().map { $0.0 })
            let tokSuppliedKeys = Set(tokWeights.map { $0.0 })
            logger.info("[cover-smoke] tokenizer supplied=\(tokSuppliedKeys.count) model=\(tokModelKeys.count) matched=\(tokSuppliedKeys.intersection(tokModelKeys).count)")
            tokenizer.update(parameters: ModuleParameters.unflattened(tokWeights))
            eval(tokenizer.parameters())

            // Audio detokenizer
            let detokWeights = await weightConverter.convertAudioTokenDetokenizerWeights(from: raw)
            let detok = AudioTokenDetokenizer()
            let detokModelKeys = Set(detok.parameters().flattened().map { $0.0 })
            let detokSuppliedKeys = Set(detokWeights.map { $0.0 })
            logger.info("[cover-smoke] detokenizer supplied=\(detokSuppliedKeys.count) model=\(detokModelKeys.count) matched=\(detokSuppliedKeys.intersection(detokModelKeys).count)")
            detok.update(parameters: ModuleParameters.unflattened(detokWeights))
            eval(detok.parameters())

            // LM (try smaller variants first, fall back upward if only a larger one is downloaded)
            let lmIDs = ["ace-step-lm-0.6b", "ace-step-lm-1.7b", "ace-step-lm-4b"]
            var lmLoaded = false
            for lmID in lmIDs {
                let allSpecs = await registry.allSpecs
                guard let lmSpec = allSpecs.first(where: { $0.id == lmID }) else { continue }
                let lmDir = lmSpec.localDirectory(base: baseDir)
                let configURL = lmDir.appending(path: "config.json")
                guard FileManager.default.fileExists(atPath: configURL.path(percentEncoded: false)) else { continue }
                let config = try AceStepLMConfig.load(from: configURL)

                let weightsURL = lmDir.appending(path: "model.safetensors")
                let indexURL = lmDir.appending(path: "model.safetensors.index.json")
                let lmRaw: [(String, MLXArray)]
                if FileManager.default.fileExists(atPath: indexURL.path(percentEncoded: false)) {
                    lmRaw = try await weightConverter.loadShardedSafetensors(indexURL: indexURL)
                } else if FileManager.default.fileExists(atPath: weightsURL.path(percentEncoded: false)) {
                    lmRaw = try await weightConverter.loadSafetensors(from: weightsURL)
                } else {
                    continue
                }
                let lmWeights = await weightConverter.convertAceStepLMWeights(from: lmRaw)
                let lm = AceStepLM(config: config)
                let lmModelKeys = Set(lm.parameters().flattened().map { $0.0 })
                let lmSuppliedKeys = Set(lmWeights.map { $0.0 })
                logger.info("[cover-smoke] LM \(lmID) supplied=\(lmSuppliedKeys.count) model=\(lmModelKeys.count) matched=\(lmSuppliedKeys.intersection(lmModelKeys).count)")
                lm.update(parameters: ModuleParameters.unflattened(lmWeights))
                eval(lm.parameters())
                lmLoaded = true
                break
            }
            if !lmLoaded {
                logger.warning("[cover-smoke] no LM variant available; download one from Settings → Models → ACE-Step LM")
            }
        } catch {
            logger.error("[cover-smoke] failed: \(error.localizedDescription)")
        }
    }
#endif
}
