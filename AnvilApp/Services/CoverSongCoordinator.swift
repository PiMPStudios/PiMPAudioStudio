// AnvilApp/Services/CoverSongCoordinator.swift
import Foundation
import AnvilCore
import MLX
import MLXNN
import OSLog

private let logger = Logger(subsystem: "com.pimp.AnvilApp", category: "CoverSongCoordinator")

/// Orchestrates cover song generation with two preprocessing paths:
///
/// **With LM (Phase 2B):**
///  1. Load + resample source audio to 48 kHz stereo
///  2. VAE-encode to latent domain (deterministic `encodeMean`)
///  3. `AudioTokenizer`: 25 Hz latents → 5 Hz FSQ indices
///  4. `AceStepLM`: FSQ indices → contextual hidden states
///  5. `AudioTokenDetokenizer`: LM hidden states → 25 Hz acoustic hints
///  6. Blend: `(1−strength)·srcLatents + strength·hints`
///  7. Standard ACE-Step diffusion via `MusicPipeline`
///
/// **Without LM (fallback):**
///  Steps 3–5 are: tokenizer quantized output → detokenizer directly (no LM).
///  This path activates when no LM variant is downloaded.
///
/// Weight management mirrors `ACEStepGenerationService`: DiT + VAE + tokenizer +
/// detokenizer + LM are loaded lazily on first call and reloaded when the active
/// model ID changes.
actor CoverSongCoordinator {
    private let pipeline: MusicPipeline
    private var conditioning: ConditioningActor
    private let weightConverter: WeightConverter
    private let registry: ModelRegistry

    private var audioTokenizer: AudioTokenizer?
    private var audioDetokenizer: AudioTokenDetokenizer?
    private var aceStepLM: AceStepLM?
    private var currentlyLoadedDiTModelID: String?
    private var currentlyLoadedVAEModelID: String?
    private var currentlyLoadedLMModelID: String?
    private var currentlyLoadedDiTConfig: DiTConfig?

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

    // MARK: - Main Entry Point

    func generate(request: GenerationRequest) -> AsyncThrowingStream<ACEStepGenerationService.Event, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    guard let referenceAudioURL = request.referenceAudioURL else {
                        throw CoverSongError.noSourceAudio
                    }

                    // Ensure DiT weights (and tokenizer/detokenizer from same bundle)
                    let activeDiTID = await self.registry.activeModelID(for: .aceStepDiT)
                    guard let activeDiTID else {
                        throw GenerationError.modelNotConfigured("ACE-Step DiT")
                    }
                    if activeDiTID != self.currentlyLoadedDiTModelID {
                        continuation.yield(.loadingWeights(progress: 0))
                        try await self.loadDiTAndTokenizerWeights()
                    }

                    // Ensure VAE weights
                    let activeVAEID = await self.registry.activeModelID(for: .aceStepVAE)
                    guard let activeVAEID else {
                        throw GenerationError.modelNotConfigured("ACE-Step VAE")
                    }
                    if activeVAEID != self.currentlyLoadedVAEModelID {
                        try await self.loadVAEWeights()
                    }

                    try Task.checkCancellation()
                    continuation.yield(.conditioning)

                    // Load source audio from disk → 48 kHz stereo MLXArray [1, frames, 2]
                    let audioNLC = try AudioFileLoader.load(from: referenceAudioURL)
                    logger.info("[CoverSongCoordinator] source audio loaded shape=\(audioNLC.shape)")

                    // VAE-encode (deterministic mean) → srcLatents [1, T, 64]
                    let srcLatents = try await self.pipeline.encodeMean(
                        audio: audioNLC,
                        precision: request.precision
                    )
                    logger.info("[CoverSongCoordinator] srcLatents shape=\(srcLatents.shape)")

                    // Cover-song preprocessing: tokenize → (LM) → detokenize → blend
                    continuation.yield(.tokenizingSource)
                    let blendedLatents = self.preprocessForCoverSong(
                        srcLatents: srcLatents,
                        coverStrength: request.coverStrength
                    )
                    logger.info("[CoverSongCoordinator] blendedLatents shape=\(blendedLatents.shape)")

                    // Build DiffusionConfig (mirrors ACEStepGenerationService logic)
                    let effectiveGuidance: Float = request.guidanceScale > 1.0
                        ? request.guidanceScale : 7.5
                    let loadedDiTID = self.currentlyLoadedDiTModelID ?? ""
                    let autoSteps: Int =
                        (self.currentlyLoadedDiTConfig?.isTurbo == true ||
                         loadedDiTID.contains("turbo")) ? 8 : 50
                    let inferSteps = request.stepsOverride ?? autoSteps
                    let shift: Float = request.shiftOverride ?? 3.0
                    let config = DiffusionConfig(
                        seed: request.seed,
                        method: request.method,
                        shift: shift,
                        steps: inferSteps,
                        guidanceScale: effectiveGuidance,
                        precision: request.precision
                    )

                    continuation.yield(.generating(step: 0, totalSteps: inferSteps, eta: 0))

                    // Progress relay from pipeline → continuation
                    let progressStream = await self.pipeline.makeProgressStream()
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

                    // Encode conditioning:
                    //   - referenceAudio → timbre tokens (via encodeSample in VAEActor)
                    //   - srcLatentsOverride → context latents (blended, bypassing
                    //     the silence-latent / reference-latent paths in ConditioningActor)
                    let cond = try await self.conditioning.encode(
                        text: request.text,
                        lyrics: request.lyrics,
                        referenceAudio: audioNLC,
                        duration: request.duration,
                        precision: request.precision,
                        srcLatentsOverride: blendedLatents
                    )

                    // Run diffusion with the pre-computed conditioning result
                    let audioResult = try await self.pipeline.generate(
                        encoderStates: cond.encoderStates,
                        contextLatents: cond.contextLatents,
                        encoderMask: cond.encoderMask,
                        duration: request.duration,
                        config: config
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

    /// Unload all cached weights. Call before model swap or on memory pressure.
    func unloadWeights() async {
        await pipeline.unloadDiTWeights()
        await pipeline.unloadVAEWeights()
        currentlyLoadedDiTModelID = nil
        currentlyLoadedDiTConfig = nil
        currentlyLoadedVAEModelID = nil
        currentlyLoadedLMModelID = nil
        audioTokenizer = nil
        audioDetokenizer = nil
        aceStepLM = nil
    }

    // MARK: - Audio Preprocessing

    /// Tokenize → (optionally LM-refine) → detokenize → blend with source latents.
    ///
    /// When `aceStepLM` is loaded, the pipeline runs through the LM for contextual
    /// hidden states before detokenizing — this produces higher-quality covers
    /// because the LM re-contextualizes each 5 Hz token against its neighbors.
    ///
    /// Falls back to `srcLatents` unchanged when the tokenizer is not yet loaded
    /// (graceful degradation — the diffusion still runs, just without the acoustic
    /// hint blending that makes it a "cover").
    private func preprocessForCoverSong(
        srcLatents: MLXArray,
        coverStrength: Float
    ) -> MLXArray {
        guard let tokenizer = audioTokenizer, let detokenizer = audioDetokenizer else {
            logger.warning("[CoverSongCoordinator] tokenizer not loaded — using srcLatents directly")
            return srcLatents
        }

        let T = srcLatents.shape[1]
        let pool = tokenizer.poolWindowSize

        // AudioTokenizer requires T divisible by poolWindowSize (5).
        // Pad the tail with zeros (silence in latent space).
        let paddedT = ((T + pool - 1) / pool) * pool
        let padded: MLXArray
        if paddedT == T {
            padded = srcLatents
        } else {
            let padding = MLXArray.zeros([1, paddedT - T, 64])
            padded = MLX.concatenated([srcLatents, padding], axis: 1)
        }

        // Tokenize: [1, paddedT, 64] → quantized [1, paddedT/pool, 2048] + indices [1, paddedT/pool, 1]
        let (quantized, indices) = tokenizer(padded)

        // Choose the detokenizer input based on whether the LM is available.
        let detokInput: MLXArray
        if let lm = aceStepLM {
            // LM path: indices [1, T/pool, 1] → squeeze → [1, T/pool] → LM → [1, T/pool, hidden]
            let squeezedIndices = indices.squeezed(axis: -1)  // [1, T/pool]
            logger.info("[CoverSongCoordinator] LM forward: indices shape=\(squeezedIndices.shape)")
            let lmHidden = lm(audioCodes: squeezedIndices, attentionMask: nil)  // [1, T/pool, lm_hidden]
            logger.info("[CoverSongCoordinator] LM output shape=\(lmHidden.shape)")
            detokInput = lmHidden
        } else {
            // No-LM fallback: pass raw quantized vectors directly.
            logger.info("[CoverSongCoordinator] no LM loaded — using quantized tokens directly")
            detokInput = quantized
        }

        // Detokenize: [1, paddedT/pool, hidden] → [1, paddedT, 64]
        let upsampled = detokenizer(detokInput)

        // Trim padding back to original T
        let lmHints = paddedT == T ? upsampled : upsampled[0..., 0..<T, 0...]

        // Blend: (1 − strength) × src + strength × acoustic hints
        let strength = MLXArray(coverStrength)
        let blended = (1 - strength) * srcLatents + strength * lmHints
        eval(blended)
        return blended.asType(.float32)
    }

    // MARK: - Weight Loading

    /// Load DiT weights and, from the same safetensors bundle, AudioTokenizer +
    /// AudioTokenDetokenizer weights. Also upgrades `conditioning` with the
    /// real tokenizer and model weights — mirrors `ACEStepGenerationService.loadDiTWeights`.
    private func loadDiTAndTokenizerWeights() async throws {
        guard let spec = await registry.activeSpec(for: .aceStepDiT) else {
            throw GenerationError.modelNotConfigured("ACE-Step DiT")
        }
        let baseDir = await registry.baseDirectory
        let previousModelID = currentlyLoadedDiTModelID

        await pipeline.unloadDiTWeights()
        currentlyLoadedDiTModelID = nil
        currentlyLoadedDiTConfig = nil

        do {
            // Load and validate DiT config
            let configPath = spec.localPath(for: "config.json", base: baseDir)
            let ditConfig = try DiTConfig.load(from: configPath)
            guard ditConfig.isSupportedBySwiftRuntime else {
                throw GenerationError.modelNotConfigured(
                    ditConfig.runtimeSupportNote ?? "Unsupported ACE-Step DiT architecture"
                )
            }
            await pipeline.configureDiT(config: ditConfig)

            // Load + remap DiT weights
            let weightPaths = ditWeightPaths(for: spec, baseDir: baseDir)
            let rawWeights = try await weightConverter.loadSafetensors(from: weightPaths)
            let ditWeights = try await weightConverter.convertDiTWeights(stateDict: rawWeights)
            await pipeline.loadDiTWeights(ditWeights, modelID: spec.id)
            currentlyLoadedDiTModelID = spec.id
            currentlyLoadedDiTConfig = ditConfig

            // Upgrade ConditioningActor with tokenizer + conditioning weights
            if let vaeSpec = await registry.activeSpec(for: .aceStepVAE) {
                let tokenizerPath = vaeSpec.localPath(for: "tokenizer.json", base: baseDir)
                if FileManager.default.fileExists(atPath: tokenizerPath.path(percentEncoded: false)),
                   let real = try? ConditioningActor.make(tokenizerURL: tokenizerPath) {
                    conditioning = real
                    logger.info("[CoverSongCoordinator] upgraded ConditioningActor with real tokenizer")
                }

                let qwen3Path = vaeSpec.localPath(for: "qwen3_model.safetensors", base: baseDir)
                if FileManager.default.fileExists(atPath: qwen3Path.path(percentEncoded: false)) {
                    let qwen3Raw = try await weightConverter.loadSafetensors(from: qwen3Path)
                    let qwen3Weights = await weightConverter.convertQwen3Weights(from: qwen3Raw)
                    let condWeights = await weightConverter.convertConditionEncoderWeights(from: rawWeights)
                    await conditioning.loadWeights(
                        qwen3Weights: qwen3Weights,
                        conditionWeights: condWeights
                    )
                    logger.info("[CoverSongCoordinator] loaded conditioning weights")
                }
            }

            // Load silence latent
            let silencePath = spec.localPath(for: "silence_latent.pt", base: baseDir)
            if FileManager.default.fileExists(atPath: silencePath.path(percentEncoded: false)),
               let silence = try? SilenceLatentLoader.load(from: silencePath) {
                await conditioning.setSilenceLatent(silence)
            } else {
                await conditioning.setSilenceLatent(nil)
            }

            // Load AudioTokenizer (tokenizer.* subtree of the same safetensors)
            let tokWeights = await weightConverter.convertAudioTokenizerWeights(from: rawWeights)
            let tok = AudioTokenizer()
            tok.update(parameters: ModuleParameters.unflattened(tokWeights))
            eval(tok.parameters())
            self.audioTokenizer = tok
            logger.info("[CoverSongCoordinator] loaded AudioTokenizer weights=\(tokWeights.count)")

            // Load AudioTokenDetokenizer (detokenizer.* subtree)
            let detokWeights = await weightConverter.convertAudioTokenDetokenizerWeights(from: rawWeights)
            let detok = AudioTokenDetokenizer()
            detok.update(parameters: ModuleParameters.unflattened(detokWeights))
            eval(detok.parameters())
            self.audioDetokenizer = detok
            logger.info("[CoverSongCoordinator] loaded AudioTokenDetokenizer weights=\(detokWeights.count)")

            // Best-effort: load AceStepLM if a variant is downloaded.
            // Non-fatal — cover generation falls back to the no-LM path.
            await self.loadLMWeightsIfAvailable()
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

    /// Attempt to load AceStepLM weights from the active (or first downloaded)
    /// LM variant. This is best-effort: if no LM variant is downloaded, or
    /// loading fails, the coordinator silently falls back to the no-LM path.
    private func loadLMWeightsIfAvailable() async {
        let baseDir = await registry.baseDirectory

        // Find the active LM spec, or the first downloaded variant.
        let activeLMID = await registry.activeModelID(for: .aceStepLM)
        let lmSpec: ModelSpec?
        if let activeLMID,
           let spec = await registry.activeSpec(for: .aceStepLM),
           await registry.status(for: activeLMID) == .ready {
            lmSpec = spec
        } else {
            // No active LM or it's not downloaded — scan for any ready variant.
            let allLMSpecs = await registry.specs(for: .aceStepLM)
            var found: ModelSpec?
            for spec in allLMSpecs {
                if await registry.status(for: spec.id) == .ready {
                    found = spec
                    break
                }
            }
            lmSpec = found
        }

        guard let spec = lmSpec else {
            logger.info("[CoverSongCoordinator] no downloaded LM variant found — cover songs will use no-LM path")
            aceStepLM = nil
            currentlyLoadedLMModelID = nil
            return
        }

        // Skip reload if we already have this variant loaded.
        if spec.id == currentlyLoadedLMModelID, aceStepLM != nil {
            return
        }

        do {
            // Load config
            let configPath = spec.localPath(for: "config.json", base: baseDir)
            let lmConfig = try AceStepLMConfig.load(from: configPath)

            // Load weights (handles both single-file and sharded)
            let weightPaths = lmWeightPaths(for: spec, baseDir: baseDir)
            let rawWeights = try await weightConverter.loadSafetensors(from: weightPaths)
            let mappedWeights = await weightConverter.convertAceStepLMWeights(from: rawWeights)

            let lm = AceStepLM(config: lmConfig)
            lm.update(parameters: ModuleParameters.unflattened(mappedWeights))
            eval(lm.parameters())

            // Dimension safety: the detokenizer's embedTokens expects input
            // matching its own hiddenSize (2048 from the DiT bundle). If the
            // LM's hidden_size differs, we can't feed its output directly.
            let detokHidden = audioDetokenizer?.hiddenSize ?? 2048
            if lmConfig.hiddenSize != detokHidden {
                logger.warning(
                    "[CoverSongCoordinator] LM hidden_size (\(lmConfig.hiddenSize)) != detokenizer hidden (\(detokHidden)) — skipping LM, needs projection layer (Phase 2D)"
                )
                aceStepLM = nil
                currentlyLoadedLMModelID = nil
                return
            }

            self.aceStepLM = lm
            self.currentlyLoadedLMModelID = spec.id
            logger.info("[CoverSongCoordinator] loaded AceStepLM (\(spec.id)) — \(lmConfig.numHiddenLayers) layers, hidden=\(lmConfig.hiddenSize)")
        } catch {
            logger.error("[CoverSongCoordinator] failed to load AceStepLM (\(spec.id)): \(error) — falling back to no-LM path")
            aceStepLM = nil
            currentlyLoadedLMModelID = nil
        }
    }

    /// Resolve safetensors paths for an LM model spec. Handles both single-file
    /// (`model.safetensors`) and sharded (`model-00001-of-00002.safetensors`) layouts.
    private func lmWeightPaths(for spec: ModelSpec, baseDir: URL) -> [URL] {
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
}

// MARK: - Errors

enum CoverSongError: Error, LocalizedError {
    case noSourceAudio
    case audioLoadFailed(String)

    var errorDescription: String? {
        switch self {
        case .noSourceAudio:
            return "No source audio file selected for cover song generation"
        case .audioLoadFailed(let message):
            return "Could not load source audio: \(message)"
        }
    }
}
