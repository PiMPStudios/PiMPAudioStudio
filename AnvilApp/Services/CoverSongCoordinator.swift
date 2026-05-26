// AnvilApp/Services/CoverSongCoordinator.swift
import Foundation
import AnvilCore
import MLX
import MLXNN
import OSLog

private let logger = Logger(subsystem: "com.pimp.AnvilApp", category: "CoverSongCoordinator")

/// Orchestrates cover song generation:
///
///  1. Load + resample source audio to 48 kHz stereo
///  2. VAE-encode to latent domain (deterministic `encodeMean`)
///  3. `AudioTokenizer`: 25 Hz latents → 5 Hz FSQ-quantized vectors (2048-dim)
///  4. `AceStepLM`: condition the 5 Hz packed audio-code stream
///  5. `ResidualFSQ` + `AudioTokenDetokenizer`: conditioned codes → 25 Hz acoustic hints
///  6. Blend: `(1−strength)·srcLatents + strength·hints`
///  7. Standard ACE-Step diffusion via `MusicPipeline`
///
/// Weight management mirrors `ACEStepGenerationService`: DiT + VAE + LM +
/// tokenizer/detokenizer are loaded lazily on first call and reloaded when the
/// active model ID changes.
actor CoverSongCoordinator {
    private let pipeline: MusicPipeline
    private var conditioning: ConditioningActor
    private let weightConverter: WeightConverter
    private let registry: ModelRegistry

    private var audioTokenizer: AudioTokenizer?
    private var audioDetokenizer: AudioTokenDetokenizer?
    private var aceStepLM: AceStepLM?
    private var audioCodeVocabulary: AudioCodeVocabulary?
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

                    // Ensure Audio→Token LM weights
                    let lmSpec = try await self.selectedLMModelSpec()
                    if lmSpec.id != self.currentlyLoadedLMModelID {
                        continuation.yield(.loadingWeights(progress: 0))
                        try await self.loadLMWeights(spec: lmSpec)
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

                    // Cover-song preprocessing: tokenize → LM code conditioning → detokenize → blend
                    continuation.yield(.tokenizingSource)
                    let tokenizedSource = try self.tokenizeSourceLatents(srcLatents)

                    continuation.yield(.lmAudioCodeConditioning)
                    let blendedLatents = try self.conditionAndBlendSourceLatents(
                        tokenizedSource,
                        coverStrength: request.coverStrength
                    )
                    logger.info("[CoverSongCoordinator] blendedLatents shape=\(blendedLatents.shape)")

                    // Build DiffusionConfig (mirrors ACEStepGenerationService logic)
                    let effectiveGuidance = request.effectiveGuidanceScale
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
                        cfgIntervalStart: request.cfgIntervalStart,
                        cfgIntervalEnd: request.cfgIntervalEnd,
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
        await conditioning.unloadWeights()
        currentlyLoadedDiTModelID = nil
        currentlyLoadedDiTConfig = nil
        currentlyLoadedVAEModelID = nil
        currentlyLoadedLMModelID = nil
        audioTokenizer = nil
        audioDetokenizer = nil
        aceStepLM = nil
        audioCodeVocabulary = nil
    }

    // MARK: - Audio Preprocessing

    private func tokenizeSourceLatents(_ srcLatents: MLXArray) throws -> CoverSongTokenizedSource {
        guard let tokenizer = audioTokenizer else {
            throw CoverSongError.coverPreprocessorNotReady
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
        let (_, indices) = tokenizer(padded)
        eval(indices)

        return CoverSongTokenizedSource(
            sourceLatents: srcLatents,
            originalLength: T,
            paddedLength: paddedT,
            packedCodes: indices
        )
    }

    /// Condition packed source audio codes with the active LM, reconstruct 5 Hz
    /// FSQ hints, detokenize to 25 Hz acoustic hints, then blend with source latents.
    private func conditionAndBlendSourceLatents(
        _ source: CoverSongTokenizedSource,
        coverStrength: Float
    ) throws -> MLXArray {
        guard let tokenizer = audioTokenizer,
              let detokenizer = audioDetokenizer,
              let lm = aceStepLM,
              let vocabulary = audioCodeVocabulary else {
            throw CoverSongError.coverPreprocessorNotReady
        }

        let conditionedCodes = lm.predictAudioCodes(
            sourceCodes: source.packedCodes,
            vocabulary: vocabulary
        )
        let lmHints5Hz = tokenizer.quantizer.getOutputFromIndices(conditionedCodes)

        // Detokenize: FSQ hints [1, paddedT/pool, 2048] → [1, paddedT, 64]
        let upsampled = detokenizer(lmHints5Hz)

        // Trim padding back to original T
        let lmHints = source.paddedLength == source.originalLength
            ? upsampled
            : upsampled[0..., 0..<source.originalLength, 0...]

        // Blend: (1 − strength) × src + strength × acoustic hints
        let strength = MLXArray(coverStrength)
        let blended = (1 - strength) * source.sourceLatents + strength * lmHints
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

    private func selectedLMModelSpec() async throws -> ModelSpec {
        if let spec = await registry.activeSpec(for: .aceStepLM) {
            let status = await registry.status(for: spec.id)
            guard case .ready = status else {
                throw CoverSongError.audioLanguageModelNotReady(spec.displayName)
            }
            return spec
        }

        let specs = await registry.specs(for: .aceStepLM)
        for spec in specs where spec.isRunnable {
            let status = await registry.status(for: spec.id)
            if case .ready = status {
                return spec
            }
        }

        throw CoverSongError.noAudioLanguageModel
    }

    private func loadLMWeights(spec: ModelSpec) async throws {
        let baseDir = await registry.baseDirectory
        aceStepLM = nil
        audioCodeVocabulary = nil
        currentlyLoadedLMModelID = nil

        let configURL = spec.localPath(for: "config.json", base: baseDir)
        let tokenizerURL = spec.localPath(for: "tokenizer.json", base: baseDir)
        guard FileManager.default.fileExists(atPath: configURL.path(percentEncoded: false)),
              FileManager.default.fileExists(atPath: tokenizerURL.path(percentEncoded: false)) else {
            throw CoverSongError.audioLanguageModelFilesMissing(spec.displayName)
        }

        let config = try AceStepLMConfig.load(from: configURL)
        let vocabulary = try AudioCodeVocabulary.load(from: tokenizerURL)
        let rawWeights = try await lmWeightList(for: spec, baseDir: baseDir)
        let weights = await weightConverter.convertAceStepLMWeights(from: rawWeights)

        let lm = AceStepLM(config: config)
        lm.update(parameters: ModuleParameters.unflattened(weights))
        eval(lm.parameters())

        aceStepLM = lm
        audioCodeVocabulary = vocabulary
        currentlyLoadedLMModelID = spec.id
        logger.info("[CoverSongCoordinator] loaded ACE-Step LM \(spec.id) weights=\(weights.count)")
    }

    private func lmWeightList(for spec: ModelSpec, baseDir: URL) async throws -> [(String, MLXArray)] {
        let indexURL = spec.localPath(for: "model.safetensors.index.json", base: baseDir)
        if FileManager.default.fileExists(atPath: indexURL.path(percentEncoded: false)) {
            return try await weightConverter.loadShardedSafetensors(indexURL: indexURL)
        }

        let singleURL = spec.localPath(for: "model.safetensors", base: baseDir)
        if FileManager.default.fileExists(atPath: singleURL.path(percentEncoded: false)) {
            return try await weightConverter.loadSafetensors(from: singleURL)
        }

        let shardURLs = spec.files
            .filter { $0.localName.hasPrefix("model-") && $0.localName.hasSuffix(".safetensors") }
            .map { spec.localPath(for: $0.localName, base: baseDir) }
        if !shardURLs.isEmpty,
           shardURLs.allSatisfy({ FileManager.default.fileExists(atPath: $0.path(percentEncoded: false)) }) {
            return try await weightConverter.loadSafetensors(from: shardURLs)
        }

        throw CoverSongError.audioLanguageModelFilesMissing(spec.displayName)
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

private struct CoverSongTokenizedSource {
    let sourceLatents: MLXArray
    let originalLength: Int
    let paddedLength: Int
    let packedCodes: MLXArray
}

// MARK: - Errors

enum CoverSongError: Error, LocalizedError {
    case noSourceAudio
    case audioLoadFailed(String)
    case coverPreprocessorNotReady
    case noAudioLanguageModel
    case audioLanguageModelNotReady(String)
    case audioLanguageModelFilesMissing(String)

    var errorDescription: String? {
        switch self {
        case .noSourceAudio:
            return "No source audio file selected for cover song generation"
        case .audioLoadFailed(let message):
            return "Could not load source audio: \(message)"
        case .coverPreprocessorNotReady:
            return "Cover song preprocessing is not ready. Load ACE-Step DiT, VAE, and LM models before generating a cover."
        case .noAudioLanguageModel:
            return "No ACE-Step LM is downloaded. Download one in Settings → Models → ACE-Step LM to generate cover songs."
        case .audioLanguageModelNotReady(let displayName):
            return "\(displayName) is not ready. Re-download or select a ready ACE-Step LM in Settings → Models."
        case .audioLanguageModelFilesMissing(let displayName):
            return "\(displayName) is missing required ACE-Step LM files. Re-download it in Settings → Models."
        }
    }
}
