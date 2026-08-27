// PiMPAudioStudio/Services/CoverSongCoordinator.swift
import Foundation
import MLX
import MLXNN
import OSLog
import PiMPAudioCore

private let logger = Logger(subsystem: "com.pimpstudios.PiMPAudioStudio", category: "CoverSongCoordinator")

/// Orchestrates cover song generation:
///
///  1. Load + resample source audio to 48 kHz stereo
///  2. VAE-encode to latent domain (deterministic `encodeMean`)
///  3. `AudioTokenizer`: 25 Hz latents → 5 Hz FSQ-quantized vectors (2048-dim)
///  4. `AudioTokenDetokenizer`: quantized vectors → 25 Hz acoustic hints
///  5. Blend: `(1−strength)·srcLatents + strength·hints`
///  6. Standard ACE-Step diffusion via `MusicPipeline`
///
/// Weight management mirrors `ACEStepGenerationService`: DiT + VAE +
/// tokenizer/detokenizer are loaded lazily on first call and reloaded when the
/// active model ID changes. ACE-Step LM variants remain available for future
/// discrete audio-code workflows, but source-audio covers follow upstream's
/// direct quantized-hint path.
actor CoverSongCoordinator {
    private let pipeline: MusicPipeline
    private var conditioning: ConditioningActor
    private let weightConverter: WeightConverter
    private let registry: ModelRegistry

    private var audioTokenizer: AudioTokenizer?
    private var audioDetokenizer: AudioTokenDetokenizer?
    private var currentlyLoadedDiTModelID: String?
    private var currentlyLoadedVAEModelID: String?
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

    /// Replace the launch-time stub with the upgraded ACE-Step conditioning actor
    /// (real tokenizer + Qwen/encoder weights) once generation has loaded models.
    func updateConditioning(_ actor: ConditioningActor) {
        conditioning = actor
    }

    func needsModelLoad() async -> Bool {
        guard let activeDiTID = await registry.activeModelID(for: .aceStepDiT),
              let activeVAEID = await registry.activeModelID(for: .aceStepVAE) else {
            return true
        }

        return activeDiTID != currentlyLoadedDiTModelID
            || activeVAEID != currentlyLoadedVAEModelID
    }

    /// Loads the complete cover-song stack without starting generation.
    /// GenerationService calls this before LoRA injection so adapters attach
    /// to the freshly active DiT and survive the cover diffusion pass.
    func ensureLoaded() async throws {
        let activeDiTID = await registry.activeModelID(for: .aceStepDiT)
        guard let activeDiTID else {
            throw GenerationError.modelNotConfigured("ACE-Step DiT")
        }
        if activeDiTID != currentlyLoadedDiTModelID {
            try await loadDiTAndTokenizerWeights()
        }

        let activeVAEID = await registry.activeModelID(for: .aceStepVAE)
        guard let activeVAEID else {
            throw GenerationError.modelNotConfigured("ACE-Step VAE")
        }
        if activeVAEID != currentlyLoadedVAEModelID {
            try await loadVAEWeights()
        }
    }

    // MARK: - Main Entry Point

    func generate(request: GenerationRequest) -> AsyncThrowingStream<ACEStepGenerationService.Event, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    guard let referenceAudioURL = request.referenceAudioURL else {
                        throw CoverSongError.noSourceAudio
                    }
                    AudioDiagnostics.memory("cover/generate/begin")
                    AudioDiagnostics.note(
                        "cover/generate/request",
                        "source=\(referenceAudioURL.lastPathComponent) duration=\(request.duration) seed=\(request.seed.map(String.init) ?? "random") method=\(request.method) guidancePreset=\(request.guidancePreset.rawValue) guidance=\(request.effectiveGuidanceScale) userGuidance=\(request.guidanceScale) cfgInterval=\(request.cfgIntervalStart)...\(request.cfgIntervalEnd) coverStrength=\(request.coverStrength) precision=\(request.precision) promptChars=\(request.text.count) lyricChars=\(request.lyrics.count)"
                    )

                    if await self.needsModelLoad() {
                        continuation.yield(.loadingWeights(progress: 0))
                    }
                    try await self.ensureLoaded()

                    try Task.checkCancellation()
                    continuation.yield(.conditioning)

                    // Load source audio from disk → 48 kHz stereo MLXArray [1, frames, 2]
                    var audioNLC = try AudioFileLoader.load(from: referenceAudioURL)
                    logger.info("[CoverSongCoordinator] source audio loaded shape=\(audioNLC.shape)")

                    // Everything past the requested duration is truncated by
                    // ConditioningActor anyway — trim the waveform up front so the
                    // VAE encode and tokenize/detokenize chain never process it.
                    let maxSourceSamples = MusicPipeline.latentFrames(forDuration: request.duration)
                        * MusicPipeline.hopLength
                    if audioNLC.shape[1] > maxSourceSamples {
                        let originalFrames = audioNLC.shape[1]
                        audioNLC = audioNLC[0..., 0 ..< maxSourceSamples, 0...]
                        AudioDiagnostics.note(
                            "cover/sourceTrim",
                            "sourceSamples=\(originalFrames) trimmedTo=\(maxSourceSamples) requestDuration=\(request.duration)"
                        )
                    }
                    AudioDiagnostics.tensor("cover/sourceAudio", audioNLC)
                    AudioDiagnostics.memory("cover/afterSourceLoad")

                    // VAE-encode (deterministic mean) → srcLatents [1, T, 64]
                    let srcLatents = try await self.pipeline.encodeMean(
                        audio: audioNLC,
                        precision: request.precision
                    )
                    logger.info("[CoverSongCoordinator] srcLatents shape=\(srcLatents.shape)")
                    AudioDiagnostics.tensor("cover/sourceLatents", srcLatents)
                    AudioDiagnostics.memory("cover/afterSourceEncode")

                    // Cover-song preprocessing: tokenize → detokenize → blend
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
                    AudioDiagnostics.note(
                        "cover/diffusion/config",
                        "steps=\(inferSteps) shift=\(shift) method=\(request.method) guidance=\(effectiveGuidance) cfgInterval=\(request.cfgIntervalStart)...\(request.cfgIntervalEnd) seed=\(request.seed.map(String.init) ?? "random")"
                    )

                    continuation.yield(.generating(step: 0, totalSteps: inferSteps, eta: 0))

                    // Progress relay from pipeline → continuation
                    let progressStream = await self.pipeline.makeProgressStream()
                    let progressTask = Task {
                        for await progress in progressStream {
                            if case let .diffusion(dp) = progress {
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
                        srcLatentsOverride: blendedLatents,
                        timbreSeed: request.seed
                    )
                    AudioDiagnostics.tensor("cover/conditioning/encoderStates", cond.encoderStates)
                    AudioDiagnostics.tensor("cover/conditioning/encoderMask", cond.encoderMask)
                    AudioDiagnostics.tensor("cover/conditioning/contextLatents", cond.contextLatents)
                    AudioDiagnostics.memory("cover/afterConditioning")

                    // Run diffusion with the pre-computed conditioning result
                    let audioResult = try await self.pipeline.generate(
                        encoderStates: cond.encoderStates,
                        contextLatents: cond.contextLatents,
                        encoderMask: cond.encoderMask,
                        duration: request.duration,
                        config: config
                    )
                    AudioDiagnostics.tensor("cover/outputAudio", audioResult.samples)
                    AudioDiagnostics.memory("cover/generate/end")

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
        audioTokenizer = nil
        audioDetokenizer = nil
    }

    // MARK: - Audio Preprocessing

    private func tokenizeSourceLatents(_ srcLatents: MLXArray) throws -> CoverSongTokenizedSource {
        guard let tokenizer = audioTokenizer else {
            throw CoverSongError.coverPreprocessorNotReady
        }
        let T = srcLatents.shape[1]
        guard T > 0 else {
            throw CoverSongError.audioLoadFailed("Source audio produced no VAE latent frames.")
        }
        let pool = tokenizer.poolWindowSize
        AudioDiagnostics.note(
            "cover/tokenize/config",
            "sourceShape=\(srcLatents.shape) pool=\(pool)"
        )

        // AudioTokenizer requires T divisible by poolWindowSize (5). Repeat
        // the final source frame for alignment; this tail is trimmed after
        // detokenization, but keeping it in-distribution avoids a zero-latent
        // edge case in the final pool window.
        let paddedT = ((T + pool - 1) / pool) * pool
        let padded: MLXArray
        if paddedT == T {
            padded = srcLatents
        } else {
            let lastFrame = srcLatents[0..., (T - 1) ..< T, 0...]
            let padding = MLX.broadcast(
                lastFrame,
                to: [srcLatents.shape[0], paddedT - T, srcLatents.shape[2]]
            )
            padded = MLX.concatenated([srcLatents, padding], axis: 1)
        }

        // Tokenize: [1, paddedT, 64] → quantized [1, paddedT/pool, 2048] + indices [1, paddedT/pool, 1]
        let (quantized, indices) = tokenizer(padded)
        AudioDiagnostics.tensor("cover/tokenizerPaddedLatents", padded)
        AudioDiagnostics.tensor("cover/tokenizerQuantized5Hz", quantized)
        AudioDiagnostics.tensor("cover/tokenizerPackedCodes", indices)
        eval(quantized)
        AudioDiagnostics.memory("cover/afterTokenize")

        return CoverSongTokenizedSource(
            sourceLatents: srcLatents,
            originalLength: T,
            paddedLength: paddedT,
            quantizedHints5Hz: quantized
        )
    }

    /// Detokenize the source's 5 Hz FSQ hints to 25 Hz acoustic hints, then
    /// blend with source latents.
    private func conditionAndBlendSourceLatents(
        _ source: CoverSongTokenizedSource,
        coverStrength: Float
    ) throws -> MLXArray {
        guard let detokenizer = audioDetokenizer else {
            throw CoverSongError.coverPreprocessorNotReady
        }
        AudioDiagnostics.note(
            "cover/blend/config",
            "sourceLength=\(source.originalLength) paddedLength=\(source.paddedLength) coverStrength=\(coverStrength)"
        )

        // Detokenize: FSQ hints [1, paddedT/pool, 2048] → [1, paddedT, 64]
        let upsampled = detokenizer(source.quantizedHints5Hz)
        AudioDiagnostics.tensor("cover/detokenizedHints25Hz", upsampled)

        // Trim padding back to original T
        let lmHints = source.paddedLength == source.originalLength
            ? upsampled
            : upsampled[0..., 0 ..< source.originalLength, 0...]
        AudioDiagnostics.tensor("cover/lmHintsTrimmed25Hz", lmHints)

        // Blend: (1 − strength) × src + strength × acoustic hints
        let strength = MLXArray(coverStrength)
        let blended = (1 - strength) * source.sourceLatents + strength * lmHints
        AudioDiagnostics.tensor("cover/blendedLatents", blended)
        eval(blended)
        AudioDiagnostics.memory("cover/afterBlend")
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

            // Conditioning is required for usable covers — fail closed like ACE text2music.
            guard let vaeSpec = await registry.activeSpec(for: .aceStepVAE) else {
                throw GenerationError.conditioningIncomplete(
                    "ACE-Step VAE / text-encoder model is not configured."
                )
            }

            let tokenizerPath = vaeSpec.localPath(for: "tokenizer.json", base: baseDir)
            let tokenizerFSPath = tokenizerPath.path(percentEncoded: false)
            guard FileManager.default.fileExists(atPath: tokenizerFSPath) else {
                throw GenerationError.conditioningIncomplete(
                    "tokenizer.json is missing. Download the ACE-Step VAE/text encoder package."
                )
            }
            do {
                conditioning = try ConditioningActor.make(tokenizerURL: tokenizerPath)
                logger.info("[CoverSongCoordinator] upgraded ConditioningActor with real tokenizer")
            } catch {
                throw GenerationError.conditioningIncomplete(
                    "Could not load tokenizer at \(tokenizerFSPath): \(error.localizedDescription)"
                )
            }

            let qwen3Path = vaeSpec.localPath(for: "qwen3_model.safetensors", base: baseDir)
            let qwen3FSPath = qwen3Path.path(percentEncoded: false)
            guard FileManager.default.fileExists(atPath: qwen3FSPath) else {
                throw GenerationError.conditioningIncomplete(
                    "qwen3_model.safetensors is missing. Download the ACE-Step VAE/text encoder package."
                )
            }
            do {
                let qwen3Raw = try await weightConverter.loadSafetensors(from: qwen3Path)
                let qwen3Weights = await weightConverter.convertQwen3Weights(from: qwen3Raw)
                let condWeights = await weightConverter.convertConditionEncoderWeights(from: rawWeights)
                await conditioning.loadWeights(
                    qwen3Weights: qwen3Weights,
                    conditionWeights: condWeights
                )
                logger.info("[CoverSongCoordinator] loaded conditioning weights")
            } catch {
                throw GenerationError.conditioningIncomplete(
                    "Failed to load Qwen3/condition encoder weights: \(error.localizedDescription)"
                )
            }

            // silence_latent preferred but not hard-required
            let silencePath = spec.localPath(for: "silence_latent.pt", base: baseDir)
            if FileManager.default.fileExists(atPath: silencePath.path(percentEncoded: false)) {
                do {
                    let silence = try SilenceLatentLoader.load(from: silencePath)
                    await conditioning.setSilenceLatent(silence)
                } catch {
                    logger.warning(
                        "[CoverSongCoordinator] silence_latent load failed — zeros: \(error.localizedDescription)"
                    )
                    await conditioning.setSilenceLatent(nil)
                }
            } else {
                await conditioning.setSilenceLatent(nil)
            }

            // Load AudioTokenizer (tokenizer.* subtree of the same safetensors)
            let tokWeights = await weightConverter.convertAudioTokenizerWeights(from: rawWeights)
            guard !tokWeights.isEmpty else {
                throw GenerationError.conditioningIncomplete(
                    "AudioTokenizer weights missing from the DiT safetensors bundle."
                )
            }
            let tok = AudioTokenizer()
            tok.update(parameters: ModuleParameters.unflattened(tokWeights))
            eval(tok.parameters())
            audioTokenizer = tok
            logger.info("[CoverSongCoordinator] loaded AudioTokenizer weights=\(tokWeights.count)")

            // Load AudioTokenDetokenizer (detokenizer.* subtree)
            let detokWeights = await weightConverter.convertAudioTokenDetokenizerWeights(from: rawWeights)
            guard !detokWeights.isEmpty else {
                throw GenerationError.conditioningIncomplete(
                    "AudioTokenDetokenizer weights missing from the DiT safetensors bundle."
                )
            }
            let detok = AudioTokenDetokenizer()
            detok.update(parameters: ModuleParameters.unflattened(detokWeights))
            eval(detok.parameters())
            audioDetokenizer = detok
            logger.info("[CoverSongCoordinator] loaded AudioTokenDetokenizer weights=\(detokWeights.count)")

            // Only mark DiT loaded after conditioning + tokenizer stack succeed.
            currentlyLoadedDiTModelID = spec.id
            currentlyLoadedDiTConfig = ditConfig

        } catch {
            currentlyLoadedDiTModelID = nil
            currentlyLoadedDiTConfig = nil
            audioTokenizer = nil
            audioDetokenizer = nil
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
    let quantizedHints5Hz: MLXArray
}

// MARK: - Errors

enum CoverSongError: Error, LocalizedError {
    case noSourceAudio
    case audioLoadFailed(String)
    case coverPreprocessorNotReady

    var errorDescription: String? {
        switch self {
        case .noSourceAudio:
            "No source audio file selected for cover song generation"
        case let .audioLoadFailed(message):
            "Could not load source audio: \(message)"
        case .coverPreprocessorNotReady:
            "Cover song preprocessing is not ready. Load ACE-Step DiT and VAE models before generating a cover."
        }
    }
}
