import Foundation
import MLX
import MLXNN
import OSLog

private let condLogger = Logger(subsystem: "com.pimp.AnvilApp", category: "ConditioningActor")

public actor ConditioningActor {
    private let vae: VAEActor
    private let qwen3: Qwen3Model
    private let lyricEncoder: LyricEncoder
    private let timbreEncoder: TimbreEncoder
    private let conditionEncoder: ConditionEncoder
    private let tokenizer: AnvilTokenizer
    /// Trained "silence" latent prior (shape `[1, 15000, 64]`) — used as the
    /// `src_latents` half of `context_latents` when generating from text
    /// only (no reference audio). Loaded from `silence_latent.pt` shipped
    /// alongside each ACE-Step DiT checkpoint. `nil` falls back to a zeros
    /// tensor (smoke test mode).
    private var silenceLatent: MLXArray?

    static let timbreFixFrame = 750
    /// Audio sample rate matching MusicPipeline.defaultSampleRate (v1.5 VAE: 48000).
    static let audioSampleRate: Int = 48000
    /// Total VAE downsampling = product of `downsampling_ratios` `[2,4,4,6,10]` from
    /// the ACE-Step v1.5 `vae/config.json` = 1920. 48000/1920 = 25 latent fps.
    static let vaeHopLength: Int = 1920

    /// Number of latent frames produced by the VAE for `duration` seconds of
    /// audio at `audioSampleRate`. Must match `MusicPipeline.latentFrames` so
    /// that `contextLatents` and the diffusion noisy latent share a sequence
    /// dimension and can be concatenated along the channel axis in DiTModel.
    static func latentFrames(forDuration duration: TimeInterval) -> Int {
        Int(ceil(Double(audioSampleRate) * duration / Double(vaeHopLength)))
    }

    public init(
        vae: VAEActor,
        qwen3: Qwen3Model,
        conditionEncoder: ConditionEncoder,
        lyricEncoder: LyricEncoder,
        timbreEncoder: TimbreEncoder,
        tokenizerURL: URL
    ) throws {
        self.vae = vae
        self.qwen3 = qwen3
        self.conditionEncoder = conditionEncoder
        self.lyricEncoder = lyricEncoder
        self.timbreEncoder = timbreEncoder
        self.tokenizer = try AnvilTokenizer(configURL: tokenizerURL)
    }

    /// Wrap a user caption with the upstream SFT_GEN_PROMPT template. Mirrors
    /// `acestep/constants.py:SFT_GEN_PROMPT` and the meta string built by
    /// `acestep/core/generation/handler/metadata_utils.py:_build_metadata_dict`
    /// when no user-supplied bpm/key/timesig/duration is available. Required:
    /// the v1.5 DiT was trained on this scaffolded prompt; raw user captions
    /// land out-of-distribution for the text encoder.
    private static func wrapPromptForSFT(caption: String, duration: TimeInterval) -> String {
        // Mirrors `acestep/constants.py:SFT_GEN_PROMPT` exactly (byte-for-byte).
        //
        // Python _create_default_meta() ends each meta line with \n, so the metas
        // block ends with "...seconds\n" and <|endoftext|> falls on its own line:
        //   "...duration: 30 seconds\n<|endoftext|>\n"
        //
        // The previous multi-line-string version produced "...seconds<|endoftext|>\n\n"
        // (metas had no trailing \n, and the blank line before the closing """ added
        // an extra \n). Use explicit concatenation to lock in the exact byte sequence.
        let instruction = "Fill the audio semantic mask based on the given conditions:"
        let metas = "- bpm: N/A\n- timesignature: N/A\n- keyscale: N/A\n- duration: \(Int(duration)) seconds\n"
        return "# Instruction\n\(instruction)\n\n# Caption\n\(caption)\n\n# Metas\n\(metas)<|endoftext|>\n"
    }

    /// Wrap lyrics with upstream's `# Languages / # Lyric` template. Mirrors
    /// `acestep/core/generation/handler/prompt_utils.py:_format_lyrics`.
    private static func wrapLyricsForSFT(lyrics: String, language: String = "en") -> String {
        return """
        # Languages
        \(language)

        # Lyric
        \(lyrics)<|endoftext|>
        """
    }

    public func encode(
        text: String,
        lyrics: String,
        referenceAudio: MLXArray?,
        duration: TimeInterval,
        precision: Precision
    ) async throws -> ConditioningResult {
        let wrappedText = Self.wrapPromptForSFT(caption: text, duration: duration)
        let wrappedLyrics = Self.wrapLyricsForSFT(lyrics: lyrics)
        let (textIds, textMask) = tokenizer.encodeText(wrappedText, maxLength: 256)
        let (lyricIds, lyricMask) = tokenizer.encodeLyrics(wrappedLyrics, maxLength: 2048)

        try Task.checkCancellation()

        let textHidden = qwen3(inputIds: textIds, attentionMask: textMask)

        // Upstream always encodes the scaffolded lyric prompt, even when the
        // user supplied no lyric body. With longest-padding this is a short,
        // real sequence (`# Languages\nen\n\n# Lyric\n<|endoftext|>`), not a
        // 2048-token PAD field, so bypassing it with zeros is out-of-distribution.
        let lyricEmbeddings = qwen3.embedTokens(lyricIds)
        let lyricHidden = lyricEncoder(embeddings: lyricEmbeddings, attentionMask: lyricMask)
        AudioDiagnostics.tensor("cond/lyricHidden", lyricHidden)

        try Task.checkCancellation()

        // referenceLatents is [B, 750, 64] — raw VAE latent space fed to timbreEncoder
        // contextLatents is [B, T, 128] = cat([src_latents, chunk_mask], axis:-1)
        let referenceLatents = try await prepareReferenceLatents(referenceAudio, precision: precision)

        // Upstream text2music still runs timbre conditioning: when there is
        // no reference audio, infer_refer_latent() feeds the first 750 frames
        // of silence_latent through AceStepTimbreEncoder and emits one timbre
        // token for the batch. The previous zero-token bypass removed that
        // learned instrument prior entirely.
        let timbreInputLatents = referenceAudio == nil
            ? silenceLatentSlice(length: Self.timbreFixFrame, precision: precision)
            : referenceLatents
        let timbreTokens = timbreEncoder(latents: timbreInputLatents, attentionMask: nil)
        AudioDiagnostics.tensor("cond/timbreTokens", timbreTokens)

        try Task.checkCancellation()

        let (encoderStates, encoderMask) = conditionEncoder(
            textHiddenStates: textHidden,
            textMask: textMask,
            lyricHiddenStates: lyricHidden,
            lyricMask: lyricMask,
            timbreTokens: timbreTokens
        )

        // tFrames must match the noisy-latent sequence length built by
        // DiffusionActor.generate (which uses MusicPipeline.latentFrames).
        // Both compute ceil(sampleRate * duration / hopLength) so the
        // [contextLatents, hiddenStates] concat in DiTModel.callAsFunction
        // has matching seq dims.
        let tFrames = Self.latentFrames(forDuration: duration)
        let contextLatents: MLXArray
        if referenceAudio != nil {
            // Use the already-prepared (padded/truncated) referenceLatents
            let chunkMask = MLXArray.ones(referenceLatents.shape).asType(precision.mlxDtype)
            contextLatents = concatenated([referenceLatents, chunkMask], axis: -1)
        } else {
            // Match upstream text2music context construction:
            //   context_latents = cat([silence_latent[:, :tFrames, :], ones], axis=-1)
            //
            // (acestep/models/turbo/modeling_acestep_v15_turbo.py:1680, 1931
            // and equivalent in base/sft variants.)
            //
            // The first half (64 channels) is the trained silence_latent —
            // a learned "no source audio" prior. The second half (64 channels)
            // is the chunk_mask = ones, telling the DiT to generate the full
            // sequence. Falling back to zeros for src_latents was the latest
            // remaining structural mismatch (caught by GPT-5.5 review):
            // matched the mask side correctly but left the source body as
            // zeros, putting the model in a degenerate prior it wasn't
            // trained for.
            let srcLatents: MLXArray
            if let silence = silenceLatent {
                // silence has shape [1, 15000, 64]; tFrames typically 750.
                // (Tile if duration would exceed 600s — upstream tiles too.)
                let available = silence.shape[1]
                if tFrames <= available {
                    srcLatents = silence[0..., 0..<tFrames, 0...]
                } else {
                    let repeats = Int(ceil(Double(tFrames) / Double(available)))
                    var tiled = silence
                    for _ in 1..<repeats {
                        tiled = concatenated([tiled, silence], axis: 1)
                    }
                    srcLatents = tiled[0..., 0..<tFrames, 0...]
                }
            } else {
                // Fall back to zeros if silence_latent.pt isn't loaded —
                // produces still-functional but slightly noisy output.
                srcLatents = MLXArray.zeros([1, tFrames, 64])
            }
            let onesMask = MLXArray.ones([1, tFrames, 64])
            contextLatents = concatenated([srcLatents, onesMask], axis: -1)
                .asType(precision.mlxDtype)
        }

        let result: ConditioningResult
        if precision == .float16 {
            result = ConditioningResult(
                encoderStates: encoderStates.asType(.float16),
                encoderMask: encoderMask,
                contextLatents: contextLatents.asType(.float16)
            )
        } else {
            result = ConditioningResult(
                encoderStates: encoderStates,
                encoderMask: encoderMask,
                contextLatents: contextLatents
            )
        }
        AudioDiagnostics.tensor("cond/encoderStates", result.encoderStates)
        AudioDiagnostics.tensor("cond/encoderMask", result.encoderMask)
        AudioDiagnostics.tensor("cond/contextLatents", result.contextLatents)
        return result
    }

    private func prepareReferenceLatents(_ referenceAudio: MLXArray?, precision: Precision) async throws -> MLXArray {
        guard let ref = referenceAudio else {
            return MLXArray.zeros([1, Self.timbreFixFrame, 64])
        }

        let latents = try await vae.encodeSample(audio: ref, precision: precision)
        let currentFrames = latents.shape[1]
        if currentFrames == Self.timbreFixFrame {
            return latents
        } else if currentFrames > Self.timbreFixFrame {
            return latents[0..<latents.shape[0], 0..<Self.timbreFixFrame]
        } else {
            let padding = MLXArray.zeros([latents.shape[0], Self.timbreFixFrame - currentFrames, 64])
            return concatenated([latents, padding], axis: 1)
        }
    }

    private func silenceLatentSlice(length: Int, precision: Precision) -> MLXArray {
        guard let silence = silenceLatent else {
            return MLXArray.zeros([1, length, 64]).asType(precision.mlxDtype)
        }

        let available = silence.shape[1]
        let sliced: MLXArray
        if length <= available {
            sliced = silence[0..., 0..<length, 0...]
        } else {
            let repeats = Int(ceil(Double(length) / Double(available)))
            var tiled = silence
            for _ in 1..<repeats {
                tiled = concatenated([tiled, silence], axis: 1)
            }
            sliced = tiled[0..., 0..<length, 0...]
        }
        return sliced.asType(precision.mlxDtype)
    }

    /// Load the per-DiT-checkpoint `silence_latent.pt` tensor. Pass `nil` to
    /// clear (e.g., when unloading weights). Shape must be `[1, ≥1, 64]`.
    public func setSilenceLatent(_ latent: MLXArray?) {
        self.silenceLatent = latent
        if let l = latent {
            condLogger.info("[ConditioningActor.setSilenceLatent] loaded shape=\(l.shape)")
        } else {
            condLogger.info("[ConditioningActor.setSilenceLatent] cleared")
        }
    }

    public func loadWeights(
        qwen3Weights: [(String, MLXArray)],
        conditionWeights: [(String, MLXArray)]
    ) async {
        // Diagnostic for each sub-module: how many supplied keys match a
        // real model parameter path? Mirrors the DiTActor / VAEActor
        // diagnostic that revealed the silent-audio root cause.
        Self.logKeyDiff(name: "qwen3", supplied: qwen3Weights, model: qwen3)

        qwen3.update(parameters: ModuleParameters.unflattened(qwen3Weights))
        eval(qwen3.parameters())

        let lyricWeights = conditionWeights.compactMap { (key, value) -> (String, MLXArray)? in
            guard key.hasPrefix("lyricEncoder.") else { return nil }
            return (String(key.dropFirst("lyricEncoder.".count)), value)
        }
        Self.logKeyDiff(name: "lyricEncoder", supplied: lyricWeights, model: lyricEncoder)
        lyricEncoder.update(parameters: ModuleParameters.unflattened(lyricWeights))
        eval(lyricEncoder.parameters())

        let timbreWeights = conditionWeights.compactMap { (key, value) -> (String, MLXArray)? in
            guard key.hasPrefix("timbreEncoder.") else { return nil }
            return (String(key.dropFirst("timbreEncoder.".count)), value)
        }
        Self.logKeyDiff(name: "timbreEncoder", supplied: timbreWeights, model: timbreEncoder)
        timbreEncoder.update(parameters: ModuleParameters.unflattened(timbreWeights))
        eval(timbreEncoder.parameters())

        let condEncWeights = conditionWeights.compactMap { (key, value) -> (String, MLXArray)? in
            guard key.hasPrefix("conditionEncoder.") else { return nil }
            return (String(key.dropFirst("conditionEncoder.".count)), value)
        }
        Self.logKeyDiff(name: "conditionEncoder", supplied: condEncWeights, model: conditionEncoder)
        conditionEncoder.update(parameters: ModuleParameters.unflattened(condEncWeights))
        eval(conditionEncoder.parameters())
    }

    private static func logKeyDiff(
        name: String, supplied: [(String, MLXArray)], model: Module
    ) {
        let modelKeys = Set(model.parameters().flattened().map { $0.0 })
        let suppliedKeys = Set(supplied.map { $0.0 })
        let matched = suppliedKeys.intersection(modelKeys).count
        let supExtra = suppliedKeys.subtracting(modelKeys)
        let modExtra = modelKeys.subtracting(suppliedKeys)
        condLogger.info("[\(name)] supplied=\(suppliedKeys.count) model=\(modelKeys.count) matched=\(matched)")
        if !supExtra.isEmpty {
            condLogger.info("[\(name)] supplied-but-no-model-param (first 8): \(Array(supExtra.sorted().prefix(8)))")
        }
        if !modExtra.isEmpty {
            condLogger.info("[\(name)] model-param-but-no-supplied (first 8): \(Array(modExtra.sorted().prefix(8)))")
        }
    }
}

extension ConditioningActor {
    /// Builds a ConditioningActor backed by a real tokenizer file on disk.
    /// Used in production to upgrade from a stub once the model's `tokenizer.json`
    /// has been downloaded, and in tests where a fixture tokenizer is available.
    public static func make(tokenizerURL: URL) throws -> ConditioningActor {
        return try ConditioningActor(
            vae: VAEActor(),
            qwen3: Qwen3Model(),
            conditionEncoder: ConditionEncoder(),
            lyricEncoder: LyricEncoder(),
            timbreEncoder: TimbreEncoder(),
            tokenizerURL: tokenizerURL
        )
    }

    /// Creates a fully functional ConditioningActor using a stub tokenizer.
    /// Suitable for app launch before model weights are downloaded — all
    /// inference calls will succeed structurally but produce random outputs
    /// until `updateTokenizer(from:)` is called after model download.
    public static func stub() throws -> ConditioningActor {
        let tokenizer = try AnvilTokenizer.forTesting()
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("AnvilTokenizerForTesting", isDirectory: true)
        let tokenizerURL = dir.appendingPathComponent("tokenizer.json")
        return try ConditioningActor(
            vae: VAEActor(),
            qwen3: Qwen3Model(),
            conditionEncoder: ConditionEncoder(),
            lyricEncoder: LyricEncoder(),
            timbreEncoder: TimbreEncoder(),
            tokenizerURL: tokenizerURL
        )
    }
}
