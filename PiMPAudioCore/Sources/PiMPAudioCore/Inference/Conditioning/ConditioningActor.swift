import Foundation
import MLX
import MLXNN
import MLXRandom
import OSLog

private let condLogger = Logger(subsystem: "com.pimpstudios.PiMPAudioStudio", category: "ConditioningActor")

public enum ConditioningActorError: Error, LocalizedError, Sendable {
    case weightsNotLoaded

    public var errorDescription: String? {
        "Conditioning weights are not loaded. Please wait for model loading to complete."
    }
}

public actor ConditioningActor {
    private let vae: VAEActor
    private var qwen3: Qwen3Model?
    private var lyricEncoder: LyricEncoder?
    private var timbreEncoder: TimbreEncoder?
    private var conditionEncoder: ConditionEncoder?
    private let tokenizer: PiMPAudioTokenizer
    /// Trained "silence" latent prior (shape `[1, 15000, 64]`) — used as the
    /// `src_latents` half of `context_latents` when generating from text
    /// only (no reference audio). Loaded from `silence_latent.pt` shipped
    /// alongside each ACE-Step DiT checkpoint. `nil` falls back to a zeros
    /// tensor (smoke test mode).
    private var silenceLatent: MLXArray?

    static let timbreFixFrame = 750
    /// Full-generation mask used in Swift's direct-DiT path.
    ///
    /// Upstream ACE-Step's default `chunk_mask_mode="auto"` maps this mask to
    /// 2.0, but a listening A/B in the native direct-DiT path showed 2.0 fixes
    /// the final-duration symptom while collapsing the output into a hum/buzz.
    /// Keep 1.0 as the musical default until the upstream LM/audio-code path or
    /// a better mask schedule is wired in.
    static let fullGenerationChunkMaskValue: Float = 1.0
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
        try self.init(
            vae: vae,
            qwen3: qwen3,
            conditionEncoder: conditionEncoder,
            lyricEncoder: lyricEncoder,
            timbreEncoder: timbreEncoder,
            tokenizer: PiMPAudioTokenizer(configURL: tokenizerURL)
        )
    }

    private init(
        vae: VAEActor,
        qwen3: Qwen3Model,
        conditionEncoder: ConditionEncoder,
        lyricEncoder: LyricEncoder,
        timbreEncoder: TimbreEncoder,
        tokenizer: PiMPAudioTokenizer
    ) {
        self.vae = vae
        self.qwen3 = qwen3
        self.conditionEncoder = conditionEncoder
        self.lyricEncoder = lyricEncoder
        self.timbreEncoder = timbreEncoder
        self.tokenizer = tokenizer
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
        let metadata = promptMetadata(for: caption, duration: duration)
        let metas = metadata.metaString
        return "# Instruction\n\(instruction)\n\n# Caption\n\(caption)\n\n# Metas\n\(metas)<|endoftext|>\n"
    }

    private struct PromptMetadata: Sendable {
        let bpm: String
        let timeSignature: String
        let keyscale: String
        let durationSeconds: Int

        var metaString: String {
            "- bpm: \(bpm)\n"
            + "- timesignature: \(timeSignature)\n"
            + "- keyscale: \(keyscale)\n"
            + "- duration: \(durationSeconds) seconds\n"
        }
    }

    private static func promptMetadata(for caption: String, duration: TimeInterval) -> PromptMetadata {
        let metadata = PromptMetadata(
            bpm: extractBPM(from: caption) ?? "N/A",
            timeSignature: extractTimeSignature(from: caption) ?? "N/A",
            keyscale: extractKeyscale(from: caption) ?? "N/A",
            durationSeconds: max(1, Int(duration.rounded()))
        )
        AudioDiagnostics.note(
            "cond/promptMetadata",
            "bpm=\(metadata.bpm) timesignature=\(metadata.timeSignature) keyscale=\(metadata.keyscale) duration=\(metadata.durationSeconds)"
        )
        return metadata
    }

    private static func extractBPM(from caption: String) -> String? {
        if let explicit = firstCapture(
            #"(?<!\d)(3[0-9]|[4-9][0-9]|[12][0-9]{2}|300)\s*(?:bpm|beats\s+per\s+minute)\b"#,
            in: caption
        ) {
            return explicit
        }

        let lower = caption.lowercased()
        if lower.contains("very slow") || lower.contains("dirge") || lower.contains("doom") {
            return "55"
        }
        if lower.contains("slow tempo") || lower.contains("slow,") || lower.contains("slow ") || lower.contains("downtempo") {
            return "70"
        }
        if lower.contains("mid-tempo") || lower.contains("medium tempo") || lower.contains("moderate tempo") {
            return "100"
        }
        if lower.contains("very fast") || lower.contains("breakneck") {
            return "180"
        }
        if lower.contains("fast tempo") || lower.contains("upbeat") || lower.contains("high energy") {
            return "140"
        }

        return nil
    }

    private static func extractTimeSignature(from caption: String) -> String? {
        firstCapture(#"\b([2346])\s*/\s*(?:4|8)\b"#, in: caption)
    }

    private static func extractKeyscale(from caption: String) -> String? {
        let pattern = #"\b(?:key(?:\s+of)?\s+)?([A-G])\s*(#|♯|b|♭|sharp|flat)?\s+(major|minor)\b"#
        guard let captures = firstCaptures(pattern, in: caption, groups: 3),
              let note = captures[0],
              let mode = captures[2] else {
            return nil
        }

        let accidental: String
        switch captures[1]?.lowercased() {
        case "#", "♯", "sharp":
            accidental = "#"
        case "b", "♭", "flat":
            accidental = "b"
        default:
            accidental = ""
        }
        return "\(note.uppercased())\(accidental) \(mode.lowercased())"
    }

    private static func firstCapture(_ pattern: String, in text: String) -> String? {
        firstCaptures(pattern, in: text, groups: 1)?.first ?? nil
    }

    private static func firstCaptures(_ pattern: String, in text: String, groups: Int) -> [String?]? {
        guard let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]) else {
            return nil
        }
        let fullRange = NSRange(text.startIndex..<text.endIndex, in: text)
        guard let match = regex.firstMatch(in: text, options: [], range: fullRange) else {
            return nil
        }
        return (1...groups).map { index in
            guard index < match.numberOfRanges,
                  let range = Range(match.range(at: index), in: text) else {
                return nil
            }
            let value = String(text[range]).trimmingCharacters(in: .whitespacesAndNewlines)
            return value.isEmpty ? nil : value
        }
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
        precision: Precision,
        srcLatentsOverride: MLXArray? = nil,
        timbreSeed: Int? = nil
    ) async throws -> ConditioningResult {
        guard
            let qwen3,
            let lyricEncoder,
            let timbreEncoder,
            let conditionEncoder
        else {
            throw ConditioningActorError.weightsNotLoaded
        }

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
        let referenceLatents = try await prepareReferenceLatents(
            referenceAudio,
            precision: precision,
            timbreSeed: timbreSeed
        )

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
        if let srcOverride = srcLatentsOverride {
            // Cover song path: caller has pre-computed blended latents via
            // tokenize → detokenize → blend. Use them directly as src_latents.
            // Pad or truncate to tFrames to match the diffusion shape.
            let overrideFrames = srcOverride.shape[1]
            let src: MLXArray
            if overrideFrames == tFrames {
                src = srcOverride
            } else if overrideFrames > tFrames {
                src = srcOverride[0..., 0..<tFrames, 0...]
            } else {
                let padding = silenceLatentPadding(
                    batchSize: srcOverride.shape[0],
                    length: tFrames - overrideFrames,
                    precision: precision
                )
                .asType(srcOverride.dtype)
                src = concatenated([srcOverride, padding], axis: 1)
            }
            let autoMask = MLXArray.full([1, tFrames, 64], values: MLXArray(Self.fullGenerationChunkMaskValue))
                .asType(precision.mlxDtype)
            contextLatents = concatenated([src.asType(precision.mlxDtype), autoMask], axis: -1)
        } else if referenceAudio != nil {
            // Use the already-prepared (padded/truncated) referenceLatents
            let chunkMask = MLXArray.full(referenceLatents.shape, values: MLXArray(Self.fullGenerationChunkMaskValue))
                .asType(precision.mlxDtype)
            contextLatents = concatenated([referenceLatents, chunkMask], axis: -1)
        } else {
            // Match upstream text2music context construction:
            //   context_latents = cat([silence_latent[:, :tFrames, :], auto_mask], axis=-1)
            //
            // The first half (64 channels) is the trained silence_latent —
            // a learned "no source audio" prior. The second half (64 channels)
            // is the full-generation chunk mask.
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
            let autoMask = MLXArray.full([1, tFrames, 64], values: MLXArray(Self.fullGenerationChunkMaskValue))
            contextLatents = concatenated([srcLatents, autoMask], axis: -1)
                .asType(precision.mlxDtype)
        }
        AudioDiagnostics.note(
            "cond/chunkMask",
            "mode=full value=\(Self.fullGenerationChunkMaskValue) frames=\(tFrames)"
        )

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

    /// Offset decorrelates the timbre posterior draw from the diffusion noise
    /// stream, which derives its key from the same seed via `MLXRandom.key(seed)`.
    private static let timbreKeyOffset: UInt64 = 0x7469_6D62  // "timb"

    private func prepareReferenceLatents(
        _ referenceAudio: MLXArray?,
        precision: Precision,
        timbreSeed: Int? = nil
    ) async throws -> MLXArray {
        guard var ref = referenceAudio else {
            return MLXArray.zeros([1, Self.timbreFixFrame, 64])
        }

        // The timbre encoder only consumes the first `timbreFixFrame` latent
        // frames; slice the waveform before encoding so we never VAE-encode
        // audio that would be truncated anyway.
        let maxSamples = Self.timbreFixFrame * Self.vaeHopLength
        if ref.shape[1] > maxSamples {
            ref = ref[0..., 0..<maxSamples, 0...]
        }

        let key = timbreSeed.map {
            MLXRandom.key(UInt64(bitPattern: Int64($0)) &+ Self.timbreKeyOffset)
        }
        let latents = try await vae.encodeSample(audio: ref, precision: precision, key: key)
        let currentFrames = latents.shape[1]
        if currentFrames == Self.timbreFixFrame {
            return latents
        } else if currentFrames > Self.timbreFixFrame {
            return latents[0..<latents.shape[0], 0..<Self.timbreFixFrame]
        } else {
            let padding = silenceLatentPadding(
                batchSize: latents.shape[0],
                length: Self.timbreFixFrame - currentFrames,
                precision: precision
            )
            .asType(latents.dtype)
            return concatenated([latents, padding], axis: 1)
        }
    }

    private func silenceLatentPadding(batchSize: Int, length: Int, precision: Precision) -> MLXArray {
        let batch = max(1, batchSize)
        let base = silenceLatentSlice(length: length, precision: precision)
        guard batch > 1 else { return base }
        return MLX.broadcast(base, to: [batch, length, 64])
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
        if qwen3 == nil {
            qwen3 = Qwen3Model()
        }
        if lyricEncoder == nil {
            lyricEncoder = LyricEncoder()
        }
        if timbreEncoder == nil {
            timbreEncoder = TimbreEncoder()
        }
        if conditionEncoder == nil {
            conditionEncoder = ConditionEncoder()
        }

        guard
            let qwen3,
            let lyricEncoder,
            let timbreEncoder,
            let conditionEncoder
        else {
            condLogger.error("[ConditioningActor.loadWeights] failed to initialise conditioning modules")
            return
        }

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

    /// Release loaded text/condition encoders and the learned silence prior.
    /// The tokenizer is cheap and remains available; modules are rebuilt on
    /// the next `loadWeights` call before generation resumes.
    public func unloadWeights() {
        qwen3 = nil
        lyricEncoder = nil
        timbreEncoder = nil
        conditionEncoder = nil
        silenceLatent = nil
        condLogger.info("[ConditioningActor.unloadWeights] cleared conditioning modules and silence_latent")
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
        return ConditioningActor(
            vae: VAEActor(),
            qwen3: Qwen3Model(),
            conditionEncoder: ConditionEncoder(),
            lyricEncoder: LyricEncoder(),
            timbreEncoder: TimbreEncoder(),
            tokenizer: try PiMPAudioTokenizer.forTesting()
        )
    }
}
