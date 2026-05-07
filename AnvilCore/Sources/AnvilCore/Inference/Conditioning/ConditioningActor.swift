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

    public func encode(
        text: String,
        lyrics: String,
        referenceAudio: MLXArray?,
        duration: TimeInterval,
        precision: Precision
    ) async throws -> ConditioningResult {
        let (textIds, textMask) = tokenizer.encodeText(text, maxLength: 256)
        let (lyricIds, lyricMask) = tokenizer.encodeLyrics(lyrics, maxLength: 2048)

        try Task.checkCancellation()

        let textHidden = qwen3(inputIds: textIds, attentionMask: textMask)

        // With longest-padding tokenization, an empty lyrics input now
        // produces a 1-token degenerate sequence (mask=0) rather than 2048
        // PAD tokens. Bypass remains: when lyrics are truly empty, emit a
        // 1-token zero hidden state matching the lyricIds shape. Avoids
        // running LyricEncoder over a PAD-only input (whose output would
        // still propagate non-trivial content via the encoder's biases).
        let trimmedLyrics = lyrics.trimmingCharacters(in: .whitespacesAndNewlines)
        let lyricHidden: MLXArray
        if trimmedLyrics.isEmpty {
            // [1, lyricSeqLen=1, hiddenSize=2048]
            let seq = lyricIds.shape[1]
            lyricHidden = MLXArray.zeros([1, seq, 2048]).asType(precision.mlxDtype)
        } else {
            let lyricEmbeddings = qwen3.embedTokens(lyricIds)
            lyricHidden = lyricEncoder(embeddings: lyricEmbeddings, attentionMask: lyricMask)
        }

        try Task.checkCancellation()

        // referenceLatents is [B, 750, 64] — raw VAE latent space fed to timbreEncoder
        // contextLatents is [B, T, 128] = cat([src_latents, chunk_mask], axis:-1)
        let referenceLatents = try await prepareReferenceLatents(referenceAudio, precision: precision)

        // The v1.5 turbo/full/sft checkpoints don't ship weights for the
        // TimbreEncoder.AttentionPooler subtree (only the special_token,
        // which is also shape-mismatched with our [1,5,2048] queryTokens
        // and gets skipped in ConditionEncoderNameMap). When the user has
        // not provided reference audio, running the timbre encoder over
        // zero latents pushes them through the default-init random
        // AttentionPooler, yielding ~5 RANDOM tokens that then get
        // concatenated into encoder_states with mask=1 (always attended).
        // For short prompts like "slow dark guitar riff" — only 5-10 real
        // text tokens vs the 5 random timbre tokens — that random signal
        // dilutes prompt-following by ~50%. Pass zeros directly when
        // there's no reference audio so the cross-attention's V-projection
        // contributes nothing from this path.
        let timbreTokens: MLXArray
        if referenceAudio == nil {
            // 5 query tokens of zeros at model hidden size (2048).
            timbreTokens = MLXArray.zeros([1, 5, 2048]).asType(precision.mlxDtype)
        } else {
            timbreTokens = timbreEncoder(latents: referenceLatents, attentionMask: nil)
        }

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
            contextLatents = MLXArray.zeros([1, tFrames, 128])
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
