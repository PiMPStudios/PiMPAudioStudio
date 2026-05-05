import Foundation
import MLX
import MLXNN

public actor ConditioningActor {
    private let vae: VAEActor
    private let qwen3: Qwen3Model
    private let lyricEncoder: LyricEncoder
    private let timbreEncoder: TimbreEncoder
    private let conditionEncoder: ConditionEncoder
    private let tokenizer: AnvilTokenizer

    static let timbreFixFrame = 750
    static let latentFrameRate: Float = 25.0

    public init(
        vae: VAEActor,
        qwen3: Qwen3Model,
        conditionEncoder: ConditionEncoder,
        lyricEncoder: LyricEncoder,
        timbreEncoder: TimbreEncoder,
        tokenizerURL: URL
    ) {
        self.vae = vae
        self.qwen3 = qwen3
        self.conditionEncoder = conditionEncoder
        self.lyricEncoder = lyricEncoder
        self.timbreEncoder = timbreEncoder
        self.tokenizer = try! AnvilTokenizer(configURL: tokenizerURL)
    }

    public func encode(
        text: String,
        lyrics: String,
        referenceAudio: MLXArray?,
        duration: TimeInterval,
        precision: Precision
    ) async -> ConditioningResult {
        let (textIds, textMask) = tokenizer.encodeText(text, maxLength: 256)
        let (lyricIds, lyricMask) = tokenizer.encodeLyrics(lyrics, maxLength: 2048)

        let textHidden = qwen3(inputIds: textIds)
        let lyricEmbeddings = qwen3.embedTokens(lyricIds)
        let lyricHidden = lyricEncoder(embeddings: lyricEmbeddings, attentionMask: lyricMask)

        let referenceLatents = await prepareReferenceLatents(referenceAudio, precision: precision)
        let timbreTokens = timbreEncoder(latents: referenceLatents, attentionMask: nil)

        let (encoderStates, encoderMask) = conditionEncoder(
            textHiddenStates: textHidden,
            textMask: textMask,
            lyricHiddenStates: lyricHidden,
            lyricMask: lyricMask,
            timbreTokens: timbreTokens
        )

        let tFrames = Int(Float(duration) * Self.latentFrameRate)
        let contextLatents: MLXArray
        if let ref = referenceAudio {
            let latents = await vae.encodeSample(audio: ref, precision: precision)
            let chunkMask = MLXArray.ones(latents.shape).asType(precision.mlxDtype)
            contextLatents = concatenated([latents, chunkMask], axis: -1)  // [B, T, 128]
        } else {
            contextLatents = MLXArray.zeros([1, tFrames, 128])
        }

        if precision == .float16 {
            return ConditioningResult(
                encoderStates: encoderStates.asType(.float16),
                encoderMask: encoderMask,
                contextLatents: contextLatents.asType(.float16)
            )
        }

        return ConditioningResult(
            encoderStates: encoderStates,
            encoderMask: encoderMask,
            contextLatents: contextLatents
        )
    }

    private func prepareReferenceLatents(_ referenceAudio: MLXArray?, precision: Precision) async -> MLXArray {
        guard let ref = referenceAudio else {
            return MLXArray.zeros([1, Self.timbreFixFrame, 64])
        }

        let latents = await vae.encodeSample(audio: ref, precision: precision)
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
        qwen3.update(parameters: ModuleParameters.unflattened(qwen3Weights))
        eval(qwen3.parameters())

        let lyricWeights = conditionWeights.compactMap { (key, value) -> (String, MLXArray)? in
            guard key.hasPrefix("lyricEncoder.") else { return nil }
            return (String(key.dropFirst("lyricEncoder.".count)), value)
        }
        lyricEncoder.update(parameters: ModuleParameters.unflattened(lyricWeights))
        eval(lyricEncoder.parameters())

        let timbreWeights = conditionWeights.compactMap { (key, value) -> (String, MLXArray)? in
            guard key.hasPrefix("timbreEncoder.") else { return nil }
            return (String(key.dropFirst("timbreEncoder.".count)), value)
        }
        timbreEncoder.update(parameters: ModuleParameters.unflattened(timbreWeights))
        eval(timbreEncoder.parameters())

        let condEncWeights = conditionWeights.compactMap { (key, value) -> (String, MLXArray)? in
            guard key.hasPrefix("conditionEncoder.") else { return nil }
            return (String(key.dropFirst("conditionEncoder.".count)), value)
        }
        conditionEncoder.update(parameters: ModuleParameters.unflattened(condEncWeights))
        eval(conditionEncoder.parameters())
    }
}

extension ConditioningActor {
    public static func forTesting(tokenizerURL: URL) -> ConditioningActor {
        return ConditioningActor(
            vae: VAEActor(),
            qwen3: Qwen3Model(),
            conditionEncoder: ConditionEncoder(),
            lyricEncoder: LyricEncoder(),
            timbreEncoder: TimbreEncoder(),
            tokenizerURL: tokenizerURL
        )
    }
}
