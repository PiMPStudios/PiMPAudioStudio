import Foundation
import MLX
import MLXNN
import MLXRandom

public class AttentionPooler: Module {
    var queryTokens: MLXArray
    let layers: [Qwen3Layer]
    let norm: RMSNorm
    let rotaryEmbedding: RotaryEmbedding
    let numQueries: Int

    public init(
        hiddenSize: Int,
        numQueries: Int,
        numLayers: Int,
        numAttentionHeads: Int,
        numKeyValueHeads: Int,
        headDim: Int,
        intermediateSize: Int,
        rmsNormEps: Float
    ) {
        self.numQueries = numQueries
        self.queryTokens = MLXRandom.normal([1, numQueries, hiddenSize]) * 0.02
        self.layers = (0..<numLayers).map { _ in
            Qwen3Layer(
                hiddenSize: hiddenSize,
                intermediateSize: intermediateSize,
                numAttentionHeads: numAttentionHeads,
                numKeyValueHeads: numKeyValueHeads,
                headDim: headDim,
                rmsNormEps: rmsNormEps
            )
        }
        self.norm = RMSNorm(dimensions: hiddenSize, eps: rmsNormEps)
        self.rotaryEmbedding = RotaryEmbedding(
            headDim: headDim,
            maxLen: 32768,
            base: 1_000_000.0
        )
    }

    public func callAsFunction(encodedSequence: MLXArray) -> MLXArray {
        let b = encodedSequence.shape[0]
        let d = encodedSequence.shape[2]
        let queries = broadcast(queryTokens, to: [b, numQueries, d])
        var hiddenStates = concatenated([queries, encodedSequence], axis: 1)

        let seqLen = hiddenStates.shape[1]
        let positionCosSin = rotaryEmbedding.forward(seqLen: seqLen)

        for layer in layers {
            hiddenStates = layer(
                hiddenStates: hiddenStates,
                positionCosSin: positionCosSin,
                attentionMask: nil
            )
        }

        hiddenStates = norm(hiddenStates)
        return hiddenStates[0..<b, 0..<numQueries]
    }
}

public class TimbreEncoder: Module {
    let inputProjection: Linear
    let layers: [Qwen3Layer]
    let norm: RMSNorm
    let rotaryEmbedding: RotaryEmbedding
    let attentionPooler: AttentionPooler

    static let inputDim = 64
    static let hiddenSize = 2048
    static let intermediateSize = 6144
    static let numEncoderLayers = 4
    static let numPoolerLayers = 2
    static let numQueries = 5
    static let numAttentionHeads = 16
    static let numKeyValueHeads = 8
    static let headDim = 128
    static let rmsNormEps: Float = 1e-6

    override public init() {
        // v1.5 checkpoint provides both weight and bias for this projection
        // (called `embed_tokens` in safetensors). Linear must allocate the
        // bias parameter slot so the loaded tensor has a place to land.
        self.inputProjection = Linear(Self.inputDim, Self.hiddenSize, bias: true)
        self.layers = (0..<Self.numEncoderLayers).map { _ in
            Qwen3Layer(
                hiddenSize: Self.hiddenSize,
                intermediateSize: Self.intermediateSize,
                numAttentionHeads: Self.numAttentionHeads,
                numKeyValueHeads: Self.numKeyValueHeads,
                headDim: Self.headDim,
                rmsNormEps: Self.rmsNormEps
            )
        }
        self.norm = RMSNorm(dimensions: Self.hiddenSize, eps: Self.rmsNormEps)
        self.rotaryEmbedding = RotaryEmbedding(
            headDim: Self.headDim,
            maxLen: 32768,
            base: 1_000_000.0
        )
        self.attentionPooler = AttentionPooler(
            hiddenSize: Self.hiddenSize,
            numQueries: Self.numQueries,
            numLayers: Self.numPoolerLayers,
            numAttentionHeads: Self.numAttentionHeads,
            numKeyValueHeads: Self.numKeyValueHeads,
            headDim: Self.headDim,
            intermediateSize: Self.intermediateSize,
            rmsNormEps: Self.rmsNormEps
        )
    }

    public func callAsFunction(
        latents: MLXArray,
        attentionMask: MLXArray?
    ) -> MLXArray {
        var hiddenStates = inputProjection(latents)
        let seqLen = hiddenStates.shape[1]
        let positionCosSin = rotaryEmbedding.forward(seqLen: seqLen)

        for layer in layers {
            hiddenStates = layer(
                hiddenStates: hiddenStates,
                positionCosSin: positionCosSin,
                attentionMask: attentionMask
            )
        }

        hiddenStates = norm(hiddenStates)
        return attentionPooler(encodedSequence: hiddenStates)
    }
}
