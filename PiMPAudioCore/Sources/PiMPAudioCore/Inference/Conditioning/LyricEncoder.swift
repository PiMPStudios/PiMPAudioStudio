import Foundation
import MLX
import MLXNN

public class LyricEncoder: Module {
    let inputProjection: Linear
    let layers: [Qwen3Layer]
    let norm: RMSNorm
    let rotaryEmbedding: RotaryEmbedding

    static let inputDim = 1024
    static let hiddenSize = 2048
    static let intermediateSize = 6144
    static let numLayers = 8
    static let numAttentionHeads = 16
    static let numKeyValueHeads = 8
    static let headDim = 128
    static let rmsNormEps: Float = 1e-6

    override public init() {
        // v1.5 checkpoint provides both weight and bias for this projection
        // (called `embed_tokens` in safetensors). Linear must allocate the
        // bias parameter slot so the loaded tensor has a place to land.
        inputProjection = Linear(Self.inputDim, Self.hiddenSize, bias: true)
        layers = (0 ..< Self.numLayers).map { _ in
            Qwen3Layer(
                hiddenSize: Self.hiddenSize,
                intermediateSize: Self.intermediateSize,
                numAttentionHeads: Self.numAttentionHeads,
                numKeyValueHeads: Self.numKeyValueHeads,
                headDim: Self.headDim,
                rmsNormEps: Self.rmsNormEps
            )
        }
        norm = RMSNorm(dimensions: Self.hiddenSize, eps: Self.rmsNormEps)
        rotaryEmbedding = RotaryEmbedding(
            headDim: Self.headDim,
            maxLen: 32768,
            base: 1_000_000.0
        )
    }

    public func callAsFunction(
        embeddings: MLXArray,
        attentionMask: MLXArray?
    ) -> MLXArray {
        var hiddenStates = inputProjection(embeddings)
        let seqLen = hiddenStates.shape[1]
        let positionCosSin = rotaryEmbedding.forward(seqLen: seqLen)

        for layer in layers {
            hiddenStates = layer(
                hiddenStates: hiddenStates,
                positionCosSin: positionCosSin,
                attentionMask: attentionMask
            )
        }

        return norm(hiddenStates)
    }
}
