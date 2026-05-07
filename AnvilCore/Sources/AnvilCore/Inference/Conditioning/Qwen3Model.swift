import Foundation
import MLX
import MLXNN

public class Qwen3Model: Module {
    let embeddingTable: Embedding
    let layers: [Qwen3Layer]
    let norm: RMSNorm
    let rotaryEmbedding: RotaryEmbedding

    static let vocabSize = 151_669
    static let hiddenSize = 1024
    static let intermediateSize = 3072
    static let numLayers = 28
    static let numAttentionHeads = 16
    static let numKeyValueHeads = 8
    static let headDim = 128
    static let rmsNormEps: Float = 1e-6

    override public init() {
        self.embeddingTable = Embedding(
            embeddingCount: Self.vocabSize,
            dimensions: Self.hiddenSize
        )
        self.layers = (0..<Self.numLayers).map { _ in
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
    }

    public func callAsFunction(inputIds: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
        var hiddenStates = embeddingTable(inputIds)
        let seqLen = hiddenStates.shape[1]
        let positionCosSin = rotaryEmbedding.forward(seqLen: seqLen)

        // Convert binary mask [B, seqLen] (1=real, 0=pad) to additive mask (0, -1e9)
        // so that attention scores for padding positions become -inf before softmax.
        let additiveMask: MLXArray? = attentionMask.map { mask in
            (1.0 - mask.asType(.float32)) * -1e9
        }

        for layer in layers {
            hiddenStates = layer(
                hiddenStates: hiddenStates,
                positionCosSin: positionCosSin,
                attentionMask: additiveMask
            )
        }

        return norm(hiddenStates)
    }

    public func embedTokens(_ inputIds: MLXArray) -> MLXArray {
        embeddingTable(inputIds)
    }
}
