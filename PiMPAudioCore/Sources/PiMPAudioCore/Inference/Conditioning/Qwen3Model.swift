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
        let b = hiddenStates.shape[0]
        let seqLen = hiddenStates.shape[1]
        let positionCosSin = rotaryEmbedding.forward(seqLen: seqLen)

        // Build causal (upper-triangular) additive mask [seqLen, seqLen].
        // The Qwen3-Embedding model is a decoder LLM — its weights were trained
        // with causal (autoregressive) attention. Position j cannot attend to a
        // future key position i > j; those entries must be −∞ before softmax.
        // Without this mask the hidden states diverge wildly from the Python
        // HuggingFace Qwen3Model forward pass (max abs diff ~45), producing
        // completely wrong text conditioning for the DiT.
        let indices = MLXArray(Array(0..<seqLen).map { Float($0) })
        let rowIdx = indices.expandedDimensions(axis: 1)  // [seqLen, 1]  (query positions)
        let colIdx = indices.expandedDimensions(axis: 0)  // [1, seqLen]  (key positions)
        let zeros  = MLXArray.zeros([seqLen, seqLen], dtype: .float32)
        let negInf = MLXArray.full([seqLen, seqLen], values: MLXArray(Float(-1e9)))
        // mask[i, j] = −1e9 when key j > query i (future token), else 0.
        let causalMask = which(colIdx .> rowIdx, negInf, zeros)  // [seqLen, seqLen]

        // Combine causal mask with optional padding mask.
        // causalMask  [seqLen, seqLen] → broadcasts as [1, 1, seqLen, seqLen]
        // paddingMask [B, seqLen]      → reshape to [B, 1, 1, seqLen]
        // Sum → [B, 1, seqLen, seqLen] (both key-padding and future positions masked)
        let combinedMask: MLXArray
        if let padMask = attentionMask {
            let paddingAdditive = (1.0 - padMask.asType(.float32)) * Float(-1e9)
            combinedMask = causalMask + paddingAdditive.reshaped([b, 1, 1, seqLen])
        } else {
            combinedMask = causalMask
        }

        for layer in layers {
            hiddenStates = layer(
                hiddenStates: hiddenStates,
                positionCosSin: positionCosSin,
                attentionMask: combinedMask
            )
        }

        return norm(hiddenStates)
    }

    public func embedTokens(_ inputIds: MLXArray) -> MLXArray {
        embeddingTable(inputIds)
    }
}
