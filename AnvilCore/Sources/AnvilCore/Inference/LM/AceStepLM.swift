// AnvilCore/Sources/AnvilCore/Inference/LM/AceStepLM.swift
import Foundation
import MLX
import MLXNN

/// 5 Hz Audio→Token Language Model used to condition packed FSQ audio-code
/// tokens. Architecture is Qwen-style decoder-only (we reuse our existing
/// Qwen3Layer). The checkpoint ships without a separate `lm_head`, so audio
/// code prediction projects hidden states through the tied embedding matrix.
public class AceStepLM: Module {
    public var embedTokens: Embedding
    public let layers: [Qwen3Layer]
    public let norm: RMSNorm
    public let rotaryEmbedding: RotaryEmbedding

    public let hiddenSize: Int

    public init(config: AceStepLMConfig) {
        self.hiddenSize = config.hiddenSize
        self.embedTokens = Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.hiddenSize
        )
        self.layers = (0..<config.numHiddenLayers).map { _ in
            Qwen3Layer(
                hiddenSize: config.hiddenSize,
                intermediateSize: config.intermediateSize,
                numAttentionHeads: config.numAttentionHeads,
                numKeyValueHeads: config.numKeyValueHeads,
                headDim: config.headDim,
                rmsNormEps: config.rmsNormEps
            )
        }
        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.rotaryEmbedding = RotaryEmbedding(
            headDim: config.headDim,
            maxLen: config.maxPositionEmbeddings,
            base: config.ropeTheta
        )
    }

    /// Forward through the LM and return final hidden states.
    /// - Parameter inputTokenIDs: `[B, T_5Hz]` tokenizer vocabulary IDs.
    /// - Parameter attentionMask: `[B, T_5Hz]` 1.0 for valid, 0.0 for PAD.
    /// - Returns: `[B, T_5Hz, hidden_size]` final hidden states (post-norm).
    public func callAsFunction(inputTokenIDs: MLXArray, attentionMask: MLXArray?) -> MLXArray {
        var h = embedTokens(inputTokenIDs)                               // [B, T, H]
        let b = h.shape[0]
        let seqLen = h.shape[1]
        let positionCosSin = rotaryEmbedding.forward(seqLen: seqLen)

        let additiveMask = Self.causalAttentionMask(
            batchSize: b,
            seqLen: seqLen,
            attentionMask: attentionMask
        )

        for layer in layers {
            h = layer(
                hiddenStates: h,
                positionCosSin: positionCosSin,
                attentionMask: additiveMask
            )
        }
        return norm(h)
    }

    /// Run the 5 Hz LM over packed FSQ audio-code IDs and greedily predict a
    /// conditioned audio-code stream. The returned IDs are still packed FSQ
    /// codes (`0..<vocabulary.codeCount`), ready for `ResidualFSQ`.
    ///
    /// `sourceCodes` may be `[B, T]` or `[B, T, 1]`. The LM is causal, so
    /// logits at position `i` predict the next token; the output preserves the
    /// first source code and fills positions `1...` from those next-token
    /// predictions.
    public func predictAudioCodes(
        sourceCodes: MLXArray,
        vocabulary: AudioCodeVocabulary
    ) -> MLXArray {
        let packedCodes = sourceCodes.ndim == 3 && sourceCodes.shape[2] == 1
            ? sourceCodes[0..., 0..., 0]
            : sourceCodes
        let seqLen = packedCodes.shape[1]
        guard seqLen > 0 else {
            return packedCodes.expandedDimensions(axis: -1)
        }

        let inputTokenIDs = packedCodes.asType(.int32) + MLXArray(Int32(vocabulary.tokenStartID))
        let hidden = self(inputTokenIDs: inputTokenIDs, attentionMask: nil)

        let audioCodeEmbeddings = embedTokens.weight[
            vocabulary.tokenStartID..<vocabulary.tokenEndID,
            0...
        ]
        let logits = matmul(hidden.asType(audioCodeEmbeddings.dtype), audioCodeEmbeddings.T)
        let greedyNextCodes = logits.argMax(axis: -1).asType(.int32)

        let conditionedCodes: MLXArray
        if seqLen == 1 {
            conditionedCodes = packedCodes.asType(.int32)
        } else {
            let firstSourceCode = packedCodes[0..., 0..<1].asType(.int32)
            let shiftedPredictions = greedyNextCodes[0..., 0..<(seqLen - 1)]
            conditionedCodes = MLX.concatenated([firstSourceCode, shiftedPredictions], axis: 1)
        }
        eval(conditionedCodes)
        return conditionedCodes.expandedDimensions(axis: -1)
    }

    private static func causalAttentionMask(
        batchSize: Int,
        seqLen: Int,
        attentionMask: MLXArray?
    ) -> MLXArray {
        let indices = MLXArray(Array(0..<seqLen).map { Float($0) })
        let rowIdx = indices.expandedDimensions(axis: 1)
        let colIdx = indices.expandedDimensions(axis: 0)
        let zeros = MLXArray.zeros([seqLen, seqLen], dtype: .float32)
        let negInf = MLXArray.full([seqLen, seqLen], values: MLXArray(Float(-1e9)))
        let causalMask = which(colIdx .> rowIdx, negInf, zeros)

        guard let attentionMask else {
            return causalMask
        }
        let paddingAdditive = (1.0 - attentionMask.asType(.float32)) * Float(-1e9)
        return causalMask + paddingAdditive.reshaped([batchSize, 1, 1, seqLen])
    }
}
