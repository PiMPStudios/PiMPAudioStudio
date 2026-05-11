// AnvilCore/Sources/AnvilCore/Inference/LM/AceStepLM.swift
import Foundation
import MLX
import MLXNN

/// 5 Hz Audio→Token Language Model used to produce LM hints from FSQ-coded
/// audio tokens. Architecture is Qwen-style decoder-only (we reuse our
/// existing Qwen3Layer). We expose final hidden states rather than next-token
/// logits — those hidden states feed AudioTokenDetokenizer to upsample to
/// 25 Hz acoustic features.
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
    /// - Parameter audioCodes: `[B, T_5Hz]` integer FSQ indices.
    /// - Parameter attentionMask: `[B, T_5Hz]` 1.0 for valid, 0.0 for PAD.
    /// - Returns: `[B, T_5Hz, hidden_size]` final hidden states (post-norm).
    public func callAsFunction(audioCodes: MLXArray, attentionMask: MLXArray?) -> MLXArray {
        var h = embedTokens(audioCodes)                                  // [B, T, H]
        let seqLen = h.shape[1]
        let positionCosSin = rotaryEmbedding.forward(seqLen: seqLen)

        // Convert padding mask to additive bias for attention.
        let additiveMask: MLXArray? = attentionMask.map { mask in
            let inverted = (1.0 - mask) * Float(-1e9)
            return inverted
        }

        for layer in layers {
            h = layer(
                hiddenStates: h,
                positionCosSin: positionCosSin,
                attentionMask: additiveMask
            )
        }
        return norm(h)
    }
}
