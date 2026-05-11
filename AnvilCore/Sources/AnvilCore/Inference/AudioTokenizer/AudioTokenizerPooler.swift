// AnvilCore/Sources/AnvilCore/Inference/AudioTokenizer/AudioTokenizerPooler.swift
import Foundation
import MLX
import MLXNN

/// The cls-token attention pooler that lives inside AudioTokenizer.
///
/// Input layout: `[B, num_windows, pool_window_size, hidden_size]` where each
/// `pool_window_size` slice is one window's worth of audio patches at the
/// VAE-latent rate. The pooler:
///   1. Reshape so each window is processed independently:
///        `[B * num_windows, pool_window_size, hidden_size]`
///   2. Prepend a learnable special_token (cls):
///        `[B * num_windows, 1 + pool_window_size, hidden_size]`
///   3. Run 2 self-attention transformer layers
///   4. Read out position 0 (the cls slot) and reshape back to
///        `[B, num_windows, hidden_size]`
///
/// The result is one pooled vector per window — the "5 Hz" representation
/// fed to the FSQ quantizer.
public class AudioTokenizerPooler: Module {
    /// `tokenizer.attention_pooler.embed_tokens` in the safetensors. Despite the
    /// name, this is a Linear (weight shape [hidden_size, hidden_size]) that
    /// projects each input patch before the cls is prepended.
    public var embedTokens: Linear
    /// `tokenizer.attention_pooler.special_token`, shape [1, 1, hidden_size].
    /// The cls query that gets prepended to each window's patches.
    public var specialToken: MLXArray
    /// Two self-attention transformer layers from `tokenizer.attention_pooler.layers.0..1`.
    public let layers: [Qwen3Layer]
    /// Final RMSNorm after the layers.
    public let norm: RMSNorm
    /// Rotary positional embedding shared across the layers.
    public let rotaryEmbedding: RotaryEmbedding

    public let hiddenSize: Int

    public init(
        hiddenSize: Int,
        intermediateSize: Int,
        numAttentionHeads: Int,
        numKeyValueHeads: Int,
        headDim: Int,
        numLayers: Int,
        rmsNormEps: Float,
        ropeTheta: Float,
        maxPositionEmbeddings: Int
    ) {
        self.hiddenSize = hiddenSize
        self.embedTokens = Linear(hiddenSize, hiddenSize, bias: true)
        self.specialToken = MLXArray.zeros([1, 1, hiddenSize])
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
            maxLen: maxPositionEmbeddings,
            base: ropeTheta
        )
    }

    /// Pool `[B, num_windows, pool_window_size, hidden_size]` →
    /// `[B, num_windows, hidden_size]` via cls-token self-attention.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let bsz = x.shape[0]
        let nWindows = x.shape[1]
        let poolWin = x.shape[2]
        let hSize = x.shape[3]

        // Linear-project each patch
        let projected = embedTokens(x.reshaped([bsz * nWindows, poolWin, hSize]))

        // Prepend cls (broadcast special_token across the [B*num_windows] axis)
        let cls = MLX.broadcast(specialToken, to: [bsz * nWindows, 1, hSize])
        var h = MLX.concatenated([cls, projected], axis: 1)               // [B*N, 1+poolWin, H]

        let seqLen = h.shape[1]
        let positionCosSin = rotaryEmbedding.forward(seqLen: seqLen)
        for layer in layers {
            h = layer(
                hiddenStates: h,
                positionCosSin: positionCosSin,
                attentionMask: nil
            )
        }
        h = norm(h)

        // Read out cls position (index 0), reshape back to [B, num_windows, H]
        let cls_out = h[0..., 0..<1, 0...]                                 // [B*N, 1, H]
        return cls_out.reshaped([bsz, nWindows, hSize])
    }
}
