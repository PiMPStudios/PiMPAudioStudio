// PiMPAudioCore/Sources/PiMPAudioCore/Inference/AudioTokenizer/AudioTokenDetokenizer.swift
import Foundation
import MLX
import MLXNN

/// Codes→acoustic upsampler. Mirrors upstream `AudioTokenDetokenizer` in
/// `acestep/models/turbo/modeling_acestep_v15_turbo.py:859`.
///
/// Input is a 5 Hz stream (one token per pool window), output is the 25 Hz
/// acoustic-feature stream:
///
///     embed_tokens          : [B, T_5Hz, hidden] → [B, T_5Hz, hidden]
///     unsqueeze + repeat    : → [B, T_5Hz, pool, hidden]
///     + special_tokens      : add learnable per-slot position tokens
///                              → [B, T_5Hz, pool, hidden]
///     reshape (b t p h)→(bt p h): for window-local self-attention
///     2 transformer layers   : self-attention within each window
///     norm                  : RMSNorm
///     proj_out              : [..., hidden] → [..., audio_acoustic_hidden_dim]
///     reshape back          : → [B, T_5Hz * pool, audio_acoustic_hidden_dim]
public class AudioTokenDetokenizer: Module {
    /// `detokenizer.embed_tokens` — Linear(hidden_size, hidden_size).
    public var embedTokens: Linear
    /// `detokenizer.special_tokens`, shape `[1, pool_window_size, hidden_size]`.
    public var specialTokens: MLXArray
    public let layers: [Qwen3Layer]
    public let norm: RMSNorm
    public let rotaryEmbedding: RotaryEmbedding
    /// `detokenizer.proj_out` — Linear(hidden_size, audio_acoustic_hidden_dim).
    public var projOut: Linear

    public let poolWindowSize: Int
    public let hiddenSize: Int

    public init(
        hiddenSize: Int = 2048,
        intermediateSize: Int = 6144,
        numAttentionHeads: Int = 16,
        numKeyValueHeads: Int = 8,
        headDim: Int = 128,
        numLayers: Int = 2,
        poolWindowSize: Int = 5,
        audioAcousticHiddenDim: Int = 64,
        rmsNormEps: Float = 1e-6,
        ropeTheta: Float = 1_000_000.0,
        maxPositionEmbeddings: Int = 32768
    ) {
        self.hiddenSize = hiddenSize
        self.poolWindowSize = poolWindowSize
        embedTokens = Linear(hiddenSize, hiddenSize, bias: true)
        specialTokens = MLXArray.zeros([1, poolWindowSize, hiddenSize])
        layers = (0 ..< numLayers).map { _ in
            Qwen3Layer(
                hiddenSize: hiddenSize,
                intermediateSize: intermediateSize,
                numAttentionHeads: numAttentionHeads,
                numKeyValueHeads: numKeyValueHeads,
                headDim: headDim,
                rmsNormEps: rmsNormEps
            )
        }
        norm = RMSNorm(dimensions: hiddenSize, eps: rmsNormEps)
        rotaryEmbedding = RotaryEmbedding(
            headDim: headDim,
            maxLen: maxPositionEmbeddings,
            base: ropeTheta
        )
        projOut = Linear(hiddenSize, audioAcousticHiddenDim, bias: true)
    }

    /// Detokenize 5 Hz hidden states into 25 Hz acoustic features.
    /// - Parameter lmHints5Hz: `[B, T_5Hz, hidden_size]`.
    /// - Returns: `[B, T_5Hz * pool_window_size, audio_acoustic_hidden_dim]`.
    public func callAsFunction(_ lmHints5Hz: MLXArray) -> MLXArray {
        let bsz = lmHints5Hz.shape[0]
        let t5Hz = lmHints5Hz.shape[1]
        let hSize = hiddenSize
        let pool = poolWindowSize

        // Linear-project the LM hidden states
        var x = embedTokens(lmHints5Hz) // [B, T, H]

        // Expand each token into `pool` patches and add learnable special_tokens
        x = x.expandedDimensions(axis: 2) // [B, T, 1, H]
        x = MLX.broadcast(x, to: [bsz, t5Hz, pool, hSize]) // [B, T, pool, H]
        let special = MLX.broadcast(specialTokens, to: [bsz, t5Hz, pool, hSize])
        x = x + special

        // Reshape so each window's `pool` patches are processed independently.
        // (b t p h) → (b*t p h)
        var h = x.reshaped([bsz * t5Hz, pool, hSize])

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
        let acoustic = projOut(h) // [B*T, pool, 64]

        // Reshape back to a flat 25 Hz stream
        let outDim = acoustic.shape[2]
        return acoustic.reshaped([bsz, t5Hz * pool, outDim])
    }
}
