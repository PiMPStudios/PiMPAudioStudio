// PiMPAudioCore/Sources/PiMPAudioCore/Inference/AudioTokenizer/AudioTokenizer.swift
import Foundation
import MLX
import MLXNN

/// Audio→token encoder. Converts a stream of VAE acoustic latents at 25 Hz
/// into 5 Hz quantized tokens via:
///
///     audio_acoustic_proj  : [B, T, 64]    → [B, T, 2048]
///     reshape windows      : [B, T, 2048]  → [B, T/pool, pool, 2048]
///     attention_pooler     : [B, T/pool, pool, 2048] → [B, T/pool, 2048]
///     residual FSQ         : [B, T/pool, 2048] → quantized [B, T/pool, 2048] + indices [B, T/pool, 1]
///
/// Mirrors upstream `AceStepAudioTokenizer` in
/// `acestep/models/turbo/modeling_acestep_v15_turbo.py:1178`.
public class AudioTokenizer: Module {
    /// `tokenizer.audio_acoustic_proj` — Linear(audio_acoustic_hidden_dim, hidden_size).
    public var audioAcousticProj: Linear
    public let attentionPooler: AudioTokenizerPooler
    public let quantizer: ResidualFSQ
    public let poolWindowSize: Int

    public init(
        audioAcousticHiddenDim: Int = 64,
        hiddenSize: Int = 2048,
        intermediateSize: Int = 6144,
        numAttentionHeads: Int = 16,
        numKeyValueHeads: Int = 8,
        headDim: Int = 128,
        numPoolerLayers: Int = 2,
        poolWindowSize: Int = 5,
        fsqLevels: [Int] = [8, 8, 8, 5, 5, 5],
        rmsNormEps: Float = 1e-6,
        ropeTheta: Float = 1_000_000.0,
        maxPositionEmbeddings: Int = 32768
    ) {
        audioAcousticProj = Linear(audioAcousticHiddenDim, hiddenSize, bias: true)
        attentionPooler = AudioTokenizerPooler(
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize,
            numAttentionHeads: numAttentionHeads,
            numKeyValueHeads: numKeyValueHeads,
            headDim: headDim,
            numLayers: numPoolerLayers,
            rmsNormEps: rmsNormEps,
            ropeTheta: ropeTheta,
            maxPositionEmbeddings: maxPositionEmbeddings
        )
        quantizer = ResidualFSQ(hiddenSize: hiddenSize, levels: fsqLevels)
        self.poolWindowSize = poolWindowSize
    }

    /// Tokenize a continuous acoustic stream.
    /// - Parameter audioLatents: `[B, T, audio_acoustic_hidden_dim]` where T must
    ///   be a multiple of `poolWindowSize`. The caller is responsible for
    ///   padding to that boundary (upstream pads with `silence_latent`).
    /// - Returns: `(quantized, indices)`:
    ///   - `quantized`: `[B, T/pool, hidden_size]` — continuous post-FSQ representation
    ///   - `indices`:   `[B, T/pool, 1]` — packed FSQ codes per window
    public func callAsFunction(_ audioLatents: MLXArray) -> (quantized: MLXArray, indices: MLXArray) {
        let bsz = audioLatents.shape[0]
        let tFrames = audioLatents.shape[1]
        let hSize = audioAcousticProj.weight.shape[0]
        precondition(
            tFrames % poolWindowSize == 0,
            "AudioTokenizer.forward: T (\(tFrames)) must be divisible by pool_window_size (\(poolWindowSize))"
        )

        let projected = audioAcousticProj(audioLatents) // [B, T, hSize]
        let nWindows = tFrames / poolWindowSize
        let windowed = projected.reshaped([bsz, nWindows, poolWindowSize, hSize]) // [B, T/pool, pool, H]
        let pooled = attentionPooler(windowed) // [B, T/pool, H]
        return quantizer(pooled)
    }
}
