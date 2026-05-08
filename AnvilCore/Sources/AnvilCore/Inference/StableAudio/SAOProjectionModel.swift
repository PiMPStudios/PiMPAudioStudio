// AnvilCore/Sources/AnvilCore/Inference/StableAudio/SAOProjectionModel.swift
//
// Stable Audio Open's "projection_model" — maps T5-base hidden states (768)
// into the DiT cross-attention conditioning embedding space and combines
// the T5 stream with the timing (seconds_start / seconds_total) stream.
//
// Skeleton: minimal Linear(768 → 768) + a placeholder timing-projection.
// Real architecture lands when Stable Audio Open re-architecture is
// enabled. Until then SAO spec is `isRunnable: false` so users can't
// accidentally run a stubbed pipeline.

import Foundation
import MLX
import MLXNN

public final class SAOProjectionConfig: Sendable {
    /// T5-base hidden size = 768 — input dimension from text encoder.
    public let textInputDim: Int
    /// DiT conditioning dimension. Stable Audio Open's transformer expects
    /// `cross_attention_embed_dim = 768` for T5 conditioning. Confirm
    /// against `transformer/config.json` once we have it loaded; the
    /// skeleton uses 768 as a safe default.
    public let outputDim: Int
    /// Number of scalar timing features (seconds_start + seconds_total).
    public let timingDim: Int

    public init(
        textInputDim: Int = 768,
        outputDim: Int = 768,
        // 256 matches the default embedding produced by `TimingEncoder`
        // (128 sin + 128 cos features split across start + duration). The
        // actual projection_model.safetensors Linear will load with its
        // own in/out shape; if that disagrees with this default at runtime
        // the SFXModelManager weight diff will surface the mismatch.
        timingDim: Int = 256
    ) {
        self.textInputDim = textInputDim
        self.outputDim = outputDim
        self.timingDim = timingDim
    }
}

public final class SAOProjectionModel: Module {
    public let config: SAOProjectionConfig
    /// Text-stream projection. Real model may use a multi-layer MLP; the
    /// skeleton uses a single Linear so weight loading still resolves.
    @ModuleInfo(key: "text_projection")
    public var textProjection: Linear
    /// Timing-stream projection. Maps `[seconds_start, seconds_total]`
    /// (or sinusoidal embedding thereof) into the same DiT embedding dim.
    @ModuleInfo(key: "timing_projection")
    public var timingProjection: Linear

    public init(config: SAOProjectionConfig = SAOProjectionConfig()) {
        self.config = config
        self._textProjection.wrappedValue =
            Linear(config.textInputDim, config.outputDim, bias: false)
        self._timingProjection.wrappedValue =
            Linear(config.timingDim, config.outputDim, bias: false)
    }

    /// Project T5 hidden states + scalar timing features into DiT
    /// conditioning. Skeleton: returns `textProjection(textHidden)` and
    /// ignores timing (until real architecture lands).
    /// - Parameters:
    ///   - textHidden: `[B, L, textInputDim]` T5 last-hidden-state.
    ///   - timing: `[B, timingDim]` raw `[seconds_start, seconds_total]`.
    /// - Returns: `(textCond, timingCond)` ready for DiT cross-attn / sum.
    public func callAsFunction(
        textHidden: MLXArray,
        timing: MLXArray
    ) -> (text: MLXArray, timing: MLXArray) {
        let textCond = textProjection(textHidden)
        let timingCond = timingProjection(timing)
        return (textCond, timingCond)
    }
}
