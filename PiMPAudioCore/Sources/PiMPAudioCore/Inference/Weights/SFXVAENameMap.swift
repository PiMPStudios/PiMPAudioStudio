// PiMPAudioCore/Sources/PiMPAudioCore/Inference/Weights/SFXVAENameMap.swift
//
// DEPRECATED — no longer used by WeightConverter.loadSFXVAEWeights.
//
// This mapper was written expecting an mlx-audiogen pre-converted vae.safetensors
// with Sequential `layers.layers.X.*` key paths.  The actual SAO vae.safetensors
// is in the standard PyTorch weight-normed Oobleck format:
//   decoder.block.N.conv_t1.{weight_g, weight_v, bias}
//   decoder.block.N.res_unit1.{conv1,conv2,snake1,snake2}.*
//   decoder.conv1.{weight_g, weight_v, bias}
//   decoder.snake1.{alpha, beta}
//   decoder.conv2.{weight_g, weight_v}    (no bias — bias=false)
//
// That format is identical to the ACE-Step music VAE, so WeightConverter now
// calls convertVAEWeights (which uses VAENameMap) for both.

import MLX

/// Retained only to avoid breaking any external callers; not used internally.
public enum SFXVAENameMap {
    /// Always returns nil.  Use VAENameMap / convertVAEWeights instead.
    public static func mapKey(_ key: String) -> String? {
        nil
    }
}
