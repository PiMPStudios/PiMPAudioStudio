// AnvilCore/Sources/AnvilCore/Inference/StableAudio/SFXVAEModel.swift
//
// Thin wrapper over OobleckDecoder for Stable Audio Open.
//
// The SAO 1.0 vae.safetensors uses the same weight-normed Oobleck format as the
// ACE-Step music VAE.  Weight loading is handled by WeightConverter.loadSFXVAEWeights
// which applies VAENameMap (weight-norm fusion + axis transpositions + snake squeeze).
//
// SAO 1.0 decoder config:
//   latentDim=64  →  channels=128  ×  channelMultiples=[1,2,4,8,16]
//   upsamplingRatios=[8,8,4,4,2]   →  total upsampling = 2048 (= hop_length)
//   audioChannels=2  (stereo; WAVWriter handles [B,T,2] correctly)
//
// The module property `decoder` is the root for weight paths, so checkpoint keys
// like `decoder.conv1.weight` and `decoder.blocks.0.convT1.weight` resolve
// directly after VAENameMap remapping.

import Foundation
import MLX
import MLXNN

public class SFXVAEModel: Module {
    let config: SFXVAEConfig
    /// OobleckEncoder with SAO strides [2,4,4,8,8] (reversed upsample ratios).
    /// Outputs `[B, T_latent, 2 × latentDim]` — split into (μ, log_scale) in encodeMean.
    /// Weights live under the `encoder.*` key prefix in `vae.safetensors`; they are
    /// loaded by `WeightConverter.loadSFXVAEWeights` (which applies the same VAENameMap
    /// as the music VAE) and were previously silently ignored because this module
    /// didn't exist yet.
    let encoder: OobleckEncoder
    let decoder: OobleckDecoder

    public init(config: SFXVAEConfig) {
        self.config = config
        // channelMultiples is fixed for SAO 1.0; ratios come from config.
        let channelMultiples = [1, 2, 4, 8, 16]
        self.encoder = OobleckEncoder(
            encoderHiddenSize: 128,
            audioChannels: config.outputChannels,
            downsamplingRatios: config.upsampleRatios.reversed(),  // [8,8,4,4,2] → [2,4,4,8,8]
            channelMultiples: channelMultiples
        )
        self.decoder = OobleckDecoder(
            channels: 128,
            inputChannels: config.latentDim,
            audioChannels: config.outputChannels,
            upsamplingRatios: config.upsampleRatios,
            channelMultiples: channelMultiples
        )
    }

    /// Encode audio `[B, T_audio, audioChannels]` → latent mean `[B, T_latent, latentDim]`.
    ///
    /// The encoder outputs `2 × latentDim` channels (μ || log_scale); we return μ only
    /// (same as `AutoEncoderOobleck.encodeMean` for the ACE-Step VAE).
    public func encodeMean(audio: MLXArray) -> MLXArray {
        let h = encoder(audio)                          // [B, T_latent, 2*latentDim]
        return MLX.split(h, parts: 2, axis: -1)[0]     // [B, T_latent, latentDim]
    }

    /// Decode diffusion latents `[B, T_latent, latentDim]` → audio `[B, T_audio, audioChannels]`.
    public func decode(latents: MLXArray) -> MLXArray {
        decoder(latents)
    }
}
