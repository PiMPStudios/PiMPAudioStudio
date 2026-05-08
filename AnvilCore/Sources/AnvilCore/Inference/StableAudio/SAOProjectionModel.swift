// AnvilCore/Sources/AnvilCore/Inference/StableAudio/SAOProjectionModel.swift
//
// Stable Audio Open projection_model.safetensors — two NumberConditioner
// instances that encode scalar timing values (seconds_start, seconds_total)
// into conditioning embeddings for the DiT.
//
// Each SAOSingleConditioner:
//   1. time_positional_embedding: learned Fourier frequency bank [128]
//   2. embedder: Linear(257 → 768, bias=True)
//      257 = 1 raw scalar value + 128 sin features + 128 cos features
// Output of each conditioner: [1, 768].
//
// T5-base hidden states [B, L, 768] pass directly to DiT cross-attention;
// there is NO text-projection layer in the real model.
//
// The two conditioner outputs are concatenated → [B, 1536] = globalCondDim
// and fed into SFXDiTModel.toGlobalEmbed.
//
// Actual safetensors key layout (pass-through in SAOProjectionNameMap):
//   start_number_conditioner.time_positional_embedding.frequencies  [128]
//   start_number_conditioner.embedder.weight                        [768, 257]
//   start_number_conditioner.embedder.bias                          [768]
//   end_number_conditioner.time_positional_embedding.frequencies    [128]
//   end_number_conditioner.embedder.weight                          [768, 257]
//   end_number_conditioner.embedder.bias                            [768]

import Foundation
import MLX
import MLXNN

// MARK: - Learned Fourier embedding

/// Wraps the `time_positional_embedding.frequencies` [numFrequencies] buffer.
/// Encodes a scalar float value into a [1 + 2·numFrequencies] feature vector:
///   [value, sin(2π·freq₀·value), …, cos(2π·freq₀·value), …]
public final class SAOFourierEmbedding: Module {
    /// Learned frequency bank, shape [numFrequencies].
    public var frequencies: MLXArray

    public init(numFrequencies: Int = 128) {
        self.frequencies = MLXArray.zeros([numFrequencies])
    }

    public func encode(_ value: Float) -> MLXArray {
        let t    = MLXArray(value)                              // scalar
        let args = 2 * Float.pi * frequencies * t              // [numFrequencies]
        return concatenated(
            [t.reshaped([1]), sin(args), cos(args)], axis: 0
        )                                                       // [1 + 2·numFrequencies]
    }
}

// MARK: - Single timing conditioner

public final class SAOSingleConditioner: Module {
    @ModuleInfo(key: "time_positional_embedding")
    public var fourier: SAOFourierEmbedding
    /// Linear(257, outputDim, bias=True).
    public var embedder: Linear

    public init(numFrequencies: Int = 128, outputDim: Int = 768) {
        let featureDim = 1 + 2 * numFrequencies               // 257
        self._fourier.wrappedValue = SAOFourierEmbedding(numFrequencies: numFrequencies)
        self.embedder = Linear(featureDim, outputDim, bias: true)
    }

    /// Encode a scalar timing value → [1, outputDim].
    public func callAsFunction(_ value: Float) -> MLXArray {
        let features = fourier.encode(value)                   // [257]
        return embedder(features.expandedDimensions(axis: 0)) // [1, outputDim]
    }
}

// MARK: - Dual conditioner: seconds_start + seconds_total

public final class SAONumberConditioner: Module {
    @ModuleInfo(key: "start_number_conditioner")
    public var startCond: SAOSingleConditioner
    @ModuleInfo(key: "end_number_conditioner")
    public var totalCond: SAOSingleConditioner

    public init(numFrequencies: Int = 128, outputDim: Int = 768) {
        self._startCond.wrappedValue = SAOSingleConditioner(
            numFrequencies: numFrequencies, outputDim: outputDim
        )
        self._totalCond.wrappedValue = SAOSingleConditioner(
            numFrequencies: numFrequencies, outputDim: outputDim
        )
    }

    /// Encode start and total duration.
    /// - Returns: `(startEmbed, totalEmbed)` each shaped `[1, outputDim]`.
    public func callAsFunction(
        startSeconds: Float,
        totalSeconds: Float
    ) -> (startEmbed: MLXArray, totalEmbed: MLXArray) {
        (startCond(startSeconds), totalCond(totalSeconds))
    }
}
