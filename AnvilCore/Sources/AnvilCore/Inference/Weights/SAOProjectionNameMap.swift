// AnvilCore/Sources/AnvilCore/Inference/Weights/SAOProjectionNameMap.swift
//
// Key mapper for `projection_model.safetensors` (Stable Audio Open).
// The file contains only the two SAONumberConditioner instances.
// T5 text conditioning flows directly to cross-attention — no projection.
//
// Actual checkpoint key layout:
//   start_number_conditioner.time_positional_embedding.0.weights  [128]
//   start_number_conditioner.time_positional_embedding.1.weight   [768, 257]
//   start_number_conditioner.time_positional_embedding.1.bias     [768]
//   end_number_conditioner.time_positional_embedding.0.weights    [128]
//   end_number_conditioner.time_positional_embedding.1.weight     [768, 257]
//   end_number_conditioner.time_positional_embedding.1.bias       [768]
//
// `time_positional_embedding` is an nn.Sequential([FourierFeatures, nn.Linear])
// in the Python model — hence the .0. and .1. indices.
//
// We remap to match our Swift module layout on SAOSingleConditioner:
//   .time_positional_embedding.0.weights  →  .time_positional_embedding.frequencies
//   .time_positional_embedding.1.weight   →  .embedder.weight
//   .time_positional_embedding.1.bias     →  .embedder.bias

import Foundation

public enum SAOProjectionNameMap {
    public static func mapKey(_ key: String) -> String? {
        guard key.hasPrefix("start_number_conditioner.") ||
              key.hasPrefix("end_number_conditioner.") else { return nil }

        var k = key
        // Unwrap Sequential .0 (FourierFeatures) → fourierEmbedding property name
        k = k.replacingOccurrences(
            of: ".time_positional_embedding.0.weights",
            with: ".time_positional_embedding.frequencies"
        )
        // Unwrap Sequential .1 (nn.Linear) → embedder sub-module
        k = k.replacingOccurrences(
            of: ".time_positional_embedding.1.weight",
            with: ".embedder.weight"
        )
        k = k.replacingOccurrences(
            of: ".time_positional_embedding.1.bias",
            with: ".embedder.bias"
        )
        return k
    }
}
