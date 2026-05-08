// AnvilCore/Sources/AnvilCore/Inference/Weights/SAOProjectionNameMap.swift
//
// Key mapper for `projection_model.safetensors` (Stable Audio Open).
// The file contains only the two SAONumberConditioner instances.
// T5 text conditioning flows directly to cross-attention — no projection.
//
// Actual key layout (matches @ModuleInfo keys on SAONumberConditioner /
// SAOSingleConditioner / SAOFourierEmbedding exactly, so all keys pass
// through unchanged):
//   start_number_conditioner.time_positional_embedding.frequencies
//   start_number_conditioner.embedder.weight / .bias
//   end_number_conditioner.time_positional_embedding.frequencies
//   end_number_conditioner.embedder.weight / .bias

import Foundation

public enum SAOProjectionNameMap {
    public static func mapKey(_ key: String) -> String? {
        guard key.hasPrefix("start_number_conditioner.") ||
              key.hasPrefix("end_number_conditioner.") else { return nil }
        return key
    }
}
