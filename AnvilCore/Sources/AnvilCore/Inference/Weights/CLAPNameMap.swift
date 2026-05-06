// AnvilCore/Sources/AnvilCore/Inference/Weights/CLAPNameMap.swift
import MLX

public enum CLAPNameMap {
    public static func mapKey(_ key: String) -> String? {
        var mapped = key
        if mapped.hasPrefix("text_model.") {
            mapped = String(mapped.dropFirst("text_model.".count))
        }
        if mapped.hasPrefix("text_projection.") {
            mapped = mapped.replacingOccurrences(of: "text_projection.", with: "projection.")
        }
        if mapped.contains("layer_norm") {
            mapped = mapped.replacingOccurrences(of: "layer_norm", with: "ln")
        }
        if mapped.contains("self_attn.q_proj") {
            mapped = mapped.replacingOccurrences(of: "self_attn.q_proj", with: "attention.query")
        }
        if mapped.contains("self_attn.k_proj") {
            mapped = mapped.replacingOccurrences(of: "self_attn.k_proj", with: "attention.key")
        }
        if mapped.contains("self_attn.v_proj") {
            mapped = mapped.replacingOccurrences(of: "self_attn.v_proj", with: "attention.value")
        }
        if mapped.contains("self_attn.out_proj") {
            mapped = mapped.replacingOccurrences(of: "self_attn.out_proj", with: "attention.out")
        }
        if mapped.contains("mlp.fc1") || mapped.contains("mlp.fc2") {
            // Already matches our naming
        }
        return mapped
    }
}
