// AnvilCore/Sources/AnvilCore/Inference/Weights/SFXDiTNameMap.swift
import MLX

public enum SFXDiTNameMap {
    public static func mapKey(_ key: String) -> String? {
        var mapped = key
        if mapped.hasPrefix("model.") {
            mapped = String(mapped.dropFirst("model.".count))
        }
        if mapped.contains("transformer_blocks") {
            mapped = mapped.replacingOccurrences(of: "transformer_blocks", with: "blocks")
        }
        if mapped.contains("norm1.linear") {
            mapped = mapped.replacingOccurrences(of: "norm1.linear", with: "adaLN")
        }
        if mapped.contains("attn1.to_q") {
            mapped = mapped.replacingOccurrences(of: "attn1.to_q", with: "attn.qkv")
        }
        if mapped.contains("attn1.to_out.0") {
            mapped = mapped.replacingOccurrences(of: "attn1.to_out.0", with: "attn.out")
        }
        if mapped.contains("attn2.to_q") {
            mapped = mapped.replacingOccurrences(of: "attn2.to_q", with: "cross_attn.q")
        }
        if mapped.contains("attn2.to_kv") {
            mapped = mapped.replacingOccurrences(of: "attn2.to_kv", with: "cross_attn.kv")
        }
        if mapped.contains("attn2.to_out.0") {
            mapped = mapped.replacingOccurrences(of: "attn2.to_out.0", with: "cross_attn.out")
        }
        if mapped.contains("ff.net.0") {
            mapped = mapped.replacingOccurrences(of: "ff.net.0", with: "mlp.fc1")
        }
        if mapped.contains("ff.net.2") {
            mapped = mapped.replacingOccurrences(of: "ff.net.2", with: "mlp.fc2")
        }
        return mapped
    }
}
