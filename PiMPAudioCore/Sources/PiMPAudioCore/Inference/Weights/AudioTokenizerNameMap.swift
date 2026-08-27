// PiMPAudioCore/Sources/PiMPAudioCore/Inference/Weights/AudioTokenizerNameMap.swift
import Foundation

public enum AudioTokenizerNameMap {
    /// Maps `tokenizer.*` keys from the DiT bundle to Swift parameter paths
    /// on the AudioTokenizer module. Returns nil for keys outside the
    /// `tokenizer.` subtree.
    public static func mapKey(_ key: String) -> String? {
        guard key.hasPrefix("tokenizer.") else { return nil }
        let stripped = String(key.dropFirst("tokenizer.".count))

        // Per-component snake→camel, preserving numeric layer indices and the
        // raw `special_token` -> `specialToken` rename which is just snake→camel.
        let components = stripped.split(separator: ".", omittingEmptySubsequences: false).map { component -> String in
            let s = String(component)
            if Int(s) != nil {
                return s
            }
            return snakeToCamel(s)
        }
        var mapped = components.joined(separator: ".")

        // Fix Qwen-style sub-paths inside attention_pooler.layers.N.* — these
        // need the same self_attn → attention rename used by Qwen3NameMap, so
        // they line up with our existing Qwen3Layer property names.
        mapped = mapped
            .replacingOccurrences(of: ".selfAttn.qProj", with: ".attention.qProj")
            .replacingOccurrences(of: ".selfAttn.kProj", with: ".attention.kProj")
            .replacingOccurrences(of: ".selfAttn.vProj", with: ".attention.vProj")
            .replacingOccurrences(of: ".selfAttn.oProj", with: ".attention.oProj")
            .replacingOccurrences(of: ".selfAttn.qNorm", with: ".attention.qNorm")
            .replacingOccurrences(of: ".selfAttn.kNorm", with: ".attention.kNorm")
            .replacingOccurrences(of: ".inputLayernorm", with: ".inputNorm")
            .replacingOccurrences(of: ".postAttentionLayernorm", with: ".postNorm")

        return mapped
    }

    private static func snakeToCamel(_ s: String) -> String {
        let parts = s.split(separator: "_")
        guard let first = parts.first else { return s }
        var result = String(first)
        for part in parts.dropFirst() {
            result += part.prefix(1).uppercased() + part.dropFirst()
        }
        return result
    }
}
