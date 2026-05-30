// PiMPAudioCore/Sources/PiMPAudioCore/Inference/Weights/AudioTokenDetokenizerNameMap.swift
import Foundation

public enum AudioTokenDetokenizerNameMap {
    /// Maps `detokenizer.*` keys from the DiT bundle to Swift parameter
    /// paths on the AudioTokenDetokenizer module. Returns nil for keys
    /// outside the `detokenizer.` subtree.
    public static func mapKey(_ key: String) -> String? {
        guard key.hasPrefix("detokenizer.") else { return nil }
        let stripped = String(key.dropFirst("detokenizer.".count))

        let components = stripped.split(separator: ".", omittingEmptySubsequences: false).map { component -> String in
            let s = String(component)
            if Int(s) != nil { return s }
            return snakeToCamel(s)
        }
        var mapped = components.joined(separator: ".")

        // Same Qwen3Layer-internal renames as AudioTokenizerNameMap.
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
