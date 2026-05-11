// AnvilCore/Sources/AnvilCore/Inference/Weights/AceStepLMNameMap.swift
import Foundation

public enum AceStepLMNameMap {
    /// Maps a key from an ACE-Step LM variant's safetensors to the parameter
    /// path expected by our Swift `AceStepLM` module. Both 0.6B (single file)
    /// and 4B (sharded) ship the same Qwen-style key naming.
    ///
    /// No `lm_head` filter — these checkpoints were saved without one
    /// (verified for 0.6B; 4B follows the same convention).
    public static func mapKey(_ key: String) -> String? {
        var mapped = key

        // Strip the optional model. prefix some Qwen checkpoints carry.
        if mapped.hasPrefix("model.") {
            mapped = String(mapped.dropFirst("model.".count))
        }

        // Filter lm_head if it ever appears.
        if mapped.hasPrefix("lm_head.") { return nil }

        // embed_tokens → embedTokens (note: AceStepLM uses Embedding, not
        // Linear, so the parameter path is .embedTokens.weight directly).
        mapped = mapped.replacingOccurrences(of: "embed_tokens.", with: "embedTokens.")

        // Standard Qwen-style sub-renames.
        mapped = mapped
            .replacingOccurrences(of: ".self_attn.q_proj", with: ".attention.qProj")
            .replacingOccurrences(of: ".self_attn.k_proj", with: ".attention.kProj")
            .replacingOccurrences(of: ".self_attn.v_proj", with: ".attention.vProj")
            .replacingOccurrences(of: ".self_attn.o_proj", with: ".attention.oProj")
            .replacingOccurrences(of: ".self_attn.q_norm", with: ".attention.qNorm")
            .replacingOccurrences(of: ".self_attn.k_norm", with: ".attention.kNorm")
            .replacingOccurrences(of: ".mlp.gate_proj", with: ".mlp.gateProj")
            .replacingOccurrences(of: ".mlp.up_proj", with: ".mlp.upProj")
            .replacingOccurrences(of: ".mlp.down_proj", with: ".mlp.downProj")
            .replacingOccurrences(of: ".input_layernorm", with: ".inputNorm")
            .replacingOccurrences(of: ".post_attention_layernorm", with: ".postNorm")

        return mapped
    }
}
