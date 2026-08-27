import Foundation

public enum Qwen3NameMap {
    public static func mapKey(_ key: String) -> String? {
        var mapped = key

        // Filter out keys we don't need
        if key.hasPrefix("lm_head.") {
            return nil
        }

        // Some Qwen3 checkpoints prefix every key with `model.` (the full
        // CausalLM packaging); the ACE-Step v1.5 `Qwen3-Embedding-0.6B`
        // checkpoint stores them flat (`embed_tokens.weight`, …). Strip
        // the prefix if present so both layouts load.
        if mapped.hasPrefix("model.") {
            mapped = String(mapped.dropFirst("model.".count))
        }

        // embed_tokens.* → embeddingTable.*
        mapped = mapped.replacingOccurrences(of: "embed_tokens.", with: "embeddingTable.")
        // (`norm.*` and `layers.N.*` already match the Swift module — no rename needed.)

        // self_attn projections → attention projections
        mapped = mapped.replacingOccurrences(of: ".self_attn.q_proj", with: ".attention.qProj")
        mapped = mapped.replacingOccurrences(of: ".self_attn.k_proj", with: ".attention.kProj")
        mapped = mapped.replacingOccurrences(of: ".self_attn.v_proj", with: ".attention.vProj")
        mapped = mapped.replacingOccurrences(of: ".self_attn.o_proj", with: ".attention.oProj")
        mapped = mapped.replacingOccurrences(of: ".self_attn.q_norm", with: ".attention.qNorm")
        mapped = mapped.replacingOccurrences(of: ".self_attn.k_norm", with: ".attention.kNorm")

        // mlp projections
        mapped = mapped.replacingOccurrences(of: ".mlp.gate_proj", with: ".mlp.gateProj")
        mapped = mapped.replacingOccurrences(of: ".mlp.up_proj", with: ".mlp.upProj")
        mapped = mapped.replacingOccurrences(of: ".mlp.down_proj", with: ".mlp.downProj")

        // layer norms
        mapped = mapped.replacingOccurrences(of: ".input_layernorm", with: ".inputNorm")
        mapped = mapped.replacingOccurrences(of: ".post_attention_layernorm", with: ".postNorm")

        return mapped
    }
}
