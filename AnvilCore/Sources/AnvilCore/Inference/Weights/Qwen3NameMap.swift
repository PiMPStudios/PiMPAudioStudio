import Foundation

public enum Qwen3NameMap {
    public static func mapKey(_ key: String) -> String? {
        var mapped = key

        // Filter out keys we don't need
        if key.hasPrefix("lm_head.") { return nil }

        // model.embed_tokens.weight → embeddingTable.weight
        mapped = mapped.replacingOccurrences(of: "model.embed_tokens", with: "embeddingTable")

        // model.norm.weight → norm.weight
        mapped = mapped.replacingOccurrences(of: "model.norm", with: "norm")

        // model.layers.N.* → layers.N.*
        mapped = mapped.replacingOccurrences(of: "model.layers.", with: "layers.")

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
