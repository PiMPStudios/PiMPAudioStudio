import Foundation

public enum ConditionEncoderNameMap {
    public static func mapKey(_ key: String) -> String? {
        // condition_encoder.text_projector.* → conditionEncoder.textProjection.*
        if key.hasPrefix("condition_encoder.text_projector") {
            return key.replacingOccurrences(
                of: "condition_encoder.text_projector",
                with: "conditionEncoder.textProjection"
            )
        }

        // condition_encoder.lyric_encoder.* → lyricEncoder.*
        if key.hasPrefix("condition_encoder.lyric_encoder.") {
            let stripped = key.replacingOccurrences(
                of: "condition_encoder.lyric_encoder.",
                with: "lyricEncoder."
            )
            return mapEncoderKey(stripped)
        }

        // condition_encoder.timbre_encoder.* → timbreEncoder.*
        if key.hasPrefix("condition_encoder.timbre_encoder.") {
            let stripped = key.replacingOccurrences(
                of: "condition_encoder.timbre_encoder.",
                with: "timbreEncoder."
            )
            return mapEncoderKey(stripped)
        }

        return nil
    }

    private static func mapEncoderKey(_ key: String) -> String {
        var mapped = key

        // embed_tokens → inputProjection
        mapped = mapped.replacingOccurrences(of: ".embed_tokens.", with: ".inputProjection.")

        // self_attn projections
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

        // special_token → attentionPooler.queryTokens
        mapped = mapped.replacingOccurrences(of: ".special_token", with: ".attentionPooler.queryTokens")

        return mapped
    }
}
