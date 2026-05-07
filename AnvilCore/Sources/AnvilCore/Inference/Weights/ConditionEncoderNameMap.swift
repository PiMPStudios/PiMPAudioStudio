import Foundation

public enum ConditionEncoderNameMap {
    public static func mapKey(_ key: String) -> String? {
        // The ACE-Step v1.5 turbo checkpoint groups conditioning weights under
        // an `encoder.` subtree (alongside the DiT under `decoder.`):
        //
        //     encoder.text_projector.weight
        //     encoder.lyric_encoder.layers.N.…
        //     encoder.timbre_encoder.layers.N.…
        //     encoder.timbre_encoder.special_token

        // Single linear: encoder.text_projector → conditionEncoder.textProjection
        if key.hasPrefix("encoder.text_projector") {
            return key.replacingOccurrences(
                of: "encoder.text_projector",
                with: "conditionEncoder.textProjection"
            )
        }

        // Lyric encoder: encoder.lyric_encoder.* → lyricEncoder.*
        if key.hasPrefix("encoder.lyric_encoder.") {
            let stripped = key.replacingOccurrences(
                of: "encoder.lyric_encoder.",
                with: "lyricEncoder."
            )
            return mapEncoderKey(stripped)
        }

        // Timbre encoder: encoder.timbre_encoder.* → timbreEncoder.*
        if key.hasPrefix("encoder.timbre_encoder.") {
            let stripped = key.replacingOccurrences(
                of: "encoder.timbre_encoder.",
                with: "timbreEncoder."
            )
            return mapEncoderKey(stripped)
        }

        return nil
    }

    /// Remaps structural sub-keys (attention projections, MLP projections, layer norms)
    /// for a key that already carries its Swift-side prefix.
    private static func mapLayerKey(_ key: String) -> String {
        var mapped = key

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

        return mapped
    }

    private static func mapEncoderKey(_ key: String) -> String? {
        // The v1.5 turbo timbre_encoder has a top-level `special_token` of
        // shape [1, 1, 2048] but our Swift `AttentionPooler.queryTokens`
        // expects [1, 5, 2048] (numQueries=5). Plus the rest of the
        // attentionPooler subtree (layers, norm) has NO corresponding
        // weights in the v1.5 checkpoint. So skip special_token entirely;
        // the AttentionPooler stays at default init. Mostly relevant only
        // when reference audio is provided.
        if key.hasSuffix(".special_token") { return nil }

        var mapped = key

        // embed_tokens → inputProjection
        mapped = mapped.replacingOccurrences(of: ".embed_tokens.", with: ".inputProjection.")

        return mapLayerKey(mapped)
    }
}
