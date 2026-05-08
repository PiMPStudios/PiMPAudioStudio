// AnvilCore/Sources/AnvilCore/Inference/Weights/T5NameMap.swift
//
// Map HuggingFace T5 safetensors keys to our `T5Encoder` parameter paths.
//
// HF naming (from `text_encoder/model.safetensors`):
//   shared.weight
//   encoder.block.0.layer.0.SelfAttention.q.weight
//   encoder.block.0.layer.0.SelfAttention.k.weight
//   encoder.block.0.layer.0.SelfAttention.v.weight
//   encoder.block.0.layer.0.SelfAttention.o.weight
//   encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight
//   encoder.block.0.layer.0.layer_norm.weight
//   encoder.block.0.layer.1.DenseReluDense.wi_0.weight
//   encoder.block.0.layer.1.DenseReluDense.wi_1.weight
//   encoder.block.0.layer.1.DenseReluDense.wo.weight
//   encoder.block.0.layer.1.layer_norm.weight
//   encoder.final_layer_norm.weight
//
// Our `T5Encoder` paths:
//   shared.weight
//   blocks.0.layer0_attn.q.weight
//   blocks.0.layer0_attn.k.weight
//   blocks.0.layer0_attn.v.weight
//   blocks.0.layer0_attn.o.weight
//   blocks.0.layer0_attn.relative_attention_bias.weight
//   blocks.0.layer0_norm.weight
//   blocks.0.layer1_mlp.wi_0.weight
//   blocks.0.layer1_mlp.wi_1.weight
//   blocks.0.layer1_mlp.wo.weight
//   blocks.0.layer1_norm.weight
//   final_layer_norm.weight
//
// HF also ships `lm_head` and `decoder.*` keys when the safetensors covers
// the full encoder-decoder T5; for SAO's text-only encoder usage we drop
// anything outside `encoder.*` + `shared`.

import Foundation

public enum T5NameMap {
    public static func mapKey(_ key: String) -> String? {
        // Always-keep top-level shared embedding.
        if key == "shared.weight" { return "shared.weight" }
        if key == "encoder.final_layer_norm.weight" { return "finalLayerNorm.weight" }

        // Drop decoder + lm_head (T5 encoder-only usage).
        if key.hasPrefix("decoder.") { return nil }
        if key.hasPrefix("lm_head.") { return nil }

        // encoder.block.N.layer.0.* → blocks.N.layer0_*
        // encoder.block.N.layer.1.* → blocks.N.layer1_*
        guard key.hasPrefix("encoder.block.") else { return nil }
        var rest = String(key.dropFirst("encoder.block.".count))
        // rest: "N.layer.X.<...>"
        guard let firstDot = rest.firstIndex(of: ".") else { return nil }
        let blockIdx = String(rest[..<firstDot])
        rest = String(rest[rest.index(after: firstDot)...])  // "layer.X.<...>"

        guard rest.hasPrefix("layer.") else { return nil }
        rest = String(rest.dropFirst("layer.".count))  // "X.<...>"
        guard let layerDot = rest.firstIndex(of: ".") else { return nil }
        let layerIdx = String(rest[..<layerDot])  // "0" or "1"
        rest = String(rest[rest.index(after: layerDot)...])  // remaining

        switch layerIdx {
        case "0":
            // Self-attention layer
            if rest == "layer_norm.weight" {
                return "blocks.\(blockIdx).layer0_norm.weight"
            }
            if rest.hasPrefix("SelfAttention.") {
                let attnPart = String(rest.dropFirst("SelfAttention.".count))
                return "blocks.\(blockIdx).layer0_attn.\(attnPart)"
            }
            return nil

        case "1":
            // FFN layer
            if rest == "layer_norm.weight" {
                return "blocks.\(blockIdx).layer1_norm.weight"
            }
            if rest.hasPrefix("DenseReluDense.") {
                let mlpPart = String(rest.dropFirst("DenseReluDense.".count))
                return "blocks.\(blockIdx).layer1_mlp.\(mlpPart)"
            }
            return nil

        default:
            return nil
        }
    }
}
