// AnvilCore/Sources/AnvilCore/Inference/Weights/SFXDiTNameMap.swift
//
// Maps HF `stabilityai/stable-audio-open-1.0` model.safetensors keys
// to Swift `SFXDiTModel` module-parameter paths.
//
// Key transformations applied in order:
//   1. Strip "model.model." double-prefix (absent in the mlx-audiogen export —
//      checkpoint keys are unprefixed at the top level).
//   2. Drop the non-learned rotary inv_freq buffer.
//   3. Remap three nn.Sequential index suffixes → fc1/fc2 aliases:
//        to_timestep_embed.0.* → to_timestep_embed.fc1.*
//        to_timestep_embed.2.* → to_timestep_embed.fc2.*
//        (same for to_cond_embed and to_global_embed)
//   4. GEGLU sub-module rerouting:
//        .ff.ff.0.proj.* → .ff.gate_up.*
//        .ff.ff.2.*      → .ff.out.*
//   5. LayerNorm parameter rename:
//        .gamma → .weight,  .beta → .bias
//   6. Transformer block remap:
//        blocks.N.* → transformer.layers.N.*
//      The mlx-audiogen export stores the 24 SAODiTLayer blocks at
//      "blocks.0…23.*" (top-level, no transformer. prefix) while our
//      Swift SAODiTTransformer holds them in `var layers: [SAODiTLayer]`
//      at the path "transformer.layers.N.*".
//
// Conv1d weights (preprocess_conv.weight, postprocess_conv.weight):
// mlx-audiogen exports store these already in MLX [out, k, in] layout —
// e.g. [64, 1, 64] for kernel=1.  WeightConverter.loadSFXDiTWeights uses
// isConv1dWeight with a shape check: only transposes when loading the
// original PyTorch [out, in, k] format (shape ends with kernel=1).

import MLX

public enum SFXDiTNameMap {

    // MARK: - Key mapping (path rename only, no value transformation)

    public static func mapKey(_ key: String) -> String? {
        // 1. Strip the double model.model. prefix.
        var k = key.hasPrefix("model.model.")
            ? String(key.dropFirst("model.model.".count))
            : key

        // 2. Drop non-trainable rotary inverse-frequency buffer.
        if k.hasSuffix("rotary_pos_emb.inv_freq") { return nil }

        // 3. Sequential index 0/2 → fc1/fc2 for the three embed MLPs.
        for prefix in ["to_timestep_embed", "to_cond_embed", "to_global_embed"] {
            let p0 = "\(prefix).0."
            let p2 = "\(prefix).2."
            if k.hasPrefix(p0) {
                k = "\(prefix).fc1." + k.dropFirst(p0.count)
                break
            }
            if k.hasPrefix(p2) {
                k = "\(prefix).fc2." + k.dropFirst(p2.count)
                break
            }
        }

        // 4. GEGLU sub-module: ff.ff.0.proj → ff.gate_up, ff.ff.2 → ff.out.
        if let r = k.range(of: ".ff.ff.0.proj.") {
            k = k.replacingCharacters(in: r, with: ".ff.gate_up.")
        } else if let r = k.range(of: ".ff.ff.2.") {
            k = k.replacingCharacters(in: r, with: ".ff.out.")
        }

        // 5. LayerNorm: gamma → weight, beta → bias.
        if k.hasSuffix(".gamma") {
            k = String(k.dropLast(".gamma".count)) + ".weight"
        } else if k.hasSuffix(".beta") {
            k = String(k.dropLast(".beta".count)) + ".bias"
        }

        // 6. Transformer block remap: "blocks.N.*" → "transformer.layers.N.*"
        //    The mlx-audiogen checkpoint stores the 24 DiT layers at top-level
        //    "blocks.0…23.*" rather than under a "transformer." parent module.
        //    Our Swift SAODiTTransformer stores them in `layers: [SAODiTLayer]`,
        //    so MLX resolves them at "transformer.layers.N.*".
        if k.hasPrefix("blocks.") {
            let rest = k.dropFirst("blocks.".count)
            if let dotIdx = rest.firstIndex(of: ".") {
                let n      = rest[..<dotIdx]
                let after  = rest[dotIdx...]
                k = "transformer.layers.\(n)\(after)"
            }
        }

        return k
    }

    // MARK: - Conv1d weight detection

    /// Returns true if the mapped key corresponds to a Conv1d weight that
    /// requires PyTorch→MLX axis transposition ([out, in, kernel] → [out, kernel, in]).
    public static func isConv1dWeight(_ key: String) -> Bool {
        key == "preprocess_conv.weight" || key == "postprocess_conv.weight"
    }

    /// Transpose a Conv1d weight from PyTorch layout to MLX layout.
    public static func conv1dToMLX(_ weight: MLXArray) -> MLXArray {
        weight.transposed(0, 2, 1)
    }
}
