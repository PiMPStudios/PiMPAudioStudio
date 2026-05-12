import Foundation
import MLX
import AnvilCore

/// Remaps PEFT-format LoRA weight keys to AnvilApp format.
///
/// HuggingFace community adapters built with the PEFT library use key patterns like:
/// `base_model.model.transformer.layers.0.self_attn.q_proj.lora_A.weight`
///
/// AnvilApp expects:
/// `layers.0.selfAttn.qProj.loraA`
///
/// Keys that cannot be mapped (embedding LoRA, bias terms, unknown projections)
/// are silently dropped — the injector skips weights it doesn't recognise anyway.
enum PEFTKeyRemapper {

    // MARK: - Detection

    /// Returns `true` if the weight dictionary looks like a PEFT checkpoint.
    static func isPEFT(_ keys: [String]) -> Bool {
        keys.contains { $0.hasPrefix("base_model.model.") }
    }

    // MARK: - Weight Remapping

    /// Remap a full PEFT weight dictionary to AnvilApp format.
    /// Unmappable keys are silently dropped.
    static func remap(_ dict: [String: MLXArray]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        for (key, value) in dict {
            if let remapped = remapKey(key) {
                result[remapped] = value
            }
        }
        return result
    }

    /// Remap a single PEFT weight key to AnvilApp format.
    /// Returns `nil` when the key has no AnvilApp equivalent.
    static func remapKey(_ key: String) -> String? {
        var k = key

        // 1. Strip `base_model.model.` PEFT prefix
        if k.hasPrefix("base_model.model.") {
            k = String(k.dropFirst("base_model.model.".count))
        }

        // 2. Strip known model-level wrappers (transformer., dit., model.)
        for prefix in ["transformer.", "dit.", "model."] {
            if k.hasPrefix(prefix) {
                k = String(k.dropFirst(prefix.count))
                break
            }
        }

        // 3. Must start with "layers.N." at this point
        let components = k.components(separatedBy: ".")
        guard components.count >= 3,
              components[0] == "layers",
              Int(components[1]) != nil else { return nil }

        let layerPrefix = "layers.\(components[1])."
        var rest = String(k.dropFirst(layerPrefix.count))

        // 4. Strip LoRA A/B suffix (PEFT uses several conventions)
        let peftSuffixes: [(String, String)] = [
            (".lora_A.default.weight", ".loraA"),
            (".lora_B.default.weight", ".loraB"),
            (".lora_A.weight",         ".loraA"),
            (".lora_B.weight",         ".loraB"),
        ]
        var loraSuffix: String?
        for (peft, anvil) in peftSuffixes {
            if rest.hasSuffix(peft) {
                rest = String(rest.dropLast(peft.count))
                loraSuffix = anvil
                break
            }
        }
        guard let loraSuffix else { return nil }

        // 5. Map the module subpath to AnvilApp camelCase convention
        guard let mappedPath = mapModulePath(rest) else { return nil }

        return layerPrefix + mappedPath + loraSuffix
    }

    // MARK: - LoRAConfig Inference

    /// Infer the closest `Set<TargetLayer>` from PEFT `target_modules`.
    ///
    /// Note: `TargetLayer.selfAttention.defaultPatterns` covers q/v only.
    /// PEFT adapters that also target k/o will have those weights stored in
    /// the safetensors file, but the k/o contributions will be silently unused
    /// until AnvilApp adds full q/k/v/o injection pattern support.
    static func inferTargetLayers(from targetModules: [String]) -> Set<TargetLayer> {
        let attnModules: Set<String> = [
            "q_proj", "k_proj", "v_proj", "o_proj", "out_proj",
            "q", "k", "v", "o", "to_q", "to_k", "to_v", "to_out",
        ]
        let mlpModules: Set<String> = [
            "gate_proj", "up_proj", "down_proj", "w1", "w2", "w3", "fc1", "fc2",
        ]

        var layers = Set<TargetLayer>()
        for module in targetModules {
            if attnModules.contains(module) { layers.insert(.selfAttention) }
            if mlpModules.contains(module)  { layers.insert(.mlp) }
        }
        // Default to self-attention when target_modules lists nothing recognisable
        return layers.isEmpty ? [.selfAttention] : layers
    }

    /// Map a PEFT HuggingFace repo ID / model path to the AnvilApp local model ID.
    ///
    /// ACE-Step v1.5 variants (turbo, base, SFT) share the same 24-layer DiT architecture,
    /// so PEFT adapters trained on any of them are compatible. The turbo checkpoint is the
    /// default and most commonly installed, so map all ACE-Step HF paths to it.
    static func resolveBaseModelID(from hfPath: String?) -> String {
        guard let path = hfPath else { return "ace-step-dit-turbo" }
        let lower = path.lowercased()
        if lower.contains("ace-step") || lower.contains("ace_step") {
            return "ace-step-dit-turbo"
        }
        // Unknown model — preserve the HF path so the UI can show a mismatch warning
        return path
    }

    // MARK: - Private Helpers

    /// Map a PEFT module subpath (e.g. `"self_attn.q_proj"`) to AnvilApp format (`"selfAttn.qProj"`).
    private static func mapModulePath(_ path: String) -> String? {
        // Self-attention (several naming conventions in the wild)
        for prefix in ["self_attn.", "attn.", "attention."] {
            if path.hasPrefix(prefix) {
                let proj = String(path.dropFirst(prefix.count))
                return mapAttnProj(proj).map { "selfAttn.\($0)" }
            }
        }
        // Cross-attention
        for prefix in ["cross_attn.", "crossattn.", "cross_attention.", "encoder_attn."] {
            if path.hasPrefix(prefix) {
                let proj = String(path.dropFirst(prefix.count))
                return mapAttnProj(proj).map { "crossAttn.\($0)" }
            }
        }
        // MLP / FFN
        for prefix in ["ffn.", "ff.", "mlp.", "feed_forward.", "feedforward.", "fc."] {
            if path.hasPrefix(prefix) {
                let proj = String(path.dropFirst(prefix.count))
                return mapMLPProj(proj).map { "mlp.\($0)" }
            }
        }
        return nil
    }

    private static func mapAttnProj(_ name: String) -> String? {
        switch name {
        case "q_proj", "q", "to_q":                          return "qProj"
        case "k_proj", "k", "to_k":                          return "kProj"
        case "v_proj", "v", "to_v":                          return "vProj"
        case "o_proj", "out_proj", "proj_out", "to_out", "o": return "oProj"
        default: return nil
        }
    }

    private static func mapMLPProj(_ name: String) -> String? {
        switch name {
        case "gate_proj", "w1", "wi_0": return "gateProj"
        case "up_proj",   "w3", "wi_1": return "upProj"
        case "down_proj", "w2", "wo":   return "downProj"
        default: return nil
        }
    }
}
