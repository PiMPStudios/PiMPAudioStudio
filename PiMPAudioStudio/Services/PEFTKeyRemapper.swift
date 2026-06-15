import Foundation
import MLX
import PiMPAudioCore

/// Remaps PEFT-format LoRA weight keys to PiMPAudioStudio format.
///
/// HuggingFace community adapters built with the PEFT library use key patterns like:
/// `base_model.model.transformer.layers.0.self_attn.q_proj.lora_A.weight`
///
/// PiMPAudioStudio expects:
/// `layers.0.selfAttn.qProj.loraA`
///
/// Keys that cannot be mapped (embedding LoRA, bias terms, unknown projections)
/// are silently dropped — the injector skips weights it doesn't recognise anyway.
enum PEFTKeyRemapper {

    // MARK: - Detection

    /// Returns `true` if the weight dictionary contains LoRA A/B matrices in either
    /// PEFT format (`base_model.model.` prefix) or Diffusers format (direct path,
    /// e.g. `transformer.layers.0.self_attn.q_proj.lora_A.weight`).
    static func isPEFT(_ keys: [String]) -> Bool {
        keys.contains { $0.hasPrefix("base_model.model.") } ||
        keys.contains { $0.contains(".lora_A.") || $0.contains(".lora_B.") }
    }

    // MARK: - Weight Remapping

    /// Remap a full PEFT weight dictionary to PiMPAudioStudio format.
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

    /// Infer exact PiMP Audio Studio injection subpaths from already-remapped LoRA keys.
    ///
    /// Remapped weights are keyed like `layers.0.selfAttn.qProj.loraA`.
    /// `LoRAConfig` needs only the layer-local path (`selfAttn.qProj`) because
    /// the injector applies the pattern across the loaded DiT layers.
    static func inferInjectionPatterns(fromRemappedKeys keys: [String]) -> [String] {
        var patterns = Set<String>()

        for key in keys {
            let parts = key.split(separator: ".")
            guard parts.count >= 5,
                  parts[0] == "layers",
                  Int(parts[1]) != nil,
                  parts.last == "loraA" || parts.last == "loraB" else { continue }

            let module = String(parts[2])
            let projection = String(parts[3])
            patterns.insert("\(module).\(projection)")
        }

        return patterns.sorted { lhs, rhs in
            patternSortKey(lhs) < patternSortKey(rhs)
        }
    }

    /// Remap a single PEFT weight key to PiMPAudioStudio format.
    /// Returns `nil` when the key has no PiMPAudioStudio equivalent.
    static func remapKey(_ key: String) -> String? {
        var k = key

        // 1. Strip `base_model.model.` PEFT prefix.
        // Some training scripts nest the wrapper multiple times
        // (e.g. `base_model.model.base_model.model.…`) — loop until clean.
        while k.hasPrefix("base_model.model.") {
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
        for (peft, pimpAudio) in peftSuffixes {
            if rest.hasSuffix(peft) {
                rest = String(rest.dropLast(peft.count))
                loraSuffix = pimpAudio
                break
            }
        }
        guard let loraSuffix else { return nil }

        // 5. Map the module subpath to PiMPAudioStudio camelCase convention
        guard let mappedPath = mapModulePath(rest) else { return nil }

        return layerPrefix + mappedPath + loraSuffix
    }

    /// Remap a PEFT module path such as `layers.0.self_attn.q_proj` to the
    /// PiMP Audio Studio path used by native adapter weights.
    static func remapModulePath(_ path: String) -> String? {
        guard let remappedA = remapKey("\(path).lora_A.weight"),
              remappedA.hasSuffix(".loraA") else {
            return nil
        }
        return String(remappedA.dropLast(".loraA".count))
    }

    // MARK: - Config Inference (weights-only repos)

    /// Synthesised configuration inferred from weight keys and tensor shapes.
    /// Used when the repo ships `adapter_model.safetensors` with no `adapter_config.json`.
    struct InferredConfig {
        let rank: Int
        let alpha: Float
        let targetModules: [String]
    }

    /// Infer rank, alpha, and target_modules from raw PEFT weight keys and shapes.
    ///
    /// PEFT stores `lora_A` with shape `[rank, in_features]`, so `shape[0]` is rank.
    /// Alpha defaults to rank (the PEFT library default when not specified).
    /// Target modules are extracted by scanning keys for known projection names.
    ///
    /// Returns `nil` if no `lora_A` weights are found (not a LoRA checkpoint).
    static func inferConfig(from dict: [String: MLXArray]) -> InferredConfig? {
        // Find any lora_A tensor — shape is [rank, in_features]
        guard let (_, loraA) = dict.first(where: { $0.key.contains("lora_A") }),
              loraA.shape.count >= 2 else { return nil }

        // rank is always the smaller dimension (rank << in_features in practice)
        let rank = min(loraA.shape[0], loraA.shape[1])

        let targetModules = extractTargetModules(from: Array(dict.keys))
        return InferredConfig(rank: rank, alpha: Float(rank), targetModules: targetModules)
    }

    // MARK: - LyCORIS / PEFT LoKR Remapping

    struct LoKRRemapResult {
        let weights: [String: MLXArray]
        let inferredLinearDim: Int
        let sourceAlpha: Float?
        let exactInjectionPatterns: [String]
        let targetLayers: Set<TargetLayer>
        let trainedPaths: [String]
        let trainedLayerIndices: [Int]
        let weightDecompose: Bool
    }

    enum LoKRRemapError: Error, LocalizedError {
        case noConvertibleWeights
        case unsupportedVariant(String)

        var errorDescription: String? {
            switch self {
            case .noConvertibleWeights:
                return "No compatible LoKR weights were found."
            case .unsupportedVariant(let detail):
                return detail
            }
        }
    }

    /// Returns `true` when a safetensors payload looks like LyCORIS/PEFT LoKR.
    static func isLyCORISLoKR(_ keys: [String]) -> Bool {
        keys.contains { key in
            splitLoKRSuffix(key) != nil ||
            key.contains(".lokr_w1") ||
            key.contains(".lokr_w2") ||
            key.contains(".dora_scale")
        }
    }

    /// Convert compatible LyCORIS/PEFT LoKR tensors to PiMP-native LoKR keys.
    ///
    /// Supported source layouts:
    /// - `prefix.lokr_w1` + `prefix.lokr_w2`
    /// - `prefix.lokr_w1_a`/`prefix.lokr_w1_b` and
    ///   `prefix.lokr_w2_a`/`prefix.lokr_w2_b` (multiplied during import)
    /// - optional `prefix.dora_scale` mapped to `.lokrMagnitude`
    ///
    /// Tucker-style `lokr_t2` tensors are rejected because the native runtime
    /// cannot represent them without changing the adapter algebra.
    static func remapLyCORISLoKR(_ dict: [String: MLXArray]) throws -> LoKRRemapResult {
        for key in dict.keys where splitLoKRSuffix(key)?.suffix.hasPrefix("lokr_t2") == true {
            throw LoKRRemapError.unsupportedVariant(
                "This LoKR checkpoint uses Tucker tensors (`lokr_t2`), which PiMP Audio Studio cannot import yet."
            )
        }

        let prefixes = Set(dict.keys.compactMap { splitLoKRSuffix($0)?.prefix })
        var remapped: [String: MLXArray] = [:]
        var sourceAlpha: Float?
        var hasDoraMagnitude = false

        for prefix in prefixes.sorted() {
            guard let nativePath = mapLyCORISLoKRPrefix(prefix),
                  let w1 = loKRTensorOrProduct(
                    dict,
                    prefix: prefix,
                    direct: "lokr_w1",
                    left: "lokr_w1_a",
                    right: "lokr_w1_b"
                  ),
                  let w2 = loKRTensorOrProduct(
                    dict,
                    prefix: prefix,
                    direct: "lokr_w2",
                    left: "lokr_w2_a",
                    right: "lokr_w2_b"
                  ) else {
                continue
            }

            remapped["\(nativePath).lokrW1"] = w1
            remapped["\(nativePath).lokrW2"] = w2

            if let doraScale = dict["\(prefix).dora_scale"] {
                remapped["\(nativePath).lokrMagnitude"] = doraScale.asType(.float32).reshaped([-1])
                hasDoraMagnitude = true
            }

            if sourceAlpha == nil, let alpha = dict["\(prefix).alpha"] {
                sourceAlpha = alpha.asType(.float32).item(Float.self)
            }
        }

        guard !remapped.isEmpty else {
            throw LoKRRemapError.noConvertibleWeights
        }

        let exactPatterns = inferLoKRInjectionPatterns(fromRemappedKeys: Array(remapped.keys))
        let targetLayers = inferTargetLayers(fromInjectionPatterns: exactPatterns)
        let trainedPaths = LoRAAdapter.trainedPaths(fromWeightKeys: Array(remapped.keys), kind: .lokr)

        return LoKRRemapResult(
            weights: remapped,
            inferredLinearDim: inferLoKRLinearDim(fromRemappedWeights: remapped),
            sourceAlpha: sourceAlpha,
            exactInjectionPatterns: exactPatterns,
            targetLayers: targetLayers,
            trainedPaths: trainedPaths,
            trainedLayerIndices: LoRAAdapter.layerIndices(from: trainedPaths),
            weightDecompose: hasDoraMagnitude
        )
    }

    /// Infer exact PiMP Audio Studio injection subpaths from remapped LoKR keys.
    static func inferLoKRInjectionPatterns(fromRemappedKeys keys: [String]) -> [String] {
        var patterns = Set<String>()

        for key in keys {
            let parts = key.split(separator: ".")
            guard parts.count >= 5,
                  parts[0] == "layers",
                  Int(parts[1]) != nil,
                  parts.last == "lokrW1" || parts.last == "lokrW2" || parts.last == "lokrMagnitude" else {
                continue
            }

            let module = String(parts[2])
            let projection = String(parts[3])
            patterns.insert("\(module).\(projection)")
        }

        return patterns.sorted { lhs, rhs in
            patternSortKey(lhs) < patternSortKey(rhs)
        }
    }

    /// Scan raw PEFT key strings for known projection names (e.g. `q_proj`, `v_proj`).
    private static func extractTargetModules(from keys: [String]) -> [String] {
        let known = ["q_proj", "k_proj", "v_proj", "o_proj", "out_proj",
                     "gate_proj", "up_proj", "down_proj"]
        var found = Set<String>()
        for key in keys {
            for name in known where key.contains(".\(name).") {
                found.insert(name)
            }
        }
        return found.sorted()
    }

    // MARK: - LoRAConfig Inference

    /// Infer the closest `Set<TargetLayer>` from PEFT `target_modules`.
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

    /// Infer target categories from exact PiMP Audio Studio module paths.
    static func inferTargetLayers(fromInjectionPatterns patterns: [String]) -> Set<TargetLayer> {
        var layers = Set<TargetLayer>()

        for pattern in patterns {
            if pattern.hasPrefix("selfAttn.") {
                layers.insert(.selfAttention)
            } else if pattern.hasPrefix("crossAttn.") {
                layers.insert(.crossAttention)
            } else if pattern.hasPrefix("mlp.") {
                layers.insert(.mlp)
            }
        }

        return layers.isEmpty ? [.selfAttention] : layers
    }

    /// Map a PEFT HuggingFace repo ID / model path to the PiMPAudioStudio local model ID.
    ///
    /// ACE-Step v1.5 variants (turbo, base, SFT, XL) share the same architecture family.
    /// We detect the variant from keywords in the HF path so the adapter list can show
    /// "Turbo", "SFT", "XL", or "Base" rather than collapsing everything to turbo.
    static func resolveBaseModelID(from hfPath: String?) -> String {
        guard let path = hfPath else { return "ace-step-dit-turbo" }
        let lower = path.lowercased()
        let looksLikeACEVariant =
            lower.contains("ace-step") ||
            lower.contains("ace_step") ||
            lower.contains("acestep") ||
            lower.hasPrefix("xl-") ||
            lower.hasPrefix("v15-xl-") ||
            lower == "xl" ||
            lower == "sft" ||
            lower == "base" ||
            lower == "turbo"
        guard looksLikeACEVariant else {
            // Unknown model — preserve the HF path so the UI can show a mismatch warning
            return path
        }
        if lower.contains("xl") {
            if lower.contains("turbo") { return "ace-step-dit-xl-turbo" }
            if lower.contains("sft") { return "ace-step-dit-xl-sft" }
            if lower.contains("base") || lower.contains("full") { return "ace-step-dit-xl-base" }
            return "ace-step-dit-xl"
        }
        if lower.contains("turbo") { return "ace-step-dit-turbo" }
        if lower.contains("sft")   { return "ace-step-dit-sft" }
        return "ace-step-dit-base"
    }

    // MARK: - Private Helpers

    private static func splitLoKRSuffix(_ key: String) -> (prefix: String, suffix: String)? {
        let suffixes = [
            "lokr_w1", "lokr_w2",
            "lokr_w1_a", "lokr_w1_b",
            "lokr_w2_a", "lokr_w2_b",
            "lokr_t2", "lokr_t2_a", "lokr_t2_b",
            "dora_scale", "alpha",
        ]

        for suffix in suffixes {
            let tail = ".\(suffix)"
            if key.hasSuffix(tail) {
                return (String(key.dropLast(tail.count)), suffix)
            }
        }

        return nil
    }

    private static func loKRTensorOrProduct(
        _ dict: [String: MLXArray],
        prefix: String,
        direct: String,
        left: String,
        right: String
    ) -> MLXArray? {
        if let directTensor = dict["\(prefix).\(direct)"] {
            return directTensor.asType(.float32)
        }

        guard let leftTensor = dict["\(prefix).\(left)"],
              let rightTensor = dict["\(prefix).\(right)"] else {
            return nil
        }

        return matmul(leftTensor.asType(.float32), rightTensor.asType(.float32))
    }

    private static func mapLyCORISLoKRPrefix(_ prefix: String) -> String? {
        var key = prefix

        while key.hasPrefix("base_model.model.") {
            key = String(key.dropFirst("base_model.model.".count))
        }

        for wrapper in ["transformer.", "dit.", "model."] {
            if key.hasPrefix(wrapper) {
                key = String(key.dropFirst(wrapper.count))
                break
            }
        }

        if key.hasPrefix("lycoris_") {
            key = String(key.dropFirst("lycoris_".count))
        }

        if let underscorePath = mapLyCORISUnderscorePrefix(key) {
            return underscorePath
        }

        let components = key.components(separatedBy: ".")
        guard components.count >= 3,
              components[0] == "layers",
              Int(components[1]) != nil else {
            return nil
        }

        let layerPrefix = "layers.\(components[1])."
        let rest = components.dropFirst(2).joined(separator: ".")
        guard let mappedPath = mapModulePath(rest) else { return nil }
        return layerPrefix + mappedPath
    }

    private static func mapLyCORISUnderscorePrefix(_ key: String) -> String? {
        guard key.hasPrefix("layers_") else { return nil }
        let afterLayers = String(key.dropFirst("layers_".count))
        guard let separator = afterLayers.firstIndex(of: "_") else { return nil }

        let layerText = String(afterLayers[..<separator])
        guard Int(layerText) != nil else { return nil }

        let restStart = afterLayers.index(after: separator)
        let rest = String(afterLayers[restStart...])
        let layerPrefix = "layers.\(layerText)."

        if rest.hasPrefix("self_attn_") {
            let projection = String(rest.dropFirst("self_attn_".count))
            return mapAttnProj(projection).map { layerPrefix + "selfAttn.\($0)" }
        }

        if rest.hasPrefix("cross_attn_") {
            let projection = String(rest.dropFirst("cross_attn_".count))
            return mapAttnProj(projection).map { layerPrefix + "crossAttn.\($0)" }
        }

        if rest.hasPrefix("mlp_") {
            let projection = String(rest.dropFirst("mlp_".count))
            return mapMLPProj(projection).map { layerPrefix + "mlp.\($0)" }
        }

        return nil
    }

    private static func inferLoKRLinearDim(fromRemappedWeights weights: [String: MLXArray]) -> Int {
        let inferred = weights
            .filter { $0.key.hasSuffix(".lokrW1") || $0.key.hasSuffix(".lokrW2") }
            .flatMap { $0.value.shape }
            .max() ?? 64
        return max(1, inferred)
    }

    /// Map a PEFT module subpath (e.g. `"self_attn.q_proj"`) to PiMPAudioStudio format (`"selfAttn.qProj"`).
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

    private static func patternSortKey(_ pattern: String) -> String {
        let moduleRank: String
        if pattern.hasPrefix("selfAttn.") {
            moduleRank = "0"
        } else if pattern.hasPrefix("crossAttn.") {
            moduleRank = "1"
        } else if pattern.hasPrefix("mlp.") {
            moduleRank = "2"
        } else {
            moduleRank = "9"
        }

        let projectionRank: String
        if pattern.hasSuffix(".qProj") {
            projectionRank = "0"
        } else if pattern.hasSuffix(".kProj") {
            projectionRank = "1"
        } else if pattern.hasSuffix(".vProj") {
            projectionRank = "2"
        } else if pattern.hasSuffix(".oProj") {
            projectionRank = "3"
        } else if pattern.hasSuffix(".gateProj") {
            projectionRank = "4"
        } else if pattern.hasSuffix(".upProj") {
            projectionRank = "5"
        } else if pattern.hasSuffix(".downProj") {
            projectionRank = "6"
        } else {
            projectionRank = "9"
        }

        return "\(moduleRank).\(projectionRank).\(pattern)"
    }
}
