import Foundation
import MLX
import MLXNN

/// Static utility for injecting and removing LoRA adapters from DiTDecoder layers.
public enum LoRAInjector {

    // MARK: - Public API

    /// Inject LoRA adapters into the model's target layers, replacing Linear with LoRALinear.
    ///
    /// - Parameters:
    ///   - model: The DiTDecoder model whose layers will be modified.
    ///   - config: LoRA configuration specifying rank, alpha, dropout, and target patterns.
    /// - Returns: List of fully-qualified key paths that were injected (e.g. "layers.0.selfAttn.qProj").
    @discardableResult
    public static func inject(
        model: DiTDecoder,
        config: LoRAConfig,
        layerIndices: Set<Int>? = nil
    ) -> [String] {
        var injectedPaths: [String] = []

        for (layerIdx, layer) in model.layers.enumerated() {
            if let layerIndices, !layerIndices.contains(layerIdx) {
                continue
            }

            for pattern in config.injectionPatterns {
                let keyPath = "layers.\(layerIdx).\(pattern)"

                if let linear = resolveLinear(in: layer, path: pattern) {
                    // Skip if already wrapped
                    if linear is LoRALinear { continue }

                    let loraLinear = LoRALinear(
                        base: linear,
                        rank: config.rank,
                        alpha: config.alpha,
                        dropout: config.dropout
                    )
                    loraLinear.freezeBase()
                    setLinear(on: layer, path: pattern, value: loraLinear)
                    injectedPaths.append(keyPath)
                }
            }
        }

        return injectedPaths
    }

    /// Remove LoRA adapters, restoring original base Linear layers.
    ///
    /// - Parameters:
    ///   - model: The DiTDecoder model to restore.
    ///   - paths: Key paths previously returned by `inject`.
    public static func remove(model: DiTDecoder, paths: [String]) {
        for path in paths {
            guard let (layerIdx, subPath) = parsePath(path),
                  layerIdx < model.layers.count else { continue }

            let layer = model.layers[layerIdx]
            if let loraLinear = resolveLinear(in: layer, path: subPath) as? LoRALinear {
                let plain = Linear(weight: loraLinear.baseWeight, bias: loraLinear.baseBias)
                setLinear(on: layer, path: subPath, value: plain)
            }
        }
    }

    /// Extract trainable LoRA weights (loraA, loraB) from injected layers for persistence.
    ///
    /// - Parameters:
    ///   - model: The DiTDecoder with injected LoRA adapters.
    ///   - paths: Key paths previously returned by `inject`.
    /// - Returns: Array of (key, weight) tuples suitable for serialization.
    public static func extractWeights(model: DiTDecoder, paths: [String]) -> [(String, MLXArray)] {
        var weights: [(String, MLXArray)] = []

        for path in paths {
            guard let (layerIdx, subPath) = parsePath(path),
                  layerIdx < model.layers.count else { continue }

            let layer = model.layers[layerIdx]
            if let loraLinear = resolveLinear(in: layer, path: subPath) as? LoRALinear {
                weights.append(("\(path).loraA", loraLinear.loraA))
                weights.append(("\(path).loraB", loraLinear.loraB))
            }
        }

        return weights
    }

    /// Load previously saved LoRA weights into existing LoRALinear layers.
    ///
    /// - Parameters:
    ///   - model: The DiTDecoder with injected LoRA adapters.
    ///   - paths: Key paths where LoRALinear layers exist.
    ///   - weights: Array of (key, weight) tuples (as returned by `extractWeights`).
    public static func loadWeights(
        model: DiTDecoder, paths: [String], weights: [(String, MLXArray)]
    ) {
        let weightDict = Dictionary(weights, uniquingKeysWith: { _, last in last })

        for path in paths {
            guard let (layerIdx, subPath) = parsePath(path),
                  layerIdx < model.layers.count else { continue }

            let layer = model.layers[layerIdx]
            if let loraLinear = resolveLinear(in: layer, path: subPath) as? LoRALinear {
                if let a = weightDict["\(path).loraA"] {
                    loraLinear.loraA._updateInternal(a)
                }
                if let b = weightDict["\(path).loraB"] {
                    loraLinear.loraB._updateInternal(b)
                }
            }
        }
    }

    /// Mark only the adapter matrices as trainable after the parent DiT has
    /// been frozen for LoRA training.
    public static func unfreezeAdapters(model: DiTDecoder, paths: [String]) {
        for path in paths {
            guard let (layerIdx, subPath) = parsePath(path),
                  layerIdx < model.layers.count else { continue }

            let layer = model.layers[layerIdx]
            if let loraLinear = resolveLinear(in: layer, path: subPath) as? LoRALinear {
                loraLinear.unfreeze(recursive: false, keys: ["loraA", "loraB"])
            }
        }
    }

    // MARK: - Private Helpers

    /// Parse "layers.N.selfAttn.qProj" into (layerIndex, "selfAttn.qProj").
    private static func parsePath(_ path: String) -> (Int, String)? {
        let components = path.split(separator: ".")
        guard components.count >= 3,
              components[0] == "layers",
              let layerIdx = Int(components[1]) else { return nil }
        let subPath = components[2...].joined(separator: ".")
        return (layerIdx, subPath)
    }

    /// Resolve a Linear layer at a known sub-path within a DiTLayer.
    private static func resolveLinear(in layer: DiTLayer, path: String) -> Linear? {
        let parts = path.split(separator: ".")
        guard parts.count == 2 else { return nil }

        let moduleName = String(parts[0])
        let projName = String(parts[1])

        switch moduleName {
        case "selfAttn":
            return resolveAttentionProj(layer.selfAttn, proj: projName)
        case "crossAttn":
            return resolveAttentionProj(layer.crossAttn, proj: projName)
        case "mlp":
            return resolveMLPProj(layer.mlp, proj: projName)
        default:
            return nil
        }
    }

    private static func resolveAttentionProj(_ attn: DiTAttention, proj: String) -> Linear? {
        switch proj {
        case "qProj": return attn.qProj
        case "kProj": return attn.kProj
        case "vProj": return attn.vProj
        case "oProj": return attn.oProj
        default: return nil
        }
    }

    private static func resolveMLPProj(_ mlp: SwiGLUMLP, proj: String) -> Linear? {
        switch proj {
        case "gateProj": return mlp.gateProj
        case "upProj": return mlp.upProj
        case "downProj": return mlp.downProj
        default: return nil
        }
    }

    /// Set a Linear value at a known sub-path within a DiTLayer.
    private static func setLinear(on layer: DiTLayer, path: String, value: Linear) {
        let parts = path.split(separator: ".")
        guard parts.count == 2 else { return }

        let moduleName = String(parts[0])
        let projName = String(parts[1])

        switch moduleName {
        case "selfAttn":
            setAttentionProj(layer.selfAttn, proj: projName, value: value)
        case "crossAttn":
            setAttentionProj(layer.crossAttn, proj: projName, value: value)
        case "mlp":
            setMLPProj(layer.mlp, proj: projName, value: value)
        default:
            break
        }
    }

    private static func setAttentionProj(_ attn: DiTAttention, proj: String, value: Linear) {
        switch proj {
        case "qProj", "kProj", "vProj", "oProj":
            attn.update(modules: ModuleChildren.unflattened([(proj, value)]))
        default: break
        }
    }

    private static func setMLPProj(_ mlp: SwiGLUMLP, proj: String, value: Linear) {
        switch proj {
        case "gateProj", "upProj", "downProj":
            mlp.update(modules: ModuleChildren.unflattened([(proj, value)]))
        default: break
        }
    }
}

// MARK: - Multi-Adapter Stacking

extension LoRAInjector {

    /// Apply a full adapter stack to the model. Replaces all current contributions
    /// on every LoRALinear layer matching the given paths.
    public static func applyStack(
        model: DiTDecoder,
        contributions: [LoRAContribution],
        paths: [String]
    ) {
        clearContributions(model: model, paths: paths)

        for path in paths {
            guard let (layerIdx, subPath) = parsePath(path),
                  layerIdx < model.layers.count else { continue }

            let layer = model.layers[layerIdx]
            guard let loraLinear = resolveLinear(in: layer, path: subPath) as? LoRALinear else {
                continue
            }

            var layerContributions: [LoRAContribution] = []
            for contribution in contributions {
                layerContributions.append(contribution)
            }
            loraLinear.contributions = layerContributions
        }
    }

    /// Apply contributions for a single adapter across all LoRALinear layers.
    /// Weights are keyed by path (e.g. "layers.0.selfAttn.qProj.loraA").
    public static func addContribution(
        model: DiTDecoder,
        adapterID: String,
        displayName: String?,
        source: AdapterSource,
        weights: [(String, MLXArray)],
        strength: Float,
        paths: [String]
    ) {
        let weightDict = Dictionary(weights, uniquingKeysWith: { _, last in last })

        for path in paths {
            guard let (layerIdx, subPath) = parsePath(path),
                  layerIdx < model.layers.count else { continue }

            let layer = model.layers[layerIdx]
            guard let loraLinear = resolveLinear(in: layer, path: subPath) as? LoRALinear else {
                continue
            }

            guard let loraA = weightDict["\(path).loraA"],
                  let loraB = weightDict["\(path).loraB"] else {
                continue
            }

            let contribution = LoRAContribution(
                adapterID: adapterID,
                displayName: displayName,
                source: source,
                loraA: loraA,
                loraB: loraB,
                strength: strength
            )
            loraLinear.contributions.append(contribution)
        }
    }

    /// Remove all contributions from every LoRALinear layer (revert to base model behavior).
    public static func clearContributions(model: DiTDecoder, paths: [String]) {
        for path in paths {
            guard let (layerIdx, subPath) = parsePath(path),
                  layerIdx < model.layers.count else { continue }

            let layer = model.layers[layerIdx]
            if let loraLinear = resolveLinear(in: layer, path: subPath) as? LoRALinear {
                loraLinear.contributions = []
            }
        }
    }

    /// Update strength for a single adapter across all LoRALinear layers.
    public static func setStrength(
        model: DiTDecoder,
        adapterID: String,
        strength: Float,
        paths: [String]
    ) {
        for path in paths {
            guard let (layerIdx, subPath) = parsePath(path),
                  layerIdx < model.layers.count else { continue }

            let layer = model.layers[layerIdx]
            if let loraLinear = resolveLinear(in: layer, path: subPath) as? LoRALinear {
                for i in loraLinear.contributions.indices {
                    if loraLinear.contributions[i].adapterID == adapterID {
                        loraLinear.contributions[i].strength = strength
                    }
                }
            }
        }
    }

    /// Remove a specific adapter's contributions from all LoRALinear layers.
    public static func removeContribution(
        model: DiTDecoder,
        adapterID: String,
        paths: [String]
    ) {
        for path in paths {
            guard let (layerIdx, subPath) = parsePath(path),
                  layerIdx < model.layers.count else { continue }

            let layer = model.layers[layerIdx]
            if let loraLinear = resolveLinear(in: layer, path: subPath) as? LoRALinear {
                loraLinear.contributions.removeAll { $0.adapterID == adapterID }
            }
        }
    }
}
