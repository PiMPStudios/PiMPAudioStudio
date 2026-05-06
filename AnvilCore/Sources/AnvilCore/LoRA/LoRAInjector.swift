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
    public static func inject(model: DiTDecoder, config: LoRAConfig) -> [String] {
        var injectedPaths: [String] = []

        for (layerIdx, layer) in model.layers.enumerated() {
            for pattern in config.targetLayers {
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
                    loraLinear.loraA = a
                }
                if let b = weightDict["\(path).loraB"] {
                    loraLinear.loraB = b
                }
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
        case "qProj": attn.qProj = value
        case "kProj": attn.kProj = value
        case "vProj": attn.vProj = value
        case "oProj": attn.oProj = value
        default: break
        }
    }

    private static func setMLPProj(_ mlp: SwiGLUMLP, proj: String, value: Linear) {
        switch proj {
        case "gateProj": mlp.gateProj = value
        case "upProj": mlp.upProj = value
        case "downProj": mlp.downProj = value
        default: break
        }
    }
}
