import Foundation
import MLX
import MLXNN

/// Static utility for injecting and removing experimental LoKR adapters.
public enum LoKRInjector {
    @discardableResult
    public static func inject(
        model: DiTDecoder,
        config: LoKRConfig,
        layerIndices: Set<Int>? = nil
    ) -> [String] {
        var injectedPaths: [String] = []
        var existingPaths: [String] = []

        for (layerIdx, layer) in model.layers.enumerated() {
            if let layerIndices, !layerIndices.contains(layerIdx) {
                continue
            }

            for pattern in config.injectionPatterns {
                let keyPath = "layers.\(layerIdx).\(pattern)"
                if let linear = resolveLinear(in: layer, path: pattern) {
                    if linear is LoKRLinear {
                        existingPaths.append(keyPath)
                        continue
                    }

                    let lokrLinear = LoKRLinear(
                        base: linear,
                        linearDim: config.linearDim,
                        alpha: config.alpha,
                        dropout: config.dropout,
                        weightDecompose: config.weightDecompose
                    )
                    lokrLinear.freezeBase()
                    setLinear(on: layer, path: pattern, value: lokrLinear)
                    injectedPaths.append(keyPath)
                }
            }
        }

        return existingPaths + injectedPaths
    }

    public static func remove(model: DiTDecoder, paths: [String]) {
        for path in paths {
            guard let (layerIdx, subPath) = parsePath(path),
                  layerIdx < model.layers.count else { continue }

            let layer = model.layers[layerIdx]
            if let lokrLinear = resolveLinear(in: layer, path: subPath) as? LoKRLinear {
                let plain = Linear(weight: lokrLinear.baseWeight, bias: lokrLinear.baseBias)
                setLinear(on: layer, path: subPath, value: plain)
            }
        }
    }

    public static func extractWeights(model: DiTDecoder, paths: [String]) -> [(String, MLXArray)] {
        var weights: [(String, MLXArray)] = []
        for path in paths {
            guard let (layerIdx, subPath) = parsePath(path),
                  layerIdx < model.layers.count else { continue }

            let layer = model.layers[layerIdx]
            if let lokrLinear = resolveLinear(in: layer, path: subPath) as? LoKRLinear {
                weights.append(("\(path).lokrW1", lokrLinear.lokrW1))
                weights.append(("\(path).lokrW2", lokrLinear.lokrW2))
                if lokrLinear.weightDecompose {
                    weights.append(("\(path).lokrMagnitude", lokrLinear.lokrMagnitude))
                }
            }
        }
        return weights
    }

    public static func loadWeights(
        model: DiTDecoder,
        paths: [String],
        weights: [(String, MLXArray)],
        adapterID: String? = nil
    ) {
        let weightDict = Dictionary(weights, uniquingKeysWith: { _, last in last })
        var loadedW1 = 0
        var loadedW2 = 0
        var loadedMagnitude = 0
        var missingMagnitude = 0
        var missingPaths: [String] = []
        var loggedFirstTensors = false

        for path in paths {
            guard let (layerIdx, subPath) = parsePath(path),
                  layerIdx < model.layers.count else { continue }

            let layer = model.layers[layerIdx]
            if let lokrLinear = resolveLinear(in: layer, path: subPath) as? LoKRLinear {
                if let w1 = weightDict["\(path).lokrW1"] {
                    lokrLinear.lokrW1._updateInternal(w1)
                    loadedW1 += 1
                }
                if let w2 = weightDict["\(path).lokrW2"] {
                    lokrLinear.lokrW2._updateInternal(w2)
                    loadedW2 += 1
                }
                if lokrLinear.weightDecompose {
                    if let magnitude = weightDict["\(path).lokrMagnitude"] {
                        lokrLinear.lokrMagnitude._updateInternal(magnitude)
                        loadedMagnitude += 1
                    } else {
                        missingMagnitude += 1
                    }
                }
                if weightDict["\(path).lokrW1"] == nil || weightDict["\(path).lokrW2"] == nil {
                    missingPaths.append(path)
                } else if !loggedFirstTensors,
                          let w1 = weightDict["\(path).lokrW1"],
                          let w2 = weightDict["\(path).lokrW2"] {
                    let adapterTag = diagnosticTagComponent(adapterID ?? "single")
                    let pathTag = diagnosticTagComponent(path)
                    AudioDiagnostics.tensor("lokr/loadWeights/\(adapterTag)/\(pathTag)/W1", w1)
                    AudioDiagnostics.tensor("lokr/loadWeights/\(adapterTag)/\(pathTag)/W2", w2)
                    if let magnitude = weightDict["\(path).lokrMagnitude"] {
                        AudioDiagnostics.tensor("lokr/loadWeights/\(adapterTag)/\(pathTag)/Magnitude", magnitude)
                    }
                    loggedFirstTensors = true
                }
            }
        }
        AudioDiagnostics.note(
            "lokr/loadWeights",
            "adapter=\(adapterID ?? "nil") paths=\(paths.count) tensors=\(weights.count) loadedW1=\(loadedW1) loadedW2=\(loadedW2) loadedMagnitude=\(loadedMagnitude) missingMagnitude=\(missingMagnitude) missing=\(missingPaths.count) firstMissing=\(Array(missingPaths.prefix(8)))"
        )
    }

    public static func unfreezeAdapters(model: DiTDecoder, paths: [String]) {
        for path in paths {
            guard let (layerIdx, subPath) = parsePath(path),
                  layerIdx < model.layers.count else { continue }

            let layer = model.layers[layerIdx]
            if let lokrLinear = resolveLinear(in: layer, path: subPath) as? LoKRLinear {
                var keys = ["lokrW1", "lokrW2"]
                if lokrLinear.weightDecompose {
                    keys.append("lokrMagnitude")
                }
                lokrLinear.unfreeze(recursive: false, keys: keys)
            }
        }
    }

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
        let requestedPaths = Set(paths)
        let availableWeightPaths = Set(weights.compactMap { key, _ -> String? in
            if key.hasSuffix(".lokrW1") {
                return String(key.dropLast(".lokrW1".count))
            }
            if key.hasSuffix(".lokrW2") {
                return String(key.dropLast(".lokrW2".count))
            }
            if key.hasSuffix(".lokrMagnitude") {
                return String(key.dropLast(".lokrMagnitude".count))
            }
            return nil
        })
        let unusedWeightPaths = availableWeightPaths.subtracting(requestedPaths)
        var matchedPaths = 0
        var missingPaths: [String] = []
        var loggedFirstTensors = false

        for path in paths {
            guard let (layerIdx, subPath) = parsePath(path),
                  layerIdx < model.layers.count else { continue }

            let layer = model.layers[layerIdx]
            guard let lokrLinear = resolveLinear(in: layer, path: subPath) as? LoKRLinear else {
                continue
            }

            guard let w1 = weightDict["\(path).lokrW1"],
                  let w2 = weightDict["\(path).lokrW2"] else {
                missingPaths.append(path)
                continue
            }
            let magnitude = weightDict["\(path).lokrMagnitude"]

            lokrLinear.contributions.append(LoKRContribution(
                adapterID: adapterID,
                displayName: displayName,
                source: source,
                lokrW1: w1,
                lokrW2: w2,
                lokrMagnitude: magnitude,
                weightDecompose: magnitude != nil,
                strength: strength,
                path: path
            ))
            matchedPaths += 1
            if !loggedFirstTensors {
                let adapterTag = diagnosticTagComponent(adapterID)
                let pathTag = diagnosticTagComponent(path)
                AudioDiagnostics.tensor("lokr/contribution/\(adapterTag)/\(pathTag)/W1", w1)
                AudioDiagnostics.tensor("lokr/contribution/\(adapterTag)/\(pathTag)/W2", w2)
                if let magnitude {
                    AudioDiagnostics.tensor("lokr/contribution/\(adapterTag)/\(pathTag)/Magnitude", magnitude)
                }
                loggedFirstTensors = true
            }
        }

        AudioDiagnostics.note(
            "lokr/contribution",
            "adapter=\(adapterID) strength=\(String(format: "%.3f", strength)) paths=\(paths.count) matched=\(matchedPaths) missing=\(missingPaths.count) unusedWeightPaths=\(unusedWeightPaths.count) firstMissing=\(Array(missingPaths.prefix(8))) firstUnused=\(Array(unusedWeightPaths.sorted().prefix(8)))"
        )
    }

    public static func clearContributions(model: DiTDecoder, paths: [String]) {
        for path in paths {
            guard let (layerIdx, subPath) = parsePath(path),
                  layerIdx < model.layers.count else { continue }

            let layer = model.layers[layerIdx]
            if let lokrLinear = resolveLinear(in: layer, path: subPath) as? LoKRLinear {
                lokrLinear.contributions = []
            }
        }
    }

    private static func parsePath(_ path: String) -> (Int, String)? {
        let components = path.split(separator: ".")
        guard components.count >= 3,
              components[0] == "layers",
              let layerIdx = Int(components[1]) else { return nil }
        let subPath = components[2...].joined(separator: ".")
        return (layerIdx, subPath)
    }

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

    private static func diagnosticTagComponent(_ value: String) -> String {
        value
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: " ", with: "_")
            .replacingOccurrences(of: ":", with: "_")
    }
}
