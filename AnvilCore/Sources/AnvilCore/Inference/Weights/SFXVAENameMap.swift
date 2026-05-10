// AnvilCore/Sources/AnvilCore/Inference/Weights/SFXVAENameMap.swift
//
// Maps SAO vae.safetensors keys (MLX Sequential paths) → OobleckDecoder property paths.
//
// The SAO mlx checkpoint uses a flat `layers.layers.X` numbering scheme (Python
// nn.Sequential serialisation).  Our Swift `OobleckDecoder` uses named properties.
//
// Decoder top-level layout (decoder.layers.layers.X):
//   0   →  conv1          (WNConv1d  64→2048, k=7)
//   1-5 →  blocks.0-4     (OobleckDecoderBlock)
//   6   →  snake1         (Snake1d)
//   7   →  conv2          (WNConv1d  128→2, k=7)
//
// OobleckDecoderBlock sub-layout (*.layers.layers.Y):
//   0   →  snake1         (Snake1d)
//   1   →  convT1         (ConvTransposed1d)
//   2-4 →  resUnit1-3     (OobleckResidualUnit)
//
// OobleckResidualUnit sub-layout (*.layers.layers.Z):
//   0   →  snake1         (Snake1d)
//   1   →  conv1          (Conv1d, k=7, dilated)
//   2   →  snake2         (Snake1d)
//   3   →  conv2          (Conv1d, k=1)
//
// Conv weights in the SAO safetensors are already in MLX NLC format [out, k, in]
// (produced by mlx-audiogen's export) — no axis transposition is applied.
//
// Only `decoder.*` keys are processed; `encoder.*` keys return nil (unused).

import MLX

public enum SFXVAENameMap {

    // MARK: - Top-level entry point

    /// Returns the Swift module path for a checkpoint key, or `nil` to skip it.
    public static func mapKey(_ key: String) -> String? {
        guard key.hasPrefix("decoder.") else { return nil }
        let rest = String(key.dropFirst("decoder.".count))
        guard let mapped = mapDecoder(rest) else { return nil }
        return "decoder." + mapped
    }

    // MARK: - Level 0: OobleckDecoder root

    private static func mapDecoder(_ path: String) -> String? {
        // layers.layers.N. → named root property
        let topMap: [(String, String)] = [
            ("layers.layers.0.", "conv1."),
            ("layers.layers.1.", "blocks.0."),
            ("layers.layers.2.", "blocks.1."),
            ("layers.layers.3.", "blocks.2."),
            ("layers.layers.4.", "blocks.3."),
            ("layers.layers.5.", "blocks.4."),
            ("layers.layers.6.", "snake1."),
            ("layers.layers.7.", "conv2."),
        ]
        for (prefix, replacement) in topMap {
            if path.hasPrefix(prefix) {
                let rest = String(path.dropFirst(prefix.count))
                // If this is a block path, map its sub-layers too.
                if replacement.hasPrefix("blocks.") {
                    let mapped = mapDecoderBlock(rest) ?? rest
                    return replacement + mapped
                }
                return replacement + rest
            }
        }
        return nil
    }

    // MARK: - Level 1: OobleckDecoderBlock

    private static func mapDecoderBlock(_ path: String) -> String? {
        let blockMap: [(String, String)] = [
            ("layers.layers.0.", "snake1."),
            ("layers.layers.1.", "convT1."),
            ("layers.layers.2.", "resUnit1."),
            ("layers.layers.3.", "resUnit2."),
            ("layers.layers.4.", "resUnit3."),
        ]
        for (prefix, replacement) in blockMap {
            if path.hasPrefix(prefix) {
                let rest = String(path.dropFirst(prefix.count))
                // If this is a ResUnit path, map its sub-layers too.
                if replacement.hasPrefix("resUnit") {
                    let mapped = mapResidualUnit(rest) ?? rest
                    return replacement + mapped
                }
                return replacement + rest
            }
        }
        return nil
    }

    // MARK: - Level 2: OobleckResidualUnit

    private static func mapResidualUnit(_ path: String) -> String? {
        let unitMap: [(String, String)] = [
            ("layers.layers.0.", "snake1."),
            ("layers.layers.1.", "conv1."),
            ("layers.layers.2.", "snake2."),
            ("layers.layers.3.", "conv2."),
        ]
        for (prefix, replacement) in unitMap {
            if path.hasPrefix(prefix) {
                let rest = String(path.dropFirst(prefix.count))
                return replacement + rest
            }
        }
        return nil
    }
}
