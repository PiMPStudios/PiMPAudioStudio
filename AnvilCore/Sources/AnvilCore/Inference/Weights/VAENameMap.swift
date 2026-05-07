import MLX

public enum VAENameMap {
    /// Maps a key from the downloaded ACE-Step VAE safetensors to the
    /// parameter path expected by our MLX-Swift `AutoEncoderOobleck`.
    ///
    /// The diffusers-format checkpoint uses Python's snake_case property
    /// names plus `block` (singular) for the `nn.ModuleList`:
    ///
    ///     decoder.block.0.conv_t1.weight
    ///     decoder.block.0.res_unit1.snake1.alpha
    ///
    /// Our Swift modules use camelCase property names plus `blocks` (plural):
    ///
    ///     decoder.blocks.0.convT1.weight
    ///     decoder.blocks.0.resUnit1.snake1.alpha
    ///
    /// MLX-Swift's `module.update(parameters:)` silently ignores keys that
    /// don't resolve to a real parameter path, so without this conversion
    /// only the handful of identically-named keys (conv1, conv2, snake1,
    /// snake2, alpha, beta, weight, bias) load — the rest stay at default
    /// init and the decoder produces near-silent output.
    public static func mapKey(_ key: String) -> String {
        let components = key.split(separator: ".", omittingEmptySubsequences: false).map { component -> String in
            let s = String(component)
            if Int(s) != nil { return s }                  // numeric layer index
            if s == "block" { return "blocks" }            // ModuleList plural
            return snakeToCamel(s)
        }
        return components.joined(separator: ".")
    }

    private static func snakeToCamel(_ s: String) -> String {
        let parts = s.split(separator: "_")
        guard let first = parts.first else { return s }
        var result = String(first)
        for part in parts.dropFirst() {
            result += part.prefix(1).uppercased() + part.dropFirst()
        }
        return result
    }

    public static func fuseWeightNorm(weightG: MLXArray, weightV: MLXArray) -> MLXArray {
        let vFlat = weightV.reshaped([weightV.shape[0], -1])
        let norm = sqrt((vFlat * vFlat).sum(axis: 1)).reshaped(weightG.shape)
        return weightG * weightV / (norm + 1e-9)
    }

    public static func conv1dToMLX(_ ptWeight: MLXArray) -> MLXArray {
        ptWeight.transposed(0, 2, 1)
    }

    public static func convTranspose1dToMLX(_ ptWeight: MLXArray) -> MLXArray {
        ptWeight.transposed(1, 2, 0)
    }

    public static func squeezeSnakeParam(_ ptParam: MLXArray) -> MLXArray {
        ptParam.squeezed()
    }

    public static func isWeightNormG(_ key: String) -> Bool {
        key.hasSuffix(".weight_g")
    }

    public static func isWeightNormV(_ key: String) -> Bool {
        key.hasSuffix(".weight_v")
    }

    public static func isSnakeParam(_ key: String) -> Bool {
        key.hasSuffix(".alpha") || key.hasSuffix(".beta")
    }

    public static func isConvTranspose(_ key: String) -> Bool {
        key.contains("conv_t1")
    }

    public static func weightNormCompanion(_ gKey: String) -> String {
        String(gKey.dropLast(".weight_g".count)) + ".weight_v"
    }

    public static func fusedKeyName(_ gKey: String) -> String {
        String(gKey.dropLast(".weight_g".count)) + ".weight"
    }
}
