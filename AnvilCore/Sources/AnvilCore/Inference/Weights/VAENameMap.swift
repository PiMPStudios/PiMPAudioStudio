import MLX

public enum VAENameMap {
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
