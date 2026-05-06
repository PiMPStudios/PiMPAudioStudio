// AnvilCore/Sources/AnvilCore/Inference/Weights/SFXVAENameMap.swift
import MLX

public enum SFXVAENameMap {
    public static func mapKey(_ key: String) -> String? {
        var mapped = key
        if mapped.hasPrefix("decoder.") {
            mapped = String(mapped.dropFirst("decoder.".count))
        }
        if mapped.contains("up_blocks") {
            mapped = mapped.replacingOccurrences(of: "up_blocks", with: "upsampleBlocks")
        }
        if mapped.contains("proj_in") || mapped.contains("proj_out") {
            // Already matches
        }
        return mapped
    }

    public static func isConv1dWeight(_ key: String) -> Bool {
        key.contains("conv.weight") || (key.contains("linear.weight") && !key.contains("norm"))
    }

    public static func conv1dToMLX(_ ptWeight: MLXArray) -> MLXArray {
        if ptWeight.ndim == 3 {
            return ptWeight.transposed(0, 2, 1)
        }
        return ptWeight
    }
}
