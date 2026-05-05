import MLX

public enum TensorNameMap {
    public static func mapDiTKey(_ key: String) -> String? {
        if key.contains("rotary_emb") {
            return nil
        }
        if key.hasPrefix("proj_in.1.") {
            return key.replacingOccurrences(of: "proj_in.1.", with: "proj_in.")
        }
        if key.hasPrefix("proj_out.1.") {
            return key.replacingOccurrences(of: "proj_out.1.", with: "proj_out.")
        }
        return key
    }

    public static func isDiTConv1dWeight(_ key: String) -> Bool {
        key.contains("proj_in.weight")
    }

    public static func isDiTConvTransposeWeight(_ key: String) -> Bool {
        key.contains("proj_out.weight")
    }

    public static func conv1dToMLX(_ ptWeight: MLXArray) -> MLXArray {
        ptWeight.transposed(0, 2, 1)
    }

    public static func convTranspose1dToMLX(_ ptWeight: MLXArray) -> MLXArray {
        ptWeight.transposed(1, 2, 0)
    }
}
