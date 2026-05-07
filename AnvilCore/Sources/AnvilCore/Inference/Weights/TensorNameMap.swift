import MLX

public enum TensorNameMap {
    /// Maps a key from the downloaded ACE-Step DiT safetensors checkpoint to
    /// the parameter path expected by our MLX-Swift `DiTDecoder` module.
    ///
    /// The checkpoint's keys use Python's snake_case naming and live under a
    /// `decoder.` prefix:
    ///
    ///     decoder.layers.0.cross_attn.q_proj.weight
    ///     decoder.condition_embedder.weight
    ///     decoder.proj_in.1.weight                    ← Sequential(Lambda, Conv1d, Lambda)
    ///     decoder.scale_shift_table
    ///
    /// Our Swift module uses Swift's camelCase property naming and the DiT
    /// decoder's properties live at the top level (no `decoder.` wrapper):
    ///
    ///     layers.0.crossAttn.qProj.weight
    ///     conditionEmbedder.weight
    ///     projIn.weight
    ///     scaleShiftTable
    ///
    /// MLX-Swift's `module.update(parameters:)` silently ignores keys that
    /// don't resolve to a real parameter path — without this conversion most
    /// of the DiT weights stay at default-init random values and the model
    /// produces near-silent noise regardless of prompt or seed.
    ///
    /// Returns nil for keys that should be skipped (encoder / tokenizer /
    /// detokenizer / null_condition_emb subtrees that belong to other parts
    /// of the upstream model graph, plus rotary embedding buffers which are
    /// recomputed on the MLX side).
    public static func mapDiTKey(_ key: String) -> String? {
        guard key.hasPrefix("decoder.") else { return nil }
        var stripped = String(key.dropFirst("decoder.".count))

        if stripped.contains("rotary_emb") { return nil }

        // PyTorch wraps proj_in / proj_out in
        //   Sequential(Lambda(transpose), Conv1d|ConvTranspose1d, Lambda(transpose))
        // so the conv weight lives at index `.1.`. Our Swift module is a
        // bare Conv1d / ConvTranspose1d at the top level, so drop the `.1.`.
        if stripped.hasPrefix("proj_in.1.") {
            stripped = stripped.replacingOccurrences(of: "proj_in.1.", with: "proj_in.")
        } else if stripped.hasPrefix("proj_out.1.") {
            stripped = stripped.replacingOccurrences(of: "proj_out.1.", with: "proj_out.")
        }

        // Convert each `.`-separated path component from snake_case to
        // camelCase. Numeric components (layer indices) are kept verbatim.
        let components = stripped.split(separator: ".", omittingEmptySubsequences: false).map { component -> String in
            let s = String(component)
            if Int(s) != nil { return s }
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

    public static func isDiTConv1dWeight(_ key: String) -> Bool {
        // After mapDiTKey, the conv1d patch projection is "projIn.weight".
        key.contains("projIn.weight")
    }

    public static func isDiTConvTransposeWeight(_ key: String) -> Bool {
        // After mapDiTKey, the conv-transpose patch un-projection is "projOut.weight".
        key.contains("projOut.weight")
    }

    public static func conv1dToMLX(_ ptWeight: MLXArray) -> MLXArray {
        ptWeight.transposed(0, 2, 1)
    }

    public static func convTranspose1dToMLX(_ ptWeight: MLXArray) -> MLXArray {
        ptWeight.transposed(1, 2, 0)
    }
}
