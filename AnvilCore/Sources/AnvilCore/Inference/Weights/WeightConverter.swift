import Foundation
import MLX

public actor WeightConverter {
    private var progressContinuation: AsyncStream<ConversionProgress>.Continuation?

    public init() {}

    public func makeProgressStream() -> AsyncStream<ConversionProgress> {
        let (stream, continuation) = AsyncStream<ConversionProgress>.makeStream()
        self.progressContinuation = continuation
        return stream
    }

    public func loadSafetensors(from url: URL) throws -> [(String, MLXArray)] {
        let dict = try MLX.loadArrays(url: url)
        return dict.sorted { $0.key < $1.key }.map { ($0.key, $0.value) }
    }

    public func loadSafetensors(from urls: [URL]) throws -> [(String, MLXArray)] {
        var result: [(String, MLXArray)] = []
        for url in urls.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            result.append(contentsOf: try loadSafetensors(from: url))
        }
        return result.sorted { $0.0 < $1.0 }
    }

    public func convertVAEWeights(stateDict: [(String, MLXArray)]) throws -> [(String, MLXArray)] {
        var result: [(String, MLXArray)] = []
        var processed: Set<String> = []
        let total = stateDict.count

        let dict = Dictionary(stateDict, uniquingKeysWith: { first, _ in first })
        let allKeys = stateDict.map { $0.0 }.sorted()

        for (idx, key) in allKeys.enumerated() {
            if idx % 50 == 0 { try Task.checkCancellation() }
            if processed.contains(key) { continue }

            progressContinuation?.yield(ConversionProgress(
                current: idx, total: total, currentTensorName: key
            ))

            if VAENameMap.isWeightNormG(key) {
                let vKey = VAENameMap.weightNormCompanion(key)
                guard let g = dict[key], let v = dict[vKey] else {
                    processed.insert(key)
                    continue
                }
                var fused = VAENameMap.fuseWeightNorm(weightG: g, weightV: v)
                let fusedKey = VAENameMap.fusedKeyName(key)
                // isConvTranspose still inspects the snake_case `conv_t1`
                // form, so axis-remap detection runs BEFORE renaming.
                if VAENameMap.isConvTranspose(fusedKey) {
                    fused = VAENameMap.convTranspose1dToMLX(fused)
                } else {
                    fused = VAENameMap.conv1dToMLX(fused)
                }
                result.append((VAENameMap.mapKey(fusedKey), fused))
                processed.insert(key)
                processed.insert(vKey)
                continue
            }

            if VAENameMap.isWeightNormV(key) { continue }

            guard let value = dict[key] else { continue }

            if VAENameMap.isSnakeParam(key) {
                result.append((VAENameMap.mapKey(key), VAENameMap.squeezeSnakeParam(value)))
            } else {
                result.append((VAENameMap.mapKey(key), value))
            }
            processed.insert(key)
        }

        progressContinuation?.finish()
        progressContinuation = nil
        return result
    }

    public func convertDiTWeights(stateDict: [(String, MLXArray)]) throws -> [(String, MLXArray)] {
        var result: [(String, MLXArray)] = []
        let total = stateDict.count

        for (idx, (key, value)) in stateDict.enumerated() {
            if idx % 50 == 0 { try Task.checkCancellation() }
            progressContinuation?.yield(ConversionProgress(
                current: idx, total: total, currentTensorName: key
            ))

            guard let mappedKey = TensorNameMap.mapDiTKey(key) else { continue }

            // The Conv1d/ConvTranspose1d predicates match by substring (`proj_in.weight` /
            // `proj_out.weight`), but the ACE-Step v1.5 checkpoint contains BOTH the
            // top-level Conv1d/ConvTranspose1d (3-D weights) AND inner `nn.Linear`
            // layers named `proj_in`/`proj_out` (2-D weights) inside encoder blocks.
            // Apply the conv axis remap only when the tensor is actually 3-D — pass
            // 2-D Linear weights through unchanged.
            if TensorNameMap.isDiTConv1dWeight(mappedKey) && value.ndim == 3 {
                result.append((mappedKey, TensorNameMap.conv1dToMLX(value)))
            } else if TensorNameMap.isDiTConvTransposeWeight(mappedKey) && value.ndim == 3 {
                result.append((mappedKey, TensorNameMap.convTranspose1dToMLX(value)))
            } else {
                result.append((mappedKey, value))
            }
        }

        progressContinuation?.finish()
        progressContinuation = nil
        return result
    }

    public func convertQwen3Weights(
        from rawWeights: [(String, MLXArray)],
        progress: ((ConversionProgress) -> Void)? = nil
    ) -> [(String, MLXArray)] {
        var result: [(String, MLXArray)] = []
        let total = rawWeights.count

        for (index, (key, value)) in rawWeights.enumerated() {
            progress?(ConversionProgress(current: index + 1, total: total, currentTensorName: key))
            guard let mappedKey = Qwen3NameMap.mapKey(key) else { continue }
            result.append((mappedKey, value))
        }

        return result
    }

    public func convertConditionEncoderWeights(
        from rawWeights: [(String, MLXArray)],
        progress: ((ConversionProgress) -> Void)? = nil
    ) -> [(String, MLXArray)] {
        var result: [(String, MLXArray)] = []
        let total = rawWeights.count

        for (index, (key, value)) in rawWeights.enumerated() {
            progress?(ConversionProgress(current: index + 1, total: total, currentTensorName: key))
            guard let mappedKey = ConditionEncoderNameMap.mapKey(key) else { continue }
            result.append((mappedKey, value))
        }

        return result
    }

    /// Load and remap SFX DiT weights from a safetensors file.
    ///
    /// Conv1d transposition is shape-conditional:
    ///   • mlx-audiogen exports (`dit.safetensors`) already store Conv1d weights in
    ///     MLX layout `[out, kernel, in]` — e.g. `[64, 1, 64]` for kernel=1.
    ///     Applying transposition again would produce the wrong `[64, 64, 1]` layout.
    ///   • Original PyTorch checkpoints use `[out, in, kernel]` — e.g. `[64, 64, 1]`.
    ///     These DO require transposition.
    ///
    /// We detect the layout by checking whether the last axis == 1 (PyTorch for
    /// kernel=1) or the middle axis == 1 (already MLX).  Only transpose if the
    /// weight is still in PyTorch format.
    public func loadSFXDiTWeights(from url: URL) throws -> [(String, MLXArray)] {
        let raw = try loadSafetensors(from: url)
        return raw.compactMap { key, value -> (String, MLXArray)? in
            guard let mapped = SFXDiTNameMap.mapKey(key) else { return nil }
            if SFXDiTNameMap.isConv1dWeight(mapped) {
                // Transpose only if in PyTorch [out, in, kernel=1] format.
                // Shape [64, 64, 1]: last dim == 1, middle dim != 1 → needs transpose.
                // Shape [64, 1, 64]: middle dim == 1 → already MLX format, skip.
                if value.ndim == 3 && value.shape[2] == 1 && value.shape[1] != 1 {
                    return (mapped, SFXDiTNameMap.conv1dToMLX(value))
                }
                return (mapped, value)  // already [out, kernel, in]
            }
            return (mapped, value)
        }
    }

    /// Load and remap SFX VAE weights from a safetensors file.
    ///
    /// The SAO vae.safetensors is in the same PyTorch weight-normed Oobleck format
    /// as the ACE-Step music VAE (decoder.block.N.*, weight_g/weight_v pairs,
    /// [out,in,k] axis order).  We reuse `convertVAEWeights` which handles:
    ///   - weight_g + weight_v fusion via WeightNorm
    ///   - Conv1d / ConvTranspose1d axis transposition ([out,in,k] → [out,k,in])
    ///   - snake alpha/beta [1,C,1] → squeezed to [C]
    ///   - snake_case → camelCase + block → blocks renaming via VAENameMap.mapKey
    /// Encoder keys pass through the converter but are silently ignored by
    /// SFXVAEModel.update(parameters:) since that model only has a decoder.
    public func loadSFXVAEWeights(from url: URL) throws -> [(String, MLXArray)] {
        let raw = try loadSafetensors(from: url)
        return try convertVAEWeights(stateDict: raw)
    }

    /// Load T5-base encoder weights for Stable Audio Open. The HF safetensors
    /// uses HuggingFace's standard T5 naming
    /// (`encoder.block.N.layer.0.SelfAttention.q.weight`, etc.). We map onto
    /// our `T5Encoder` Module's path layout
    /// (`blocks.N.layer0_attn.q.weight`, etc.). Skeleton mapping is
    /// best-effort — keys that don't resolve are dropped (logged via the
    /// existing `compactMap` filter and surfaced by the per-module supplied/
    /// matched diagnostic on load).
    public func loadT5Weights(from url: URL) throws -> [(String, MLXArray)] {
        let raw = try loadSafetensors(from: url)
        return raw.compactMap { key, value -> (String, MLXArray)? in
            guard let mapped = T5NameMap.mapKey(key) else { return nil }
            return (mapped, value)
        }
    }

    /// Load SAO projection_model weights from a safetensors file. The HF
    /// keys are typically simple (`text_projection.weight`,
    /// `timing_projection.weight`) and align directly with our
    /// `SAOProjectionModel` ModuleInfo keys.
    public func loadSAOProjectionWeights(from url: URL) throws -> [(String, MLXArray)] {
        let raw = try loadSafetensors(from: url)
        return raw.compactMap { key, value -> (String, MLXArray)? in
            guard let mapped = SAOProjectionNameMap.mapKey(key) else { return nil }
            return (mapped, value)
        }
    }
}
