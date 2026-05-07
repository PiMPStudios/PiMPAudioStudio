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
                if VAENameMap.isConvTranspose(fusedKey) {
                    fused = VAENameMap.convTranspose1dToMLX(fused)
                } else {
                    fused = VAENameMap.conv1dToMLX(fused)
                }
                result.append((fusedKey, fused))
                processed.insert(key)
                processed.insert(vKey)
                continue
            }

            if VAENameMap.isWeightNormV(key) { continue }

            guard let value = dict[key] else { continue }

            if VAENameMap.isSnakeParam(key) {
                result.append((key, VAENameMap.squeezeSnakeParam(value)))
            } else {
                result.append((key, value))
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
    public func loadSFXDiTWeights(from url: URL) throws -> [(String, MLXArray)] {
        let raw = try loadSafetensors(from: url)
        return raw.compactMap { key, value -> (String, MLXArray)? in
            guard let mapped = SFXDiTNameMap.mapKey(key) else { return nil }
            return (mapped, value)
        }
    }

    /// Load and remap SFX VAE weights from a safetensors file,
    /// transposing conv1d weights to MLX layout.
    public func loadSFXVAEWeights(from url: URL) throws -> [(String, MLXArray)] {
        let raw = try loadSafetensors(from: url)
        return raw.compactMap { key, value -> (String, MLXArray)? in
            guard let mapped = SFXVAENameMap.mapKey(key) else { return nil }
            if SFXVAENameMap.isConv1dWeight(mapped) {
                return (mapped, SFXVAENameMap.conv1dToMLX(value))
            }
            return (mapped, value)
        }
    }

    /// Load and remap CLAP encoder weights from a safetensors file.
    public func loadCLAPWeights(from url: URL) throws -> [(String, MLXArray)] {
        let raw = try loadSafetensors(from: url)
        return raw.compactMap { key, value -> (String, MLXArray)? in
            guard let mapped = CLAPNameMap.mapKey(key) else { return nil }
            return (mapped, value)
        }
    }
}
