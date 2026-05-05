import Foundation
import MLX

public actor WeightConverter {
    private var progressContinuation: AsyncStream<ConversionProgress>.Continuation?
    private var _progress: AsyncStream<ConversionProgress>?

    public init() {}

    public var progress: AsyncStream<ConversionProgress> {
        if let existing = _progress { return existing }
        let (stream, continuation) = AsyncStream<ConversionProgress>.makeStream()
        self.progressContinuation = continuation
        self._progress = stream
        return stream
    }

    public func convertVAEWeights(stateDict: [(String, MLXArray)]) -> [(String, MLXArray)] {
        var result: [(String, MLXArray)] = []
        var processed: Set<String> = []
        let total = stateDict.count

        let dict = Dictionary(stateDict, uniquingKeysWith: { first, _ in first })
        let allKeys = stateDict.map { $0.0 }.sorted()

        for (idx, key) in allKeys.enumerated() {
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

        return result
    }

    public func convertDiTWeights(stateDict: [(String, MLXArray)]) -> [(String, MLXArray)] {
        var result: [(String, MLXArray)] = []
        let total = stateDict.count

        for (idx, (key, value)) in stateDict.enumerated() {
            progressContinuation?.yield(ConversionProgress(
                current: idx, total: total, currentTensorName: key
            ))

            guard let mappedKey = TensorNameMap.mapDiTKey(key) else { continue }

            if TensorNameMap.isDiTConv1dWeight(mappedKey) {
                result.append((mappedKey, TensorNameMap.conv1dToMLX(value)))
            } else if TensorNameMap.isDiTConvTransposeWeight(mappedKey) {
                result.append((mappedKey, TensorNameMap.convTranspose1dToMLX(value)))
            } else {
                result.append((mappedKey, value))
            }
        }

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
}
