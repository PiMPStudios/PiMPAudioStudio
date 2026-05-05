import MLX

public final class CrossAttentionCache: @unchecked Sendable {
    private var keys: [Int: MLXArray] = [:]
    private var values: [Int: MLXArray] = [:]

    public init() {}

    public func update(key: MLXArray, value: MLXArray, layerIdx: Int) {
        keys[layerIdx] = key
        values[layerIdx] = value
    }

    public func get(layerIdx: Int) -> (key: MLXArray, value: MLXArray)? {
        guard let k = keys[layerIdx], let v = values[layerIdx] else {
            return nil
        }
        return (k, v)
    }

    public func isUpdated(layerIdx: Int) -> Bool {
        keys[layerIdx] != nil
    }

    public func reset() {
        keys.removeAll()
        values.removeAll()
    }
}
