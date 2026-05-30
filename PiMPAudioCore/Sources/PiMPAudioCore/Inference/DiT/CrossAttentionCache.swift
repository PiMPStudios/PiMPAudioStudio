import MLX

/// Cache for cross-attention key/value tensors computed from encoder conditioning.
///
/// Stores the projected K and V tensors per DiT layer so they can be reused
/// across diffusion timesteps without recomputing the text/lyric encoding.
///
/// ## Threading contract — why `@unchecked Sendable` is safe
///
/// This class is `@unchecked Sendable` because it is a mutable reference type
/// that crosses actor boundaries (created in `DiffusionActor`, passed into
/// `DiTActor.step()`). Swift cannot verify its safety automatically, but it IS
/// safe under the following invariants that the diffusion sampler enforces:
///
/// 1. **One cache per generation call.** A fresh instance is created at the
///    start of each generation and discarded when it finishes.
/// 2. **Sequential access only.** `DiTActor.step()` is called one timestep at
///    a time — never concurrently. The cache is never shared across parallel tasks.
/// 3. **Clean ownership transfer.** The actor returns the updated cache to the
///    caller; the caller does not retain a concurrent reference while the actor
///    is using it.
///
/// - Warning: Do **not** pass a single `CrossAttentionCache` to concurrent tasks.
///   Concurrent mutation of `keys` and `values` is not thread-safe.
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

    /// Clear all cached tensors. Must be called between independent generation calls.
    public func reset() {
        keys.removeAll()
        values.removeAll()
    }
}
