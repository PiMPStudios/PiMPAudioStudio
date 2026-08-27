import Foundation
import MLX
import MLXRandom

/// Synchronous key context used by local training to keep adapter-dropout RNG
/// explicit during MLX valueAndGrad graph construction.
public final class AdapterDropoutKeyStream: @unchecked Sendable {
    private let lock = NSLock()
    private let keys: [MLXArray]
    private var nextIndex = 0

    public var count: Int {
        keys.count
    }

    init(keys: [MLXArray]) {
        self.keys = keys
    }

    func nextKey() -> MLXArray? {
        lock.lock()
        defer { lock.unlock() }

        guard !keys.isEmpty else { return nil }
        let index = min(nextIndex, keys.count - 1)
        nextIndex += 1
        return keys[index]
    }
}

public enum AdapterDropoutKeyContext {
    private static let lock = NSLock()
    private static var currentStream: AdapterDropoutKeyStream?

    public static func seed(epoch: Int, globalStep: Int, sampleIndex: Int, pathCount: Int) -> UInt64 {
        var value = UInt64(truncatingIfNeeded: epoch) &* 0x9E37_79B9_7F4A_7C15
        value ^= UInt64(truncatingIfNeeded: globalStep) &* 0xBF58_476D_1CE4_E5B9
        value ^= UInt64(truncatingIfNeeded: sampleIndex) &* 0x94D0_49BB_1331_11EB
        value ^= UInt64(truncatingIfNeeded: pathCount) &* 0xD6E8_FEB8_6659_FD93
        return value == 0 ? 1 : value
    }

    public static func makeKeyStream(seed: UInt64, count: Int) -> AdapterDropoutKeyStream {
        let keyCount = max(1, count)
        let keys = MLXRandom.split(key: MLXRandom.key(seed), into: keyCount)
        eval(keys)
        return AdapterDropoutKeyStream(keys: keys)
    }

    public static func withKeyStream<T>(
        _ stream: AdapterDropoutKeyStream?,
        operation: () throws -> T
    ) rethrows -> T {
        lock.lock()
        let previousStream = currentStream
        currentStream = stream
        lock.unlock()

        defer {
            lock.lock()
            currentStream = previousStream
            lock.unlock()
        }

        return try operation()
    }

    static func nextKey() -> MLXArray? {
        lock.lock()
        let stream = currentStream
        lock.unlock()
        return stream?.nextKey()
    }
}
