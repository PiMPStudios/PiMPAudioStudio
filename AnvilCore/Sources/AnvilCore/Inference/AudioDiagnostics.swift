import Foundation
import MLX
import OSLog

/// Lightweight runtime-toggleable instrumentation for the audio inference
/// pipeline. Every call is a cheap UserDefaults read + early return when the
/// flag is off, so it's safe to leave callsites in place.
///
/// Toggle via UI (Advanced → Verbose audio diagnostics) or directly:
///   defaults write com.pimp.AnvilApp audioDebugVerbose -bool YES
///
/// In Xcode, filter the console with: subsystem:com.pimp.AnvilApp [AUDIODIAG]
///
/// Output format (single line per boundary):
///   [AUDIODIAG] <tag> shape=[…] dtype=… abs_mean=… abs_max=… any_nan=… any_inf=… first5=[…]
public enum AudioDiagnostics {
    private static let logger = Logger(subsystem: "com.pimp.AnvilApp", category: "AudioDiag")
    private static let userDefaultsKey = "audioDebugVerbose"

    public static var isEnabled: Bool {
        UserDefaults.standard.bool(forKey: userDefaultsKey)
    }

    /// Emit tensor stats for an MLXArray. Forces eval to materialize values.
    public static func tensor(_ tag: String, _ array: MLXArray) {
        guard isEnabled else { return }
        eval(array)

        let shape = array.shape
        let dtype = "\(array.dtype)"
        // Compute reductions in float32. Large finite float16 tensors can
        // overflow the reduction accumulator and report `abs_mean=inf` even
        // when `any_inf=false`, which makes diagnostics look scarier than the
        // tensor actually is.
        let statsArray = array.asType(.float32)
        let absArr = MLX.abs(statsArray)
        let absMean = absArr.mean().item(Float.self)
        let absMax = absArr.max().item(Float.self)

        // NaN/Inf checks — relevant for diagnosing exploded gradients/embeddings.
        let nanMask = (statsArray .!= statsArray)
        let anyNaN = nanMask.any().item(Bool.self)
        let infMask = MLX.isInf(statsArray)
        let anyInf = infMask.any().item(Bool.self)

        // First 5 scalar values from the flattened array, for sanity-eyeballing.
        let flat = statsArray.reshaped([-1])
        let count = min(5, flat.shape[0])
        var firstN: [Float] = []
        for i in 0..<count {
            firstN.append(flat[i].item(Float.self))
        }

        logger.info("""
        [AUDIODIAG] \(tag) shape=\(shape) dtype=\(dtype) \
        abs_mean=\(String(format: "%.4g", absMean)) \
        abs_max=\(String(format: "%.4g", absMax)) \
        any_nan=\(anyNaN) any_inf=\(anyInf) \
        first5=\(firstN.map { String(format: "%.4g", $0) })
        """)
    }

    /// Emit a free-form note. Useful for schedule values, step counts, config flags.
    public static func note(_ tag: String, _ message: String) {
        guard isEnabled else { return }
        logger.info("[AUDIODIAG] \(tag) \(message)")
    }

    /// Emit an MLX allocator snapshot so long runs can be checked for growth.
    public static func memory(_ tag: String) {
        guard isEnabled else { return }
        let snapshot = Memory.snapshot()
        logger.info("""
        [AUDIODIAG] \(tag) memory \
        active=\(snapshot.activeMemory) (\(formatMiB(snapshot.activeMemory)) MiB) \
        cache=\(snapshot.cacheMemory) (\(formatMiB(snapshot.cacheMemory)) MiB) \
        peak=\(snapshot.peakMemory) (\(formatMiB(snapshot.peakMemory)) MiB)
        """)
    }

    /// Emit a [Float] array's full contents (for short arrays like timestep schedules).
    public static func floats(_ tag: String, _ values: [Float]) {
        guard isEnabled else { return }
        logger.info("[AUDIODIAG] \(tag) count=\(values.count) values=\(values)")
    }

    private static func formatMiB(_ bytes: Int) -> String {
        String(format: "%.1f", Double(bytes) / 1_048_576.0)
    }
}
