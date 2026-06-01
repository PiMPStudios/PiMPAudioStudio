import Foundation
import MLX
import OSLog

/// Lightweight runtime-toggleable instrumentation for the audio inference
/// pipeline. Every call is a cheap UserDefaults read + early return when the
/// flag is off, so it's safe to leave callsites in place.
///
/// Toggle via UI (Advanced → Verbose audio diagnostics) or directly:
///   defaults write com.pimpstudios.PiMPAudioStudio com.pimpstudios.PiMPAudioStudio.audioDebugVerbose -bool YES
///
/// In Xcode, filter the console with: subsystem:com.pimpstudios.PiMPAudioStudio [AUDIODIAG]
/// When enabled, the same lines are mirrored to /tmp/PiMPAudioStudio-AUDIODIAG-*.log.
///
/// Output format (single line per boundary):
///   [AUDIODIAG] <tag> shape=[…] dtype=… abs_mean=… abs_max=… any_nan=… any_inf=… first5=[…]
public enum AudioDiagnostics {
    private static let logger = Logger(subsystem: "com.pimpstudios.PiMPAudioStudio", category: "AudioDiag")
    private static let userDefaultsKey = "com.pimpstudios.PiMPAudioStudio.audioDebugVerbose"
    private static let legacyUserDefaultsKey = "audioDebugVerbose"
    private static let fileSink = AudioDiagnosticsFileSink()

    public static var isEnabled: Bool {
        let enabled = UserDefaults.standard.bool(forKey: userDefaultsKey) ||
            UserDefaults.standard.bool(forKey: legacyUserDefaultsKey)
        fileSink.setEnabled(enabled)
        return enabled
    }

    public static var activeLogFileURL: URL? {
        fileSink.activeLogFileURL
    }

    public static func setVerboseLoggingEnabled(_ enabled: Bool) {
        UserDefaults.standard.set(enabled, forKey: userDefaultsKey)
        fileSink.setEnabled(enabled)
        guard enabled, let url = activeLogFileURL else { return }
        emit("[AUDIODIAG] log/file path=\(url.path)")
    }

    public static func refreshFileLoggingState() {
        fileSink.setEnabled(
            UserDefaults.standard.bool(forKey: userDefaultsKey) ||
                UserDefaults.standard.bool(forKey: legacyUserDefaultsKey)
        )
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

        emit(
            "[AUDIODIAG] \(tag) shape=\(shape) dtype=\(dtype) "
                + "abs_mean=\(String(format: "%.4g", absMean)) "
                + "abs_max=\(String(format: "%.4g", absMax)) "
                + "any_nan=\(anyNaN) any_inf=\(anyInf) "
                + "first5=\(firstN.map { String(format: "%.4g", $0) })"
        )
    }

    /// Emit a free-form note. Useful for schedule values, step counts, config flags.
    public static func note(_ tag: String, _ message: String) {
        guard isEnabled else { return }
        emit("[AUDIODIAG] \(tag) \(message)")
    }

    /// Emit an MLX allocator snapshot so long runs can be checked for growth.
    public static func memory(_ tag: String) {
        guard isEnabled else { return }
        let snapshot = Memory.snapshot()
        emit(
            "[AUDIODIAG] \(tag) memory "
                + "active=\(snapshot.activeMemory) (\(formatMiB(snapshot.activeMemory)) MiB) "
                + "cache=\(snapshot.cacheMemory) (\(formatMiB(snapshot.cacheMemory)) MiB) "
                + "peak=\(snapshot.peakMemory) (\(formatMiB(snapshot.peakMemory)) MiB)"
        )
    }

    /// Emit a [Float] array's full contents (for short arrays like timestep schedules).
    public static func floats(_ tag: String, _ values: [Float]) {
        guard isEnabled else { return }
        emit("[AUDIODIAG] \(tag) count=\(values.count) values=\(values)")
    }

    private static func emit(_ line: String) {
        logger.info("\(line, privacy: .public)")
        fileSink.write(line)
    }

    private static func formatMiB(_ bytes: Int) -> String {
        String(format: "%.1f", Double(bytes) / 1_048_576.0)
    }
}

private final class AudioDiagnosticsFileSink: @unchecked Sendable {
    private let lock = NSLock()
    private let fileManager = FileManager.default
    private let logDirectory = URL(fileURLWithPath: "/tmp", isDirectory: true)
    private let fileNameFormatter: DateFormatter
    private let lineFormatter: DateFormatter

    private var isEnabled = false
    private var fileHandle: FileHandle?
    private var logFileURL: URL?

    init() {
        let locale = Locale(identifier: "en_US_POSIX")

        let fileNameFormatter = DateFormatter()
        fileNameFormatter.locale = locale
        fileNameFormatter.timeZone = .current
        fileNameFormatter.dateFormat = "yyyy-MM-dd_HH-mm-ss-SSS"
        self.fileNameFormatter = fileNameFormatter

        let lineFormatter = DateFormatter()
        lineFormatter.locale = locale
        lineFormatter.timeZone = .current
        lineFormatter.dateFormat = "yyyy-MM-dd HH:mm:ss.SSSZZZZZ"
        self.lineFormatter = lineFormatter
    }

    var activeLogFileURL: URL? {
        withLock { logFileURL }
    }

    func setEnabled(_ enabled: Bool) {
        withLock {
            guard enabled != isEnabled else { return }
            isEnabled = enabled
            if enabled {
                openNewLogFileLocked()
            } else {
                closeLogFileLocked()
            }
        }
    }

    func write(_ line: String) {
        withLock {
            guard isEnabled else { return }
            if fileHandle == nil {
                openNewLogFileLocked()
            }
            writeLocked("\(lineFormatter.string(from: Date())) \(line)")
        }
    }

    private func openNewLogFileLocked() {
        closeLogFileLocked()

        let timestamp = fileNameFormatter.string(from: Date())
        let suffix = UUID().uuidString.prefix(8)
        let fileName = "PiMPAudioStudio-AUDIODIAG-\(timestamp)-\(suffix).log"
        let url = logDirectory.appendingPathComponent(fileName, isDirectory: false)

        guard fileManager.createFile(atPath: url.path, contents: nil) else {
            logFileURL = nil
            fileHandle = nil
            return
        }

        do {
            fileHandle = try FileHandle(forWritingTo: url)
            logFileURL = url
            writeLocked("# PiMP Audio Studio audio diagnostics")
            writeLocked("# Started \(lineFormatter.string(from: Date()))")
        } catch {
            logFileURL = nil
            fileHandle = nil
        }
    }

    private func closeLogFileLocked() {
        if let fileHandle {
            try? fileHandle.close()
        }
        fileHandle = nil
        logFileURL = nil
    }

    private func writeLocked(_ line: String) {
        guard let data = "\(line)\n".data(using: .utf8), let fileHandle else { return }
        try? fileHandle.write(contentsOf: data)
    }

    private func withLock<T>(_ operation: () -> T) -> T {
        lock.lock()
        defer { lock.unlock() }
        return operation()
    }
}
