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
/// When enabled, the same lines are mirrored to PiMPAudioStudio-AUDIODIAG-*.log
/// files in the app-configured diagnostics folder.
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

    public static var configuredLogDirectoryURL: URL? {
        fileSink.configuredLogDirectoryURL
    }

    public static func setLogDirectory(_ url: URL?) {
        fileSink.setLogDirectory(url)
        // Opportunistic cleanup whenever the folder is (re)configured (app launch).
        _ = pruneLogFiles(in: url)
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

    // MARK: - Log retention

    /// Default retention: 14 days, 40 files, 64 MiB total (keeps the active file).
    public static let defaultLogMaxAgeDays = 14
    public static let defaultLogMaxFiles = 40
    public static let defaultLogMaxTotalBytes = 64 * 1024 * 1024

    /// Filename prefix for mirrored verbose diagnostics logs.
    public static let logFileNamePrefix = "PiMPAudioStudio-AUDIODIAG-"

    /// Delete old AUDIODIAG log files under the diagnostics folder.
    ///
    /// Keeps the currently active log (if any). Among the rest, drops files older
    /// than `maxAgeDays`, then newest-first until under `maxFiles` and
    /// `maxTotalBytes`. Safe to call often (e.g. on launch and when opening a new log).
    @discardableResult
    public static func pruneLogFiles(
        in directory: URL? = nil,
        maxAgeDays: Int = defaultLogMaxAgeDays,
        maxFiles: Int = defaultLogMaxFiles,
        maxTotalBytes: Int = defaultLogMaxTotalBytes,
        now: Date = Date(),
        fileManager: FileManager = .default
    ) -> DiagnosticsPruneResult {
        let dir = (directory ?? configuredLogDirectoryURL)?.standardizedFileURL
        guard let dir else {
            return DiagnosticsPruneResult(deletedCount: 0, deletedBytes: 0, remainingCount: 0)
        }

        let keepPaths = Set(
            [activeLogFileURL].compactMap { $0?.standardizedFileURL.path }
        )

        let contents: [URL]
        do {
            contents = try fileManager.contentsOfDirectory(
                at: dir,
                includingPropertiesForKeys: [.contentModificationDateKey, .fileSizeKey, .isRegularFileKey],
                options: [.skipsHiddenFiles]
            )
        } catch {
            return DiagnosticsPruneResult(deletedCount: 0, deletedBytes: 0, remainingCount: 0)
        }

        var candidates: [DiagnosticsLogFileInfo] = []
        for url in contents {
            let name = url.lastPathComponent
            guard name.hasPrefix(logFileNamePrefix), name.hasSuffix(".log") else { continue }
            let values = try? url.resourceValues(forKeys: [
                .contentModificationDateKey,
                .fileSizeKey,
                .isRegularFileKey,
            ])
            guard values?.isRegularFile == true else { continue }
            let modified = values?.contentModificationDate ?? .distantPast
            let size = values?.fileSize ?? 0
            candidates.append(
                DiagnosticsLogFileInfo(
                    url: url.standardizedFileURL,
                    modified: modified,
                    size: size
                )
            )
        }

        let toDelete = selectLogFilesToDelete(
            candidates: candidates,
            keepPaths: keepPaths,
            maxAgeDays: maxAgeDays,
            maxFiles: maxFiles,
            maxTotalBytes: maxTotalBytes,
            now: now
        )

        var deletedCount = 0
        var deletedBytes = 0
        for info in toDelete {
            do {
                try fileManager.removeItem(at: info.url)
                deletedCount += 1
                deletedBytes += max(0, info.size)
            } catch {
                // Best-effort; leave the file if locked or mid-write.
            }
        }

        let remaining = max(0, candidates.count - deletedCount)
        if deletedCount > 0 {
            // Use logger only — do not write to the sink (would re-open files during prune).
            logger.info(
                "Pruned \(deletedCount, privacy: .public) AUDIODIAG log(s), freed \(deletedBytes, privacy: .public) bytes; \(remaining, privacy: .public) remaining"
            )
        }

        return DiagnosticsPruneResult(
            deletedCount: deletedCount,
            deletedBytes: deletedBytes,
            remainingCount: remaining
        )
    }

    /// Pure selection used by ``pruneLogFiles`` (unit-tested).
    static func selectLogFilesToDelete(
        candidates: [DiagnosticsLogFileInfo],
        keepPaths: Set<String>,
        maxAgeDays: Int,
        maxFiles: Int,
        maxTotalBytes: Int,
        now: Date
    ) -> [DiagnosticsLogFileInfo] {
        let ageLimit = max(0, maxAgeDays)
        let fileLimit = max(1, maxFiles)
        let byteLimit = max(0, maxTotalBytes)
        let cutoff = Calendar.current.date(byAdding: .day, value: -ageLimit, to: now) ?? now

        let sortedNewestFirst = candidates.sorted { lhs, rhs in
            if lhs.modified != rhs.modified {
                return lhs.modified > rhs.modified
            }
            return lhs.url.path > rhs.url.path
        }

        var keep = Set<String>()
        var keptBytes = 0

        for info in sortedNewestFirst {
            let path = info.url.path
            if keepPaths.contains(path) {
                keep.insert(path)
                keptBytes += max(0, info.size)
                continue
            }
        }

        for info in sortedNewestFirst {
            let path = info.url.path
            if keep.contains(path) { continue }
            if info.modified < cutoff { continue } // too old → delete
            if keep.count >= fileLimit { continue }
            let size = max(0, info.size)
            if keptBytes + size > byteLimit, !keep.isEmpty { continue }
            keep.insert(path)
            keptBytes += size
        }

        return sortedNewestFirst.filter { !keep.contains($0.url.path) }
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

/// Result of pruning AUDIODIAG log files.
public struct DiagnosticsPruneResult: Equatable, Sendable {
    public let deletedCount: Int
    public let deletedBytes: Int
    public let remainingCount: Int

    public init(deletedCount: Int, deletedBytes: Int, remainingCount: Int) {
        self.deletedCount = deletedCount
        self.deletedBytes = deletedBytes
        self.remainingCount = remainingCount
    }
}

/// One diagnostics log file considered for retention.
struct DiagnosticsLogFileInfo: Equatable, Sendable {
    let url: URL
    let modified: Date
    let size: Int
}

private final class AudioDiagnosticsFileSink: @unchecked Sendable {
    private let lock = NSLock()
    private let fileManager = FileManager.default
    private let fileNameFormatter: DateFormatter
    private let lineFormatter: DateFormatter

    private var isEnabled = false
    private var fileHandle: FileHandle?
    private var logFileURL: URL?
    private var configuredLogDirectory: URL?

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

    var configuredLogDirectoryURL: URL? {
        withLock { configuredLogDirectory }
    }

    func setLogDirectory(_ url: URL?) {
        let standardizedURL = url?.standardizedFileURL
        withLock {
            guard configuredLogDirectory != standardizedURL else { return }
            configuredLogDirectory = standardizedURL
            if isEnabled {
                openNewLogFileLocked()
            }
        }
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

        for directory in candidateLogDirectories() {
            let url = directory.appendingPathComponent(fileName, isDirectory: false)

            do {
                try fileManager.createDirectory(
                    at: directory,
                    withIntermediateDirectories: true
                )
                try Data().write(to: url, options: .withoutOverwriting)
                fileHandle = try FileHandle(forWritingTo: url)
                logFileURL = url
                writeLocked("# PiMP Audio Studio audio diagnostics")
                writeLocked("# Started \(lineFormatter.string(from: Date()))")
                writeLocked("# Path \(url.path)")
                // Defer prune off this lock so we never delete while still opening.
                let directoryForPrune = directory
                DispatchQueue.global(qos: .utility).async {
                    _ = AudioDiagnostics.pruneLogFiles(in: directoryForPrune)
                }
                return
            } catch {
                logFileURL = nil
                fileHandle = nil
            }
        }
    }

    private func candidateLogDirectories() -> [URL] {
        var seenPaths = Set<String>()
        return [configuredLogDirectory, fileManager.temporaryDirectory].compactMap(\.self).filter { url in
            let path = url.standardizedFileURL.path
            guard !seenPaths.contains(path) else { return false }
            seenPaths.insert(path)
            return true
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
