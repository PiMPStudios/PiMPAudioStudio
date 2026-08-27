import Foundation
@testable import PiMPAudioCore
import Testing

@Suite("AudioDiagnostics", .serialized)
struct AudioDiagnosticsTests {
    @Test("Verbose diagnostics mirror notes to configured log folder")
    func verboseDiagnosticsMirrorNotesToConfiguredLogFolder() throws {
        let defaults = UserDefaults.standard
        let defaultsKey = "com.pimpstudios.PiMPAudioStudio.audioDebugVerbose"
        let legacyDefaultsKey = "audioDebugVerbose"
        let originalValue = defaults.object(forKey: defaultsKey)
        let originalLegacyValue = defaults.object(forKey: legacyDefaultsKey)
        let originalLogDirectory = AudioDiagnostics.configuredLogDirectoryURL
        let logDirectory = FileManager.default.temporaryDirectory
            .appending(path: "AudioDiagnosticsTests-\(UUID().uuidString)", directoryHint: .isDirectory)
        var createdURLs: [URL] = []

        defer {
            AudioDiagnostics.setVerboseLoggingEnabled(false)
            AudioDiagnostics.setLogDirectory(originalLogDirectory)
            restore(originalValue, forKey: defaultsKey, in: defaults)
            restore(originalLegacyValue, forKey: legacyDefaultsKey, in: defaults)
            for url in createdURLs {
                try? FileManager.default.removeItem(at: url)
            }
            try? FileManager.default.removeItem(at: logDirectory)
        }

        defaults.removeObject(forKey: legacyDefaultsKey)
        AudioDiagnostics.setLogDirectory(logDirectory)
        AudioDiagnostics.setVerboseLoggingEnabled(true)

        guard let firstURL = AudioDiagnostics.activeLogFileURL else {
            Issue.record("Expected verbose diagnostics to open a tmp log file")
            return
        }
        createdURLs.append(firstURL)

        let marker = UUID().uuidString
        AudioDiagnostics.note("test/file", "marker=\(marker)")
        AudioDiagnostics.setVerboseLoggingEnabled(false)

        #expect(AudioDiagnostics.activeLogFileURL == nil)
        #expect(firstURL.path.hasPrefix(logDirectory.path))
        #expect(firstURL.lastPathComponent.hasPrefix("PiMPAudioStudio-AUDIODIAG-"))
        #expect(firstURL.pathExtension == "log")

        let firstContents = try String(contentsOf: firstURL, encoding: .utf8)
        #expect(firstContents.contains("# PiMP Audio Studio audio diagnostics"))
        #expect(firstContents.contains("[AUDIODIAG] test/file marker=\(marker)"))

        AudioDiagnostics.setVerboseLoggingEnabled(true)
        guard let secondURL = AudioDiagnostics.activeLogFileURL else {
            Issue.record("Expected re-enabling verbose diagnostics to open a new tmp log file")
            return
        }
        createdURLs.append(secondURL)

        #expect(secondURL != firstURL)
        #expect(secondURL.lastPathComponent.hasPrefix("PiMPAudioStudio-AUDIODIAG-"))
    }

    @Test("selectLogFilesToDelete drops aged files and enforces count/size caps")
    func selectLogFilesToDeleteEnforcesRetention() {
        let now = Date(timeIntervalSince1970: 1_800_000_000)
        let dir = URL(fileURLWithPath: "/tmp/diag-test", isDirectory: true)
        func info(_ name: String, daysAgo: Int, size: Int) -> DiagnosticsLogFileInfo {
            let modified = Calendar.current.date(byAdding: .day, value: -daysAgo, to: now) ?? now
            return DiagnosticsLogFileInfo(
                url: dir.appendingPathComponent(name),
                modified: modified,
                size: size
            )
        }

        let active = info("PiMPAudioStudio-AUDIODIAG-active.log", daysAgo: 0, size: 1000)
        let recent = info("PiMPAudioStudio-AUDIODIAG-recent.log", daysAgo: 1, size: 1000)
        let old = info("PiMPAudioStudio-AUDIODIAG-old.log", daysAgo: 30, size: 1000)
        let bulky = info("PiMPAudioStudio-AUDIODIAG-bulky.log", daysAgo: 2, size: 50000)

        let deleted = AudioDiagnostics.selectLogFilesToDelete(
            candidates: [active, recent, old, bulky],
            keepPaths: [active.url.path],
            maxAgeDays: 14,
            maxFiles: 2,
            maxTotalBytes: 10000,
            now: now
        )
        let deletedNames = Set(deleted.map(\.url.lastPathComponent))
        #expect(deletedNames.contains("PiMPAudioStudio-AUDIODIAG-old.log"))
        #expect(deletedNames.contains("PiMPAudioStudio-AUDIODIAG-bulky.log"))
        #expect(!deletedNames.contains("PiMPAudioStudio-AUDIODIAG-active.log"))
        #expect(!deletedNames.contains("PiMPAudioStudio-AUDIODIAG-recent.log"))
    }

    @Test("pruneLogFiles removes excess logs on disk")
    func pruneLogFilesRemovesExcessLogsOnDisk() throws {
        let fm = FileManager.default
        let dir = fm.temporaryDirectory
            .appending(path: "AudioDiagnosticsPrune-\(UUID().uuidString)", directoryHint: .isDirectory)
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: dir) }

        let now = Date()
        for i in 0 ..< 5 {
            let url = dir.appendingPathComponent(
                "PiMPAudioStudio-AUDIODIAG-2020-01-0\(i + 1)_00-00-00-000-AAAAAAA\(i).log"
            )
            try "old log \(i)".write(to: url, atomically: true, encoding: .utf8)
            try fm.setAttributes(
                [.modificationDate: #require(Calendar.current.date(byAdding: .day, value: -60, to: now))],
                ofItemAtPath: url.path
            )
        }
        let keepURL = dir.appendingPathComponent(
            "PiMPAudioStudio-AUDIODIAG-keep-me.log"
        )
        try "keep".write(to: keepURL, atomically: true, encoding: .utf8)

        let result = AudioDiagnostics.pruneLogFiles(
            in: dir,
            maxAgeDays: 14,
            maxFiles: 40,
            maxTotalBytes: 64 * 1024 * 1024,
            now: now,
            fileManager: fm
        )
        #expect(result.deletedCount == 5)
        #expect(fm.fileExists(atPath: keepURL.path))
        let remaining = try fm.contentsOfDirectory(atPath: dir.path)
            .filter { $0.hasPrefix("PiMPAudioStudio-AUDIODIAG-") }
        #expect(remaining.count == 1)
        #expect(remaining[0] == keepURL.lastPathComponent)
    }

    private func restore(_ value: Any?, forKey key: String, in defaults: UserDefaults) {
        if let value {
            defaults.set(value, forKey: key)
        } else {
            defaults.removeObject(forKey: key)
        }
    }
}
