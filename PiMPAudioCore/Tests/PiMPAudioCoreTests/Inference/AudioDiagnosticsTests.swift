import Foundation
import Testing
@testable import PiMPAudioCore

@Suite("AudioDiagnostics", .serialized)
struct AudioDiagnosticsTests {
    @Test("Verbose diagnostics mirror notes to timestamped tmp log files")
    func verboseDiagnosticsMirrorNotesToTimestampedTmpLogFiles() throws {
        let defaults = UserDefaults.standard
        let defaultsKey = "com.pimpstudios.PiMPAudioStudio.audioDebugVerbose"
        let legacyDefaultsKey = "audioDebugVerbose"
        let originalValue = defaults.object(forKey: defaultsKey)
        let originalLegacyValue = defaults.object(forKey: legacyDefaultsKey)
        var createdURLs: [URL] = []

        defer {
            AudioDiagnostics.setVerboseLoggingEnabled(false)
            restore(originalValue, forKey: defaultsKey, in: defaults)
            restore(originalLegacyValue, forKey: legacyDefaultsKey, in: defaults)
            for url in createdURLs {
                try? FileManager.default.removeItem(at: url)
            }
        }

        defaults.removeObject(forKey: legacyDefaultsKey)
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
        #expect(firstURL.path.hasPrefix("/tmp/"))
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

    private func restore(_ value: Any?, forKey key: String, in defaults: UserDefaults) {
        if let value {
            defaults.set(value, forKey: key)
        } else {
            defaults.removeObject(forKey: key)
        }
    }
}
