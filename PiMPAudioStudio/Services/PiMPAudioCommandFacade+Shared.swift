import AVFoundation
import Foundation
import PiMPAudioCore

extension PiMPAudioCommandFacade {
    func stringValue(_ value: Any?) -> String? {
        switch value {
        case is NSNull:
            nil
        case let value as String:
            value
        case let value as UUID:
            value.uuidString
        case let value as CustomStringConvertible:
            value.description
        default:
            nil
        }
    }

    func boolValue(_ value: Any?) -> Bool? {
        switch value {
        case let value as Bool:
            value
        case let value as NSNumber:
            value.boolValue
        case let value as String:
            switch value.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
            case "true", "yes", "1":
                true
            case "false", "no", "0":
                false
            default:
                nil
            }
        default:
            nil
        }
    }

    func integerValue(_ value: Any?) -> Int? {
        switch value {
        case let value as Int:
            value
        case let value as NSNumber:
            value.intValue
        case let value as String:
            Int(value.trimmingCharacters(in: .whitespacesAndNewlines))
        default:
            nil
        }
    }

    func hasArgument(_ key: String, in arguments: [String: Any]) -> Bool {
        arguments.keys.contains(key)
    }

    func doubleValue(_ value: Any?) -> Double? {
        switch value {
        case let value as Double:
            value
        case let value as Float:
            Double(value)
        case let value as Int:
            Double(value)
        case let value as NSNumber:
            value.doubleValue
        case let value as String:
            Double(value.trimmingCharacters(in: .whitespacesAndNewlines))
        default:
            nil
        }
    }

    func stringArrayValue(_ value: Any?) -> [String]? {
        switch value {
        case is NSNull:
            nil
        case let values as [String]:
            values
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
        case let values as [Any]:
            values
                .compactMap(stringValue)
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
        case let raw as String:
            raw
                .split { character in
                    character == "," || character == "\n" || character == ";"
                }
                .map { String($0).trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
        default:
            nil
        }
    }

    func requiredString(_ key: String, in arguments: [String: Any]) throws -> String {
        let value = stringValue(arguments[key])?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !value.isEmpty else {
            throw PiMPAudioCommandFacadeError.missingRequiredArgument(key)
        }
        return value
    }

    func requiredString(
        _ key: String,
        aliases: [String],
        in arguments: [String: Any]
    ) throws -> String {
        for candidate in [key] + aliases {
            let value = stringValue(arguments[candidate])?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            if !value.isEmpty {
                return value
            }
        }
        throw PiMPAudioCommandFacadeError.missingRequiredArgument(key)
    }

    func requiredDuration(in arguments: [String: Any]) throws -> Double {
        guard let duration = doubleValue(arguments["duration"] ?? arguments["durationSeconds"]),
              duration > 0 else {
            throw PiMPAudioCommandFacadeError.invalidDuration
        }
        return duration
    }

    func generationMode(from value: Any?) -> GenerationMode? {
        guard let rawValue = stringValue(value) else { return nil }
        if let mode = GenerationMode(rawValue: rawValue) {
            return mode
        }
        return GenerationMode(fromClassification: rawValue)
    }

    func pipelineChoice(from value: Any?) -> PipelineChoice? {
        guard let rawValue = stringValue(value)?.trimmingCharacters(in: .whitespacesAndNewlines) else {
            return nil
        }
        if let pipeline = PipelineChoice(rawValue: rawValue) {
            return pipeline
        }
        switch rawValue.lowercased() {
        case "music", "acestep", "ace-step", "ace_step":
            return .aceStep
        case "sfx", "soundfx", "sound_fx", "sound fx", "stableaudio", "stable-audio", "stable_audio":
            return .stableAudio
        default:
            return nil
        }
    }

    func inferMethod(from value: Any?) -> InferMethod? {
        guard let rawValue = stringValue(value)?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() else {
            return nil
        }
        return InferMethod(rawValue: rawValue)
    }

    func precision(from value: Any?) -> Precision? {
        guard let rawValue = stringValue(value)?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() else {
            return nil
        }
        return Precision(rawValue: rawValue)
    }

    func guidancePreset(from value: Any?) -> GuidancePreset? {
        guard let rawValue = stringValue(value)?.trimmingCharacters(in: .whitespacesAndNewlines) else {
            return nil
        }
        if let preset = GuidancePreset(rawValue: rawValue) {
            return preset
        }
        let compact = rawValue
            .lowercased()
            .replacingOccurrences(of: " ", with: "")
            .replacingOccurrences(of: "-", with: "")
        switch compact {
        case "constant":
            return .constant
        case "lateemphasis", "late":
            return .lateEmphasis
        case "midtolate", "midlate":
            return .midToLate
        case "softlate", "soft":
            return .softLate
        default:
            return nil
        }
    }

    func normalizedText(_ text: String) -> String {
        text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    func lyricLineCount(_ lyrics: String) -> Int {
        lyrics
            .components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty && !$0.hasPrefix("[") }
            .count
    }

    func progressFraction(current: Int, total: Int) -> Double {
        guard total > 0 else { return 0 }
        return min(1, max(0, Double(current) / Double(total)))
    }

    func diagnosticsStatusPayload() -> [String: Any] {
        [
            "enabled": AudioDiagnostics.isEnabled,
            "directory": (AudioDiagnostics.configuredLogDirectoryURL ?? AppStorageLocations.diagnosticsDirectory)
                .path(percentEncoded: false),
            "activeLogPath": AudioDiagnostics.activeLogFileURL?.path(percentEncoded: false) ?? NSNull(),
            "latestLogPath": latestDiagnosticLogURL()?.path(percentEncoded: false) ?? NSNull(),
        ]
    }

    func resolveDiagnosticLogURL(arguments: [String: Any]) throws -> URL {
        let diagnosticsDirectory = AudioDiagnostics.configuredLogDirectoryURL
            ?? AppStorageLocations.diagnosticsDirectory

        if let path = stringValue(arguments["path"] ?? arguments["logPath"])?
            .trimmingCharacters(in: .whitespacesAndNewlines),
            !path.isEmpty {
            let url = URL(fileURLWithPath: path)
            guard isInDiagnosticsDirectory(url) else {
                throw PiMPAudioCommandFacadeError.resourceNotFound(path)
            }
            guard FileManager.default.fileExists(atPath: url.path(percentEncoded: false)) else {
                throw PiMPAudioCommandFacadeError.resourceNotFound(path)
            }
            return url
        }

        if let fileName = stringValue(arguments["fileName"] ?? arguments["name"])?
            .trimmingCharacters(in: .whitespacesAndNewlines),
            !fileName.isEmpty {
            let url = diagnosticsDirectory.appendingPathComponent(fileName)
            guard FileManager.default.fileExists(atPath: url.path(percentEncoded: false)) else {
                throw PiMPAudioCommandFacadeError.resourceNotFound(fileName)
            }
            return url
        }

        if let activeURL = AudioDiagnostics.activeLogFileURL,
           FileManager.default.fileExists(atPath: activeURL.path(percentEncoded: false)) {
            return activeURL
        }

        if let latestURL = latestDiagnosticLogURL() {
            return latestURL
        }

        throw PiMPAudioCommandFacadeError.resourceNotFound("latest diagnostics log")
    }

    func latestDiagnosticLogURL() -> URL? {
        let diagnosticsDirectory = AudioDiagnostics.configuredLogDirectoryURL
            ?? AppStorageLocations.diagnosticsDirectory
        guard let contents = try? FileManager.default.contentsOfDirectory(
            at: diagnosticsDirectory,
            includingPropertiesForKeys: [.contentModificationDateKey, .isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            return nil
        }

        return contents
            .filter { url in
                let values = try? url.resourceValues(forKeys: [.isRegularFileKey])
                return values?.isRegularFile == true
                    && url.lastPathComponent.hasPrefix("PiMPAudioStudio-AUDIODIAG-")
                    && url.pathExtension == "log"
            }
            .sorted { modificationDate($0) > modificationDate($1) }
            .first
    }

    func tailLines(
        from url: URL,
        limit: Int,
        containing filter: String?
    ) throws -> [String] {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }

        let fileSize = try handle.seekToEnd()
        let maxBytes: UInt64 = 1_048_576
        let readSize = min(fileSize, maxBytes)
        try handle.seek(toOffset: fileSize - readSize)
        let data = handle.readDataToEndOfFile()

        var text = String(decoding: data, as: UTF8.self)
        if readSize < fileSize, let firstNewline = text.firstIndex(of: "\n") {
            text = String(text[text.index(after: firstNewline)...])
        }

        let allLines = text
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map(String.init)
        let filteredLines: [String] = if let filter, !filter.isEmpty {
            allLines.filter {
                $0.localizedCaseInsensitiveContains(filter)
            }
        } else {
            allLines
        }
        return Array(filteredLines.suffix(limit))
    }

    func isInDiagnosticsDirectory(_ url: URL) -> Bool {
        let diagnosticsDirectory = (AudioDiagnostics.configuredLogDirectoryURL
            ?? AppStorageLocations.diagnosticsDirectory)
            .standardizedFileURL
            .path(percentEncoded: false)
        let candidate = url.standardizedFileURL.path(percentEncoded: false)
        return candidate == diagnosticsDirectory
            || candidate.hasPrefix(diagnosticsDirectory + "/")
    }

    func modificationDate(_ url: URL) -> Date {
        (try? url.resourceValues(forKeys: [.contentModificationDateKey]))?
            .contentModificationDate ?? .distantPast
    }

    func increment(_ dictionary: inout [String: Int], key: String) {
        dictionary[key, default: 0] += 1
    }

    func compact(_ text: String, limit: Int) -> String {
        let oneLine = text
            .replacingOccurrences(of: "\n", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard oneLine.count > limit else { return oneLine }
        return String(oneLine.prefix(max(0, limit - 3))) + "..."
    }

    func formatGiB(_ bytes: UInt64) -> String {
        String(format: "%.1f GiB", Double(bytes) / 1_073_741_824)
    }

    func resource(
        uri: String,
        name: String,
        title: String,
        description: String
    ) -> [String: Any] {
        [
            "uri": uri,
            "name": name,
            "title": title,
            "description": description,
            "mimeType": "application/json",
        ]
    }

    func resourceURI(kind: String, identifier: String? = nil) -> String {
        guard let identifier else { return "\(AppRuntimeIdentity.mcpScheme)://\(kind)" }
        let encoded = identifier.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? identifier
        return "\(AppRuntimeIdentity.mcpScheme)://\(kind)/\(encoded)"
    }

    func confirmationRequirementPayload(toolName: String) -> [String: Any] {
        [
            "required": true,
            "confirmArgument": "confirm",
            "confirmValue": true,
            "supportedModes": [
                "confirmArgument",
                "elicitation/form",
            ],
            "fallback": "Clients without MCP elicitation support must call \(toolName) with confirm: true after explicit user approval.",
        ]
    }
}
