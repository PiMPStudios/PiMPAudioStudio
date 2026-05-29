import Foundation
import AnvilCore

enum SourceSeparationBackend: String, CaseIterable, Equatable {
    case nativeDemucs = "demucs-mlx-swift"
    case externalAudioSeparator = "audio-separator"
}

struct SourceSeparationOutput: Equatable {
    let backend: String
    let model: String
    let backendVersion: String
    let stems: [String: URL]
    let warnings: [String]

    func addingWarnings(_ additionalWarnings: [String]) -> SourceSeparationOutput {
        SourceSeparationOutput(
            backend: backend,
            model: model,
            backendVersion: backendVersion,
            stems: stems,
            warnings: additionalWarnings + warnings
        )
    }
}

enum SourceSeparationError: LocalizedError {
    case allBackendsFailed([String])
    case invalidStemOutput(String)

    var errorDescription: String? {
        switch self {
        case .allBackendsFailed(let failures):
            let details = failures.joined(separator: "\n")
            return "No source-separation backend completed successfully.\n\(details)"
        case .invalidStemOutput(let message):
            return "Stem separation output is invalid: \(message)"
        }
    }
}

final class SourceSeparationService {
    static let defaultBackends: [SourceSeparationBackend] = [
        .nativeDemucs,
        .externalAudioSeparator,
    ]

    let backends: [SourceSeparationBackend]

    init(backends: [SourceSeparationBackend] = SourceSeparationService.defaultBackends) {
        self.backends = backends
    }

    func separate(
        inputURL: URL,
        outputDirectory: URL,
        force: Bool = false
    ) async throws -> SourceSeparationOutput {
        var failures: [String] = []

        for backend in backends {
            try Task.checkCancellation()
            do {
                var output: SourceSeparationOutput
                switch backend {
                case .nativeDemucs:
                    output = try await DemucsMLXSourceSeparator().separate(
                        inputURL: inputURL,
                        outputDirectory: outputDirectory,
                        force: force
                    )
                case .externalAudioSeparator:
                    output = try await ExternalAudioSeparatorSourceSeparator().separate(
                        inputURL: inputURL,
                        outputDirectory: outputDirectory,
                        force: force
                    )
                }
                return output.addingWarnings(failures.map { "fallback after failure: \($0)" })
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                failures.append("\(backend.rawValue): \(error.localizedDescription)")
            }
        }

        throw SourceSeparationError.allBackendsFailed(failures)
    }
}

struct DemucsMLXSourceSeparator {
    static let backendName = SourceSeparationBackend.nativeDemucs.rawValue
    static let defaultModelName = DemucsStemSeparator.defaultModelName

    let modelName: String
    let modelDirectory: URL?

    init(
        modelName: String = DemucsMLXSourceSeparator.configuredModelName(),
        modelDirectory: URL? = DemucsMLXSourceSeparator.configuredModelDirectory()
    ) {
        self.modelName = modelName
        self.modelDirectory = modelDirectory
    }

    func separate(
        inputURL: URL,
        outputDirectory: URL,
        force: Bool
    ) async throws -> SourceSeparationOutput {
        try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)
        let vocalURL = outputDirectory.appendingPathComponent("vocals.wav")
        let instrumentalURL = outputDirectory.appendingPathComponent("instrumental.wav")
        if force {
            try removeIfExists(vocalURL)
            try removeIfExists(instrumentalURL)
        }

        let result = try await DemucsStemSeparator(
            modelName: modelName,
            modelDirectory: modelDirectory
        )
        .separateVocalsAndInstrumental(
            inputURL: inputURL,
            outputDirectory: outputDirectory,
            force: force
        )

        return SourceSeparationOutput(
            backend: result.backend,
            model: result.model,
            backendVersion: result.backendVersion,
            stems: [
                "vocals": result.vocalsURL,
                "instrumental": result.instrumentalURL,
            ],
            warnings: []
        )
    }

    private static func configuredModelName() -> String {
        let configured = ProcessInfo.processInfo.environment["ANVIL_DEMUCS_MODEL_NAME"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return configured?.isEmpty == false ? configured! : defaultModelName
    }

    private static func configuredModelDirectory() -> URL? {
        guard let configured = ProcessInfo.processInfo.environment["ANVIL_DEMUCS_MODEL_DIR"]?
            .trimmingCharacters(in: .whitespacesAndNewlines),
            !configured.isEmpty else {
            return nil
        }
        return URL(fileURLWithPath: (configured as NSString).expandingTildeInPath, isDirectory: true)
    }
}

struct ExternalAudioSeparatorSourceSeparator {
    static let backendName = SourceSeparationBackend.externalAudioSeparator.rawValue

    func separate(
        inputURL: URL,
        outputDirectory: URL,
        force: Bool
    ) async throws -> SourceSeparationOutput {
        let executableURL = try Self.resolveExecutableURL()
        let backendVersion = Self.version(executableURL: executableURL) ?? "unknown"
        let tempOutputDirectory = outputDirectory.appendingPathComponent(".audio_separator_tmp", isDirectory: true)
        if FileManager.default.fileExists(atPath: tempOutputDirectory.path) {
            try FileManager.default.removeItem(at: tempOutputDirectory)
        }
        try FileManager.default.createDirectory(at: tempOutputDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempOutputDirectory) }

        let command = Self.command(
            executableURL: executableURL,
            inputURL: inputURL,
            outputDirectory: tempOutputDirectory
        )
        let runner = LocalDatasetCapturedProcess(
            command: command,
            environment: Self.environment(executableURL: executableURL)
        )
        _ = try await runner.run()

        let stems = try Self.canonicalizeStemOutputs(
            from: tempOutputDirectory,
            to: outputDirectory
        )
        return SourceSeparationOutput(
            backend: Self.backendName,
            model: "auto",
            backendVersion: backendVersion,
            stems: stems,
            warnings: []
        )
    }

    static func command(
        executableURL: URL,
        inputURL: URL,
        outputDirectory: URL
    ) -> LocalDatasetEnrichmentCommand {
        LocalDatasetEnrichmentCommand(
            step: .separatingStems,
            executableURL: executableURL,
            arguments: [
                inputURL.path,
                "--output_dir",
                outputDirectory.path,
                "--output_format",
                "WAV",
            ]
        )
    }

    static func resolveExecutableURL(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> URL {
        for url in candidateExecutableURLs(environment: environment)
            where FileManager.default.isExecutableFile(atPath: url.path) {
            return url
        }
        throw LocalDatasetEnrichmentError.missingAudioSeparatorCLI
    }

    static func candidateExecutableURLs(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> [URL] {
        var candidates: [URL] = []
        if let configured = environment["ANVIL_AUDIO_SEPARATOR_BIN"]?.trimmingCharacters(in: .whitespacesAndNewlines),
           !configured.isEmpty {
            candidates.append(URL(fileURLWithPath: (configured as NSString).expandingTildeInPath))
        }

        candidates.append(contentsOf: searchPath(environment: environment).map {
            URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath)
                .appendingPathComponent("audio-separator")
        })
        return candidates.deduplicatedByPath()
    }

    static func version(executableURL: URL) -> String? {
        let process = Process()
        let pipe = Pipe()
        process.executableURL = executableURL
        process.arguments = ["--version"]
        process.standardOutput = pipe
        process.standardError = Pipe()
        process.environment = environment(executableURL: executableURL)
        do {
            try process.run()
            process.waitUntilExit()
        } catch {
            return nil
        }
        guard process.terminationStatus == 0 else { return nil }
        return String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func environment(
        executableURL: URL,
        base: [String: String] = ProcessInfo.processInfo.environment
    ) -> [String: String] {
        var environment = base
        environment["PATH"] = ([executableURL.deletingLastPathComponent().path] + searchPath(environment: base))
            .deduplicated()
            .joined(separator: ":")
        environment["PYTHONUNBUFFERED"] = "1"
        environment["NO_COLOR"] = "1"
        environment["CLICOLOR"] = "0"
        environment["TERM"] = "dumb"
        return environment
    }

    private static func searchPath(environment: [String: String]) -> [String] {
        let pathParts = (environment["PATH"] ?? "")
            .split(separator: ":")
            .map(String.init)
        return [
            "/opt/homebrew/bin",
            "/usr/local/bin",
            "/opt/local/bin",
            "/usr/bin",
            "/bin",
            "/usr/sbin",
            "/sbin",
        ] + pathParts
    }

    private static func canonicalizeStemOutputs(
        from tempOutputDirectory: URL,
        to outputDirectory: URL
    ) throws -> [String: URL] {
        let files = (try? FileManager.default.contentsOfDirectory(
            at: tempOutputDirectory,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        )) ?? []

        guard let vocals = findStemFile(named: "vocals", in: files),
              let instrumental = findStemFile(named: "instrumental", in: files) else {
            let names = files.map(\.lastPathComponent).joined(separator: ", ")
            throw LocalDatasetEnrichmentError.noStemOutput(names.isEmpty ? "no files were written" : names)
        }

        try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)
        let targets = [
            "vocals": outputDirectory.appendingPathComponent("vocals.wav"),
            "instrumental": outputDirectory.appendingPathComponent("instrumental.wav"),
        ]
        try copyStem(vocals, to: targets["vocals"]!)
        try copyStem(instrumental, to: targets["instrumental"]!)
        return targets
    }

    private static func findStemFile(named stem: String, in files: [URL]) -> URL? {
        let aliases: [String]
        switch stem {
        case "vocals":
            aliases = ["vocals", "vocal", "voice"]
        case "instrumental":
            aliases = ["instrumental", "inst", "no_vocals", "no-vocals"]
        default:
            aliases = [stem]
        }

        return files.first { file in
            let base = file.deletingPathExtension().lastPathComponent.lowercased()
            return aliases.contains { base.contains($0) }
        }
    }

    private static func copyStem(_ source: URL, to target: URL) throws {
        try removeIfExists(target)
        try FileManager.default.copyItem(at: source, to: target)
    }
}

private func removeIfExists(_ url: URL) throws {
    if FileManager.default.fileExists(atPath: url.path) {
        try FileManager.default.removeItem(at: url)
    }
}

private extension Array where Element == URL {
    func deduplicatedByPath() -> [URL] {
        var seen = Set<String>()
        return filter { url in
            seen.insert(url.standardizedFileURL.path).inserted
        }
    }
}

private extension Array where Element == String {
    func deduplicated() -> [String] {
        var seen = Set<String>()
        return filter { seen.insert($0).inserted }
    }
}
