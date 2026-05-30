import Foundation
import PiMPAudioCore

struct SourceSeparationOutput: Equatable {
    let backend: String
    let model: String
    let backendVersion: String
    let stems: [String: URL]
    let warnings: [String]
}

final class SourceSeparationService {
    let separator: DemucsMLXSourceSeparator

    init(separator: DemucsMLXSourceSeparator = DemucsMLXSourceSeparator()) {
        self.separator = separator
    }

    func separate(
        inputURL: URL,
        outputDirectory: URL,
        force: Bool = false
    ) async throws -> SourceSeparationOutput {
        try await separator.separate(
            inputURL: inputURL,
            outputDirectory: outputDirectory,
            force: force
        )
    }
}

struct DemucsMLXSourceSeparator {
    static let backendName = DemucsStemSeparator.backendName
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
        let environment = ProcessInfo.processInfo.environment
        let configured = environment["PIMP_AUDIO_DEMUCS_MODEL_NAME"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return configured?.isEmpty == false ? configured! : defaultModelName
    }

    private static func configuredModelDirectory() -> URL? {
        let environment = ProcessInfo.processInfo.environment
        guard let configured = environment["PIMP_AUDIO_DEMUCS_MODEL_DIR"]?
            .trimmingCharacters(in: .whitespacesAndNewlines),
            !configured.isEmpty else {
            return nil
        }
        return URL(fileURLWithPath: (configured as NSString).expandingTildeInPath, isDirectory: true)
    }
}

private func removeIfExists(_ url: URL) throws {
    if FileManager.default.fileExists(atPath: url.path) {
        try FileManager.default.removeItem(at: url)
    }
}
