import Foundation
@preconcurrency import DemucsMLX

public struct DemucsStemSeparationOutput: Sendable, Equatable {
    public let backend: String
    public let model: String
    public let backendVersion: String
    public let vocalsURL: URL
    public let instrumentalURL: URL

    public init(
        backend: String,
        model: String,
        backendVersion: String,
        vocalsURL: URL,
        instrumentalURL: URL
    ) {
        self.backend = backend
        self.model = model
        self.backendVersion = backendVersion
        self.vocalsURL = vocalsURL
        self.instrumentalURL = instrumentalURL
    }
}

public enum DemucsStemSeparatorError: LocalizedError {
    case invalidOutput(String)

    public var errorDescription: String? {
        switch self {
        case .invalidOutput(let message):
            return "Demucs stem output is invalid: \(message)"
        }
    }
}

public struct DemucsStemSeparator: Sendable {
    public static let backendName = "demucs-mlx-swift"
    public static let defaultModelName = "htdemucs"

    public let modelName: String
    public let modelDirectory: URL?
    public let shifts: Int
    public let overlap: Float
    public let batchSize: Int

    public init(
        modelName: String = DemucsStemSeparator.defaultModelName,
        modelDirectory: URL? = nil,
        shifts: Int = 1,
        overlap: Float = 0.25,
        batchSize: Int = 1
    ) {
        self.modelName = modelName
        self.modelDirectory = modelDirectory
        self.shifts = shifts
        self.overlap = overlap
        self.batchSize = batchSize
    }

    public func separateVocalsAndInstrumental(
        inputURL: URL,
        outputDirectory: URL,
        force: Bool = false
    ) async throws -> DemucsStemSeparationOutput {
        try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)
        let vocalURL = outputDirectory.appendingPathComponent("vocals.wav")
        let instrumentalURL = outputDirectory.appendingPathComponent("instrumental.wav")
        if force {
            try Self.removeIfExists(vocalURL)
            try Self.removeIfExists(instrumentalURL)
        }

        let modelName = modelName
        let modelDirectory = modelDirectory
        let parameters = DemucsSeparationParameters(
            shifts: shifts,
            overlap: overlap,
            split: true,
            segmentSeconds: nil,
            batchSize: batchSize,
            seed: nil
        )

        let result = try await Task.detached(priority: .userInitiated) {
            try Task.checkCancellation()
            let separator = try DemucsSeparator(
                modelName: modelName,
                parameters: parameters,
                modelDirectory: modelDirectory
            )
            return try separator.separate(fileAt: inputURL)
        }.value

        try Task.checkCancellation()
        guard let vocals = result.stems["vocals"] else {
            throw DemucsStemSeparatorError.invalidOutput("result did not include a vocals stem")
        }
        let instrumental = try Self.instrumentalComplement(input: result.input, vocals: vocals)

        try Self.writeAudioReplacingExisting(vocals, to: vocalURL)
        try Self.writeAudioReplacingExisting(instrumental, to: instrumentalURL)

        return DemucsStemSeparationOutput(
            backend: Self.backendName,
            model: modelName,
            backendVersion: "DemucsMLX",
            vocalsURL: vocalURL,
            instrumentalURL: instrumentalURL
        )
    }

    private static func instrumentalComplement(input: DemucsAudio, vocals: DemucsAudio) throws -> DemucsAudio {
        let mixSamples = input.channelMajorSamples
        let vocalSamples = vocals.channelMajorSamples
        guard mixSamples.count == vocalSamples.count else {
            throw DemucsStemSeparatorError.invalidOutput(
                "mix sample count \(mixSamples.count) did not match vocals sample count \(vocalSamples.count)"
            )
        }

        var complementSamples = [Float](repeating: 0, count: mixSamples.count)
        for index in mixSamples.indices {
            complementSamples[index] = mixSamples[index] - vocalSamples[index]
        }
        return try DemucsAudio(
            channelMajor: complementSamples,
            channels: vocals.channels,
            sampleRate: vocals.sampleRate
        )
    }

    private static func writeAudioReplacingExisting(_ audio: DemucsAudio, to url: URL) throws {
        try removeIfExists(url)
        try AudioIO.writeAudio(audio, to: url, format: .wav(bitDepth: .int16))
    }

    private static func removeIfExists(_ url: URL) throws {
        if FileManager.default.fileExists(atPath: url.path) {
            try FileManager.default.removeItem(at: url)
        }
    }
}
