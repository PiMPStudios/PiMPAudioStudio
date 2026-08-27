import Foundation
import PiMPAudioCore

struct DatasetLyricsTranscriptionArtifact: Equatable, Sendable {
    let text: String
    let source: String
    let backend: String
    let model: String
    let alignerModel: String?
    let lyricsURL: URL
    let timedLyricsURL: URL
    let lrcURL: URL
    let wordCount: Int
    let elapsedSeconds: Double
}

protocol DatasetLyricsTranscribing: Sendable {
    func transcribe(
        vocalsURL: URL,
        outputDirectory: URL,
        force: Bool
    ) async throws -> DatasetLyricsTranscriptionArtifact
}

protocol DatasetLyricsTranscriptionEvicting: Sendable {
    func evictModels() async
}

protocol DatasetLyricsAligning: Sendable {
    func align(
        lyrics: String,
        audioURL: URL,
        outputDirectory: URL,
        force: Bool
    ) async throws -> DatasetLyricsTranscriptionArtifact
}

actor LyricsTranscriptionService:
    DatasetLyricsTranscribing,
    DatasetLyricsTranscriptionEvicting,
    DatasetLyricsAligning {
    private let transcriber: QwenLyricsTranscriber
    private let options: LyricsTranscriptionOptions

    init(
        transcriber: QwenLyricsTranscriber = QwenLyricsTranscriber(),
        options: LyricsTranscriptionOptions = .qualityDataset
    ) {
        self.transcriber = transcriber
        self.options = options
    }

    func evictModels() async {
        await transcriber.evictModels()
    }

    func transcribe(
        vocalsURL: URL,
        outputDirectory: URL,
        force: Bool
    ) async throws -> DatasetLyricsTranscriptionArtifact {
        try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)

        let lyricsURL = outputDirectory.appendingPathComponent("lyrics.txt")
        let timedLyricsURL = outputDirectory.appendingPathComponent("lyrics.timed.json")
        let lrcURL = outputDirectory.appendingPathComponent("lyrics.lrc")
        if !force,
           FileManager.default.fileExists(atPath: lyricsURL.path(percentEncoded: false)),
           let cachedText = try? String(contentsOf: lyricsURL, encoding: .utf8),
           !cachedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            let text = cachedText.trimmingCharacters(in: .whitespacesAndNewlines)
            if !FileManager.default.fileExists(atPath: timedLyricsURL.path(percentEncoded: false)) {
                try Self.writeJSONObject(
                    cachedTimedLyricsPayload(text: text, vocalsURL: vocalsURL),
                    to: timedLyricsURL
                )
            }
            if !FileManager.default.fileExists(atPath: lrcURL.path(percentEncoded: false)) {
                try Self.lrcText(from: text).write(to: lrcURL, atomically: true, encoding: .utf8)
            }
            return DatasetLyricsTranscriptionArtifact(
                text: text,
                source: "cached",
                backend: QwenLyricsTranscriber.backendName,
                model: options.modelID,
                alignerModel: options.alignerModelID,
                lyricsURL: lyricsURL,
                timedLyricsURL: timedLyricsURL,
                lrcURL: lrcURL,
                wordCount: text.split(whereSeparator: \.isWhitespace).count,
                elapsedSeconds: 0
            )
        }

        let result = try await transcriber.transcribe(audioURL: vocalsURL, options: options)
        try result.text.write(to: lyricsURL, atomically: true, encoding: .utf8)
        try Self.writeJSONObject(
            timedLyricsPayload(result: result, vocalsURL: vocalsURL),
            to: timedLyricsURL
        )
        try Self.lrcText(from: result).write(to: lrcURL, atomically: true, encoding: .utf8)

        return DatasetLyricsTranscriptionArtifact(
            text: result.text,
            source: "qwen3-asr",
            backend: result.backend,
            model: result.modelID,
            alignerModel: result.alignerModelID,
            lyricsURL: lyricsURL,
            timedLyricsURL: timedLyricsURL,
            lrcURL: lrcURL,
            wordCount: result.words.count,
            elapsedSeconds: result.elapsedSeconds
        )
    }

    func align(
        lyrics: String,
        audioURL: URL,
        outputDirectory: URL,
        force: Bool
    ) async throws -> DatasetLyricsTranscriptionArtifact {
        try FileManager.default.createDirectory(
            at: outputDirectory,
            withIntermediateDirectories: true
        )
        let lyricsURL = outputDirectory.appendingPathComponent("lyrics.txt")
        let timedLyricsURL = outputDirectory.appendingPathComponent(
            "lyrics.timed.json"
        )
        let lrcURL = outputDirectory.appendingPathComponent("lyrics.lrc")
        if !force,
           FileManager.default.fileExists(
               atPath: timedLyricsURL.path(percentEncoded: false)
           ),
           FileManager.default.fileExists(
               atPath: lrcURL.path(percentEncoded: false)
           ) {
            if !FileManager.default.fileExists(
                atPath: lyricsURL.path(percentEncoded: false)
            ) {
                try lyrics.write(
                    to: lyricsURL,
                    atomically: true,
                    encoding: .utf8
                )
            }
            return DatasetLyricsTranscriptionArtifact(
                text: lyrics,
                source: "cached-alignment",
                backend: QwenLyricsTranscriber.alignerBackendName,
                model: options.alignerModelID ?? "",
                alignerModel: options.alignerModelID,
                lyricsURL: lyricsURL,
                timedLyricsURL: timedLyricsURL,
                lrcURL: lrcURL,
                wordCount: lyrics.split(whereSeparator: \.isWhitespace).count,
                elapsedSeconds: 0
            )
        }

        let result = try await transcriber.align(
            audioURL: audioURL,
            text: lyrics,
            options: options
        )
        let compatibleResult = LyricsTranscriptionResult(
            backend: result.backend,
            modelID: result.modelID,
            alignerModelID: result.modelID,
            text: result.text,
            words: result.words,
            sampleRate: result.sampleRate,
            trim: result.trim,
            elapsedSeconds: result.elapsedSeconds
        )
        try result.text.write(
            to: lyricsURL,
            atomically: true,
            encoding: .utf8
        )
        try Self.writeJSONObject(
            timedLyricsPayload(
                result: compatibleResult,
                vocalsURL: audioURL
            ),
            to: timedLyricsURL
        )
        try Self.lrcText(from: compatibleResult).write(
            to: lrcURL,
            atomically: true,
            encoding: .utf8
        )
        return DatasetLyricsTranscriptionArtifact(
            text: result.text,
            source: "qwen3-forced-alignment",
            backend: result.backend,
            model: result.modelID,
            alignerModel: result.modelID,
            lyricsURL: lyricsURL,
            timedLyricsURL: timedLyricsURL,
            lrcURL: lrcURL,
            wordCount: result.words.count,
            elapsedSeconds: result.elapsedSeconds
        )
    }

    private func timedLyricsPayload(result: LyricsTranscriptionResult, vocalsURL: URL) -> [String: Any] {
        var payload: [String: Any] = [
            "pimp_audio_lyrics_transcription_version": "1.0",
            "generated_at": ISO8601DateFormatter().string(from: Date()),
            "source_vocals": vocalsURL.lastPathComponent,
            "backend": result.backend,
            "model": result.modelID,
            "text": result.text,
            "sample_rate": result.sampleRate,
            "elapsed_seconds": Self.rounded(result.elapsedSeconds, places: 3),
            "trim": [
                "leading_seconds": Self.rounded(result.trim.leadingSeconds, places: 3),
                "trailing_seconds": Self.rounded(result.trim.trailingSeconds, places: 3),
            ],
            "words": result.words.map { word in
                [
                    "text": word.text,
                    "start": Self.rounded(word.startTime, places: 3),
                    "end": Self.rounded(word.endTime, places: 3),
                ]
            },
        ]
        if let alignerModelID = result.alignerModelID {
            payload["aligner_model"] = alignerModelID
        }
        return payload
    }

    private func cachedTimedLyricsPayload(text: String, vocalsURL: URL) -> [String: Any] {
        [
            "pimp_audio_lyrics_transcription_version": "1.0",
            "generated_at": ISO8601DateFormatter().string(from: Date()),
            "source_vocals": vocalsURL.lastPathComponent,
            "backend": QwenLyricsTranscriber.backendName,
            "model": options.modelID,
            "source": "cached",
            "text": text,
            "sample_rate": options.sampleRate,
            "elapsed_seconds": 0,
            "trim": [
                "leading_seconds": 0,
                "trailing_seconds": 0,
            ],
            "words": [],
        ]
    }

    private static func lrcText(from result: LyricsTranscriptionResult) -> String {
        guard !result.words.isEmpty else {
            let text = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
            return lrcText(from: text)
        }

        var lines: [String] = []
        var currentWords: [LyricsTranscriptionWord] = []
        for word in result.words {
            currentWords.append(word)
            let shouldBreak = currentWords.count >= 8 || word.text
                .range(of: #"[.!?]$"#, options: .regularExpression) != nil
            if shouldBreak {
                lines.append(lrcLine(from: currentWords))
                currentWords.removeAll(keepingCapacity: true)
            }
        }
        if !currentWords.isEmpty {
            lines.append(lrcLine(from: currentWords))
        }
        return lines.joined(separator: "\n") + "\n"
    }

    private static func lrcText(from text: String) -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? "" : "[00:00.00]\(trimmed)\n"
    }

    private static func lrcLine(from words: [LyricsTranscriptionWord]) -> String {
        let timestamp = lrcTimestamp(words.first?.startTime ?? 0)
        let text = words.map(\.text).joined(separator: " ")
        return "[\(timestamp)]\(text)"
    }

    private static func lrcTimestamp(_ seconds: Double) -> String {
        let clamped = max(0, seconds)
        let minutes = Int(clamped / 60)
        let remaining = clamped - Double(minutes * 60)
        return String(format: "%02d:%05.2f", minutes, remaining)
    }

    private static func writeJSONObject(_ object: [String: Any], to url: URL) throws {
        let data = try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: url, options: .atomic)
    }

    private static func rounded(_ value: Double, places: Int) -> Double {
        let scale = pow(10, Double(places))
        return (value * scale).rounded() / scale
    }
}
