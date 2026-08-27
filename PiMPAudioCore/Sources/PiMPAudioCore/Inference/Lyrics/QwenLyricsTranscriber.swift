import AudioCommon
import Foundation
import MLX
import Qwen3ASR

public struct LyricsTranscriptionOptions: Sendable, Equatable {
    public static let fastDataset = LyricsTranscriptionOptions(
        modelID: "aufklarer/Qwen3-ASR-0.6B-MLX-4bit"
    )

    public static let qualityDataset = LyricsTranscriptionOptions(
        modelID: "aufklarer/Qwen3-ASR-1.7B-MLX-8bit"
    )

    public var modelID: String
    public var alignerModelID: String?
    public var language: String?
    public var alignmentLanguage: String
    public var sampleRate: Int
    public var maxTokens: Int
    public var context: String?
    public var offlineMode: Bool
    public var trimsSilence: Bool
    public var silenceThreshold: Float

    public init(
        modelID: String,
        alignerModelID: String? = "aufklarer/Qwen3-ForcedAligner-0.6B-4bit",
        language: String? = nil,
        alignmentLanguage: String = "English",
        sampleRate: Int = 24000,
        maxTokens: Int = 4096,
        context: String? = nil,
        offlineMode: Bool = false,
        trimsSilence: Bool = true,
        silenceThreshold: Float = 0.003
    ) {
        self.modelID = modelID
        self.alignerModelID = alignerModelID
        self.language = language
        self.alignmentLanguage = alignmentLanguage
        self.sampleRate = sampleRate
        self.maxTokens = maxTokens
        self.context = context
        self.offlineMode = offlineMode
        self.trimsSilence = trimsSilence
        self.silenceThreshold = silenceThreshold
    }
}

public struct LyricsTranscriptionWord: Sendable, Equatable {
    public let text: String
    public let startTime: Double
    public let endTime: Double

    public init(text: String, startTime: Double, endTime: Double) {
        self.text = text
        self.startTime = startTime
        self.endTime = endTime
    }
}

public struct LyricsTranscriptionTrim: Sendable, Equatable {
    public let leadingSeconds: Double
    public let trailingSeconds: Double

    public init(leadingSeconds: Double, trailingSeconds: Double) {
        self.leadingSeconds = leadingSeconds
        self.trailingSeconds = trailingSeconds
    }
}

public struct LyricsTranscriptionResult: Sendable, Equatable {
    public let backend: String
    public let modelID: String
    public let alignerModelID: String?
    public let text: String
    public let words: [LyricsTranscriptionWord]
    public let sampleRate: Int
    public let trim: LyricsTranscriptionTrim
    public let elapsedSeconds: Double

    public init(
        backend: String,
        modelID: String,
        alignerModelID: String?,
        text: String,
        words: [LyricsTranscriptionWord],
        sampleRate: Int,
        trim: LyricsTranscriptionTrim,
        elapsedSeconds: Double
    ) {
        self.backend = backend
        self.modelID = modelID
        self.alignerModelID = alignerModelID
        self.text = text
        self.words = words
        self.sampleRate = sampleRate
        self.trim = trim
        self.elapsedSeconds = elapsedSeconds
    }
}

public struct LyricsAlignmentResult: Sendable, Equatable {
    public let backend: String
    public let modelID: String
    public let text: String
    public let words: [LyricsTranscriptionWord]
    public let sampleRate: Int
    public let trim: LyricsTranscriptionTrim
    public let elapsedSeconds: Double

    public init(
        backend: String,
        modelID: String,
        text: String,
        words: [LyricsTranscriptionWord],
        sampleRate: Int,
        trim: LyricsTranscriptionTrim,
        elapsedSeconds: Double
    ) {
        self.backend = backend
        self.modelID = modelID
        self.text = text
        self.words = words
        self.sampleRate = sampleRate
        self.trim = trim
        self.elapsedSeconds = elapsedSeconds
    }
}

public enum LyricsAlignmentError: LocalizedError {
    case missingAlignerModel
    case emptyText

    public var errorDescription: String? {
        switch self {
        case .missingAlignerModel:
            "A Qwen3 forced-aligner model is required."
        case .emptyText:
            "Lyrics text is required for forced alignment."
        }
    }
}

public actor QwenLyricsTranscriber {
    public static let backendName = "speech-swift/qwen3-asr"
    public static let alignerBackendName = "speech-swift/qwen3-forced-aligner"

    private var asrModel: Qwen3ASRModel?
    private var asrModelID: String?
    private var aligner: Qwen3ForcedAligner?
    private var alignerModelID: String?

    public init() {}

    public func evictModels() {
        asrModel = nil
        asrModelID = nil
        aligner = nil
        alignerModelID = nil
        Memory.clearCache()
    }

    public func transcribe(
        audioURL: URL,
        options: LyricsTranscriptionOptions = .qualityDataset
    ) async throws -> LyricsTranscriptionResult {
        let rawSamples = try AudioFileLoader.load(url: audioURL, targetSampleRate: options.sampleRate)
        let prepared = options.trimsSilence
            ? Self.trimmed(samples: rawSamples, sampleRate: options.sampleRate, threshold: options.silenceThreshold)
            : PreparedAudio(samples: rawSamples, leadingSeconds: 0, trailingSeconds: 0)

        let startedAt = Date()
        let model = try await loadASRModel(options: options)
        let text = model.transcribe(
            audio: prepared.samples,
            sampleRate: options.sampleRate,
            language: options.language,
            maxTokens: options.maxTokens,
            context: options.context
        )
        .trimmingCharacters(in: .whitespacesAndNewlines)

        let alignedWords: [LyricsTranscriptionWord]
        if let alignerModelID = options.alignerModelID,
           !alignerModelID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
           !text.isEmpty {
            let aligner = try await loadAligner(modelID: alignerModelID, offlineMode: options.offlineMode)
            alignedWords = aligner.alignLong(
                audio: prepared.samples,
                text: text,
                sampleRate: options.sampleRate,
                language: options.alignmentLanguage
            )
            .map {
                LyricsTranscriptionWord(
                    text: $0.text,
                    startTime: Double($0.startTime) + prepared.leadingSeconds,
                    endTime: Double($0.endTime) + prepared.leadingSeconds
                )
            }
        } else {
            alignedWords = []
        }

        return LyricsTranscriptionResult(
            backend: Self.backendName,
            modelID: options.modelID,
            alignerModelID: options.alignerModelID,
            text: text,
            words: alignedWords,
            sampleRate: options.sampleRate,
            trim: LyricsTranscriptionTrim(
                leadingSeconds: prepared.leadingSeconds,
                trailingSeconds: prepared.trailingSeconds
            ),
            elapsedSeconds: Date().timeIntervalSince(startedAt)
        )
    }

    public func align(
        audioURL: URL,
        text: String,
        options: LyricsTranscriptionOptions = .qualityDataset
    ) async throws -> LyricsAlignmentResult {
        let trimmedText = text.trimmingCharacters(
            in: .whitespacesAndNewlines
        )
        guard !trimmedText.isEmpty else {
            throw LyricsAlignmentError.emptyText
        }
        guard let alignerModelID = options.alignerModelID,
              !alignerModelID.trimmingCharacters(
                  in: .whitespacesAndNewlines
              ).isEmpty
        else {
            throw LyricsAlignmentError.missingAlignerModel
        }

        let rawSamples = try AudioFileLoader.load(
            url: audioURL,
            targetSampleRate: options.sampleRate
        )
        let prepared = options.trimsSilence
            ? Self.trimmed(
                samples: rawSamples,
                sampleRate: options.sampleRate,
                threshold: options.silenceThreshold
            )
            : PreparedAudio(
                samples: rawSamples,
                leadingSeconds: 0,
                trailingSeconds: 0
            )
        try Task.checkCancellation()
        let startedAt = Date()
        let aligner = try await loadAligner(
            modelID: alignerModelID,
            offlineMode: options.offlineMode
        )
        try Task.checkCancellation()
        let words = aligner.alignLong(
            audio: prepared.samples,
            text: trimmedText,
            sampleRate: options.sampleRate,
            language: options.alignmentLanguage
        )
        .map {
            LyricsTranscriptionWord(
                text: $0.text,
                startTime: Double($0.startTime) + prepared.leadingSeconds,
                endTime: Double($0.endTime) + prepared.leadingSeconds
            )
        }
        try Task.checkCancellation()
        return LyricsAlignmentResult(
            backend: Self.alignerBackendName,
            modelID: alignerModelID,
            text: trimmedText,
            words: words,
            sampleRate: options.sampleRate,
            trim: LyricsTranscriptionTrim(
                leadingSeconds: prepared.leadingSeconds,
                trailingSeconds: prepared.trailingSeconds
            ),
            elapsedSeconds: Date().timeIntervalSince(startedAt)
        )
    }

    private func loadASRModel(options: LyricsTranscriptionOptions) async throws -> Qwen3ASRModel {
        if let asrModel, asrModelID == options.modelID {
            return asrModel
        }
        let model = try await Qwen3ASRModel.fromPretrained(
            modelId: options.modelID,
            offlineMode: options.offlineMode
        )
        asrModel = model
        asrModelID = options.modelID
        return model
    }

    private func loadAligner(modelID: String, offlineMode: Bool) async throws -> Qwen3ForcedAligner {
        if let aligner, alignerModelID == modelID {
            return aligner
        }
        let model = try await Qwen3ForcedAligner.fromPretrained(
            modelId: modelID,
            offlineMode: offlineMode
        )
        aligner = model
        alignerModelID = modelID
        return model
    }

    private static func trimmed(samples: [Float], sampleRate: Int, threshold: Float) -> PreparedAudio {
        guard !samples.isEmpty else {
            return PreparedAudio(samples: [], leadingSeconds: 0, trailingSeconds: 0)
        }

        let peak = samples.map { abs($0) }.max() ?? 0
        let effectiveThreshold = max(threshold, peak * 0.01)
        guard peak > effectiveThreshold else {
            return PreparedAudio(samples: samples, leadingSeconds: 0, trailingSeconds: 0)
        }

        let first = samples.firstIndex { abs($0) >= effectiveThreshold } ?? samples.startIndex
        let last = samples.lastIndex { abs($0) >= effectiveThreshold } ?? samples.index(before: samples.endIndex)
        guard first <= last else {
            return PreparedAudio(samples: samples, leadingSeconds: 0, trailingSeconds: 0)
        }

        let padding = max(0, Int((Double(sampleRate) * 0.1).rounded()))
        let start = max(samples.startIndex, first - padding)
        let end = min(samples.index(before: samples.endIndex), last + padding)
        let trimmed = Array(samples[start ... end])
        let leading = Double(start) / Double(sampleRate)
        let trailing = Double(samples.count - end - 1) / Double(sampleRate)
        return PreparedAudio(samples: trimmed, leadingSeconds: leading, trailingSeconds: trailing)
    }
}

private struct PreparedAudio {
    let samples: [Float]
    let leadingSeconds: Double
    let trailingSeconds: Double
}
