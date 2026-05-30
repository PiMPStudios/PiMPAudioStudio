import Foundation

struct HistoryModelSnapshot: Codable, Sendable, Equatable {
    let category: String
    let id: String
    let displayName: String?
}

struct HistoryAdapterSnapshot: Codable, Sendable, Equatable {
    let id: String
    let name: String?
    let weight: Float
    let source: String?
    let rank: Int?
    let baseModelID: String?
}

struct GenerationTraceRequestSnapshot: Codable, Sendable, Equatable {
    let duration: TimeInterval
    let seed: Int?
    let method: String
    let guidanceScale: Float
    let guidancePreset: String
    let guidancePresetName: String
    let cfgIntervalStart: Float
    let cfgIntervalEnd: Float
    let precision: String
    let stepsOverride: Int?
    let shiftOverride: Float?
    let resolvedMode: String
    let pipelineOverride: String
    let coverStrength: Float
    let referenceAudioFileName: String?
}

struct GenerationTraceFeatureFlags: Codable, Sendable, Equatable {
    let lmThinkingEnabled: Bool
    let cotMetadataEnabled: Bool
    let cotCaptionEnabled: Bool
    let dcwEnabled: Bool
}

struct GenerationTraceTimingSnapshot: Codable, Sendable, Equatable {
    let diffusionTime: TimeInterval
    let perStepTime: TimeInterval
    let totalTime: TimeInterval
}

struct GenerationTraceAudioSnapshot: Codable, Sendable, Equatable {
    let sampleRate: Int
    let duration: TimeInterval
    let sampleShape: [Int]
    let absMean: Float
    let absMax: Float
    let waveformPeaks: [Float]
    let perSecondAbsMean: [Float]
    let perSecondAbsMax: [Float]
}

struct GenerationTraceEventSnapshot: Codable, Sendable, Equatable {
    let timestamp: Date
    let stage: String
    let message: String
    let detail: String?
    let progress: Double?
    let level: String
}

struct GenerationTrace: Codable, Sendable, Equatable {
    static let currentSchemaVersion = 2

    let schemaVersion: Int
    let historyEntryID: UUID
    let createdAt: Date
    let audioFileName: String
    let traceFileName: String
    let pipeline: String
    let prompt: String
    let originalPrompt: String?
    let lyrics: String
    let request: GenerationTraceRequestSnapshot
    let primaryModel: HistoryModelSnapshot?
    let supportingModels: [HistoryModelSnapshot]
    let adapters: [HistoryAdapterSnapshot]
    let featureFlags: GenerationTraceFeatureFlags
    let timing: GenerationTraceTimingSnapshot
    let audio: GenerationTraceAudioSnapshot
    let listeningNote: String?
    let isPreferred: Bool?
    let status: String?
    let errorMessage: String?
    let events: [GenerationTraceEventSnapshot]?

    init(
        schemaVersion: Int = GenerationTrace.currentSchemaVersion,
        historyEntryID: UUID,
        createdAt: Date,
        audioFileName: String,
        traceFileName: String,
        pipeline: String,
        prompt: String,
        originalPrompt: String?,
        lyrics: String,
        request: GenerationTraceRequestSnapshot,
        primaryModel: HistoryModelSnapshot?,
        supportingModels: [HistoryModelSnapshot],
        adapters: [HistoryAdapterSnapshot],
        featureFlags: GenerationTraceFeatureFlags,
        timing: GenerationTraceTimingSnapshot,
        audio: GenerationTraceAudioSnapshot,
        listeningNote: String?,
        isPreferred: Bool? = nil,
        status: String? = "complete",
        errorMessage: String? = nil,
        events: [GenerationTraceEventSnapshot]? = nil
    ) {
        self.schemaVersion = schemaVersion
        self.historyEntryID = historyEntryID
        self.createdAt = createdAt
        self.audioFileName = audioFileName
        self.traceFileName = traceFileName
        self.pipeline = pipeline
        self.prompt = prompt
        self.originalPrompt = originalPrompt
        self.lyrics = lyrics
        self.request = request
        self.primaryModel = primaryModel
        self.supportingModels = supportingModels
        self.adapters = adapters
        self.featureFlags = featureFlags
        self.timing = timing
        self.audio = audio
        self.listeningNote = listeningNote
        self.isPreferred = isPreferred
        self.status = status
        self.errorMessage = errorMessage
        self.events = events
    }

    private enum CodingKeys: String, CodingKey {
        case schemaVersion
        case historyEntryID
        case createdAt
        case audioFileName
        case traceFileName
        case pipeline
        case prompt
        case originalPrompt
        case lyrics
        case request
        case primaryModel
        case supportingModels
        case adapters
        case featureFlags
        case timing
        case audio
        case listeningNote
        case isPreferred
        case status
        case errorMessage
        case events
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.schemaVersion = try container.decodeIfPresent(Int.self, forKey: .schemaVersion) ?? 1
        self.historyEntryID = try container.decode(UUID.self, forKey: .historyEntryID)
        self.createdAt = try container.decode(Date.self, forKey: .createdAt)
        self.audioFileName = try container.decode(String.self, forKey: .audioFileName)
        self.traceFileName = try container.decode(String.self, forKey: .traceFileName)
        self.pipeline = try container.decode(String.self, forKey: .pipeline)
        self.prompt = try container.decode(String.self, forKey: .prompt)
        self.originalPrompt = try container.decodeIfPresent(String.self, forKey: .originalPrompt)
        self.lyrics = try container.decode(String.self, forKey: .lyrics)
        self.request = try container.decode(GenerationTraceRequestSnapshot.self, forKey: .request)
        self.primaryModel = try container.decodeIfPresent(HistoryModelSnapshot.self, forKey: .primaryModel)
        self.supportingModels = try container.decodeIfPresent([HistoryModelSnapshot].self, forKey: .supportingModels) ?? []
        self.adapters = try container.decodeIfPresent([HistoryAdapterSnapshot].self, forKey: .adapters) ?? []
        self.featureFlags = try container.decodeIfPresent(GenerationTraceFeatureFlags.self, forKey: .featureFlags) ?? GenerationTraceFeatureFlags(
            lmThinkingEnabled: false,
            cotMetadataEnabled: false,
            cotCaptionEnabled: false,
            dcwEnabled: false
        )
        self.timing = try container.decodeIfPresent(GenerationTraceTimingSnapshot.self, forKey: .timing) ?? GenerationTraceTimingSnapshot(
            diffusionTime: 0,
            perStepTime: 0,
            totalTime: 0
        )
        self.audio = try container.decode(GenerationTraceAudioSnapshot.self, forKey: .audio)
        self.listeningNote = try container.decodeIfPresent(String.self, forKey: .listeningNote)
        self.isPreferred = try container.decodeIfPresent(Bool.self, forKey: .isPreferred)
        self.status = try container.decodeIfPresent(String.self, forKey: .status)
        self.errorMessage = try container.decodeIfPresent(String.self, forKey: .errorMessage)
        self.events = try container.decodeIfPresent([GenerationTraceEventSnapshot].self, forKey: .events)
    }

    func withReview(listeningNote: String?, isPreferred: Bool?) -> GenerationTrace {
        GenerationTrace(
            schemaVersion: GenerationTrace.currentSchemaVersion,
            historyEntryID: historyEntryID,
            createdAt: createdAt,
            audioFileName: audioFileName,
            traceFileName: traceFileName,
            pipeline: pipeline,
            prompt: prompt,
            originalPrompt: originalPrompt,
            lyrics: lyrics,
            request: request,
            primaryModel: primaryModel,
            supportingModels: supportingModels,
            adapters: adapters,
            featureFlags: featureFlags,
            timing: timing,
            audio: audio,
            listeningNote: listeningNote,
            isPreferred: isPreferred,
            status: status,
            errorMessage: errorMessage,
            events: events
        )
    }
}

struct GenerationFailureTrace: Codable, Sendable, Equatable {
    static let currentSchemaVersion = 1

    let schemaVersion: Int
    let traceType: String
    let id: UUID
    let createdAt: Date
    let completedAt: Date?
    let traceFileName: String
    let pipeline: String
    let prompt: String
    let lyrics: String
    let request: GenerationTraceRequestSnapshot
    let primaryModel: HistoryModelSnapshot?
    let supportingModels: [HistoryModelSnapshot]
    let adapters: [HistoryAdapterSnapshot]
    let errorMessage: String
    let events: [GenerationTraceEventSnapshot]
}

struct HistoryEntry: Identifiable, Codable, Sendable, Equatable {
    let id: UUID
    let createdAt: Date
    let prompt: String
    let lyrics: String
    let duration: TimeInterval
    let seed: Int
    let method: String
    let guidanceScale: Float
    let precision: String
    let fileName: String
    let traceFileName: String?
    let pipeline: String?
    let originalPrompt: String?
    let modelCategory: String?
    let modelID: String?
    let modelName: String?
    let supportingModels: [HistoryModelSnapshot]?
    let adapters: [HistoryAdapterSnapshot]?
    let listeningNote: String?
    let isPreferred: Bool?

    func withReview(listeningNote: String?, isPreferred: Bool?) -> HistoryEntry {
        HistoryEntry(
            id: id,
            createdAt: createdAt,
            prompt: prompt,
            lyrics: lyrics,
            duration: duration,
            seed: seed,
            method: method,
            guidanceScale: guidanceScale,
            precision: precision,
            fileName: fileName,
            traceFileName: traceFileName,
            pipeline: pipeline,
            originalPrompt: originalPrompt,
            modelCategory: modelCategory,
            modelID: modelID,
            modelName: modelName,
            supportingModels: supportingModels,
            adapters: adapters,
            listeningNote: listeningNote,
            isPreferred: isPreferred
        )
    }

    static func traceFileName(forAudioFileName fileName: String) -> String {
        let baseName = (fileName as NSString).deletingPathExtension
        return "\(baseName).trace.json"
    }

    static func fileName(prompt: String, date: Date = Date()) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd_HHmmss"
        let timestamp = formatter.string(from: date)

        let words = prompt
            .lowercased()
            .components(separatedBy: .whitespaces)
            .prefix(5)
            .joined(separator: "-")
            .replacingOccurrences(
                of: "[^a-z0-9\\-]",
                with: "",
                options: .regularExpression
            )
            .prefix(50)

        return "pimpaudio_\(timestamp)_\(words).wav"
    }
}
