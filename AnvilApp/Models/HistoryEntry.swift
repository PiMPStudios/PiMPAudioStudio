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

struct GenerationTrace: Codable, Sendable, Equatable {
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

        return "anvil_\(timestamp)_\(words).wav"
    }
}
