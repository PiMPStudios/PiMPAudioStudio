import Foundation
import PiMPAudioCore

// MARK: - Codable conformance for app-local generation mode

extension GenerationMode: Codable {}

// MARK: - GenerationPreset

/// Snapshot of all generation settings that can be saved and recalled.
struct GenerationPreset: Identifiable, Codable, Sendable, Equatable {
    let id: UUID
    var name: String
    let createdAt: Date

    // Content
    var prompt: String
    var lyrics: String

    /// Generation mode override. `nil` = Auto (use intent router).
    var mode: GenerationMode?

    // Parameters
    var duration: Double
    var seedText: String
    var guidanceScale: Double
    var guidancePreset: GuidancePreset
    var pipelineOverride: PipelineChoice

    // Advanced
    var method: InferMethod
    var precision: Precision
    var stepsOverride: String
    var shiftOverride: Double

    // Adapter
    var adapterID: String?
    var adapterStrength: Double

    /// Cover song
    var coverStrength: Double

    init(
        id: UUID = UUID(),
        name: String,
        createdAt: Date = Date(),
        prompt: String,
        lyrics: String = "",
        mode: GenerationMode? = nil,
        duration: Double = 30,
        seedText: String = "",
        guidanceScale: Double = 7.5,
        guidancePreset: GuidancePreset = GuidancePreset.defaultSelection,
        pipelineOverride: PipelineChoice = .auto,
        method: InferMethod = .ode,
        precision: Precision = .float32,
        stepsOverride: String = "",
        shiftOverride: Double = 3.0,
        adapterID: String? = nil,
        adapterStrength: Double = 1.0,
        coverStrength: Double = 0.7
    ) {
        self.id = id
        self.name = name
        self.createdAt = createdAt
        self.prompt = prompt
        self.lyrics = lyrics
        self.mode = mode
        self.duration = duration
        self.seedText = seedText
        self.guidanceScale = guidanceScale
        self.guidancePreset = guidancePreset
        self.pipelineOverride = pipelineOverride
        self.method = method
        self.precision = precision
        self.stepsOverride = stepsOverride
        self.shiftOverride = shiftOverride
        self.adapterID = adapterID
        self.adapterStrength = adapterStrength
        self.coverStrength = coverStrength
    }

    private enum CodingKeys: String, CodingKey {
        case id
        case name
        case createdAt
        case prompt
        case lyrics
        case mode
        case duration
        case seedText
        case guidanceScale
        case guidancePreset
        case pipelineOverride
        case method
        case precision
        case stepsOverride
        case shiftOverride
        case adapterID
        case adapterStrength
        case coverStrength
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(UUID.self, forKey: .id)
        name = try container.decode(String.self, forKey: .name)
        createdAt = try container.decode(Date.self, forKey: .createdAt)
        prompt = try container.decode(String.self, forKey: .prompt)
        lyrics = try container.decodeIfPresent(String.self, forKey: .lyrics) ?? ""
        mode = try container.decodeIfPresent(GenerationMode.self, forKey: .mode)
        duration = try container.decodeIfPresent(Double.self, forKey: .duration) ?? 30
        seedText = try container.decodeIfPresent(String.self, forKey: .seedText) ?? ""
        guidanceScale = try container.decodeIfPresent(Double.self, forKey: .guidanceScale) ?? 7.5
        guidancePreset = try container.decodeIfPresent(GuidancePreset.self, forKey: .guidancePreset) ?? .constant
        pipelineOverride = try container.decodeIfPresent(PipelineChoice.self, forKey: .pipelineOverride) ?? .auto
        method = try container.decodeIfPresent(InferMethod.self, forKey: .method) ?? .ode
        precision = try container.decodeIfPresent(Precision.self, forKey: .precision) ?? .float32
        stepsOverride = try container.decodeIfPresent(String.self, forKey: .stepsOverride) ?? ""
        shiftOverride = try container.decodeIfPresent(Double.self, forKey: .shiftOverride) ?? 3.0
        adapterID = try container.decodeIfPresent(String.self, forKey: .adapterID)
        adapterStrength = try container.decodeIfPresent(Double.self, forKey: .adapterStrength) ?? 1.0
        coverStrength = try container.decodeIfPresent(Double.self, forKey: .coverStrength) ?? 0.7
    }
}
