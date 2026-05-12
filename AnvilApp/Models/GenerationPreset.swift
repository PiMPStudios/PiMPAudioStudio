import Foundation
import AnvilCore

// MARK: - Codable conformance for raw-representable enums used in presets

extension GenerationMode: Codable {}
extension PipelineChoice: Codable {}
extension InferMethod: Codable {}
extension Precision: Codable {}

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
    var pipelineOverride: PipelineChoice

    // Advanced
    var method: InferMethod
    var precision: Precision
    var stepsOverride: String
    var shiftOverride: Double

    // Adapter
    var adapterID: String?

    // Cover song
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
        pipelineOverride: PipelineChoice = .auto,
        method: InferMethod = .ode,
        precision: Precision = .float32,
        stepsOverride: String = "",
        shiftOverride: Double = 3.0,
        adapterID: String? = nil,
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
        self.pipelineOverride = pipelineOverride
        self.method = method
        self.precision = precision
        self.stepsOverride = stepsOverride
        self.shiftOverride = shiftOverride
        self.adapterID = adapterID
        self.coverStrength = coverStrength
    }
}
