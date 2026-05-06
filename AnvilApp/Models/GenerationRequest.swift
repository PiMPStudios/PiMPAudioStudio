import Foundation
import AnvilCore

struct GenerationRequest: Sendable {
    let text: String
    let lyrics: String
    let referenceAudioURL: URL?
    let duration: TimeInterval
    let seed: Int?
    let method: InferMethod
    let guidanceScale: Float
    let precision: Precision
    let resolvedMode: GenerationMode
    let pipelineOverride: PipelineChoice

    init(
        text: String,
        lyrics: String = "",
        referenceAudioURL: URL? = nil,
        duration: TimeInterval = 30.0,
        seed: Int? = nil,
        method: InferMethod = .ode,
        guidanceScale: Float = 1.0,
        precision: Precision = .float16,
        resolvedMode: GenerationMode = .music,
        pipelineOverride: PipelineChoice = .auto
    ) {
        self.text = text
        self.lyrics = lyrics
        self.referenceAudioURL = referenceAudioURL
        self.duration = duration
        self.seed = seed
        self.method = method
        self.guidanceScale = guidanceScale
        self.precision = precision
        self.resolvedMode = resolvedMode
        self.pipelineOverride = pipelineOverride
    }
}
