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
    let guidancePreset: GuidancePreset
    let precision: Precision
    let stepsOverride: Int?
    let shiftOverride: Float?
    let resolvedMode: GenerationMode
    let pipelineOverride: PipelineChoice
    /// Blend weight for cover song generation (0 = source only, 1 = fully
    /// acoustic-hint driven). Only used when `referenceAudioURL` is non-nil
    /// and the request routes to `CoverSongCoordinator`.
    let coverStrength: Float

    init(
        text: String,
        lyrics: String = "",
        referenceAudioURL: URL? = nil,
        duration: TimeInterval = 30.0,
        seed: Int? = nil,
        method: InferMethod = .ode,
        guidanceScale: Float = 1.0,
        guidancePreset: GuidancePreset = GuidancePreset.defaultSelection,
        precision: Precision = .float32,
        stepsOverride: Int? = nil,
        shiftOverride: Float? = nil,
        resolvedMode: GenerationMode = .music,
        pipelineOverride: PipelineChoice = .auto,
        coverStrength: Float = 0.7
    ) {
        self.text = text
        self.lyrics = lyrics
        self.referenceAudioURL = referenceAudioURL
        self.duration = duration
        self.seed = seed
        self.method = method
        self.guidanceScale = guidanceScale
        self.guidancePreset = guidancePreset
        self.precision = precision
        self.stepsOverride = stepsOverride
        self.shiftOverride = shiftOverride
        self.resolvedMode = resolvedMode
        self.pipelineOverride = pipelineOverride
        self.coverStrength = coverStrength
    }

    var effectiveGuidanceScale: Float {
        guidancePreset.effectiveGuidanceScale(userValue: guidanceScale)
    }

    var cfgIntervalStart: Float {
        guidancePreset.cfgInterval.start
    }

    var cfgIntervalEnd: Float {
        guidancePreset.cfgInterval.end
    }
}
