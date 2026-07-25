import Foundation
import PiMPAudioCore

extension HistoryEntry {
    /// Build a Generate preset from this history take, enriched by the optional
    /// sibling trace when available (guidance preset, steps, cover strength, etc.).
    func asGenerationPreset(
        name: String = "From History",
        trace: GenerationTrace? = nil
    ) -> GenerationPreset {
        let method = InferMethod(rawValue: method) ?? .ode
        let precision = Precision(rawValue: precision) ?? .float32

        var guidancePreset = GuidancePreset.defaultSelection
        var stepsOverride = ""
        var shiftOverride = 3.0
        var coverStrength = 0.7
        var mode: GenerationMode?
        var pipelineOverride = pipelineChoice(from: pipeline)
        var resolvedGuidance = Double(guidanceScale)

        if let request = trace?.request {
            if let parsed = GuidancePreset(rawValue: request.guidancePreset) {
                guidancePreset = parsed
            }
            if let steps = request.stepsOverride {
                stepsOverride = String(steps)
            }
            if let shift = request.shiftOverride {
                shiftOverride = Double(shift)
            }
            coverStrength = Double(request.coverStrength)
            resolvedGuidance = Double(request.guidanceScale)
            mode = GenerationMode(fromClassification: request.resolvedMode)
            let fromRequest = pipelineChoice(from: request.pipelineOverride)
            if fromRequest != .auto || pipelineOverride == .auto {
                pipelineOverride = fromRequest
            }
        }

        let adapters = trace?.adapters ?? self.adapters ?? []
        let primary = adapters.first

        return GenerationPreset(
            name: name,
            prompt: prompt,
            lyrics: lyrics,
            mode: mode,
            duration: duration,
            seedText: "\(seed)",
            guidanceScale: resolvedGuidance,
            guidancePreset: guidancePreset,
            pipelineOverride: pipelineOverride,
            method: method,
            precision: precision,
            stepsOverride: stepsOverride,
            shiftOverride: shiftOverride,
            adapterID: primary?.id,
            adapterStrength: Double(primary?.weight ?? 1.0),
            coverStrength: coverStrength
        )
    }

    /// Adapter stack payloads for GenerateView's stack restore path.
    func adapterStackPayload(trace: GenerationTrace? = nil) -> [[String: Any]] {
        let adapters = trace?.adapters ?? self.adapters ?? []
        return adapters.map { adapter in
            [
                "id": adapter.id,
                "weight": adapter.weight,
            ]
        }
    }

    private func pipelineChoice(from raw: String?) -> PipelineChoice {
        guard let raw else { return .auto }
        let lowered = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        switch lowered {
        case "stableaudio", "stable_audio", "sfx":
            return .stableAudio
        case "acestep", "ace_step", "ace-step", "coversong", "cover_song", "cover":
            return .aceStep
        case "auto":
            return .auto
        default:
            if let exact = PipelineChoice(rawValue: raw) {
                return exact
            }
            return .auto
        }
    }
}
