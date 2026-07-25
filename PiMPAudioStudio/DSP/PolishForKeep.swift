import Foundation

// MARK: - Recipe

/// One-shot “Polish for Keep” controls (editor P2).
struct PolishForKeepRecipe: Equatable, Sendable {
    enum Target: String, CaseIterable, Sendable {
        case streaming
        case louder
        case gentle

        var displayName: String {
            switch self {
            case .streaming: return "Streaming"
            case .louder: return "Louder"
            case .gentle: return "Gentle"
            }
        }

        var detail: String {
            switch self {
            case .streaming: return "≈ −14 LUFS, −1 dB ceiling"
            case .louder: return "≈ −10 LUFS, −0.5 dB ceiling"
            case .gentle: return "Clean + −3 dB peak headroom"
            }
        }
    }

    enum Tone: String, CaseIterable, Sendable {
        case neutral
        case brighter
        case darker

        var displayName: String {
            rawValue.capitalized
        }
    }

    enum Glue: String, CaseIterable, Sendable {
        case off
        case light
        case medium

        var displayName: String {
            rawValue.capitalized
        }
    }

    enum Space: String, CaseIterable, Sendable {
        case off
        case room
        case hall

        var displayName: String {
            rawValue.capitalized
        }
    }

    var target: Target
    var tone: Tone
    var glue: Glue
    var space: Space
    var trimSilenceAndFades: Bool

    static let `default` = PolishForKeepRecipe(
        target: .streaming,
        tone: .neutral,
        glue: .light,
        space: .off,
        trimSilenceAndFades: true
    )
}

// MARK: - Processor

/// Runs the keep-polish chain as a single offline pass (one undo step in the editor).
enum PolishForKeepProcessor {
    enum PolishError: LocalizedError {
        case bufferEmpty

        var errorDescription: String? {
            switch self {
            case .bufferEmpty:
                return "The audio buffer is empty."
            }
        }
    }

    static func apply(
        _ buffer: AudioBuffer,
        recipe: PolishForKeepRecipe = .default
    ) throws -> AudioBuffer {
        guard buffer.frameCount > 0 else { throw PolishError.bufferEmpty }
        return try apply(FloatAudio(buffer), recipe: recipe).toAudioBuffer()
    }

    /// Chain order: trim → HPF → tone EQ → glue → subtle space → loudness/limit or gentle peak.
    ///
    /// The whole chain runs in float and quantizes once, in the `AudioBuffer` overload
    /// above. Requantizing between stages would clip every intermediate overshoot —
    /// compressor makeup gain, filter ringing, the reverb wet sum, the loudness boost —
    /// at ±1.0 before the limiter ever got a chance to catch them.
    static func apply(
        _ audio: FloatAudio,
        recipe: PolishForKeepRecipe = .default
    ) throws -> FloatAudio {
        guard audio.frameCount > 0 else { throw PolishError.bufferEmpty }

        var current = audio

        if recipe.trimSilenceAndFades {
            current = try SimpleDSP.trimSilence(
                current,
                parameters: TrimSilenceParameters(
                    thresholdDBFS: -50,
                    maxLeadingSeconds: 5,
                    maxTrailingSeconds: 5,
                    edgeFadeMilliseconds: 30
                )
            )
        }

        current = try SimpleDSP.highPass(
            current,
            parameters: HighPassParameters(cutoffHz: highPassCutoff(for: recipe.tone))
        )

        if let eq = toneEQParameters(for: recipe.tone) {
            current = try AVAudioEngineDSP.applyEQ(
                to: current,
                parameters: eq,
                target: .wholeTrack
            )
        }

        if let compressor = compressorParameters(for: recipe.glue) {
            current = try DynamicsProcessor.compress(current, parameters: compressor)
        }

        // Space before loudness/limit so wet peaks are still controlled. This is a
        // deliberate reordering of the spec's step 8 — running reverb after the
        // limiter would put unlimited wet signal past the ceiling.
        if let reverb = reverbParameters(for: recipe.space) {
            current = try AVAudioEngineDSP.applyReverb(
                to: current,
                parameters: reverb,
                target: .wholeTrack
            )
        }

        switch recipe.target {
        case .gentle:
            current = try SimpleDSP.normalize(
                current,
                mode: .peak(targetDBFS: -3.0)
            )
        case .streaming:
            current = try DynamicsProcessor.aimLoudness(
                current,
                parameters: .streaming
            )
        case .louder:
            current = try DynamicsProcessor.aimLoudness(
                current,
                parameters: .louder
            )
        }

        return current
    }

    // MARK: - Recipe mapping

    static func highPassCutoff(for tone: PolishForKeepRecipe.Tone) -> Float {
        switch tone {
        case .neutral, .brighter: return 40
        case .darker: return 35
        }
    }

    /// Mild corrective EQ only (1–2 dB moves). `nil` for pure Neutral (HPF alone).
    static func toneEQParameters(for tone: PolishForKeepRecipe.Tone) -> EQParameters? {
        switch tone {
        case .neutral:
            return nil
        case .brighter:
            return EQParameters(bands: [
                EQBand(frequency: 300, gain: 0, q: 1.0),
                EQBand(frequency: 1_200, gain: 0.5, q: 1.2),
                EQBand(frequency: 10_000, gain: 1.5, q: 0.7),
            ])
        case .darker:
            return EQParameters(bands: [
                EQBand(frequency: 280, gain: -1.5, q: 1.0),
                EQBand(frequency: 1_200, gain: 0, q: 1.2),
                EQBand(frequency: 8_000, gain: -0.5, q: 0.8),
            ])
        }
    }

    static func compressorParameters(for glue: PolishForKeepRecipe.Glue) -> CompressorParameters? {
        switch glue {
        case .off: return nil
        case .light: return .light
        case .medium: return .medium
        }
    }

    /// Full-mix-safe wet levels (well under creative reverb defaults).
    static func reverbParameters(for space: PolishForKeepRecipe.Space) -> ReverbParameters? {
        switch space {
        case .off:
            return nil
        case .room:
            return ReverbParameters(
                decayTime: 0.7,
                wetDryMix: 0.08,
                preDelay: 8,
                damping: 0.55
            )
        case .hall:
            return ReverbParameters(
                decayTime: 1.6,
                wetDryMix: 0.10,
                preDelay: 15,
                damping: 0.6
            )
        }
    }

    static func description(for recipe: PolishForKeepRecipe) -> String {
        var parts = ["Polish for Keep (\(recipe.target.displayName))"]
        if recipe.trimSilenceAndFades { parts.append("trim") }
        parts.append("HPF")
        if recipe.tone != .neutral { parts.append(recipe.tone.displayName.lowercased()) }
        if recipe.glue != .off { parts.append("glue \(recipe.glue.displayName.lowercased())") }
        if recipe.space != .off { parts.append(recipe.space.displayName.lowercased()) }
        return parts.joined(separator: " · ")
    }
}
