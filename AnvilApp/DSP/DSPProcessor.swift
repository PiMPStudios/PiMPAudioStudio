import Foundation

struct DSPProcessor {
    enum ProcessingError: Error {
        case selectionRequired
        case unknownEffect
    }

    func apply(
        effect: DSPEffect,
        to buffer: AudioBuffer,
        target: ApplyTarget,
        parameters: DSPParameters = .none
    ) throws -> AudioBuffer {
        switch effect {
        case .trimToSelection:
            guard case .selection(let start, let end) = target else {
                throw ProcessingError.selectionRequired
            }
            return try SimpleDSP.trimToSelection(buffer, start: start, end: end)

        case .deleteSelection:
            guard case .selection(let start, let end) = target else {
                throw ProcessingError.selectionRequired
            }
            return try SimpleDSP.deleteSelection(buffer, start: start, end: end)

        case .fadeIn:
            guard case .selection(let start, let end) = target else {
                throw ProcessingError.selectionRequired
            }
            return try SimpleDSP.fadeIn(buffer, start: start, end: end)

        case .fadeOut:
            guard case .selection(let start, let end) = target else {
                throw ProcessingError.selectionRequired
            }
            return try SimpleDSP.fadeOut(buffer, start: start, end: end)

        case .normalize:
            return try SimpleDSP.normalize(buffer)

        case .eq:
            guard case .eq(let eqParams) = parameters else {
                return buffer
            }
            return try AVAudioEngineDSP.applyEQ(to: buffer, parameters: eqParams, target: target)

        case .reverb:
            guard case .reverb(let reverbParams) = parameters else {
                return buffer
            }
            return try AVAudioEngineDSP.applyReverb(to: buffer, parameters: reverbParams, target: target)

        case .timeStretch:
            guard case .timeStretch(let params) = parameters else {
                return buffer
            }
            return try AVAudioEngineDSP.applyTimeStretch(to: buffer, parameters: params, target: target)

        case .pitchShift:
            guard case .pitchShift(let params) = parameters else {
                return buffer
            }
            return try AVAudioEngineDSP.applyPitchShift(to: buffer, parameters: params, target: target)
        }
    }

    func makeDescription(effect: DSPEffect, target: ApplyTarget, parameters: DSPParameters = .none) -> DSPOperation {
        let targetLabel: String
        switch target {
        case .selection(let start, let end):
            targetLabel = "(\(formatTime(start))–\(formatTime(end)))"
        case .wholeTrack:
            targetLabel = "(whole track)"
        }

        let paramLabel: String
        switch parameters {
        case .timeStretch(let params):
            paramLabel = " \(String(format: "%.2f", params.rate))×"
        case .pitchShift(let params):
            let semitones = Int(params.pitch / 100)
            let sign = semitones >= 0 ? "+" : ""
            paramLabel = " \(sign)\(semitones) st"
        default:
            paramLabel = ""
        }

        return DSPOperation(
            effect: effect,
            target: target,
            description: "\(effect.displayName)\(paramLabel) \(targetLabel)"
        )
    }

    private func formatTime(_ time: TimeInterval) -> String {
        let minutes = Int(time) / 60
        let seconds = Int(time) % 60
        let tenths = Int((time - Double(Int(time))) * 10)
        if minutes > 0 {
            return String(format: "%d:%02d.%d", minutes, seconds, tenths)
        }
        return String(format: "%d.%ds", seconds, tenths)
    }
}
