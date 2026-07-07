import Foundation

struct DSPProcessor {
    enum ProcessingError: LocalizedError {
        case selectionRequired
        case unknownEffect

        var errorDescription: String? {
            switch self {
            case .selectionRequired:
                return "Select a region before applying this effect."
            case .unknownEffect:
                return "The selected DSP effect is not supported."
            }
        }
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
            return try processRegion(buffer, target: target) { regionBuffer in
                let floats = AudioBufferConverter.toFloats(regionBuffer)
                let stretched = WSOLAProcessor.stretch(floats, rate: params.rate,
                                                       sampleRate: regionBuffer.sampleRate,
                                                       channels: regionBuffer.channels)
                return AudioBufferConverter.fromFloats(stretched,
                                                       sampleRate: regionBuffer.sampleRate,
                                                       channels: regionBuffer.channels)
            }

        case .pitchShift:
            guard case .pitchShift(let params) = parameters else {
                return buffer
            }
            return try processRegion(buffer, target: target) { regionBuffer in
                let floats = AudioBufferConverter.toFloats(regionBuffer)
                let pitchRatio = pow(2.0, params.pitch / 1200.0)
                let resampled = Resampler.resample(floats, ratio: pitchRatio,
                                                    channels: regionBuffer.channels)
                let timeCorrected = WSOLAProcessor.stretch(resampled,
                                                            rate: 1.0 / pitchRatio,
                                                            sampleRate: regionBuffer.sampleRate,
                                                            channels: regionBuffer.channels)
                var matched = timeCorrected
                if matched.count < floats.count {
                    matched.append(contentsOf: [Float](repeating: 0, count: floats.count - matched.count))
                } else if matched.count > floats.count {
                    matched = Array(matched.prefix(floats.count))
                }
                let corrected = LPCFormantShifter.correct(original: floats,
                                                          shifted: matched,
                                                          sampleRate: regionBuffer.sampleRate,
                                                          channels: regionBuffer.channels)
                return AudioBufferConverter.fromFloats(corrected,
                                                       sampleRate: regionBuffer.sampleRate,
                                                       channels: regionBuffer.channels)
            }
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

    private func processRegion(
        _ buffer: AudioBuffer,
        target: ApplyTarget,
        transform: (AudioBuffer) throws -> AudioBuffer
    ) throws -> AudioBuffer {
        switch target {
        case .wholeTrack:
            return try transform(buffer)
        case .selection(let start, let end):
            let bytesPerSample = MemoryLayout<Int16>.size
            let startSample = SimpleDSP.sampleIndex(time: start, sampleRate: buffer.sampleRate, channels: buffer.channels)
            let endSample = min(
                SimpleDSP.sampleIndex(time: end, sampleRate: buffer.sampleRate, channels: buffer.channels),
                buffer.data.count / bytesPerSample
            )
            let startByte = startSample * bytesPerSample
            let endByte = endSample * bytesPerSample

            let regionData = Data(buffer.data[startByte..<endByte])
            let regionBuffer = AudioBuffer(data: regionData, sampleRate: buffer.sampleRate, channels: buffer.channels)
            let processed = try transform(regionBuffer)

            var result = Data()
            result.append(buffer.data[0..<startByte])
            result.append(processed.data)
            if endByte < buffer.data.count {
                result.append(buffer.data[endByte...])
            }
            return AudioBuffer(data: result, sampleRate: buffer.sampleRate, channels: buffer.channels)
        }
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
