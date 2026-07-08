import Foundation
import Accelerate

struct SimpleDSP {
    enum DSPError: LocalizedError {
        case invalidSelection
        case bufferEmpty

        var errorDescription: String? {
            switch self {
            case .invalidSelection:
                return "The selected audio range is not valid."
            case .bufferEmpty:
                return "The audio buffer is empty."
            }
        }
    }

    static func trimToSelection(_ buffer: AudioBuffer, start: TimeInterval, end: TimeInterval) throws -> AudioBuffer {
        let startSample = sampleIndex(time: start, sampleRate: buffer.sampleRate, channels: buffer.channels)
        let endSample = sampleIndex(time: end, sampleRate: buffer.sampleRate, channels: buffer.channels)

        guard startSample < endSample else { throw DSPError.invalidSelection }

        let bytesPerSample = MemoryLayout<Int16>.size
        let startByte = startSample * bytesPerSample
        let endByte = min(endSample * bytesPerSample, buffer.data.count)

        guard startByte < buffer.data.count else { throw DSPError.invalidSelection }

        let trimmedData = buffer.data[startByte..<endByte]
        return AudioBuffer(data: Data(trimmedData), sampleRate: buffer.sampleRate, channels: buffer.channels)
    }

    static func deleteSelection(_ buffer: AudioBuffer, start: TimeInterval, end: TimeInterval) throws -> AudioBuffer {
        let startSample = sampleIndex(time: start, sampleRate: buffer.sampleRate, channels: buffer.channels)
        let endSample = sampleIndex(time: end, sampleRate: buffer.sampleRate, channels: buffer.channels)

        guard startSample < endSample else { throw DSPError.invalidSelection }

        let bytesPerSample = MemoryLayout<Int16>.size
        let startByte = startSample * bytesPerSample
        let endByte = min(endSample * bytesPerSample, buffer.data.count)

        var result = Data()
        result.append(buffer.data[0..<startByte])
        if endByte < buffer.data.count {
            result.append(buffer.data[endByte...])
        }

        guard !result.isEmpty else { throw DSPError.bufferEmpty }
        return AudioBuffer(data: result, sampleRate: buffer.sampleRate, channels: buffer.channels)
    }

    static func fadeIn(_ buffer: AudioBuffer, start: TimeInterval, end: TimeInterval) throws -> AudioBuffer {
        try applyFade(buffer, start: start, end: end, from: 0.0, to: 1.0)
    }

    static func fadeOut(_ buffer: AudioBuffer, start: TimeInterval, end: TimeInterval) throws -> AudioBuffer {
        try applyFade(buffer, start: start, end: end, from: 1.0, to: 0.0)
    }

    /// Apply a linear gain ramp over the selected time range.
    ///
    /// The ramp is built **per frame** and applied to every channel in that
    /// frame. The previous implementation generated a ramp over interleaved
    /// samples, so stereo L/R received adjacent ramp values (skewed fade).
    private static func applyFade(
        _ buffer: AudioBuffer,
        start: TimeInterval,
        end: TimeInterval,
        from startGain: Float,
        to endGain: Float
    ) throws -> AudioBuffer {
        var floats = AudioBufferConverter.toFloats(buffer)
        let channels = max(buffer.channels, 1)
        let startFrame = Int(start * Double(buffer.sampleRate))
        let endFrame = min(
            Int(end * Double(buffer.sampleRate)),
            floats.count / channels
        )

        guard startFrame < endFrame else { throw DSPError.invalidSelection }

        let frameCount = endFrame - startFrame
        var frameRamp = [Float](repeating: 0, count: frameCount)
        var startVal = startGain
        var endVal = endGain
        vDSP_vgen(&startVal, &endVal, &frameRamp, 1, vDSP_Length(frameCount))

        for frame in 0..<frameCount {
            let gain = frameRamp[frame]
            let base = (startFrame + frame) * channels
            for ch in 0..<channels {
                floats[base + ch] *= gain
            }
        }

        return AudioBufferConverter.fromFloats(floats, sampleRate: buffer.sampleRate, channels: buffer.channels)
    }

    static func normalize(_ buffer: AudioBuffer, mode: NormalizeMode = .peak) throws -> AudioBuffer {
        var floats = AudioBufferConverter.toFloats(buffer)
        guard !floats.isEmpty else { throw DSPError.bufferEmpty }

        switch mode {
        case .peak:
            var peak: Float = 0
            vDSP_maxmgv(floats, 1, &peak, vDSP_Length(floats.count))
            guard peak > 0.001 else { return buffer }
            guard peak < 0.99 else { return buffer }
            var scale = 1.0 / peak
            vDSP_vsmul(floats, 1, &scale, &floats, 1, vDSP_Length(floats.count))

        case .rms(let targetDB):
            guard targetDB.isFinite else { return buffer }

            let currentRMS = rmsAmplitude(floats)
            guard currentRMS > 0.000_001 else { return buffer }

            let targetRMS = pow(10.0, targetDB / 20.0)
            var peak: Float = 0
            vDSP_maxmgv(floats, 1, &peak, vDSP_Length(floats.count))
            let unclippedScale = targetRMS / currentRMS
            var scale = peak > 0 ? min(unclippedScale, 1.0 / peak) : unclippedScale
            vDSP_vsmul(floats, 1, &scale, &floats, 1, vDSP_Length(floats.count))
        }

        return AudioBufferConverter.fromFloats(floats, sampleRate: buffer.sampleRate, channels: buffer.channels)
    }

    // MARK: - Helpers

    static func rmsAmplitude(_ samples: [Float]) -> Float {
        guard !samples.isEmpty else { return 0 }

        var rms: Float = 0
        samples.withUnsafeBufferPointer { pointer in
            vDSP_rmsqv(pointer.baseAddress!, 1, &rms, vDSP_Length(samples.count))
        }
        return rms
    }

    static func rmsDBFS(_ samples: [Float]) -> Float {
        20.0 * log10(max(rmsAmplitude(samples), 1e-10))
    }

    static func sampleIndex(time: TimeInterval, sampleRate: Int, channels: Int) -> Int {
        Int(time * Double(sampleRate)) * channels
    }
}
