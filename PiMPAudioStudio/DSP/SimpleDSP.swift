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

    /// Normalize peak to a headroom target (default −1 dBFS) or scale by RMS.
    static func normalize(
        _ buffer: AudioBuffer,
        mode: NormalizeMode = .peakDefault
    ) throws -> AudioBuffer {
        var floats = AudioBufferConverter.toFloats(buffer)
        guard !floats.isEmpty else { throw DSPError.bufferEmpty }

        switch mode {
        case .peak(let targetDBFS):
            var peak: Float = 0
            vDSP_maxmgv(floats, 1, &peak, vDSP_Length(floats.count))
            guard peak > 0.000_001 else { return buffer }

            let clampedTargetDB = min(max(targetDBFS, -120), 0)
            let targetLinear = pow(10.0, clampedTargetDB / 20.0)
            var scale = min(targetLinear / peak, 1.0 / peak)
            // Avoid no-op numerical noise when already on target.
            if abs(scale - 1.0) < 0.000_5 { return buffer }
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

    /// Remove leading/trailing silence and optionally apply short edge fades.
    static func trimSilence(
        _ buffer: AudioBuffer,
        parameters: TrimSilenceParameters = .default
    ) throws -> AudioBuffer {
        let floats = AudioBufferConverter.toFloats(buffer)
        guard !floats.isEmpty else { throw DSPError.bufferEmpty }

        let channels = max(buffer.channels, 1)
        let frameCount = floats.count / channels
        guard frameCount > 1 else { return buffer }

        let thresholdLinear = pow(10.0, min(parameters.thresholdDBFS, 0) / 20.0)
        let maxLeadFrames = max(0, Int(Double(parameters.maxLeadingSeconds) * Double(buffer.sampleRate)))
        let maxTrailFrames = max(0, Int(Double(parameters.maxTrailingSeconds) * Double(buffer.sampleRate)))

        var firstSound = 0
        while firstSound < frameCount {
            if framePeak(floats, frame: firstSound, channels: channels) >= thresholdLinear {
                break
            }
            firstSound += 1
        }

        var lastSound = frameCount - 1
        while lastSound > firstSound {
            if framePeak(floats, frame: lastSound, channels: channels) >= thresholdLinear {
                break
            }
            lastSound -= 1
        }

        // Entire buffer below threshold — keep as-is (avoid erasing quiet pads).
        if firstSound >= frameCount {
            return buffer
        }

        let leadTrim = min(firstSound, maxLeadFrames)
        let trailCandidate = frameCount - 1 - lastSound
        let trailTrim = min(max(0, trailCandidate), maxTrailFrames)
        let keepStart = leadTrim
        let keepEnd = frameCount - trailTrim
        guard keepStart < keepEnd else { return buffer }

        if keepStart == 0, keepEnd == frameCount, parameters.edgeFadeMilliseconds <= 0 {
            return buffer
        }

        let startTime = Double(keepStart) / Double(buffer.sampleRate)
        let endTime = Double(keepEnd) / Double(buffer.sampleRate)
        var result = try trimToSelection(buffer, start: startTime, end: endTime)

        let fadeSeconds = max(0, Double(parameters.edgeFadeMilliseconds) / 1_000.0)
        if fadeSeconds > 0, result.duration > fadeSeconds * 2 {
            result = try fadeIn(result, start: 0, end: fadeSeconds)
            let fadeStart = max(0, result.duration - fadeSeconds)
            result = try fadeOut(result, start: fadeStart, end: result.duration)
        }

        return result
    }

    /// 2-pole high-pass filter (Butterworth Q ≈ 0.707), per channel.
    static func highPass(
        _ buffer: AudioBuffer,
        parameters: HighPassParameters = .default
    ) throws -> AudioBuffer {
        var floats = AudioBufferConverter.toFloats(buffer)
        guard !floats.isEmpty else { throw DSPError.bufferEmpty }

        let sampleRate = Float(max(buffer.sampleRate, 1))
        let nyquist = sampleRate * 0.49
        let cutoff = min(max(parameters.cutoffHz, 5), nyquist)
        let coeffs = butterworthHighPassCoefficients(cutoffHz: cutoff, sampleRate: sampleRate)
        let channels = max(buffer.channels, 1)
        let frameCount = floats.count / channels

        for ch in 0..<channels {
            var x1: Float = 0
            var x2: Float = 0
            var y1: Float = 0
            var y2: Float = 0
            for frame in 0..<frameCount {
                let index = frame * channels + ch
                let x0 = floats[index]
                let y0 = coeffs.b0 * x0 + coeffs.b1 * x1 + coeffs.b2 * x2
                    - coeffs.a1 * y1 - coeffs.a2 * y2
                floats[index] = y0
                x2 = x1
                x1 = x0
                y2 = y1
                y1 = y0
            }
        }

        return AudioBufferConverter.fromFloats(floats, sampleRate: buffer.sampleRate, channels: buffer.channels)
    }

    // MARK: - Filter helpers

    private struct BiquadCoefficients {
        var b0: Float
        var b1: Float
        var b2: Float
        var a1: Float
        var a2: Float
    }

    private static func butterworthHighPassCoefficients(
        cutoffHz: Float,
        sampleRate: Float
    ) -> BiquadCoefficients {
        let w0 = 2.0 * Double.pi * Double(cutoffHz) / Double(sampleRate)
        let cosw = cos(w0)
        let sinw = sin(w0)
        let q = 1.0 / sqrt(2.0)
        let alpha = sinw / (2.0 * q)
        let b0 = (1.0 + cosw) / 2.0
        let b1 = -(1.0 + cosw)
        let b2 = (1.0 + cosw) / 2.0
        let a0 = 1.0 + alpha
        let a1 = -2.0 * cosw
        let a2 = 1.0 - alpha
        return BiquadCoefficients(
            b0: Float(b0 / a0),
            b1: Float(b1 / a0),
            b2: Float(b2 / a0),
            a1: Float(a1 / a0),
            a2: Float(a2 / a0)
        )
    }

    private static func framePeak(_ floats: [Float], frame: Int, channels: Int) -> Float {
        var peak: Float = 0
        let base = frame * channels
        for ch in 0..<channels {
            peak = max(peak, abs(floats[base + ch]))
        }
        return peak
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
