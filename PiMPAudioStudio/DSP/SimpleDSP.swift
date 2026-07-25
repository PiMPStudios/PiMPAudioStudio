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

    static func normalize(
        _ buffer: AudioBuffer,
        mode: NormalizeMode = .peakDefault
    ) throws -> AudioBuffer {
        try normalize(FloatAudio(buffer), mode: mode).toAudioBuffer()
    }

    /// Normalize peak to a headroom target (default −1 dBFS) or scale by RMS.
    static func normalize(
        _ audio: FloatAudio,
        mode: NormalizeMode = .peakDefault
    ) throws -> FloatAudio {
        guard !audio.isEmpty else { throw DSPError.bufferEmpty }
        var floats = audio.samples

        switch mode {
        case .peak(let targetDBFS):
            var peak: Float = 0
            vDSP_maxmgv(floats, 1, &peak, vDSP_Length(floats.count))
            guard peak > 0.000_001 else { return audio }

            // Target is clamped at or below full scale, so `targetLinear / peak` is
            // already the safe scale — no separate clip guard is needed.
            let clampedTargetDB = min(max(targetDBFS, -120), 0)
            let targetLinear = pow(10.0, clampedTargetDB / 20.0)
            var scale = targetLinear / peak
            // Avoid no-op numerical noise when already on target.
            if abs(scale - 1.0) < 0.000_5 { return audio }
            vDSP_vsmul(floats, 1, &scale, &floats, 1, vDSP_Length(floats.count))

        case .rms(let targetDB):
            guard targetDB.isFinite else { return audio }

            let currentRMS = rmsAmplitude(floats)
            guard currentRMS > 0.000_001 else { return audio }

            let targetRMS = pow(10.0, targetDB / 20.0)
            var peak: Float = 0
            vDSP_maxmgv(floats, 1, &peak, vDSP_Length(floats.count))
            let unclippedScale = targetRMS / currentRMS
            var scale = peak > 0 ? min(unclippedScale, 1.0 / peak) : unclippedScale
            vDSP_vsmul(floats, 1, &scale, &floats, 1, vDSP_Length(floats.count))
        }

        return audio.replacingSamples(floats)
    }

    static func trimSilence(
        _ buffer: AudioBuffer,
        parameters: TrimSilenceParameters = .default
    ) throws -> AudioBuffer {
        try trimSilence(FloatAudio(buffer), parameters: parameters).toAudioBuffer()
    }

    /// Remove leading/trailing silence and optionally apply short edge fades.
    static func trimSilence(
        _ audio: FloatAudio,
        parameters: TrimSilenceParameters = .default
    ) throws -> FloatAudio {
        guard !audio.isEmpty else { throw DSPError.bufferEmpty }

        let channels = audio.channels
        let frameCount = audio.frameCount
        guard frameCount > 1 else { return audio }

        let thresholdLinear = pow(10.0, min(parameters.thresholdDBFS, 0) / 20.0)
        let maxLeadFrames = max(0, Int(Double(parameters.maxLeadingSeconds) * Double(audio.sampleRate)))
        let maxTrailFrames = max(0, Int(Double(parameters.maxTrailingSeconds) * Double(audio.sampleRate)))

        var firstSound = 0
        while firstSound < frameCount, audio.framePeak(firstSound) < thresholdLinear {
            firstSound += 1
        }

        // Entire buffer below threshold — keep as-is (avoid erasing quiet pads).
        if firstSound >= frameCount { return audio }

        var lastSound = frameCount - 1
        while lastSound > firstSound, audio.framePeak(lastSound) < thresholdLinear {
            lastSound -= 1
        }

        let leadTrim = min(firstSound, maxLeadFrames)
        let trailCandidate = frameCount - 1 - lastSound
        let trailTrim = min(max(0, trailCandidate), maxTrailFrames)
        let keepStart = leadTrim
        let keepEnd = frameCount - trailTrim
        guard keepStart < keepEnd else { return audio }

        // Frame-indexed slice: the old path round-tripped frame → seconds → frame
        // through `trimToSelection`, which could drift a frame on rounding.
        var result = (keepStart == 0 && keepEnd == frameCount)
            ? audio
            : audio.frames(keepStart, keepEnd)

        let fadeFrames = Int(
            (Double(max(0, parameters.edgeFadeMilliseconds)) / 1_000.0) * Double(audio.sampleRate)
        )
        if fadeFrames > 0, result.frameCount > fadeFrames * 2 {
            applyLinearFade(
                &result.samples,
                channels: channels,
                startFrame: 0,
                frameCount: fadeFrames,
                from: 0,
                to: 1
            )
            applyLinearFade(
                &result.samples,
                channels: channels,
                startFrame: result.frameCount - fadeFrames,
                frameCount: fadeFrames,
                from: 1,
                to: 0
            )
        }

        return result
    }

    static func highPass(
        _ buffer: AudioBuffer,
        parameters: HighPassParameters = .default
    ) throws -> AudioBuffer {
        try highPass(FloatAudio(buffer), parameters: parameters).toAudioBuffer()
    }

    /// 2-pole high-pass filter (Butterworth Q ≈ 0.707), per channel.
    static func highPass(
        _ audio: FloatAudio,
        parameters: HighPassParameters = .default
    ) throws -> FloatAudio {
        guard !audio.isEmpty else { throw DSPError.bufferEmpty }

        var floats = audio.samples
        let sampleRate = Float(max(audio.sampleRate, 1))
        let nyquist = sampleRate * 0.49
        let cutoff = min(max(parameters.cutoffHz, 5), nyquist)
        let coeffs = butterworthHighPassCoefficients(cutoffHz: cutoff, sampleRate: sampleRate)
        let channels = audio.channels
        let frameCount = audio.frameCount

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

        return audio.replacingSamples(floats)
    }

    /// Linear gain ramp over `frameCount` frames, applied identically to every channel.
    static func applyLinearFade(
        _ samples: inout [Float],
        channels: Int,
        startFrame: Int,
        frameCount: Int,
        from startGain: Float,
        to endGain: Float
    ) {
        guard frameCount > 0, channels > 0 else { return }
        let totalFrames = samples.count / channels
        let first = max(0, startFrame)
        let last = min(totalFrames, startFrame + frameCount)
        guard first < last else { return }

        var ramp = [Float](repeating: 0, count: frameCount)
        var startVal = startGain
        var endVal = endGain
        vDSP_vgen(&startVal, &endVal, &ramp, 1, vDSP_Length(frameCount))

        for frame in first..<last {
            let gain = ramp[frame - startFrame]
            let base = frame * channels
            for ch in 0..<channels {
                samples[base + ch] *= gain
            }
        }
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
