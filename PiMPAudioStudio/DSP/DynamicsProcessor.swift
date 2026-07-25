import Foundation
import Accelerate

/// Stereo-linked glue compressor, peak limiter, and loudness staging for editor polish P1.
enum DynamicsProcessor {
    enum DynamicsError: LocalizedError {
        case bufferEmpty

        var errorDescription: String? {
            switch self {
            case .bufferEmpty:
                return "The audio buffer is empty."
            }
        }
    }

    // MARK: - Compress

    /// Feed-forward peak compressor with soft knee; stereo channels share one gain envelope.
    static func compress(
        _ buffer: AudioBuffer,
        parameters: CompressorParameters = .default
    ) throws -> AudioBuffer {
        var floats = AudioBufferConverter.toFloats(buffer)
        guard !floats.isEmpty else { throw DynamicsError.bufferEmpty }

        let channels = max(buffer.channels, 1)
        let frameCount = floats.count / channels
        let sampleRate = max(Float(buffer.sampleRate), 1)
        let threshold = parameters.thresholdDBFS
        let ratio = max(parameters.ratio, 1.001)
        let knee = max(parameters.kneeDB, 0)
        let makeupLinear = pow(10.0, parameters.makeupGainDB / 20.0)

        let attackCoef = envelopeCoefficient(
            timeMilliseconds: max(parameters.attackMilliseconds, 0.1),
            sampleRate: sampleRate
        )
        let releaseCoef = envelopeCoefficient(
            timeMilliseconds: max(parameters.releaseMilliseconds, 1),
            sampleRate: sampleRate
        )

        var envelopeDB: Float = -120
        for frame in 0..<frameCount {
            var peak: Float = 0
            let base = frame * channels
            for ch in 0..<channels {
                peak = max(peak, abs(floats[base + ch]))
            }
            let inputDB = peak > 1e-10 ? 20.0 * log10(peak) : -120

            if inputDB > envelopeDB {
                envelopeDB = attackCoef * envelopeDB + (1 - attackCoef) * inputDB
            } else {
                envelopeDB = releaseCoef * envelopeDB + (1 - releaseCoef) * inputDB
            }

            let grDB = gainReductionDB(
                levelDB: envelopeDB,
                thresholdDB: threshold,
                ratio: ratio,
                kneeDB: knee
            )
            let gain = pow(10.0, grDB / 20.0) * makeupLinear
            for ch in 0..<channels {
                floats[base + ch] *= gain
            }
        }

        return AudioBufferConverter.fromFloats(
            floats,
            sampleRate: buffer.sampleRate,
            channels: buffer.channels
        )
    }

    // MARK: - Limit

    /// Peak limiter toward a ceiling with fast attack and configurable release.
    static func limit(
        _ buffer: AudioBuffer,
        parameters: LimiterParameters = .default
    ) throws -> AudioBuffer {
        var floats = AudioBufferConverter.toFloats(buffer)
        guard !floats.isEmpty else { throw DynamicsError.bufferEmpty }

        let channels = max(buffer.channels, 1)
        let frameCount = floats.count / channels
        let sampleRate = max(Float(buffer.sampleRate), 1)
        let ceilingDB = min(max(parameters.ceilingDBFS, -24), 0)
        let ceilingLinear = pow(10.0, ceilingDB / 20.0)

        // Very fast attack so peaks are caught; release smoothed.
        let attackCoef = envelopeCoefficient(timeMilliseconds: 0.5, sampleRate: sampleRate)
        let releaseCoef = envelopeCoefficient(
            timeMilliseconds: max(parameters.releaseMilliseconds, 5),
            sampleRate: sampleRate
        )

        var envelope: Float = 0
        for frame in 0..<frameCount {
            var peak: Float = 0
            let base = frame * channels
            for ch in 0..<channels {
                peak = max(peak, abs(floats[base + ch]))
            }

            if peak > envelope {
                envelope = attackCoef * envelope + (1 - attackCoef) * peak
            } else {
                envelope = releaseCoef * envelope + (1 - releaseCoef) * peak
            }

            var gain: Float = 1
            if envelope > ceilingLinear && envelope > 1e-10 {
                gain = ceilingLinear / envelope
            }
            for ch in 0..<channels {
                var sample = floats[base + ch] * gain
                // Hard safety clamp at ceiling (after smooth GR).
                if sample > ceilingLinear { sample = ceilingLinear }
                if sample < -ceilingLinear { sample = -ceilingLinear }
                floats[base + ch] = sample
            }
        }

        return AudioBufferConverter.fromFloats(
            floats,
            sampleRate: buffer.sampleRate,
            channels: buffer.channels
        )
    }

    // MARK: - Loudness staging

    /// Gain toward a target approximate LUFS, then limit to a peak ceiling.
    static func aimLoudness(
        _ buffer: AudioBuffer,
        parameters: LoudnessParameters = .default
    ) throws -> AudioBuffer {
        var floats = AudioBufferConverter.toFloats(buffer)
        guard !floats.isEmpty else { throw DynamicsError.bufferEmpty }

        let current = LoudnessMeter.approximateLUFS(floats)
        let target = min(max(parameters.targetLUFS, -70), 0)
        var deltaDB = target - current
        let maxBoost = max(parameters.maxBoostDB, 0)
        // Cap boost; allow large cuts so hot masters can still come down.
        if deltaDB > maxBoost { deltaDB = maxBoost }
        if deltaDB < -36 { deltaDB = -36 }

        if abs(deltaDB) > 0.05 {
            var scale = pow(10.0, deltaDB / 20.0)
            vDSP_vsmul(floats, 1, &scale, &floats, 1, vDSP_Length(floats.count))
        }

        let gained = AudioBufferConverter.fromFloats(
            floats,
            sampleRate: buffer.sampleRate,
            channels: buffer.channels
        )
        return try limit(
            gained,
            parameters: LimiterParameters(
                ceilingDBFS: parameters.ceilingDBFS,
                releaseMilliseconds: 50
            )
        )
    }

    // MARK: - Helpers

    /// Soft-knee compressor gain reduction in dB (negative or zero).
    static func gainReductionDB(
        levelDB: Float,
        thresholdDB: Float,
        ratio: Float,
        kneeDB: Float
    ) -> Float {
        let safeRatio = max(ratio, 1.001)
        let halfKnee = max(kneeDB, 0) * 0.5
        let over = levelDB - thresholdDB

        let compressedOver: Float
        if kneeDB <= 0.001 {
            compressedOver = over <= 0 ? 0 : over
        } else if over < -halfKnee {
            compressedOver = 0
        } else if over > halfKnee {
            compressedOver = over
        } else {
            // Soft knee quadratic region
            let x = over + halfKnee
            compressedOver = (x * x) / (2 * kneeDB)
        }

        if compressedOver <= 0 { return 0 }
        // Output overshoot = compressedOver / ratio; GR = output - input overshoot
        return compressedOver * (1.0 / safeRatio - 1.0)
    }

    private static func envelopeCoefficient(timeMilliseconds: Float, sampleRate: Float) -> Float {
        // Standard one-pole: exp(-1 / (seconds * sampleRate))
        let seconds = max(Double(timeMilliseconds), 0.01) / 1_000.0
        return Float(exp(-1.0 / (seconds * Double(sampleRate))))
    }
}
