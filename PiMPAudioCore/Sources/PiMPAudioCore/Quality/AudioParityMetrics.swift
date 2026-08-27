import Foundation
import MLX

public enum AudioParityMetricsError: Error, LocalizedError, Sendable {
    case unsupportedAudioShape([Int])
    case channelMismatch(swiftChannels: Int, referenceChannels: Int)
    case emptyAudio

    public var errorDescription: String? {
        switch self {
        case let .unsupportedAudioShape(shape):
            "Unsupported audio tensor shape \(shape). Expected [1, frames, channels], [frames, channels], or [channels, frames]."
        case let .channelMismatch(swiftChannels, referenceChannels):
            "Swift audio has \(swiftChannels) channels but reference audio has \(referenceChannels)."
        case .emptyAudio:
            "Cannot compare empty audio."
        }
    }
}

public struct AudioComparisonMetrics: Sendable {
    public let comparedFrames: Int
    public let comparedChannels: Int
    public let sampleRate: Int
    public let meanAbsoluteError: Float
    public let meanSquaredError: Float
    public let rootMeanSquaredError: Float
    public let maxAbsoluteDifference: Float
    public let swiftRMS: Float
    public let referenceRMS: Float
    public let swiftHighFrequencyProxy: Float
    public let referenceHighFrequencyProxy: Float
    public let swiftStereoCorrelation: Float?
    public let referenceStereoCorrelation: Float?
    public let rmsEnvelopeCorrelation: Float?

    public var durationSeconds: Double {
        Double(comparedFrames) / Double(sampleRate)
    }
}

public enum AudioParityMetrics {
    public static func normalizeToNLC(_ audio: MLXArray) throws -> MLXArray {
        switch audio.ndim {
        case 3:
            guard audio.shape[0] == 1, audio.shape[1] > 0, audio.shape[2] > 0 else {
                throw AudioParityMetricsError.unsupportedAudioShape(audio.shape)
            }
            return audio
        case 2:
            let first = audio.shape[0]
            let second = audio.shape[1]
            guard first > 0, second > 0 else {
                throw AudioParityMetricsError.unsupportedAudioShape(audio.shape)
            }

            if second <= 8 {
                return audio.reshaped([1, first, second])
            }
            if first <= 8 {
                return audio.transposed(0, 1).reshaped([1, second, first])
            }
            throw AudioParityMetricsError.unsupportedAudioShape(audio.shape)
        default:
            throw AudioParityMetricsError.unsupportedAudioShape(audio.shape)
        }
    }

    public static func compare(
        swiftAudio rawSwiftAudio: MLXArray,
        referenceAudio rawReferenceAudio: MLXArray,
        sampleRate: Int,
        envelopeWindowFrames: Int = 2048
    ) throws -> AudioComparisonMetrics {
        let swiftAudio = try normalizeToNLC(rawSwiftAudio).asType(.float32)
        let referenceAudio = try normalizeToNLC(rawReferenceAudio).asType(.float32)

        let swiftFrames = swiftAudio.shape[1]
        let referenceFrames = referenceAudio.shape[1]
        let frames = min(swiftFrames, referenceFrames)
        let swiftChannels = swiftAudio.shape[2]
        let referenceChannels = referenceAudio.shape[2]
        guard frames > 0 else { throw AudioParityMetricsError.emptyAudio }
        guard swiftChannels == referenceChannels else {
            throw AudioParityMetricsError.channelMismatch(
                swiftChannels: swiftChannels,
                referenceChannels: referenceChannels
            )
        }

        let channels = swiftChannels
        let swiftComparable = swiftAudio[0..., 0 ..< frames, 0 ..< channels]
        let referenceComparable = referenceAudio[0..., 0 ..< frames, 0 ..< channels]
        eval(swiftComparable, referenceComparable)

        let diff = swiftComparable - referenceComparable
        let absDiff = MLX.abs(diff)
        let squaredDiff = diff * diff
        eval(absDiff, squaredDiff)

        let mae = absDiff.mean().item(Float.self)
        let mse = squaredDiff.mean().item(Float.self)

        return AudioComparisonMetrics(
            comparedFrames: frames,
            comparedChannels: channels,
            sampleRate: sampleRate,
            meanAbsoluteError: mae,
            meanSquaredError: mse,
            rootMeanSquaredError: sqrt(max(mse, 0)),
            maxAbsoluteDifference: absDiff.max().item(Float.self),
            swiftRMS: rms(swiftComparable),
            referenceRMS: rms(referenceComparable),
            swiftHighFrequencyProxy: highFrequencyProxy(swiftComparable),
            referenceHighFrequencyProxy: highFrequencyProxy(referenceComparable),
            swiftStereoCorrelation: stereoCorrelation(swiftComparable),
            referenceStereoCorrelation: stereoCorrelation(referenceComparable),
            rmsEnvelopeCorrelation: rmsEnvelopeCorrelation(
                swiftComparable,
                referenceComparable,
                windowFrames: envelopeWindowFrames
            )
        )
    }

    private static func rms(_ audio: MLXArray) -> Float {
        sqrt(max((audio * audio).mean().item(Float.self), 0))
    }

    private static func highFrequencyProxy(_ audio: MLXArray) -> Float {
        let frames = audio.shape[1]
        guard frames > 1 else { return 0 }
        let next = audio[0..., 1 ..< frames, 0 ..< audio.shape[2]]
        let previous = audio[0..., 0 ..< (frames - 1), 0 ..< audio.shape[2]]
        let diff = next - previous
        return rms(diff)
    }

    private static func stereoCorrelation(_ audio: MLXArray) -> Float? {
        guard audio.shape[2] >= 2 else { return nil }
        let frames = audio.shape[1]
        let left = audio[0, 0 ..< frames, 0]
        let right = audio[0, 0 ..< frames, 1]
        return correlation(left, right)
    }

    private static func rmsEnvelopeCorrelation(
        _ swiftAudio: MLXArray,
        _ referenceAudio: MLXArray,
        windowFrames: Int
    ) -> Float? {
        guard windowFrames > 0 else { return nil }
        let frames = min(swiftAudio.shape[1], referenceAudio.shape[1])
        let windows = frames / windowFrames
        guard windows >= 2 else { return nil }

        let usableFrames = windows * windowFrames
        let channels = swiftAudio.shape[2]
        let swiftWindowed = (swiftAudio[0..., 0 ..< usableFrames, 0 ..< channels] * swiftAudio[
            0...,
            0 ..< usableFrames,
            0 ..< channels
        ])
        .reshaped([windows, windowFrames * channels])
        .mean(axis: 1)
        let referenceWindowed = (referenceAudio[0..., 0 ..< usableFrames, 0 ..< channels] * referenceAudio[
            0...,
            0 ..< usableFrames,
            0 ..< channels
        ])
        .reshaped([windows, windowFrames * channels])
        .mean(axis: 1)

        return correlation(sqrt(swiftWindowed), sqrt(referenceWindowed))
    }

    private static func correlation(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
        let centeredLHS = lhs - lhs.mean()
        let centeredRHS = rhs - rhs.mean()
        let numerator = (centeredLHS * centeredRHS).mean().item(Float.self)
        let lhsPower = (centeredLHS * centeredLHS).mean().item(Float.self)
        let rhsPower = (centeredRHS * centeredRHS).mean().item(Float.self)
        let denominator = sqrt(max(lhsPower * rhsPower, 1e-20))
        return numerator / denominator
    }
}
