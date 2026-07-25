import Foundation
import Accelerate

/// Lightweight level readouts for the editor polish path.
///
/// Peak is true sample peak (not intersample true-peak). Approximate LUFS is an
/// ungated mean-square estimate with the BS.1770 offset — good for aiming, not
/// for broadcast certification.
enum LoudnessMeter {
    struct Metrics: Equatable, Sendable {
        var peakDBFS: Float
        var approximateLUFS: Float
        var duration: TimeInterval
        var sampleRate: Int
        var channels: Int
    }

    static func measure(_ buffer: AudioBuffer) -> Metrics {
        let samples = AudioBufferConverter.toFloats(buffer)
        return Metrics(
            peakDBFS: peakDBFS(samples),
            approximateLUFS: approximateLUFS(samples),
            duration: buffer.duration,
            sampleRate: buffer.sampleRate,
            channels: buffer.channels
        )
    }

    static func peakDBFS(_ samples: [Float]) -> Float {
        guard !samples.isEmpty else { return -120 }
        var peak: Float = 0
        vDSP_maxmgv(samples, 1, &peak, vDSP_Length(samples.count))
        guard peak > 1e-10 else { return -120 }
        return 20.0 * log10(peak)
    }

    /// Ungated mean-square loudness with the −0.691 LUFS offset (no full K-filter).
    static func approximateLUFS(_ samples: [Float]) -> Float {
        guard !samples.isEmpty else { return -70 }
        var meanSquare: Float = 0
        vDSP_measqv(samples, 1, &meanSquare, vDSP_Length(samples.count))
        guard meanSquare > 1e-12 else { return -70 }
        return -0.691 + 10.0 * log10(meanSquare)
    }

    static func formatDBFS(_ value: Float) -> String {
        if value <= -119.5 { return "−∞ dB" }
        return String(format: "%.1f dB", value)
    }

    static func formatLUFS(_ value: Float) -> String {
        if value <= -69.5 { return "−∞ LUFS" }
        return String(format: "%.1f LUFS", value)
    }
}
