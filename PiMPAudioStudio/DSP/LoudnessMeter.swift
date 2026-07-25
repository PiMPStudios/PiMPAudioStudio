import Foundation
import Accelerate

/// Level readouts for the editor polish path.
///
/// - Sample peak: max |sample| in the buffer
/// - True-peak estimate: max after 4× linear oversampling (not full ITU-R BS.1770
///   true-peak; good enough to catch many intersample peaks)
/// - Mono peak: peak of (L+R)/2 on stereo (mono-compatibility cue)
/// - Approximate LUFS: ungated mean-square + −0.691 offset (aiming only)
enum LoudnessMeter {
    struct Metrics: Equatable, Sendable {
        var peakDBFS: Float
        var truePeakDBFS: Float
        var monoPeakDBFS: Float?
        var approximateLUFS: Float
        var duration: TimeInterval
        var sampleRate: Int
        var channels: Int

        var isStereo: Bool { channels >= 2 }

        /// Headroom under a 0 dBFS full-scale sample peak.
        var samplePeakHeadroomDB: Float { -peakDBFS }

        /// True-peak overshoot vs sample peak (dB); positive means TP hotter than sample peak.
        var truePeakOvershootDB: Float { truePeakDBFS - peakDBFS }
    }

    static func measure(_ buffer: AudioBuffer) -> Metrics {
        let samples = AudioBufferConverter.toFloats(buffer)
        let channels = max(buffer.channels, 1)
        return Metrics(
            peakDBFS: peakDBFS(samples),
            truePeakDBFS: truePeakDBFS(samples, channels: channels),
            monoPeakDBFS: channels >= 2 ? monoPeakDBFS(samples, channels: channels) : nil,
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
        return linearToDBFS(peak)
    }

    /// 4× linear-oversampled peak estimate (cheap true-peak proxy).
    static func truePeakDBFS(_ samples: [Float], channels: Int = 1) -> Float {
        guard !samples.isEmpty else { return -120 }
        let ch = max(channels, 1)
        let frameCount = samples.count / ch
        guard frameCount > 1 else { return peakDBFS(samples) }

        var peak: Float = 0
        for c in 0..<ch {
            var prev = samples[c]
            peak = max(peak, abs(prev))
            for frame in 1..<frameCount {
                let next = samples[frame * ch + c]
                // Linear intermediates at 1/4, 2/4, 3/4 between samples
                let d = next - prev
                peak = max(peak, abs(prev + 0.25 * d))
                peak = max(peak, abs(prev + 0.50 * d))
                peak = max(peak, abs(prev + 0.75 * d))
                peak = max(peak, abs(next))
                prev = next
            }
        }
        return linearToDBFS(peak)
    }

    /// Peak of the mono mid signal (L+R)/2 for stereo compatibility checks.
    static func monoPeakDBFS(_ samples: [Float], channels: Int) -> Float {
        let ch = max(channels, 1)
        guard ch >= 2, !samples.isEmpty else { return peakDBFS(samples) }
        let frameCount = samples.count / ch
        var peak: Float = 0
        for frame in 0..<frameCount {
            let base = frame * ch
            let mid = 0.5 * (samples[base] + samples[base + 1])
            peak = max(peak, abs(mid))
        }
        return linearToDBFS(peak)
    }

    /// Fold interleaved stereo (or multi-channel) to mono mid for preview.
    static func foldToMono(_ buffer: AudioBuffer) -> AudioBuffer {
        let samples = AudioBufferConverter.toFloats(buffer)
        let channels = max(buffer.channels, 1)
        guard channels >= 2 else { return buffer }

        let frameCount = samples.count / channels
        var mono = [Float](repeating: 0, count: frameCount)
        for frame in 0..<frameCount {
            let base = frame * channels
            // Average first two channels (typical stereo L/R)
            mono[frame] = 0.5 * (samples[base] + samples[base + 1])
        }
        return AudioBufferConverter.fromFloats(
            mono,
            sampleRate: buffer.sampleRate,
            channels: 1
        )
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

    private static func linearToDBFS(_ linear: Float) -> Float {
        guard linear > 1e-10 else { return -120 }
        return 20.0 * log10(linear)
    }
}
