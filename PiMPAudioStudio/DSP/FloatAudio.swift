import Accelerate
import Foundation

/// Interleaved float working format for the editor DSP chain.
///
/// Multi-stage processing (notably `PolishForKeepProcessor`) converts to `FloatAudio`
/// **once**, stays in float across every stage, and quantizes back to the Int16
/// `AudioBuffer` **once** at the end.
///
/// Converting between stages is not merely lossy — it is destructive. `fromFloats`
/// clamps to ±1.0, so any stage that legitimately overshoots unity (a loudness
/// boost, compressor makeup gain, EQ/HPF ringing, reverb wet sum) would be hard
/// clipped before the next stage ever saw the peaks. A limiter fed that way can
/// only report success while flattening transients it was supposed to catch.
struct FloatAudio: Equatable, Sendable {
    /// Interleaved samples: `[frame0_ch0, frame0_ch1, frame1_ch0, ...]`.
    var samples: [Float]
    var sampleRate: Int
    var channels: Int

    init(samples: [Float], sampleRate: Int, channels: Int) {
        self.samples = samples
        self.sampleRate = sampleRate
        self.channels = max(channels, 1)
    }

    init(_ buffer: AudioBuffer) {
        self.init(
            samples: AudioBufferConverter.toFloats(buffer),
            sampleRate: buffer.sampleRate,
            channels: buffer.channels
        )
    }

    var frameCount: Int {
        samples.count / channels
    }

    var duration: TimeInterval {
        Double(frameCount) / Double(max(sampleRate, 1))
    }

    var isEmpty: Bool {
        samples.isEmpty
    }

    /// Quantize back to the editor's Int16 buffer format. Clips at ±1.0 — call once,
    /// at the end of a chain.
    func toAudioBuffer() -> AudioBuffer {
        AudioBufferConverter.fromFloats(samples, sampleRate: sampleRate, channels: channels)
    }

    /// Replace the sample payload while keeping rate/channel layout.
    func replacingSamples(_ newSamples: [Float]) -> FloatAudio {
        FloatAudio(samples: newSamples, sampleRate: sampleRate, channels: channels)
    }

    /// Max |sample| across all channels.
    func peakAmplitude() -> Float {
        guard !samples.isEmpty else { return 0 }
        var peak: Float = 0
        vDSP_maxmgv(samples, 1, &peak, vDSP_Length(samples.count))
        return peak
    }

    /// Max |sample| within a single frame, across channels.
    func framePeak(_ frame: Int) -> Float {
        var peak: Float = 0
        let base = frame * channels
        for channel in 0 ..< channels {
            peak = max(peak, abs(samples[base + channel]))
        }
        return peak
    }

    /// Frames `start..<end` as a new value. Bounds are clamped.
    func frames(_ start: Int, _ end: Int) -> FloatAudio {
        let lower = max(0, min(start, frameCount))
        let upper = max(lower, min(end, frameCount))
        return replacingSamples(Array(samples[(lower * channels) ..< (upper * channels)]))
    }
}
