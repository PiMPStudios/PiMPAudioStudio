import MLX
import Testing
@testable import PiMPAudioCore

@Suite("Audio parity metrics", .disabled("SwiftPM test bundles cannot load MLX default.metallib reliably; QualityEval is build-verified instead."))
struct AudioParityMetricsTests {
    @Test("Identical audio has zero difference")
    func identicalAudioHasZeroDifference() throws {
        try Device.withDefaultDevice(.cpu) {
            let audio = MLXArray.zeros([1, 8, 2])

            let metrics = try AudioParityMetrics.compare(
                swiftAudio: audio,
                referenceAudio: audio,
                sampleRate: 48_000,
                envelopeWindowFrames: 4
            )

            #expect(metrics.meanAbsoluteError == 0)
            #expect(metrics.meanSquaredError == 0)
            #expect(metrics.rootMeanSquaredError == 0)
            #expect(metrics.maxAbsoluteDifference == 0)
            #expect(metrics.comparedFrames == 8)
            #expect(metrics.comparedChannels == 2)
        }
    }

    @Test("Layout normalization accepts channel-first audio")
    func normalizesChannelFirstAudio() throws {
        try Device.withDefaultDevice(.cpu) {
            let channelFirst = MLXArray(converting: [
                0.1, 0.2, 0.3,
                0.4, 0.5, 0.6,
            ], [2, 3])

            let normalized = try AudioParityMetrics.normalizeToNLC(channelFirst)

            #expect(normalized.shape == [1, 3, 2])
        }
    }

    @Test("Reports channel mismatch")
    func reportsChannelMismatch() throws {
        Device.withDefaultDevice(.cpu) {
            let stereo = MLXArray.zeros([1, 8, 2])
            let mono = MLXArray.zeros([1, 8, 1])

            #expect(throws: AudioParityMetricsError.self) {
                _ = try AudioParityMetrics.compare(
                    swiftAudio: stereo,
                    referenceAudio: mono,
                    sampleRate: 48_000
                )
            }
        }
    }
}
