import Testing
import MLX
@testable import AnvilCore

@Suite("TimingEncoder")
struct TimingEncoderTests {
    @Test("Produces correct output shape")
    func outputShape() {
        let encoder = TimingEncoder(embeddingDim: 768)
        let result = encoder.encode(startSeconds: 0.0, durationSeconds: 30.0)
        #expect(result.shape == [1, 768])
    }

    @Test("Different durations produce different embeddings")
    func differentDurations() {
        let encoder = TimingEncoder(embeddingDim: 768)
        let a = encoder.encode(startSeconds: 0.0, durationSeconds: 10.0)
        let b = encoder.encode(startSeconds: 0.0, durationSeconds: 30.0)
        let diff = abs(a - b).sum()
        #expect(diff.item(Float.self) > 0.0)
    }

    @Test("Different start times produce different embeddings")
    func differentStartTimes() {
        let encoder = TimingEncoder(embeddingDim: 768)
        let a = encoder.encode(startSeconds: 0.0, durationSeconds: 30.0)
        let b = encoder.encode(startSeconds: 5.0, durationSeconds: 30.0)
        let diff = abs(a - b).sum()
        #expect(diff.item(Float.self) > 0.0)
    }

    @Test("Output values are bounded")
    func valuesBounded() {
        let encoder = TimingEncoder(embeddingDim: 768)
        let result = encoder.encode(startSeconds: 0.0, durationSeconds: 47.0)
        let maxVal = abs(result).max().item(Float.self)
        #expect(maxVal <= 1.0)
    }
}
