import Testing
import MLX
import MLXNN
@testable import AnvilCore

@Suite("Timbre Encoder")
struct TimbreEncoderTests {
    @Test("TimbreEncoder produces [B, 5, 2048] from [B, 750, 64]")
    func timbreEncoderOutputShape() {
        let encoder = TimbreEncoder()
        let latents = MLXArray.zeros([1, 750, 64])
        let output = encoder(latents: latents, attentionMask: nil)
        #expect(output.shape == [1, 5, 2048])
    }

    @Test("TimbreEncoder silence fallback (zeros input)")
    func timbreEncoderSilenceFallback() {
        let encoder = TimbreEncoder()
        let silence = MLXArray.zeros([1, 750, 64])
        let output = encoder(latents: silence, attentionMask: nil)
        #expect(output.shape == [1, 5, 2048])
    }

    @Test("AttentionPooler compresses sequence to query count")
    func attentionPoolerShape() {
        let pooler = AttentionPooler(
            hiddenSize: 2048,
            numQueries: 5,
            numLayers: 2,
            numAttentionHeads: 16,
            numKeyValueHeads: 8,
            headDim: 128,
            intermediateSize: 6144,
            rmsNormEps: 1e-6
        )
        let encoded = MLXArray.zeros([1, 750, 2048])
        let output = pooler(encodedSequence: encoded)
        #expect(output.shape == [1, 5, 2048])
    }
}
