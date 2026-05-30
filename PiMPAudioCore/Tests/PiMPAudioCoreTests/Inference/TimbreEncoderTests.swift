import Testing
import MLX
import MLXNN
@testable import PiMPAudioCore

@Suite("Timbre Encoder")
struct TimbreEncoderTests {
    @Test("TimbreEncoder produces [B, 1, 2048] from [B, 750, 64]")
    func timbreEncoderOutputShape() {
        let encoder = TimbreEncoder()
        let latents = MLXArray.zeros([1, 750, 64])
        let output = encoder(latents: latents, attentionMask: nil)
        #expect(output.shape == [1, 1, 2048])
    }

    @Test("TimbreEncoder silence fallback (zeros input)")
    func timbreEncoderSilenceFallback() {
        let encoder = TimbreEncoder()
        let silence = MLXArray.zeros([1, 750, 64])
        let output = encoder(latents: silence, attentionMask: nil)
        #expect(output.shape == [1, 1, 2048])
    }
}
