import Testing
import MLX
import MLXNN
@testable import AnvilCore

@Suite("Lyric Encoder")
struct LyricEncoderTests {
    @Test("LyricEncoder produces [B, T, 2048] from [B, T, 1024] input")
    func lyricEncoderOutputShape() {
        let encoder = LyricEncoder()
        let embeddings = MLXArray.zeros([1, 100, 1024])
        let output = encoder(embeddings: embeddings, attentionMask: nil)
        #expect(output.shape == [1, 100, 2048])
    }

    @Test("LyricEncoder handles short sequences")
    func lyricEncoderShortSeq() {
        let encoder = LyricEncoder()
        let embeddings = MLXArray.zeros([1, 5, 1024])
        let output = encoder(embeddings: embeddings, attentionMask: nil)
        #expect(output.shape == [1, 5, 2048])
    }

    @Test("LyricEncoder respects max sequence length 2048")
    func lyricEncoderMaxLength() {
        let encoder = LyricEncoder()
        let embeddings = MLXArray.zeros([1, 2048, 1024])
        let output = encoder(embeddings: embeddings, attentionMask: nil)
        #expect(output.shape == [1, 2048, 2048])
    }
}
