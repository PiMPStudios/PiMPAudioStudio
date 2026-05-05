import Testing
import MLX
import MLXNN
@testable import AnvilCore

@Suite("Qwen3 Components")
struct Qwen3ModelTests {
    @Test("Qwen3Attention produces correct output shape")
    func qwen3AttentionShape() {
        let attn = Qwen3Attention(
            hiddenSize: 1024,
            numAttentionHeads: 16,
            numKeyValueHeads: 8,
            headDim: 128,
            rmsNormEps: 1e-6
        )
        let input = MLXArray.zeros([1, 10, 1024])
        let rope = RotaryEmbedding(headDim: 128, maxLen: 32768, base: 1_000_000.0)
        let (cos, sin) = rope.forward(seqLen: 10)
        let output = attn(hiddenStates: input, positionCosSin: (cos, sin), attentionMask: nil)
        #expect(output.shape == [1, 10, 1024])
    }

    @Test("Qwen3Layer produces correct output shape")
    func qwen3LayerShape() {
        let layer = Qwen3Layer(
            hiddenSize: 1024,
            intermediateSize: 3072,
            numAttentionHeads: 16,
            numKeyValueHeads: 8,
            headDim: 128,
            rmsNormEps: 1e-6
        )
        let input = MLXArray.zeros([1, 10, 1024])
        let rope = RotaryEmbedding(headDim: 128, maxLen: 32768, base: 1_000_000.0)
        let (cos, sin) = rope.forward(seqLen: 10)
        let output = layer(hiddenStates: input, positionCosSin: (cos, sin), attentionMask: nil)
        #expect(output.shape == [1, 10, 1024])
    }
}
