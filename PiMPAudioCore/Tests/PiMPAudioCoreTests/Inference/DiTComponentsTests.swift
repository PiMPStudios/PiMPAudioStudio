import Testing
import MLX
import MLXNN
import MLXFast
@testable import PiMPAudioCore

@Suite("DiT Components")
struct DiTComponentsTests {
    @Test("RotaryEmbedding produces correct shapes")
    func rotaryEmbeddingShape() {
        let rope = RotaryEmbedding(headDim: 128, maxLen: 32768, base: 1_000_000.0)
        let (cos, sin) = rope.forward(seqLen: 100)
        #expect(cos.shape == [1, 1, 100, 128])
        #expect(sin.shape == [1, 1, 100, 128])
    }

    @Test("SwiGLUMLP output shape matches input")
    func swiGLUShape() {
        let mlp = SwiGLUMLP(hiddenSize: 2048, intermediateSize: 6144)
        let input = MLXArray.zeros([1, 10, 2048])
        let output = mlp(input)
        #expect(output.shape == [1, 10, 2048])
    }

    @Test("DiTAttention self-attention shape")
    func selfAttentionShape() {
        let attn = DiTAttention(
            hiddenSize: 2048,
            numAttentionHeads: 16,
            numKeyValueHeads: 8,
            headDim: 128,
            rmsNormEps: 1e-6,
            attentionBias: false,
            layerIdx: 0,
            isCrossAttention: false,
            slidingWindow: nil
        )
        let input = MLXArray.zeros([1, 10, 2048])
        let rope = RotaryEmbedding(headDim: 128, maxLen: 32768, base: 1_000_000.0)
        let (cos, sin) = rope.forward(seqLen: 10)
        let output = attn(
            hiddenStates: input,
            positionCosSin: (cos, sin),
            attentionMask: nil,
            encoderHiddenStates: nil,
            cache: nil,
            useCache: false
        )
        #expect(output.shape == [1, 10, 2048])
    }

    @Test("DiTAttention cross-attention shape")
    func crossAttentionShape() {
        let attn = DiTAttention(
            hiddenSize: 2048,
            numAttentionHeads: 16,
            numKeyValueHeads: 8,
            headDim: 128,
            rmsNormEps: 1e-6,
            attentionBias: false,
            layerIdx: 0,
            isCrossAttention: true,
            slidingWindow: nil
        )
        let input = MLXArray.zeros([1, 10, 2048])
        let encoderStates = MLXArray.zeros([1, 50, 2048])
        let output = attn(
            hiddenStates: input,
            positionCosSin: nil,
            attentionMask: nil,
            encoderHiddenStates: encoderStates,
            cache: nil,
            useCache: false
        )
        #expect(output.shape == [1, 10, 2048])
    }

    @Test("DiTAttention cross-attention uses cache on second call")
    func crossAttentionCache() {
        let attn = DiTAttention(
            hiddenSize: 2048,
            numAttentionHeads: 16,
            numKeyValueHeads: 8,
            headDim: 128,
            rmsNormEps: 1e-6,
            attentionBias: false,
            layerIdx: 3,
            isCrossAttention: true,
            slidingWindow: nil
        )
        let input = MLXArray.zeros([1, 10, 2048])
        let encoderStates = MLXArray.zeros([1, 50, 2048])
        let cache = CrossAttentionCache()

        _ = attn(
            hiddenStates: input, positionCosSin: nil, attentionMask: nil,
            encoderHiddenStates: encoderStates, cache: cache, useCache: true
        )
        #expect(cache.isUpdated(layerIdx: 3))

        let output2 = attn(
            hiddenStates: input, positionCosSin: nil, attentionMask: nil,
            encoderHiddenStates: encoderStates, cache: cache, useCache: true
        )
        #expect(output2.shape == [1, 10, 2048])
    }

    @Test("TimestepEmbedding output shapes")
    func timestepEmbeddingShape() {
        let embed = TimestepEmbedding(inChannels: 256, timeEmbedDim: 2048)
        let t = MLXArray(converting: [0.5, 0.3])
        let (temb, proj) = embed(t)
        #expect(temb.shape == [2, 2048])
        #expect(proj.shape == [2, 6, 2048])
    }

    @Test("DiTLayer output shape")
    func ditLayerShape() {
        let layer = DiTLayer(
            hiddenSize: 2048,
            intermediateSize: 6144,
            numAttentionHeads: 16,
            numKeyValueHeads: 8,
            headDim: 128,
            rmsNormEps: 1e-6,
            attentionBias: false,
            layerIdx: 0,
            layerType: "full_attention",
            slidingWindow: 128
        )
        let input = MLXArray.zeros([1, 10, 2048])
        let temb = MLXArray.zeros([1, 6, 2048])
        let encoder = MLXArray.zeros([1, 50, 2048])
        let rope = RotaryEmbedding(headDim: 128, maxLen: 32768, base: 1_000_000.0)
        let (cos, sin) = rope.forward(seqLen: 10)
        let output = layer(
            hiddenStates: input,
            positionCosSin: (cos, sin),
            temb: temb,
            selfAttnMask: nil,
            encoderHiddenStates: encoder,
            encoderAttentionMask: nil,
            cache: nil,
            useCache: false
        )
        #expect(output.shape == [1, 10, 2048])
    }

    @Test("createSlidingWindowMask shape and values")
    func slidingWindowMask() {
        let mask = createSlidingWindowMask(seqLen: 8, windowSize: 2)
        #expect(mask.shape == [1, 1, 8, 8])
        eval(mask)
    }
}
