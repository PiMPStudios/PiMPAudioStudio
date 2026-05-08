import Testing
import MLX
@testable import AnvilCore

@Suite("DiT Model")
struct DiTModelTests {
    @Test("DiTDecoder forward pass shape")
    func forwardPassShape() {
        let decoder = DiTDecoder(
            hiddenSize: 256,
            intermediateSize: 512,
            numHiddenLayers: 2,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            headDim: 64,
            rmsNormEps: 1e-6,
            attentionBias: false,
            inChannels: 192,
            audioAcousticHiddenDim: 64,
            patchSize: 2,
            slidingWindow: 128,
            layerTypes: ["sliding_attention", "full_attention"],
            ropeTheta: 1_000_000.0,
            maxPositionEmbeddings: 32768
        )
        let hiddenStates = MLXArray.zeros([1, 20, 64])
        let timestep = MLXArray.zeros([1])
        let timestepR = MLXArray.zeros([1])
        let encoderStates = MLXArray.zeros([1, 50, 256])
        let contextLatents = MLXArray.zeros([1, 20, 128])

        let (output, _) = decoder(
            hiddenStates: hiddenStates,
            timestep: timestep,
            timestepR: timestepR,
            encoderHiddenStates: encoderStates,
            contextLatents: contextLatents,
            encoderAttentionMask: nil,
            cache: nil,
            useCache: false
        )
        #expect(output.shape == [1, 20, 64])
    }

    @Test("DiTDecoder with cache returns cache")
    func forwardWithCache() {
        let decoder = DiTDecoder(
            hiddenSize: 256,
            intermediateSize: 512,
            numHiddenLayers: 2,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            headDim: 64,
            rmsNormEps: 1e-6,
            attentionBias: false,
            inChannels: 192,
            audioAcousticHiddenDim: 64,
            patchSize: 2,
            slidingWindow: 128,
            layerTypes: ["sliding_attention", "full_attention"],
            ropeTheta: 1_000_000.0,
            maxPositionEmbeddings: 32768
        )
        let hiddenStates = MLXArray.zeros([1, 20, 64])
        let timestep = MLXArray.zeros([1])
        let timestepR = MLXArray.zeros([1])
        let encoderStates = MLXArray.zeros([1, 50, 256])
        let contextLatents = MLXArray.zeros([1, 20, 128])
        let cache = CrossAttentionCache()

        let (_, returnedCache) = decoder(
            hiddenStates: hiddenStates,
            timestep: timestep,
            timestepR: timestepR,
            encoderHiddenStates: encoderStates,
            contextLatents: contextLatents,
            encoderAttentionMask: nil,
            cache: cache,
            useCache: true
        )
        #expect(returnedCache != nil)
        #expect(cache.isUpdated(layerIdx: 0))
        #expect(cache.isUpdated(layerIdx: 1))
    }

    @Test("DiTDecoder handles odd sequence lengths (padding)")
    func oddSequenceLength() {
        let decoder = DiTDecoder(
            hiddenSize: 256,
            intermediateSize: 512,
            numHiddenLayers: 2,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            headDim: 64,
            rmsNormEps: 1e-6,
            attentionBias: false,
            inChannels: 192,
            audioAcousticHiddenDim: 64,
            patchSize: 2,
            slidingWindow: 128,
            layerTypes: ["sliding_attention", "full_attention"],
            ropeTheta: 1_000_000.0,
            maxPositionEmbeddings: 32768
        )
        let hiddenStates = MLXArray.zeros([1, 21, 64])
        let timestep = MLXArray.zeros([1])
        let timestepR = MLXArray.zeros([1])
        let encoderStates = MLXArray.zeros([1, 50, 256])
        let contextLatents = MLXArray.zeros([1, 21, 128])

        let (output, _) = decoder(
            hiddenStates: hiddenStates,
            timestep: timestep,
            timestepR: timestepR,
            encoderHiddenStates: encoderStates,
            contextLatents: contextLatents,
            encoderAttentionMask: nil,
            cache: nil,
            useCache: false
        )
        #expect(output.shape == [1, 21, 64])
    }

    @Test("DiTActor step returns correct shape")
    func actorStep() async throws {
        let decoder = DiTDecoder(
            hiddenSize: 256,
            intermediateSize: 512,
            numHiddenLayers: 2,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            headDim: 64,
            rmsNormEps: 1e-6,
            attentionBias: false,
            inChannels: 192,
            audioAcousticHiddenDim: 64,
            patchSize: 2,
            slidingWindow: 128,
            layerTypes: ["sliding_attention", "full_attention"],
            ropeTheta: 1_000_000.0,
            maxPositionEmbeddings: 32768
        )
        let actor = DiTActor(decoder: decoder)
        let noisyLatents = MLXArray.zeros([1, 20, 64])
        let encoderStates = MLXArray.zeros([1, 50, 256])
        let contextLatents = MLXArray.zeros([1, 20, 128])

        let (output, cache) = try await actor.step(
            noisyLatents: noisyLatents,
            timestep: 0.5,
            timestepR: 0.5,
            encoderStates: encoderStates,
            contextLatents: contextLatents,
            encoderAttentionMask: nil,
            cache: nil,
            useCache: true
        )
        #expect(output.shape == [1, 20, 64])
        #expect(cache != nil)
    }
}
