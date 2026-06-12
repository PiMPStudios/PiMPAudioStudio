import Testing
import Foundation
import MLX
import MLXNN
@testable import PiMPAudioCore

private func makeTestDecoder() -> DiTDecoder {
    DiTDecoder(
        hiddenSize: 256,
        encoderHiddenSize: 256,
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
}

@Suite("LoRA Injector")
struct LoRAInjectorTests {

    private func makeContribution(path: String, id: String) -> LoRAContribution {
        LoRAContribution(
            adapterID: id,
            displayName: nil,
            source: .local,
            loraA: MLXArray.zeros([4, 256]),
            loraB: MLXArray.zeros([256, 4]),
            strength: 1.0,
            scaling: 1.0,
            path: path
        )
    }

    @Test("applyStack attaches only contributions matching each layer's path")
    func applyStackFiltersByPath() {
        let decoder = makeTestDecoder()
        let config = LoRAConfig(rank: 4, targetLayers: [.selfAttention])
        let paths = LoRAInjector.inject(model: decoder, config: config)
        #expect(paths.contains("layers.0.selfAttn.qProj"))
        #expect(paths.contains("layers.1.selfAttn.qProj"))

        let c0 = makeContribution(path: "layers.0.selfAttn.qProj", id: "adapter-a")
        let c1 = makeContribution(path: "layers.1.selfAttn.qProj", id: "adapter-b")
        LoRAInjector.applyStack(model: decoder, contributions: [c0, c1], paths: paths)

        let q0 = decoder.layers[0].selfAttn.qProj as? LoRALinear
        let q1 = decoder.layers[1].selfAttn.qProj as? LoRALinear
        let v0 = decoder.layers[0].selfAttn.vProj as? LoRALinear
        #expect(q0?.contributions.map(\.adapterID) == ["adapter-a"])
        #expect(q1?.contributions.map(\.adapterID) == ["adapter-b"])
        #expect(v0?.contributions.isEmpty == true)
    }

    @Test("LoRA inject refuses to wrap a layer already wrapped by LoKR")
    func loraInjectSkipsLoKRWrappedLayer() {
        let decoder = makeTestDecoder()
        let lokrPaths = LoKRInjector.inject(
            model: decoder,
            config: LoKRConfig(linearDim: 4, targetLayers: [.selfAttention])
        )
        #expect(!lokrPaths.isEmpty)

        let loraPaths = LoRAInjector.inject(
            model: decoder,
            config: LoRAConfig(rank: 4, targetLayers: [.selfAttention])
        )
        #expect(loraPaths.isEmpty)
        #expect(decoder.layers[0].selfAttn.qProj is LoKRLinear)
    }

    @Test("LoKR inject refuses to wrap a layer already wrapped by LoRA")
    func lokrInjectSkipsLoRAWrappedLayer() {
        let decoder = makeTestDecoder()
        let loraPaths = LoRAInjector.inject(
            model: decoder,
            config: LoRAConfig(rank: 4, targetLayers: [.selfAttention])
        )
        #expect(!loraPaths.isEmpty)

        let lokrPaths = LoKRInjector.inject(
            model: decoder,
            config: LoKRConfig(linearDim: 4, targetLayers: [.selfAttention])
        )
        #expect(lokrPaths.isEmpty)
        #expect(decoder.layers[0].selfAttn.qProj is LoRALinear)
    }
}
