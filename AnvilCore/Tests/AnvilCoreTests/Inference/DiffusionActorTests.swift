import Testing
import MLX
@testable import AnvilCore

@Suite("Diffusion Actor")
struct DiffusionActorTests {
    private func makeSmallDiT() -> DiTActor {
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
        return DiTActor(decoder: decoder)
    }

    @Test("Generate returns latents with correct shape")
    func generateShape() async {
        let dit = makeSmallDiT()
        let actor = DiffusionActor(dit: dit)
        let encoderStates = MLXArray.zeros([1, 50, 256])
        let contextLatents = MLXArray.zeros([1, 20, 128])
        let config = DiffusionConfig(seed: 42, steps: 4)

        let result = await actor.generate(
            encoderStates: encoderStates,
            contextLatents: contextLatents,
            shape: (batch: 1, frames: 20, channels: 64),
            config: config
        )
        #expect(result.latents.shape == [1, 20, 64])
        #expect(result.timeCosts.totalTime > 0)
    }

    @Test("Deterministic with same seed")
    func deterministic() async {
        let dit = makeSmallDiT()
        let actor = DiffusionActor(dit: dit)
        let encoderStates = MLXArray.zeros([1, 20, 256])
        let contextLatents = MLXArray.zeros([1, 10, 128])
        let config = DiffusionConfig(seed: 123, steps: 2)

        let r1 = await actor.generate(
            encoderStates: encoderStates, contextLatents: contextLatents,
            shape: (batch: 1, frames: 10, channels: 64), config: config
        )
        let r2 = await actor.generate(
            encoderStates: encoderStates, contextLatents: contextLatents,
            shape: (batch: 1, frames: 10, channels: 64), config: config
        )
        eval(r1.latents, r2.latents)
        let diff = abs(r1.latents - r2.latents).sum()
        eval(diff)
        #expect(diff.item(Float.self) < 1e-4)
    }

    @Test("SDE method produces different results each run without seed")
    func sdeVariation() async {
        let dit = makeSmallDiT()
        let actor = DiffusionActor(dit: dit)
        let encoderStates = MLXArray.zeros([1, 20, 256])
        let contextLatents = MLXArray.zeros([1, 10, 128])
        let config = DiffusionConfig(seed: nil, method: .sde, steps: 2)

        let r1 = await actor.generate(
            encoderStates: encoderStates, contextLatents: contextLatents,
            shape: (batch: 1, frames: 10, channels: 64), config: config
        )
        let r2 = await actor.generate(
            encoderStates: encoderStates, contextLatents: contextLatents,
            shape: (batch: 1, frames: 10, channels: 64), config: config
        )
        eval(r1.latents, r2.latents)
        let diff = abs(r1.latents - r2.latents).sum()
        eval(diff)
        #expect(diff.item(Float.self) > 0.01)
    }
}
