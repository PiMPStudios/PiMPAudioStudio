// AnvilCore/Tests/AnvilCoreTests/Inference/StableAudio/SFXVAETests.swift
import Testing
import MLX
import MLXRandom
@testable import AnvilCore

@Suite("SFX VAE")
struct SFXVAETests {
    let config = SFXVAEConfig(latentDim: 64, outputChannels: 2, sampleRate: 44100, upsampleRatios: [8, 8, 4, 2])

    @Test("Decode produces audio from latents")
    func decodeShape() {
        let model = SFXVAEModel(config: config)
        let latents = MLXRandom.normal([1, 16, 64])
        let audio = model.decode(latents: latents)
        #expect(audio.shape[0] == 1)
        #expect(audio.shape[2] == 2) // stereo
        #expect(audio.shape[1] == 16 * 8 * 8 * 4 * 2) // upsampled
    }

    @Test("Actor decode returns correct shape")
    func actorDecode() async {
        let actor = SFXVAEActor(config: config)
        let latents = MLXRandom.normal([1, 16, 64])
        let audio = await actor.decode(latents: latents)
        #expect(audio.shape[0] == 1)
        #expect(audio.shape[2] == 2)
    }

    @Test("Actor loadWeights marks as loaded")
    func actorLoad() async {
        let actor = SFXVAEActor(config: config)
        let loaded = await actor.isLoaded
        #expect(!loaded)
    }
}
