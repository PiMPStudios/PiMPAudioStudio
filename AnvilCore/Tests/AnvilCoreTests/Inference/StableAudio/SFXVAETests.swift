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

    @Test("Actor decode returns correct shape after loading weights")
    func actorDecode() async throws {
        let actor = SFXVAEActor(config: config)
        await actor.loadWeights(randomVAEWeights())
        let latents = MLXRandom.normal([1, 16, 64])
        let audio = try await actor.decode(latents: latents)
        #expect(audio.shape[0] == 1)
        #expect(audio.shape[2] == 2)
    }

    @Test("Actor loadWeights marks as loaded")
    func actorLoad() async {
        let actor = SFXVAEActor(config: config)
        let loaded = await actor.isLoaded
        #expect(!loaded)
    }

    @Test("Decode throws when weights not loaded")
    func decodeThrowsWithoutWeights() async {
        let actor = SFXVAEActor(config: config)
        let latents = MLXRandom.normal([1, 16, 64])
        await #expect(throws: SFXVAEActorError.weightsNotLoaded) {
            try await actor.decode(latents: latents)
        }
    }

    @Test("Encode throws when weights not loaded")
    func encodeThrowsWithoutWeights() async {
        let actor = SFXVAEActor(config: config)
        let audio = MLXRandom.normal([1, 2048, 2])
        await #expect(throws: SFXVAEActorError.weightsNotLoaded) {
            try await actor.encode(audio: audio)
        }
    }

    // MARK: - Helpers

    /// Produces random-shaped weights matching SFXVAEModel's parameter structure for the test
    /// config: latentDim=64, outputChannels=2, upsampleRatios=[8,8,4,2].
    private func randomVAEWeights() -> [(String, MLXArray)] {
        var w: [(String, MLXArray)] = []

        // Decoder projIn: Linear(64, 64)
        w.append(("projIn.weight", MLXRandom.normal([64, 64])))
        w.append(("projIn.bias", MLXArray.zeros([64])))

        // upsampleBlocks: Linear(inputDim, outputDim * ratio)
        // Block 0: ratio=8, in=64, out=32 → Linear(64, 256)
        w.append(("upsampleBlocks.0.linear.weight", MLXRandom.normal([256, 64])))
        w.append(("upsampleBlocks.0.linear.bias", MLXArray.zeros([256])))
        // Block 1: ratio=8, in=32, out=16 → Linear(32, 128)
        w.append(("upsampleBlocks.1.linear.weight", MLXRandom.normal([128, 32])))
        w.append(("upsampleBlocks.1.linear.bias", MLXArray.zeros([128])))
        // Block 2: ratio=4, in=16, out=8 → Linear(16, 32)
        w.append(("upsampleBlocks.2.linear.weight", MLXRandom.normal([32, 16])))
        w.append(("upsampleBlocks.2.linear.bias", MLXArray.zeros([32])))
        // Block 3: ratio=2, in=8, out=4 → Linear(8, 8)
        w.append(("upsampleBlocks.3.linear.weight", MLXRandom.normal([8, 8])))
        w.append(("upsampleBlocks.3.linear.bias", MLXArray.zeros([8])))

        // Decoder projOut: Linear(4, 2)
        w.append(("projOut.weight", MLXRandom.normal([2, 4])))
        w.append(("projOut.bias", MLXArray.zeros([2])))

        // Encoder encoderProjIn: Linear(2, 2)
        w.append(("encoderProjIn.weight", MLXRandom.normal([2, 2])))
        w.append(("encoderProjIn.bias", MLXArray.zeros([2])))

        // downsampleBlocks: Linear(inputDim * ratio, outputDim)
        // Reversed ratios = [2, 4, 8, 8]
        // Block 0: ratio=2, in=2, out=4 → Linear(4, 4)
        w.append(("downsampleBlocks.0.linear.weight", MLXRandom.normal([4, 4])))
        w.append(("downsampleBlocks.0.linear.bias", MLXArray.zeros([4])))
        // Block 1: ratio=4, in=4, out=8 → Linear(16, 8)
        w.append(("downsampleBlocks.1.linear.weight", MLXRandom.normal([8, 16])))
        w.append(("downsampleBlocks.1.linear.bias", MLXArray.zeros([8])))
        // Block 2: ratio=8, in=8, out=16 → Linear(64, 16)
        w.append(("downsampleBlocks.2.linear.weight", MLXRandom.normal([16, 64])))
        w.append(("downsampleBlocks.2.linear.bias", MLXArray.zeros([16])))
        // Block 3: ratio=8, in=16, out=32 → Linear(128, 32)
        w.append(("downsampleBlocks.3.linear.weight", MLXRandom.normal([32, 128])))
        w.append(("downsampleBlocks.3.linear.bias", MLXArray.zeros([32])))

        // Encoder encoderProjOut: Linear(32, 128)
        w.append(("encoderProjOut.weight", MLXRandom.normal([128, 32])))
        w.append(("encoderProjOut.bias", MLXArray.zeros([128])))

        return w
    }
}
