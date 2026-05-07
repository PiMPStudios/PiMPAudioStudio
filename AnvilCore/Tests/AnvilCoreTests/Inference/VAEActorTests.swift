import Testing
import MLX
@testable import AnvilCore

@Suite("VAE Actor")
struct VAEActorTests {
    @Test("Decode produces audio from latents")
    func decodeProducesAudio() async throws {
        let actor = VAEActor()
        let latents = MLXArray.zeros([1, 4, 64])
        let audio = try await actor.decode(latents: latents, precision: .float32)
        #expect(audio.shape[0] == 1)
        #expect(audio.shape[2] == 2) // stereo
    }

    @Test("Encode produces latents from audio")
    func encodeProducesLatents() async throws {
        let actor = VAEActor()
        let audio = MLXArray.zeros([1, 4096, 2])
        let latents = try await actor.encodeSample(audio: audio, precision: .float32)
        #expect(latents.shape[0] == 1)
        #expect(latents.shape[2] == 64)
    }

    @Test("Decode with fp16 precision")
    func decodeFp16() async throws {
        let actor = VAEActor()
        let latents = MLXArray.zeros([1, 4, 64])
        let audio = try await actor.decode(latents: latents, precision: .float16)
        #expect(audio.shape[0] == 1)
        #expect(audio.shape[2] == 2)
    }

    @Test("Long latents trigger tiled decode")
    func longLatentsTiled() async throws {
        let actor = VAEActor()
        let latents = MLXArray.zeros([1, 3000, 64])
        let audio = try await actor.decode(latents: latents, precision: .float32)
        #expect(audio.shape[0] == 1)
        #expect(audio.shape[2] == 2)
        #expect(audio.shape[1] > 5_000_000)
    }
}
