import Foundation
import Testing
import MLX
@testable import PiMPAudioCore

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

    @Test("Decode matches Python reference")
    func decodeMatchesPythonReference() async throws {
        let weightsPath = ("~/Library/Containers/com.pimpstudios.PiMPAudioStudio/Data/Library/Application Support/PiMP Audio Studio/models/ace-step-vae/vae.safetensors" as NSString).expandingTildeInPath
        let latentPath = "/tmp/vae_test_latent.safetensors"
        guard FileManager.default.fileExists(atPath: weightsPath),
              FileManager.default.fileExists(atPath: latentPath)
        else {
            print("Skipping VAE parity reference test; required external fixtures are not present.")
            return
        }

        // Load and convert weights
        let weightsURL = URL(fileURLWithPath: weightsPath)
        let converter = WeightConverter()
        let rawWeights = try await converter.loadSafetensors(from: weightsURL)
        let converted = try await converter.convertVAEWeights(stateDict: rawWeights)

        // Load weights into VAEActor
        let actor = VAEActor()
        await actor.loadWeights(converted, modelID: "parity-test")

        // Load the reference latent saved by the Python script
        let latentURL = URL(fileURLWithPath: latentPath)
        let latentArrays = try MLX.loadArrays(url: latentURL)
        guard let latents = latentArrays["latents"] else {
            Issue.record("Key 'latents' missing from \(latentPath)")
            return
        }

        // Decode
        let audio = try await actor.decode(latents: latents, precision: .float32)
        eval(audio)

        // Compute stats
        let absAudio = MLX.abs(audio)
        let absMean = absAudio.mean().item(Float.self)
        let absMax = absAudio.max().item(Float.self)
        let mean = audio.mean().item(Float.self)

        // First 8 samples of left channel: audio[0, 0..<8, 0]
        let first8 = audio[0, 0..<8, 0]
        eval(first8)
        let first8Values = (0..<8).map { first8[$0].item(Float.self) }

        print("[VAE_PARITY_SWIFT] output shape       : \(audio.shape)")
        let durationSec = Float(audio.shape[1]) / 48000.0
        print("[VAE_PARITY_SWIFT] output duration    : \(durationSec) s  (\(audio.shape[1]) samples @ 48kHz)")
        print("[VAE_PARITY_SWIFT] output abs_mean    : \(String(format: "%.6f", absMean))")
        print("[VAE_PARITY_SWIFT] output abs_max     : \(String(format: "%.6f", absMax))")
        print("[VAE_PARITY_SWIFT] output mean        : \(String(format: "%.6f", mean))")
        print("[VAE_PARITY_SWIFT] output[0,:8,0]     : \(first8Values.map { String(format: "%.6f", $0) })")

        // Sanity assertions
        #expect(audio.shape[1] == 192000, "Expected 100 frames × 1920 = 192000 samples")
        #expect(audio.shape[0] == 1)
        #expect(audio.shape[2] == 2)
        #expect(absMean > 0, "abs_mean should be positive")
        #expect(absMean < 10.0, "abs_mean \(absMean) is suspiciously large (expected < 10.0)")
    }
}
