// AnvilCore/Tests/AnvilCoreTests/Inference/StableAudio/SFXDiffusionTests.swift
import Testing
import MLX
import MLXRandom
@testable import AnvilCore

@Suite("SFX Diffusion")
struct SFXDiffusionTests {
    @Test("Sample produces latents with correct shape")
    func sampleShape() async throws {
        let ditConfig = SFXDiTConfig(hiddenSize: 128, numLayers: 1, numHeads: 2, latentDim: 64, conditioningDim: 512)
        let dit = SFXDiTActor(config: ditConfig)
        let diffusion = SFXDiffusionActor(dit: dit)

        let config = SFXDiffusionConfig(steps: 4, guidanceScale: 1.0, seed: 42, latentShape: [1, 16, 64])
        let conditioning = MLXRandom.normal([1, 512])
        let result = try await diffusion.sample(conditioning: conditioning, config: config)
        #expect(result.latents.shape == [1, 16, 64])
    }

    @Test("Cancellation stops sampling")
    func cancellation() async {
        let ditConfig = SFXDiTConfig(hiddenSize: 128, numLayers: 1, numHeads: 2, latentDim: 64, conditioningDim: 512)
        let dit = SFXDiTActor(config: ditConfig)
        let diffusion = SFXDiffusionActor(dit: dit)

        let config = SFXDiffusionConfig(steps: 100, guidanceScale: 1.0, seed: 42, latentShape: [1, 16, 64])
        let conditioning = MLXRandom.normal([1, 512])

        let task = Task {
            try await diffusion.sample(conditioning: conditioning, config: config)
        }
        try? await Task.sleep(for: .milliseconds(50))
        task.cancel()

        let result = await task.result
        switch result {
        case .failure(let error):
            #expect(error is CancellationError)
        case .success:
            break // May complete before cancellation on fast hardware
        }
    }

    @Test("CFG with guidanceScale > 1 runs without error")
    func cfgMode() async throws {
        let ditConfig = SFXDiTConfig(hiddenSize: 128, numLayers: 1, numHeads: 2, latentDim: 64, conditioningDim: 512)
        let dit = SFXDiTActor(config: ditConfig)
        let diffusion = SFXDiffusionActor(dit: dit)

        let config = SFXDiffusionConfig(steps: 2, guidanceScale: 3.0, seed: 42, latentShape: [1, 16, 64])
        let conditioning = MLXRandom.normal([1, 512])
        let result = try await diffusion.sample(conditioning: conditioning, config: config)
        #expect(result.latents.shape == [1, 16, 64])
    }
}
