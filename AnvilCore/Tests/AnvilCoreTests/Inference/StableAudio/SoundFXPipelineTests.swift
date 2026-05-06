// AnvilCore/Tests/AnvilCoreTests/Inference/StableAudio/SoundFXPipelineTests.swift
import Testing
import MLX
import MLXRandom
@testable import AnvilCore

@Suite("SoundFXPipeline")
struct SoundFXPipelineTests {
    @Test("Generate produces AudioResult")
    func generateProducesResult() async throws {
        let pipeline = makeTestPipeline()
        let tokenizer = CLAPTokenizer.forTesting()

        let config = SFXDiffusionConfig(steps: 2, guidanceScale: 1.0, seed: 42, latentShape: [1, 16, 64])
        let result = try await pipeline.generate(
            text: "thunder",
            duration: 5.0,
            config: config,
            tokenizer: tokenizer
        )
        #expect(result.sampleRate == 44100)
        #expect(result.samples.shape[0] == 1)
        #expect(result.duration == 5.0)
    }

    @Test("Duration clamped to 47 seconds max")
    func durationClamped() async throws {
        let pipeline = makeTestPipeline()
        let tokenizer = CLAPTokenizer.forTesting()

        let config = SFXDiffusionConfig(steps: 2, guidanceScale: 1.0, seed: 42, latentShape: [1, 16, 64])
        let result = try await pipeline.generate(
            text: "long ambient",
            duration: 60.0,
            config: config,
            tokenizer: tokenizer
        )
        #expect(result.duration == 47.0)
    }

    @Test("Progress stream emits events")
    func progressStream() async throws {
        let pipeline = makeTestPipeline()
        let tokenizer = CLAPTokenizer.forTesting()
        let progressStream = await pipeline.progress

        let config = SFXDiffusionConfig(steps: 2, guidanceScale: 1.0, seed: 42, latentShape: [1, 16, 64])

        let progressTask = Task {
            var count = 0
            for await _ in progressStream {
                count += 1
                if count >= 2 { break }
            }
            return count
        }

        _ = try await pipeline.generate(
            text: "rain",
            duration: 5.0,
            config: config,
            tokenizer: tokenizer
        )

        let count = await progressTask.value
        #expect(count >= 1)
    }

    private func makeTestPipeline() -> SoundFXPipeline {
        let clapEncoder = CLAPEncoder(embeddingDim: 512, numLayers: 1, numHeads: 4, vocabSize: 49408)
        let timingEncoder = TimingEncoder(embeddingDim: 256)
        let ditConfig = SFXDiTConfig(hiddenSize: 128, numLayers: 1, numHeads: 2, latentDim: 64, conditioningDim: 768)
        let dit = SFXDiTActor(config: ditConfig)
        let diffusion = SFXDiffusionActor(dit: dit)
        let vaeConfig = SFXVAEConfig(latentDim: 64, outputChannels: 2, sampleRate: 44100, upsampleRatios: [2, 2])
        let vae = SFXVAEActor(config: vaeConfig)

        return SoundFXPipeline(
            clap: clapEncoder,
            diffusion: diffusion,
            vae: vae,
            timingEncoder: timingEncoder
        )
    }
}
