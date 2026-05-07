// AnvilCore/Sources/AnvilCore/Inference/StableAudio/SoundFXPipeline.swift
import Foundation
import MLX

public actor SoundFXPipeline {
    private let clap: CLAPEncoder
    private let timingEncoder: TimingEncoder
    private let diffusion: SFXDiffusionActor
    private let vae: SFXVAEActor
    private var progressContinuation: AsyncStream<GenerationProgress>.Continuation?

    public static let maxDuration: TimeInterval = 47.0
    public static let defaultSampleRate = 44100
    public static let hopLength = 1024

    public init(
        clap: CLAPEncoder,
        diffusion: SFXDiffusionActor,
        vae: SFXVAEActor,
        timingEncoder: TimingEncoder
    ) {
        self.clap = clap
        self.diffusion = diffusion
        self.vae = vae
        self.timingEncoder = timingEncoder
    }

    public func makeProgressStream() -> AsyncStream<GenerationProgress> {
        let (stream, continuation) = AsyncStream<GenerationProgress>.makeStream()
        self.progressContinuation = continuation
        return stream
    }

    public func generate(
        text: String,
        duration: TimeInterval,
        config: SFXDiffusionConfig,
        tokenizer: CLAPTokenizer
    ) async throws -> AudioResult {
        let totalStart = Date()
        let clampedDuration = min(duration, Self.maxDuration)

        // 1. CLAP text encoding
        let clapEmbedding: MLXArray
        let clapLoaded = await clap.isLoaded
        if clapLoaded {
            clapEmbedding = try await clap.encode(text: text, tokenizer: tokenizer)
        } else {
            // Fallback: use zero embedding for unloaded model (testing)
            clapEmbedding = MLXArray.zeros([1, 512])
        }

        // 2. Timing conditioning
        let timingEmbedding = timingEncoder.encode(startSeconds: 0.0, durationSeconds: Float(clampedDuration))

        // 3. Combine conditioning
        let conditioning = concatenated([clapEmbedding, timingEmbedding], axis: 1)

        // 4. Diffusion
        progressContinuation?.yield(.diffusion(DiffusionProgress(
            currentStep: 0, totalSteps: config.steps, elapsedTime: 0
        )))

        let diffResult = try await diffusion.sample(conditioning: conditioning, config: config)

        // 5. VAE decode
        progressContinuation?.yield(.decoding)
        let audio = try await vae.decode(latents: diffResult.latents)

        let totalTime = Date().timeIntervalSince(totalStart)
        let costs = TimeCosts(
            diffusionTime: diffResult.timeCosts.diffusionTime,
            perStepTime: diffResult.timeCosts.perStepTime,
            totalTime: totalTime
        )

        progressContinuation?.yield(.complete)
        progressContinuation?.finish()
        progressContinuation = nil

        return AudioResult(
            samples: audio,
            sampleRate: Self.defaultSampleRate,
            duration: clampedDuration,
            timeCosts: costs
        )
    }
}
