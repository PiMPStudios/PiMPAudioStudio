// AnvilCore/Sources/AnvilCore/Inference/StableAudio/SoundFXPipeline.swift
import Foundation
import MLX

public actor SoundFXPipeline {
    /// Stable Audio Open conditioning stack: T5-base text encoder →
    /// projection model → cross-attn into DiT. Replaces the prior CLAP
    /// path; CLAPEncoder/Tokenizer remain in the codebase for backward-
    /// compatible tests but are no longer on the runtime path.
    private let t5: T5Encoder
    private let projection: SAOProjectionModel
    private let timingEncoder: TimingEncoder
    private let diffusion: SFXDiffusionActor
    private let vae: SFXVAEActor
    private var progressContinuation: AsyncStream<GenerationProgress>.Continuation?

    public static let maxDuration: TimeInterval = 47.0
    public static let defaultSampleRate = 44100
    public static let hopLength = 1024

    public init(
        t5: T5Encoder,
        projection: SAOProjectionModel,
        diffusion: SFXDiffusionActor,
        vae: SFXVAEActor,
        timingEncoder: TimingEncoder
    ) {
        self.t5 = t5
        self.projection = projection
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
        tokenizer: AnvilTokenizer
    ) async throws -> AudioResult {
        let totalStart = Date()
        let clampedDuration = min(duration, Self.maxDuration)

        // 1. T5 text encoding (max_length 128 per SAO model_config.json).
        let (textIds, textMask) = tokenizer.encodeText(text, maxLength: 128)
        let textHidden = t5(inputIds: textIds, attentionMask: textMask)

        // 2. Sinusoidal/Fourier timing features for [seconds_start,
        // seconds_total]. `TimingEncoder.encode` returns shape
        // `[1, embeddingDim]` with sin/cos of log-spaced frequencies for
        // each input scalar — the standard Fourier-feature embedding the
        // projection_model is trained to consume.
        let timingEmbed = timingEncoder.encode(
            startSeconds: 0.0,
            durationSeconds: Float(clampedDuration)
        )

        // 3. Project both streams into the DiT conditioning dim.
        let projected = projection(textHidden: textHidden, timing: timingEmbed)
        let textCond = projected.text          // [1, L, embed]
        let timingCond = projected.timing      // [1, embed]

        // 4. Combine for downstream cross-attn. Real SAO concatenates the
        // projected timing token onto the front of the text sequence; the
        // skeleton matches that shape so the DiT call signature is stable.
        let timingTok = timingCond.expandedDimensions(axis: 1)  // [1, 1, embed]
        let conditioning = concatenated([timingTok, textCond], axis: 1)

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
