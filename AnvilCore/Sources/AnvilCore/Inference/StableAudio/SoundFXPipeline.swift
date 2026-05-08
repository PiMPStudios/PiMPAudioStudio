// AnvilCore/Sources/AnvilCore/Inference/StableAudio/SoundFXPipeline.swift
//
// Stable Audio Open end-to-end pipeline:
//   1. T5-base tokenise + encode text → [B, L, 768] crossCond
//   2. SAONumberConditioner encode (seconds_start=0, seconds_total=duration)
//      → concat [B, 1536] globalCond
//   3. Diffusion ODE (crossCond + globalCond → predicted velocity → latents)
//   4. Oobleck VAE decode → audio samples
//
// The public `generate(text:duration:config:tokenizer:)` entry point handles
// steps 1–4.  The overload `generate(precomputedText:duration:config:)` accepts
// already-encoded T5 hidden states and is used in unit tests to avoid needing
// a real tokenizer file.

import Foundation
import MLX

public actor SoundFXPipeline {
    private let t5: T5Encoder
    private let conditioner: SAONumberConditioner
    private let diffusion: SFXDiffusionActor
    private let vae: SFXVAEActor
    private var progressContinuation: AsyncStream<GenerationProgress>.Continuation?

    public static let maxDuration: TimeInterval = 47.0
    public static let defaultSampleRate = 44100
    public static let hopLength = 1024

    public init(
        t5: T5Encoder,
        conditioner: SAONumberConditioner,
        diffusion: SFXDiffusionActor,
        vae: SFXVAEActor
    ) {
        self.t5          = t5
        self.conditioner = conditioner
        self.diffusion   = diffusion
        self.vae         = vae
    }

    public func makeProgressStream() -> AsyncStream<GenerationProgress> {
        let (stream, continuation) = AsyncStream<GenerationProgress>.makeStream()
        self.progressContinuation = continuation
        return stream
    }

    // MARK: - Primary entry point (tokenizes text internally)

    public func generate(
        text: String,
        duration: TimeInterval,
        config: SFXDiffusionConfig,
        tokenizer: AnvilTokenizer
    ) async throws -> AudioResult {
        // 1. T5 text encoding.
        let (textIds, textMask) = tokenizer.encodeText(text, maxLength: 128)
        let textHidden = t5(inputIds: textIds, attentionMask: textMask) // [1, L, 768]

        return try await generate(precomputedText: textHidden, duration: duration, config: config)
    }

    // MARK: - Overload for pre-encoded text (tests + external callers)

    /// Generate audio from pre-encoded T5 hidden states.
    /// - Parameters:
    ///   - precomputedText: `[B, L, conditioningDim]` T5-base last hidden state
    ///   - duration: output duration; clamped to `maxDuration`
    ///   - config: diffusion sampler configuration
    public func generate(
        precomputedText textHidden: MLXArray,
        duration: TimeInterval,
        config: SFXDiffusionConfig
    ) async throws -> AudioResult {
        let totalStart = Date()
        let clampedDuration = min(duration, Self.maxDuration)

        // 2. Number conditioner: encode seconds_start=0 and seconds_total.
        let (startEmb, totalEmb) = conditioner(
            startSeconds: 0.0,
            totalSeconds: Float(clampedDuration)
        )
        // Concatenate along the last axis: [1, 768] + [1, 768] → [1, 1536].
        let globalCond = concatenated([startEmb, totalEmb], axis: -1)  // [1, globalCondDim]

        // crossCond: [B, L, conditioningDim] = T5 hidden states.
        let crossCond = textHidden

        // 3. Diffusion.
        progressContinuation?.yield(.diffusion(DiffusionProgress(
            currentStep: 0, totalSteps: config.steps, elapsedTime: 0
        )))

        let diffResult = try await diffusion.sample(
            crossCond: crossCond,
            globalCond: globalCond,
            config: config
        )

        // 4. VAE decode.
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
