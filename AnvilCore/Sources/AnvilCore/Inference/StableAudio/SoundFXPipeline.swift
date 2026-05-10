// AnvilCore/Sources/AnvilCore/Inference/StableAudio/SoundFXPipeline.swift
//
// Stable Audio Open end-to-end pipeline:
//   1. T5-base tokenise + encode text → [B, 128, 768] hidden states
//   2. SAONumberConditioner encode (seconds_start=0, seconds_total=duration)
//      → startEmb [1, 768], totalEmb [1, 768]
//   3. Build conditioning tensors:
//      crossCond  = T5 hidden states  [1, 128, 768]  (text only — no time tokens)
//      globalCond = cat([totalEmb, startEmb], axis=-1)  [1, 1536]  ← total FIRST
//      uncrossCond = T5("") hidden states  [1, 128, 768]  (for CFG null pass)
//   4. Diffusion ODE (crossCond + globalCond → predicted velocity → latents)
//   5. Oobleck VAE decode → audio samples
//
// The public `generate(text:duration:config:tokenizer:)` entry point handles
// steps 1–5.  The overload `generate(precomputedText:uncondText:duration:config:)`
// accepts already-encoded T5 hidden states and is used in unit tests.
//
// Note on conditioning layout (matches stable-audio-tools training):
//   crossCond  = T5 hidden states [1, 128, 768]  (text only — no time tokens)
//   globalCond = cat([totalEmb, startEmb], axis=-1)  [1, 1536]  → prepend token
// Time tokens must NOT be appended to crossCond; they belong only in globalCond.

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
    /// SAO VAE strides [2, 4, 4, 8, 8] → product = 2048 samples per latent frame.
    public static let hopLength = 2048

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
        // T5 encoding of prompt and empty string (for CFG null conditioning).
        let (textIds, textMask) = tokenizer.encodeText(text, maxLength: 128)
        let textHidden = t5(inputIds: textIds, attentionMask: textMask)   // [1, 128, 768]

        let (emptyIds, emptyMask) = tokenizer.encodeText("", maxLength: 128)
        let emptyHidden = t5(inputIds: emptyIds, attentionMask: emptyMask) // [1, 128, 768]

        return try await generate(
            precomputedText: textHidden,
            uncondText: emptyHidden,
            duration: duration,
            config: config
        )
    }

    // MARK: - Overload for pre-encoded text (tests + external callers)

    /// Generate audio from pre-encoded T5 hidden states.
    /// - Parameters:
    ///   - precomputedText: `[B, 128, 768]` T5-base last hidden state for the prompt.
    ///                      Used directly as cross-attention conditioning (no time tokens appended).
    ///   - uncondText:      `[B, 128, 768]` T5-base hidden state for CFG null (empty string).
    ///                      Pass `nil` to disable CFG (treats `guidanceScale` as 1.0).
    ///   - duration: output duration; clamped to `maxDuration`
    ///   - config: diffusion sampler configuration
    public func generate(
        precomputedText textHidden: MLXArray,
        uncondText: MLXArray? = nil,
        duration: TimeInterval,
        config: SFXDiffusionConfig
    ) async throws -> AudioResult {
        let totalStart = Date()
        let clampedDuration = min(duration, Self.maxDuration)

        // Number conditioner: encode seconds_start=0 and seconds_total.
        let (startEmb, totalEmb) = conditioner(
            startSeconds: 0.0,
            totalSeconds: Float(clampedDuration)
        )
        AudioDiagnostics.note("SFX_COND", "duration=\(clampedDuration)s steps=\(config.steps) guidance=\(config.guidanceScale) sigmaMax=\(config.sigmaMax)")
        AudioDiagnostics.tensor("SFX_START_EMB", startEmb)
        AudioDiagnostics.tensor("SFX_TOTAL_EMB", totalEmb)

        // globalCond: [total, start] order matches mlx-audiogen Conditioners output.
        let globalCond = concatenated([totalEmb, startEmb], axis: -1)  // [1, 1536]
        AudioDiagnostics.tensor("SFX_GLOBAL_COND", globalCond)

        // Cross-attention tokens: full T5 hidden states only — [1, 128, 768].
        // stable-audio-tools was trained with T5-only cross-attention.  The time
        // tokens (seconds_start / seconds_total) flow exclusively through
        // globalCond → to_global_embed → prepend token.  Appending them here
        // introduces a fixed cross-attention bias the model was never trained
        // to handle, which manifests as a large DC offset in the decoded audio.
        let crossCond = textHidden  // [1, 128, 768]
        AudioDiagnostics.tensor("SFX_CROSS_COND", crossCond)

        // Unconditioned tokens for CFG: T5("") hidden states (same shape).
        let uncrossCondTokens: MLXArray? = uncondText  // [1, 128, 768] or nil
        if let uc = uncrossCondTokens {
            AudioDiagnostics.tensor("SFX_UNCROSS_COND", uc)
        } else {
            AudioDiagnostics.note("SFX_UNCROSS_COND", "nil (CFG disabled)")
        }

        // Diffusion.
        progressContinuation?.yield(.diffusion(DiffusionProgress(
            currentStep: 0, totalSteps: config.steps, elapsedTime: 0
        )))

        let diffResult = try await diffusion.sample(
            crossCond: crossCond,
            globalCond: globalCond,
            config: config,
            uncrossCondTokens: uncrossCondTokens
        )

        // VAE decode.
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
