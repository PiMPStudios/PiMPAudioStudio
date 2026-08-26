// PiMPAudioCore/Sources/PiMPAudioCore/Inference/StableAudio/SoundFXPipeline.swift
//
// Stable Audio Open end-to-end pipeline:
//   1. T5-base tokenise + encode text → [B, 128, 768] hidden states
//   2. SAONumberConditioner encode (seconds_start=0, seconds_total=duration)
//      → startEmb [1, 768], totalEmb [1, 768]
//   3. Build conditioning tensors:
//      crossCond   = cat([T5[:64], totalToken, startToken])  [1, 66, 768]
//                    Matches mlx-audiogen: first 64 T5 tokens + 2 time tokens.
//      globalCond  = cat([totalEmb, startEmb], axis=-1)  [1, 1536]  ← total FIRST
//      uncrossCond = cat([T5("")[:64], totalToken, startToken])  [1, 66, 768]
//   4. Diffusion ODE (crossCond + globalCond → predicted velocity → latents)
//   5. Oobleck VAE decode → audio samples
//
// The public `generate(text:duration:config:tokenizer:)` entry point handles
// steps 1–5.  The overload `generate(precomputedText:uncondText:duration:config:)`
// accepts already-encoded T5 hidden states and is used in unit tests.
//
// Note on conditioning layout (matches stable-audio-tools / mlx-audiogen):
//   crossCond  = cat([T5[:64], total_time_token, start_time_token])  [1, 66, 768]
//   globalCond = cat([totalEmb, startEmb], axis=-1)  [1, 1536]  → prepend token
// Time tokens are APPENDED to cross-attention (positions 64 and 65), matching
// the stable-audio-tools training layout.  The same time tokens are used for
// both the conditional and unconditional CFG branches.

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
        tokenizer: PiMPAudioTokenizer
    ) async throws -> AudioResult {
        // T5 encoding — pad to max_length=128 BEFORE encoding so T5 processes
        // PAD tokens in context (matches Python `padding="max_length"`).
        //
        // Why this matters: without pre-padding, T5("") receives only 1 token
        // (the EOS token) in complete isolation.  T5's FFN layers transform
        // that single EOS embedding into a hidden state with much larger
        // magnitude than PAD-token hidden states.  Python always passes 128
        // tokens (EOS + 127 PADs) so the PAD token representations at
        // positions 1–63 are small and near-uniform, making v_uncond ≈ v_cond
        // and keeping CFG-amplification in-bounds.  With our Swift zero-pad
        // approach the EOS hidden state dominates cross-attention for the
        // uncond branch and v_uncond blows up ~72 % above v_cond.
        let (rawTextIds,  rawTextMask)  = tokenizer.encodeText(text, maxLength: 128)
        let (rawEmptyIds, rawEmptyMask) = tokenizer.encodeText("",   maxLength: 128)

        let (paddedTextIds,  paddedTextMask)  = Self.padTokensTo128(rawTextIds,  mask: rawTextMask)
        let (paddedEmptyIds, paddedEmptyMask) = Self.padTokensTo128(rawEmptyIds, mask: rawEmptyMask)

        let textHidden  = t5(inputIds: paddedTextIds,  attentionMask: paddedTextMask)  // [1, 128, 768]
        let emptyHidden = t5(inputIds: paddedEmptyIds, attentionMask: paddedEmptyMask) // [1, 128, 768]
        AudioDiagnostics.tensor("SFX_T5_TEXT_HIDDEN", textHidden)
        AudioDiagnostics.tensor("SFX_T5_EMPTY_HIDDEN", emptyHidden)

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
        defer {
            progressContinuation?.finish()
            progressContinuation = nil
        }
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

        // Cross-attention tokens: T5[:64] + total_time_token + start_time_token.
        //
        // Matches stable-audio-tools training layout and mlx-audiogen Conditioners:
        //   cross_attn = cat([t5_tokens[:64], total_emb[:, None, :], start_emb[:, None, :]])
        //              → [1, 66, 768]
        //
        // IMPORTANT: The T5 encoder returns variable-length output (only non-padding
        // tokens).  Python always pads to max_length=128 before slicing [:64], so the
        // cross-attention is always exactly 66 tokens.  We must zero-pad here if the
        // T5 output is shorter than 64 tokens.  Passing fewer tokens shifts the time
        // tokens to wrong positions and corrupts every cross-attention block, causing
        // the velocity field to diverge (latents explode).
        let totalToken = totalEmb.expandedDimensions(axis: 1)   // [1, 1, 768]
        let startToken = startEmb.expandedDimensions(axis: 1)   // [1, 1, 768]
        let t5Text  = Self.t5Pad64(textHidden)                   // [1, 64, 768]
        let crossCond = concatenated([t5Text, totalToken, startToken], axis: 1) // [1, 66, 768]
        AudioDiagnostics.tensor("SFX_CROSS_COND", crossCond)

        // Unconditioned tokens: same time tokens, empty-string T5 prefix.
        // Python: uncond_tokens uses the same seconds_total/start values.
        let uncrossCondTokens: MLXArray? = uncondText.map { uc -> MLXArray in
            let t5Uncond = Self.t5Pad64(uc)                      // [1, 64, 768]
            return concatenated([t5Uncond, totalToken, startToken], axis: 1) // [1, 66, 768]
        }
        if let uc = uncrossCondTokens {
            AudioDiagnostics.tensor("SFX_UNCROSS_COND", uc)
        } else {
            AudioDiagnostics.note("SFX_UNCROSS_COND", "nil (CFG disabled)")
        }

        // Diffusion.
        progressContinuation?.yield(.diffusion(DiffusionProgress(
            currentStep: 0, totalSteps: config.steps, elapsedTime: 0
        )))

        // Wire up per-step progress relay from SFXDiffusionActor — same pattern
        // as MusicPipeline/DiffusionActor. Without this the upstream progressTask
        // only ever sees step-0 then .decoding, so the UI bar never advances.
        let diffProgressStream = await diffusion.makeProgressStream()
        let outerContinuation = self.progressContinuation   // Sendable capture
        let diffRelayTask = Task {
            for await progress in diffProgressStream {
                outerContinuation?.yield(progress)
            }
        }
        defer { diffRelayTask.cancel() }

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

        return AudioResult(
            samples: audio,
            sampleRate: Self.defaultSampleRate,
            duration: clampedDuration,
            timeCosts: costs
        )
    }

    // MARK: - Inpainting conditioning

    /// Encode text + timing into the three conditioning tensors needed for
    /// `SFXDiffusionActor.inpaint`.  Mirrors the conditioning build in
    /// `generate(precomputedText:…)` without running diffusion.
    public func buildConditioning(
        text: String,
        duration: TimeInterval,
        tokenizer: PiMPAudioTokenizer
    ) async -> SFXInpaintConditioning {
        let clampedDuration = min(duration, Self.maxDuration)

        let (rawTextIds,  rawTextMask)  = tokenizer.encodeText(text, maxLength: 128)
        let (rawEmptyIds, rawEmptyMask) = tokenizer.encodeText("",   maxLength: 128)
        let (paddedTextIds,  paddedTextMask)  = Self.padTokensTo128(rawTextIds,  mask: rawTextMask)
        let (paddedEmptyIds, paddedEmptyMask) = Self.padTokensTo128(rawEmptyIds, mask: rawEmptyMask)

        let textHidden  = t5(inputIds: paddedTextIds,  attentionMask: paddedTextMask)
        let emptyHidden = t5(inputIds: paddedEmptyIds, attentionMask: paddedEmptyMask)

        let (startEmb, totalEmb) = conditioner(
            startSeconds: 0.0,
            totalSeconds: Float(clampedDuration)
        )

        let globalCond  = concatenated([totalEmb, startEmb], axis: -1)  // [1, 1536]
        let totalToken  = totalEmb.expandedDimensions(axis: 1)           // [1, 1, 768]
        let startToken  = startEmb.expandedDimensions(axis: 1)           // [1, 1, 768]
        let crossCond   = concatenated([Self.t5Pad64(textHidden),  totalToken, startToken], axis: 1) // [1, 66, 768]
        let uncrossCond = concatenated([Self.t5Pad64(emptyHidden), totalToken, startToken], axis: 1) // [1, 66, 768]

        return SFXInpaintConditioning(
            crossCond: crossCond,
            globalCond: globalCond,
            uncrossCondTokens: uncrossCond
        )
    }

    // MARK: - Helpers

    /// Pad or truncate T5 hidden states to exactly 64 tokens along axis 1.
    ///
    /// After pre-padding tokenizer output to 128 tokens before T5 encoding,
    /// `hidden` will always have seqLen=128 and this just slices to the first 64.
    /// The zero-pad branch is kept as a safe fallback for callers that pass
    /// pre-encoded hidden states shorter than 64 (e.g. unit tests).
    ///
    /// - Parameter hidden: `[B, seqLen, 768]` T5 last hidden state (any seqLen).
    /// - Returns: `[B, 64, 768]` tensor, zero-padded if seqLen < 64.
    private static func t5Pad64(_ hidden: MLXArray) -> MLXArray {
        let seqLen = hidden.shape[1]
        let target = 64
        if seqLen >= target {
            return hidden[0..., 0..<target, 0...]   // truncate
        }
        let B = hidden.shape[0]
        let D = hidden.shape[2]
        let pad = MLXArray.zeros([B, target - seqLen, D])
        return concatenated([hidden, pad], axis: 1) // zero-pad (fallback only)
    }

    /// Pad tokenizer output to exactly 128 tokens using PAD token (ID=0, mask=0).
    ///
    /// Matches Python `tokenizer(text, padding="max_length", max_length=128)`.
    /// If the input is already ≥ 128 tokens it is returned unchanged (encodeText
    /// truncates at 128, so this branch is effectively dead in normal use).
    private static func padTokensTo128(_ ids: MLXArray, mask: MLXArray) -> (MLXArray, MLXArray) {
        let currentLen = ids.shape[1]
        let target = 128
        if currentLen >= target { return (ids, mask) }
        let B    = ids.shape[0]
        let nPad = target - currentLen
        // PAD token ID = 0, PAD mask = 0 (matches T5 tokenizer convention).
        let padIds  = MLXArray.zeros([B, nPad]).asType(.int32)
        let padMask = MLXArray.zeros([B, nPad]).asType(.int32)
        return (
            concatenated([ids,  padIds],  axis: 1),
            concatenated([mask, padMask], axis: 1)
        )
    }
}
