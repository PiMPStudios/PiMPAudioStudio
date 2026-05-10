// AnvilCore/Sources/AnvilCore/Inference/StableAudio/SFXDiffusionActor.swift
import Foundation
import MLX
import MLXRandom

public actor SFXDiffusionActor {
    private let dit: SFXDiTActor
    private var progressContinuation: AsyncStream<GenerationProgress>.Continuation?

    public init(dit: SFXDiTActor) {
        self.dit = dit
    }

    public func makeProgressStream() -> AsyncStream<GenerationProgress> {
        let (stream, continuation) = AsyncStream<GenerationProgress>.makeStream()
        self.progressContinuation = continuation
        return stream
    }

    /// Run the ODE diffusion sampler.
    ///
    /// SAO (mlx-audiogen / mlx-stable-audio) uses **rectified flow** with t ∈ [1, 0].
    /// Schedule: logSNR sigmoid schedule (see `generateRFSchedule`).
    /// ODE step:  x_next = x_t + (t_next − t_curr) · v_pred   (simple Euler, same as ACE-Step).
    ///
    /// - Parameters:
    ///   - crossCond:          `[B, S, condTokenDim]` T5[:64] + time tokens (66 tokens)
    ///   - globalCond:         `[B, globalCondDim]` concat(totalEmbed, startEmbed)
    ///   - uncrossCondTokens:  `[B, S, condTokenDim]` T5("")+time tokens for CFG null;
    ///                         pass `nil` to disable CFG entirely
    public func sample(
        crossCond: MLXArray,
        globalCond: MLXArray,
        config: SFXDiffusionConfig,
        uncrossCondTokens: MLXArray? = nil  // T5("")+time tokens for CFG; nil → no CFG
    ) async throws -> DiffusionResult {
        let totalStart = Date()

        // Rectified flow: initial noise is unit-variance N(0,I) — do NOT scale by any σ_max.
        var xt = DiffusionSampler.generateNoise(shape: config.latentShape, seed: config.seed)

        // LogSNR sigmoid schedule: t descends from sigmaMax to near 0 in `steps` values.
        // The final sigmaNext is 0 (handled by the existing fallback in the loop below).
        let sigmas = generateRFSchedule(steps: config.steps, sigmaMax: config.sigmaMax)
        let doCFG  = config.guidanceScale > 1.0 && uncrossCondTokens != nil

        AudioDiagnostics.note("SFX_DIFF", "latentShape=\(config.latentShape) steps=\(config.steps) sigmaMax=\(config.sigmaMax) doCFG=\(doCFG)")
        // Log first 10 schedule values so we can verify the logSNR sigmoid shape.
        AudioDiagnostics.floats("SFX_SCHEDULE_FIRST10", Array(sigmas.prefix(10)))
        AudioDiagnostics.floats("SFX_SCHEDULE_LAST5",   Array(sigmas.suffix(5)))
        AudioDiagnostics.tensor("SFX_NOISE_XT", xt)

        // Reset the DiT internal-diag counter so the first two forward calls
        // (step 0 cond + uncond) emit their full stage probes.
        await dit.resetDiagCounter()

        let diffStart = Date()

        for stepIdx in 0..<config.steps {
            try Task.checkCancellation()

            let sigma     = sigmas[stepIdx]
            let sigmaNext = stepIdx + 1 < sigmas.count ? sigmas[stepIdx + 1] : Float(0)
            let timestep  = MLXArray([sigma])

            var vt: MLXArray
            if doCFG {
                let condPred = try await dit.predict(
                    latents: xt, timestep: timestep,
                    crossCond: crossCond, globalCond: globalCond
                )
                let uncondPred = try await dit.predict(
                    latents: xt, timestep: timestep,
                    crossCond: uncrossCondTokens!,
                    globalCond: globalCond
                )
                let diff = condPred - uncondPred
                vt = uncondPred + config.guidanceScale * diff

                // Log cond/uncond diff at step 0 — compare to Python reference:
                //   Python diff.abs_mean ≈ 0.019 (ratio ≈ 11% of v_cond)
                //   Swift should match within ~2× if cross-attention is working.
                if stepIdx == 0 {
                    let diffMean = abs(diff).mean().item(Float.self)
                    let condMean = abs(condPred).mean().item(Float.self)
                    let ratio    = condMean > 0 ? diffMean / condMean : 0
                    print(String(format: "[SFXDIFF] step0 cond_abs_mean=%.4f uncond_abs_mean=%.4f diff_abs_mean=%.4f ratio=%.4f (Python ref: diff≈0.019 ratio≈0.11)",
                                 condMean, abs(uncondPred).mean().item(Float.self), diffMean, ratio))
                }
            } else {
                vt = try await dit.predict(
                    latents: xt, timestep: timestep,
                    crossCond: crossCond, globalCond: globalCond
                )
            }

            // Rectified flow Euler step: x_next = x_t + (t_next - t_curr) * v_t
            xt = DiffusionSampler.odeStep(xt: xt, vt: vt, currentT: sigma, nextT: sigmaNext)
            eval(xt)

            // [AUDIODIAG] verbose: log steps 0, 1, and final to trace denoising progress.
            if stepIdx <= 1 || stepIdx == config.steps - 1 {
                AudioDiagnostics.note("SFX_STEP[\(stepIdx)]", "sigma=\(String(format:"%.4f",sigma)) sigmaNext=\(String(format:"%.4f",sigmaNext))")
                AudioDiagnostics.tensor("SFX_VT[\(stepIdx)]", vt)
                AudioDiagnostics.tensor("SFX_XT[\(stepIdx)]", xt)
            }

            let elapsed = Date().timeIntervalSince(diffStart)
            progressContinuation?.yield(.diffusion(DiffusionProgress(
                currentStep: stepIdx + 1, totalSteps: config.steps, elapsedTime: elapsed
            )))
        }

        let diffTime = Date().timeIntervalSince(diffStart)
        let totalTime = Date().timeIntervalSince(totalStart)
        let costs = TimeCosts(
            diffusionTime: diffTime,
            perStepTime: diffTime / Double(max(config.steps, 1)),
            totalTime: totalTime
        )

        AudioDiagnostics.tensor("SFX_FINAL_LATENTS", xt)
        AudioDiagnostics.note("SFX_DIFF_DONE", "diffTime=\(String(format:"%.2f",diffTime))s perStep=\(String(format:"%.3f",diffTime/Double(max(config.steps,1))))s")

        // Unconditional diagnostic — always printed regardless of AudioDiagnostics flag.
        // Tells us whether the ODE converged to structured latents or stayed noisy.
        let finalAbsMean = abs(xt).mean().item(Float.self)
        let finalAbsMax  = abs(xt).max().item(Float.self)
        let finalMean    = xt.mean().item(Float.self)
        print("[SFXDIFF] final_latents abs_mean=\(String(format:"%.4f",finalAbsMean)) abs_max=\(String(format:"%.4f",finalAbsMax)) mean=\(String(format:"%.4f",finalMean)) shape=\(xt.shape)")

        progressContinuation?.finish()
        progressContinuation = nil
        return DiffusionResult(latents: xt, timeCosts: costs)
    }

    /// Inpaint using a partial-noise schedule.  Stable Audio inpainting is
    /// currently gated in `InpaintCoordinator`; this method signature is
    /// preserved for future use.
    ///
    /// `schedule` must be a rectified-flow t-value sequence descending from some
    /// t_start ∈ (0, 1] down to a value near 0 (e.g. a slice of `generateRFSchedule`).
    public func inpaint(
        startLatents: MLXArray,
        schedule: [Float],
        crossCond: MLXArray,
        globalCond: MLXArray,
        config: SFXDiffusionConfig,
        uncrossCondTokens: MLXArray? = nil  // T5("")+time tokens for CFG; nil → no CFG
    ) async throws -> DiffusionResult {
        let totalStart = Date()

        guard !schedule.isEmpty else {
            let costs = TimeCosts(diffusionTime: 0, perStepTime: 0, totalTime: 0)
            return DiffusionResult(latents: startLatents, timeCosts: costs)
        }

        var xt = startLatents
        let stepCount = schedule.count
        let doCFG = config.guidanceScale > 1.0 && uncrossCondTokens != nil

        let diffStart = Date()

        for stepIdx in 0..<stepCount {
            try Task.checkCancellation()

            let sigma     = schedule[stepIdx]
            let sigmaNext = stepIdx + 1 < stepCount ? schedule[stepIdx + 1] : Float(0)
            let timestep  = MLXArray([sigma])

            var vt: MLXArray
            if doCFG {
                let condPred = try await dit.predict(
                    latents: xt, timestep: timestep,
                    crossCond: crossCond, globalCond: globalCond
                )
                let uncondPred = try await dit.predict(
                    latents: xt, timestep: timestep,
                    crossCond: uncrossCondTokens!,
                    globalCond: globalCond
                )
                let diff = condPred - uncondPred
                vt = uncondPred + config.guidanceScale * diff
                if stepIdx == 0 {
                    let diffMean = abs(diff).mean().item(Float.self)
                    let condMean = abs(condPred).mean().item(Float.self)
                    let ratio    = condMean > 0 ? diffMean / condMean : 0
                    print(String(format: "[SFXDIFF] step0 cond_abs_mean=%.4f uncond_abs_mean=%.4f diff_abs_mean=%.4f ratio=%.4f (Python ref: diff≈0.019 ratio≈0.11)",
                                 condMean, abs(uncondPred).mean().item(Float.self), diffMean, ratio))
                }
            } else {
                vt = try await dit.predict(
                    latents: xt, timestep: timestep,
                    crossCond: crossCond, globalCond: globalCond
                )
            }

            xt = DiffusionSampler.odeStep(xt: xt, vt: vt, currentT: sigma, nextT: sigmaNext)
            eval(xt)

            let elapsed = Date().timeIntervalSince(diffStart)
            progressContinuation?.yield(.diffusion(DiffusionProgress(
                currentStep: stepIdx + 1, totalSteps: stepCount, elapsedTime: elapsed
            )))
        }

        let diffTime = Date().timeIntervalSince(diffStart)
        let totalTime = Date().timeIntervalSince(totalStart)
        let costs = TimeCosts(
            diffusionTime: diffTime,
            perStepTime: diffTime / Double(stepCount),
            totalTime: totalTime
        )

        progressContinuation?.finish()
        progressContinuation = nil
        return DiffusionResult(latents: xt, timeCosts: costs)
    }

    /// Rectified-flow logSNR timestep schedule.
    ///
    /// Matches mlx-audiogen `get_rf_schedule(steps, sigma_max)`:
    ///   `logsnr = linspace(logsnr_max, 2.0, steps+1)`
    ///   `t = sigmoid(-logsnr) = 1/(1+exp(logsnr))`
    ///   with `t[0]` pinned to `sigmaMax` and `t[-1]` to 0.
    ///
    /// For SAO `sigmaMax = 1.0`, `logsnr_max = -6.0`.
    ///
    /// Returns `steps` values (the `t[-1]=0.0` sentinel is the loop's `sigmaNext`
    /// fallback when `stepIdx + 1 >= sigmas.count`).
    func generateRFSchedule(steps: Int, sigmaMax: Float = 1.0) -> [Float] {
        let logsnrMax: Float = sigmaMax < 1.0
            ? logf((1.0 - sigmaMax) / sigmaMax + 1e-6)
            : -6.0
        let logsnrMin: Float = 2.0

        return (0..<steps).map { i in
            if i == 0 { return sigmaMax }  // pin to exact sigmaMax
            let logsnr = logsnrMax + Float(i) / Float(steps) * (logsnrMin - logsnrMax)
            return 1.0 / (1.0 + expf(logsnr))
        }
    }
}
