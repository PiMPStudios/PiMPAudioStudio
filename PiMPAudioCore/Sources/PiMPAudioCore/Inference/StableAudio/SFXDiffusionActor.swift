// PiMPAudioCore/Sources/PiMPAudioCore/Inference/StableAudio/SFXDiffusionActor.swift
import Foundation
import MLX
import MLXRandom

private func sfxAudioDiagnosticStep(_ stepIdx: Int, totalSteps: Int) -> Bool {
    stepIdx <= 1 || stepIdx == totalSteps - 1 || stepIdx % 10 == 0
}

private func sfxAudioDiagnosticFloat(_ value: Float, format: String = "%.4g") -> String {
    String(format: format, value)
}

private func sfxAudioDiagnosticAbsMean(_ array: MLXArray) -> Float {
    MLX.abs(array.asType(.float32)).mean().item(Float.self)
}

private func sfxAudioDiagnosticCFGStats(
    stepIdx: Int,
    sigma: Float,
    sigmaNext: Float,
    guidanceScale: Float,
    condPred: MLXArray,
    uncondPred: MLXArray,
    velocity: MLXArray
) -> String {
    let diff = condPred - uncondPred
    let condMean = sfxAudioDiagnosticAbsMean(condPred)
    let uncondMean = sfxAudioDiagnosticAbsMean(uncondPred)
    let diffMean = sfxAudioDiagnosticAbsMean(diff)
    let velocityMean = sfxAudioDiagnosticAbsMean(velocity)
    let ratio = condMean > 0 ? diffMean / condMean : 0
    return "step=\(stepIdx) sigma=\(sfxAudioDiagnosticFloat(sigma, format: "%.4f")) sigmaNext=\(sfxAudioDiagnosticFloat(sigmaNext, format: "%.4f")) guidance=\(sfxAudioDiagnosticFloat(guidanceScale, format: "%.3f")) cond_abs_mean=\(sfxAudioDiagnosticFloat(condMean)) uncond_abs_mean=\(sfxAudioDiagnosticFloat(uncondMean)) diff_abs_mean=\(sfxAudioDiagnosticFloat(diffMean)) diff_to_cond=\(sfxAudioDiagnosticFloat(ratio)) velocity_abs_mean=\(sfxAudioDiagnosticFloat(velocityMean))"
}

public actor SFXDiffusionActor {
    private let dit: SFXDiTActor
    private var progressContinuation: AsyncStream<GenerationProgress>.Continuation?

    public init(dit: SFXDiTActor) {
        self.dit = dit
    }

    public func makeProgressStream() -> AsyncStream<GenerationProgress> {
        let (stream, continuation) = AsyncStream<GenerationProgress>.makeStream()
        progressContinuation = continuation
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
        uncrossCondTokens: MLXArray? = nil // T5("")+time tokens for CFG; nil → no CFG
    ) async throws -> DiffusionResult {
        let totalStart = Date()
        defer {
            progressContinuation?.finish()
            progressContinuation = nil
        }

        // Rectified flow: initial noise is unit-variance N(0,I) — do NOT scale by any σ_max.
        var xt = DiffusionSampler.generateNoise(shape: config.latentShape, seed: config.seed)

        // LogSNR sigmoid schedule: t descends from sigmaMax to near 0 in `steps` values.
        // The final sigmaNext is 0 (handled by the existing fallback in the loop below).
        let sigmas = generateRFSchedule(steps: config.steps, sigmaMax: config.sigmaMax)
        let doCFG = config.guidanceScale > 1.0 && uncrossCondTokens != nil

        AudioDiagnostics.memory("sfx/diffusion/begin")
        AudioDiagnostics.note(
            "SFX_DIFF",
            "latentShape=\(config.latentShape) steps=\(config.steps) sigmaMax=\(config.sigmaMax) doCFG=\(doCFG)"
        )
        AudioDiagnostics.note(
            "sfx/diffusion/config",
            "latentShape=\(config.latentShape) steps=\(config.steps) sigmaMax=\(config.sigmaMax) guidance=\(config.guidanceScale) doCFG=\(doCFG) seed=\(config.seed.map(String.init) ?? "random")"
        )
        // Log first 10 schedule values so we can verify the logSNR sigmoid shape.
        AudioDiagnostics.floats("SFX_SCHEDULE_FIRST10", Array(sigmas.prefix(10)))
        AudioDiagnostics.floats("SFX_SCHEDULE_LAST5", Array(sigmas.suffix(5)))
        AudioDiagnostics.floats("sfx/diffusion/schedule", sigmas)
        AudioDiagnostics.tensor("sfx/diffusion/crossCond", crossCond)
        AudioDiagnostics.tensor("sfx/diffusion/globalCond", globalCond)
        if let uncrossCondTokens {
            AudioDiagnostics.tensor("sfx/diffusion/uncrossCondTokens", uncrossCondTokens)
        } else {
            AudioDiagnostics.note("sfx/diffusion/uncrossCondTokens", "nil")
        }
        AudioDiagnostics.tensor("SFX_NOISE_XT", xt)

        // Reset the DiT internal-diag counter so the first two forward calls
        // (step 0 cond + uncond) emit their full stage probes.
        await dit.resetDiagCounter()

        let diffStart = Date()

        for stepIdx in 0 ..< config.steps {
            try Task.checkCancellation()

            let sigma = sigmas[stepIdx]
            let sigmaNext = stepIdx + 1 < sigmas.count ? sigmas[stepIdx + 1] : Float(0)
            let timestep = MLXArray([sigma])
            let logDetail = AudioDiagnostics.isEnabled && sfxAudioDiagnosticStep(stepIdx, totalSteps: config.steps)
            if logDetail {
                AudioDiagnostics.tensor("sfx/diffusion/step\(stepIdx)/xtBefore", xt)
            }

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
                if logDetail {
                    AudioDiagnostics.note(
                        "sfx/diffusion/cfgStats",
                        sfxAudioDiagnosticCFGStats(
                            stepIdx: stepIdx,
                            sigma: sigma,
                            sigmaNext: sigmaNext,
                            guidanceScale: config.guidanceScale,
                            condPred: condPred,
                            uncondPred: uncondPred,
                            velocity: vt
                        )
                    )
                    AudioDiagnostics.tensor("sfx/diffusion/step\(stepIdx)/condPred", condPred)
                    AudioDiagnostics.tensor("sfx/diffusion/step\(stepIdx)/uncondPred", uncondPred)
                    AudioDiagnostics.tensor("sfx/diffusion/step\(stepIdx)/cfgDelta", diff)
                }

                if stepIdx == 0, AudioDiagnostics.isEnabled {
                    let diffMean = MLX.abs(diff).mean().item(Float.self)
                    let condMean = MLX.abs(condPred).mean().item(Float.self)
                    let ratio = condMean > 0 ? diffMean / condMean : 0
                    AudioDiagnostics.note(
                        "sfx/diffusion/step0cfg",
                        String(
                            format: "cond_abs_mean=%.4f uncond_abs_mean=%.4f diff_abs_mean=%.4f ratio=%.4f (Python ref: diff≈0.019 ratio≈0.11)",
                            condMean,
                            MLX.abs(uncondPred).mean().item(Float.self),
                            diffMean,
                            ratio
                        )
                    )
                }
            } else {
                vt = try await dit.predict(
                    latents: xt, timestep: timestep,
                    crossCond: crossCond, globalCond: globalCond
                )
                if logDetail {
                    AudioDiagnostics.note(
                        "sfx/diffusion/cfgStats",
                        "step=\(stepIdx) sigma=\(sfxAudioDiagnosticFloat(sigma, format: "%.4f")) sigmaNext=\(sfxAudioDiagnosticFloat(sigmaNext, format: "%.4f")) doCFG=false guidance=\(sfxAudioDiagnosticFloat(config.guidanceScale, format: "%.3f"))"
                    )
                }
            }

            // Rectified flow Euler step: x_next = x_t + (t_next - t_curr) * v_t
            xt = DiffusionSampler.odeStep(xt: xt, vt: vt, currentT: sigma, nextT: sigmaNext)
            eval(xt)

            // [AUDIODIAG] verbose: log steps 0, 1, and final to trace denoising progress.
            if stepIdx <= 1 || stepIdx == config.steps - 1 {
                AudioDiagnostics.note(
                    "SFX_STEP[\(stepIdx)]",
                    "sigma=\(String(format: "%.4f", sigma)) sigmaNext=\(String(format: "%.4f", sigmaNext))"
                )
                AudioDiagnostics.tensor("SFX_VT[\(stepIdx)]", vt)
                AudioDiagnostics.tensor("SFX_XT[\(stepIdx)]", xt)
            }
            if logDetail {
                AudioDiagnostics.tensor("sfx/diffusion/step\(stepIdx)/vt", vt)
                AudioDiagnostics.tensor("sfx/diffusion/step\(stepIdx)/xtAfter", xt)
                AudioDiagnostics.memory("sfx/diffusion/step\(stepIdx)/after")
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
        AudioDiagnostics.note(
            "SFX_DIFF_DONE",
            "diffTime=\(String(format: "%.2f", diffTime))s perStep=\(String(format: "%.3f", diffTime / Double(max(config.steps, 1))))s"
        )
        AudioDiagnostics.memory("sfx/diffusion/end")

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
        uncrossCondTokens: MLXArray? = nil // T5("")+time tokens for CFG; nil → no CFG
    ) async throws -> DiffusionResult {
        let totalStart = Date()
        defer {
            progressContinuation?.finish()
            progressContinuation = nil
        }

        guard !schedule.isEmpty else {
            let costs = TimeCosts(diffusionTime: 0, perStepTime: 0, totalTime: 0)
            return DiffusionResult(latents: startLatents, timeCosts: costs)
        }

        var xt = startLatents
        let stepCount = schedule.count
        let doCFG = config.guidanceScale > 1.0 && uncrossCondTokens != nil

        AudioDiagnostics.memory("sfx/inpaint/begin")
        AudioDiagnostics.note(
            "sfx/inpaint/config",
            "startShape=\(startLatents.shape) steps=\(stepCount) sigmaMax=\(config.sigmaMax) guidance=\(config.guidanceScale) doCFG=\(doCFG) seed=\(config.seed.map(String.init) ?? "random")"
        )
        AudioDiagnostics.floats("sfx/inpaint/schedule", schedule)
        AudioDiagnostics.tensor("sfx/inpaint/startLatents", startLatents)
        AudioDiagnostics.tensor("sfx/inpaint/crossCond", crossCond)
        AudioDiagnostics.tensor("sfx/inpaint/globalCond", globalCond)
        if let uncrossCondTokens {
            AudioDiagnostics.tensor("sfx/inpaint/uncrossCondTokens", uncrossCondTokens)
        } else {
            AudioDiagnostics.note("sfx/inpaint/uncrossCondTokens", "nil")
        }

        let diffStart = Date()

        for stepIdx in 0 ..< stepCount {
            try Task.checkCancellation()

            let sigma = schedule[stepIdx]
            let sigmaNext = stepIdx + 1 < stepCount ? schedule[stepIdx + 1] : Float(0)
            let timestep = MLXArray([sigma])
            let logDetail = AudioDiagnostics.isEnabled && sfxAudioDiagnosticStep(stepIdx, totalSteps: stepCount)
            if logDetail {
                AudioDiagnostics.tensor("sfx/inpaint/step\(stepIdx)/xtBefore", xt)
            }

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
                if logDetail {
                    AudioDiagnostics.note(
                        "sfx/inpaint/cfgStats",
                        sfxAudioDiagnosticCFGStats(
                            stepIdx: stepIdx,
                            sigma: sigma,
                            sigmaNext: sigmaNext,
                            guidanceScale: config.guidanceScale,
                            condPred: condPred,
                            uncondPred: uncondPred,
                            velocity: vt
                        )
                    )
                    AudioDiagnostics.tensor("sfx/inpaint/step\(stepIdx)/condPred", condPred)
                    AudioDiagnostics.tensor("sfx/inpaint/step\(stepIdx)/uncondPred", uncondPred)
                    AudioDiagnostics.tensor("sfx/inpaint/step\(stepIdx)/cfgDelta", diff)
                }
                if stepIdx == 0, AudioDiagnostics.isEnabled {
                    let diffMean = MLX.abs(diff).mean().item(Float.self)
                    let condMean = MLX.abs(condPred).mean().item(Float.self)
                    let ratio = condMean > 0 ? diffMean / condMean : 0
                    AudioDiagnostics.note(
                        "sfx/inpaint/step0cfg",
                        String(
                            format: "cond_abs_mean=%.4f uncond_abs_mean=%.4f diff_abs_mean=%.4f ratio=%.4f (Python ref: diff≈0.019 ratio≈0.11)",
                            condMean,
                            MLX.abs(uncondPred).mean().item(Float.self),
                            diffMean,
                            ratio
                        )
                    )
                }
            } else {
                vt = try await dit.predict(
                    latents: xt, timestep: timestep,
                    crossCond: crossCond, globalCond: globalCond
                )
                if logDetail {
                    AudioDiagnostics.note(
                        "sfx/inpaint/cfgStats",
                        "step=\(stepIdx) sigma=\(sfxAudioDiagnosticFloat(sigma, format: "%.4f")) sigmaNext=\(sfxAudioDiagnosticFloat(sigmaNext, format: "%.4f")) doCFG=false guidance=\(sfxAudioDiagnosticFloat(config.guidanceScale, format: "%.3f"))"
                    )
                }
            }

            xt = DiffusionSampler.odeStep(xt: xt, vt: vt, currentT: sigma, nextT: sigmaNext)
            eval(xt)
            if logDetail {
                AudioDiagnostics.tensor("sfx/inpaint/step\(stepIdx)/vt", vt)
                AudioDiagnostics.tensor("sfx/inpaint/step\(stepIdx)/xtAfter", xt)
                AudioDiagnostics.memory("sfx/inpaint/step\(stepIdx)/after")
            }

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

        AudioDiagnostics.tensor("sfx/inpaint/finalLatents", xt)
        AudioDiagnostics.memory("sfx/inpaint/end")

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

        return (0 ..< steps).map { i in
            if i == 0 {
                return sigmaMax
            } // pin to exact sigmaMax
            let logsnr = logsnrMax + Float(i) / Float(steps) * (logsnrMin - logsnrMax)
            return 1.0 / (1.0 + expf(logsnr))
        }
    }
}
