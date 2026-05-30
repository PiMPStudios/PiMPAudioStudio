import Foundation
import MLX
import MLXRandom
import OSLog

private let diagLogger = Logger(subsystem: "com.pimpstudios.PiMPAudioStudio", category: "DiffusionActor")

private func audioDiagnosticStep(_ stepIdx: Int, totalSteps: Int) -> Bool {
    stepIdx <= 1 || stepIdx == totalSteps - 1 || stepIdx % 10 == 0
}

private func audioDiagnosticFloat(_ value: Float, format: String = "%.4g") -> String {
    String(format: format, value)
}

private func audioDiagnosticAbsMean(_ array: MLXArray) -> Float {
    MLX.abs(array.asType(.float32)).mean().item(Float.self)
}

private func audioDiagnosticCFGPlan(
    schedule: [Float],
    intervalStart: Float,
    intervalEnd: Float
) -> String {
    var applied: [String] = []
    var skipped: [String] = []
    for (idx, timestep) in schedule.enumerated() {
        let label = "\(idx)@\(audioDiagnosticFloat(timestep, format: "%.4f"))"
        if timestep >= intervalStart && timestep <= intervalEnd {
            applied.append(label)
        } else {
            skipped.append(label)
        }
    }
    return "interval=\(audioDiagnosticFloat(intervalStart, format: "%.4f"))...\(audioDiagnosticFloat(intervalEnd, format: "%.4f")) appliedCount=\(applied.count) applied=[\(applied.joined(separator: ","))] skippedCount=\(skipped.count) skipped=[\(skipped.joined(separator: ","))]"
}

private func audioDiagnosticCFGStats(
    stepIdx: Int,
    timestep: Float,
    guidanceScale: Float,
    applyCFG: Bool,
    predCond: MLXArray,
    predUncond: MLXArray,
    velocity: MLXArray
) -> String {
    let diff = predCond - predUncond
    let condMean = audioDiagnosticAbsMean(predCond)
    let uncondMean = audioDiagnosticAbsMean(predUncond)
    let diffMean = audioDiagnosticAbsMean(diff)
    let velocityMean = audioDiagnosticAbsMean(velocity)
    let ratio = condMean > 0 ? diffMean / condMean : 0

    return "step=\(stepIdx) t=\(audioDiagnosticFloat(timestep, format: "%.4f")) applyCFG=\(applyCFG) guidance=\(audioDiagnosticFloat(guidanceScale, format: "%.3f")) cond_abs_mean=\(audioDiagnosticFloat(condMean)) uncond_abs_mean=\(audioDiagnosticFloat(uncondMean)) diff_abs_mean=\(audioDiagnosticFloat(diffMean)) diff_to_cond=\(audioDiagnosticFloat(ratio)) velocity_abs_mean=\(audioDiagnosticFloat(velocityMean))"
}

public actor DiffusionActor {
    private let dit: DiTActor
    private var progressContinuation: AsyncStream<DiffusionProgress>.Continuation?

    public init(dit: DiTActor) {
        self.dit = dit
    }

    public func makeProgressStream() -> AsyncStream<DiffusionProgress> {
        let (stream, continuation) = AsyncStream<DiffusionProgress>.makeStream()
        self.progressContinuation = continuation
        return stream
    }

    public func generate(
        encoderStates: MLXArray,
        contextLatents: MLXArray,
        encoderMask: MLXArray?,
        shape: (batch: Int, frames: Int, channels: Int),
        config: DiffusionConfig
    ) async throws -> DiffusionResult {
        let totalStart = Date()

        let bsz = shape.batch
        let T = shape.frames
        let C = shape.channels

        var xt = DiffusionSampler.generateNoise(shape: [bsz, T, C], seed: config.seed)
        AudioDiagnostics.memory("diff/generate/begin")
        AudioDiagnostics.note("config", "shift=\(config.shift) steps=\(config.steps ?? -1) method=\(config.method) guidance=\(config.guidanceScale) cfgInterval=\(config.cfgIntervalStart)...\(config.cfgIntervalEnd) precision=\(config.precision) seed=\(config.seed.map(String.init) ?? "random")")
        AudioDiagnostics.note(
            "diff/generate/config",
            "shape=[\(bsz),\(T),\(C)] shift=\(config.shift) steps=\(config.steps ?? -1) method=\(config.method) guidance=\(config.guidanceScale) cfgInterval=\(config.cfgIntervalStart)...\(config.cfgIntervalEnd) audioCoverStrength=\(config.audioCoverStrength) precision=\(config.precision) seed=\(config.seed.map(String.init) ?? "random")"
        )
        AudioDiagnostics.tensor("init/xt(noise)", xt)
        AudioDiagnostics.tensor("init/encoderStates", encoderStates)
        AudioDiagnostics.tensor("init/contextLatents", contextLatents)
        if let mask = encoderMask {
            AudioDiagnostics.tensor("init/encoderMask", mask)
        } else {
            AudioDiagnostics.note("init/encoderMask", "nil")
        }

        let schedule = TimestepSchedule.compute(
            shift: config.shift, timesteps: nil, inferSteps: config.steps
        )
        let numSteps = schedule.count
        AudioDiagnostics.floats("schedule", schedule)

        let doCFG = config.guidanceScale > 1.0
        if doCFG {
            AudioDiagnostics.note(
                "diff/generate/cfgPlan",
                audioDiagnosticCFGPlan(
                    schedule: schedule,
                    intervalStart: config.cfgIntervalStart,
                    intervalEnd: config.cfgIntervalEnd
                )
            )
        } else {
            AudioDiagnostics.note(
                "diff/generate/cfgPlan",
                "disabled guidanceScale=\(audioDiagnosticFloat(config.guidanceScale, format: "%.3f"))"
            )
        }
        var encHS = encoderStates
        var ctx = contextLatents
        var encMask = encoderMask
        if doCFG {
            // Match upstream MLX (acestep/models/mlx/dit_generate.py:188): the
            // CFG unconditional branch uses the LEARNED `null_condition_emb`
            // token broadcast to encoder_states shape, NOT zeros. Zeros are
            // out-of-distribution for the trained DiT, making the unconditional
            // prediction garbage so CFG (cond - uncond) injects noise on top
            // of the conditional pred instead of isolating the prompt direction.
            let nullEmb: MLXArray
            if let nullCond = await dit.nullConditionEmbedding() {
                nullEmb = MLX.broadcast(nullCond, to: encHS.shape).asType(encHS.dtype)
                AudioDiagnostics.tensor("diff/generate/nullConditionEmbedding", nullEmb)
            } else {
                nullEmb = MLXArray.zeros(like: encHS)
                AudioDiagnostics.note("diff/generate/nullConditionEmbedding", "missing; falling back to zeros")
            }
            encHS = concatenated([encHS, nullEmb], axis: 0)
            ctx = concatenated([ctx, ctx], axis: 0)
            encMask = encoderMask.map { concatenated([$0, $0], axis: 0) }
        }
        let momentum: APGMomentum? = doCFG ? APGMomentum() : nil

        let coverSteps = Int(Float(numSteps) * config.audioCoverStrength)

        let cache = CrossAttentionCache()
        let diffStart = Date()
        AudioDiagnostics.memory("diff/generate/diffusionBegin")

        for stepIdx in 0..<numSteps {
            try Task.checkCancellation()
            let currentT = schedule[stepIdx]
            let logDetail = AudioDiagnostics.isEnabled && audioDiagnosticStep(stepIdx, totalSteps: numSteps)

            if stepIdx >= coverSteps {
                cache.reset()
            }

            let xIn = doCFG ? concatenated([xt, xt], axis: 0) : xt
            if logDetail {
                AudioDiagnostics.tensor("diff/generate/step\(stepIdx)/xtBefore", xt)
            }

            let (vt, _) = try await dit.step(
                noisyLatents: xIn,
                timestep: currentT,
                timestepR: currentT,
                encoderStates: encHS,
                contextLatents: ctx,
                encoderAttentionMask: encMask,
                cache: cache,
                useCache: !doCFG
            )
            AudioDiagnostics.tensor("step\(stepIdx)/vt(t=\(String(format: "%.4f", currentT)))", vt)

            var velocity = vt

            if doCFG {
                let predCond = vt[0..<bsz]
                let predUncond = vt[bsz...]
                let applyCFG = currentT >= config.cfgIntervalStart && currentT <= config.cfgIntervalEnd
                if applyCFG {
                    velocity = DiffusionSampler.apgForward(
                        predCond: predCond, predUncond: predUncond,
                        guidanceScale: config.guidanceScale, momentum: momentum
                    )
                } else {
                    velocity = predCond
                }
                if logDetail {
                    AudioDiagnostics.note(
                        "diff/generate/cfgStats",
                        audioDiagnosticCFGStats(
                            stepIdx: stepIdx,
                            timestep: currentT,
                            guidanceScale: config.guidanceScale,
                            applyCFG: applyCFG,
                            predCond: predCond,
                            predUncond: predUncond,
                            velocity: velocity
                        )
                    )
                    AudioDiagnostics.tensor("diff/generate/step\(stepIdx)/predCond", predCond)
                    AudioDiagnostics.tensor("diff/generate/step\(stepIdx)/predUncond", predUncond)
                    AudioDiagnostics.tensor("diff/generate/step\(stepIdx)/cfgDelta", predCond - predUncond)
                    AudioDiagnostics.tensor("diff/generate/step\(stepIdx)/velocity", velocity)
                }
            } else if logDetail {
                AudioDiagnostics.note(
                    "diff/generate/cfgStats",
                    "step=\(stepIdx) t=\(audioDiagnosticFloat(currentT, format: "%.4f")) doCFG=false guidance=\(audioDiagnosticFloat(config.guidanceScale, format: "%.3f"))"
                )
                AudioDiagnostics.tensor("diff/generate/step\(stepIdx)/velocity", velocity)
            }

            if stepIdx == numSteps - 1 {
                xt = DiffusionSampler.finalStep(xt: xt, vt: velocity, currentT: currentT)
            } else {
                let nextT = schedule[stepIdx + 1]
                switch config.method {
                case .ode:
                    xt = DiffusionSampler.odeStep(xt: xt, vt: velocity, currentT: currentT, nextT: nextT)
                case .sde:
                    xt = DiffusionSampler.sdeStep(
                        xt: xt,
                        vt: velocity,
                        currentT: currentT,
                        nextT: nextT,
                        seed: config.seed,
                        stepIndex: stepIdx
                    )
                }
            }
            eval(xt)
            if logDetail {
                AudioDiagnostics.tensor("diff/generate/step\(stepIdx)/xtAfter", xt)
                AudioDiagnostics.memory("diff/generate/step\(stepIdx)/after")
            }

            let elapsed = Date().timeIntervalSince(diffStart)
            progressContinuation?.yield(DiffusionProgress(
                currentStep: stepIdx + 1, totalSteps: numSteps, elapsedTime: elapsed
            ))
        }

        let diffTime = Date().timeIntervalSince(diffStart)
        let totalTime = Date().timeIntervalSince(totalStart)
        let costs = TimeCosts(
            diffusionTime: diffTime,
            perStepTime: diffTime / Double(max(numSteps, 1)),
            totalTime: totalTime
        )

        // Diagnostic: dump final latent magnitude so we can tell if the
        // diffusion is producing reasonable-magnitude latents (~1-5 typical
        // for VAE latent space) or near-zero latents (which would explain
        // silent audio output).
        let absXt = MLX.abs(xt)
        let absMean = absXt.mean().item(Float.self)
        let absMax = absXt.max().item(Float.self)
        diagLogger.info("[DiffusionActor] final latent shape=\(xt.shape) abs_mean=\(absMean) abs_max=\(absMax)")

        // Per-second latent diagnostics: dump abs_mean for each 25-frame
        // window (1 second at 25fps). This tells us exactly where content
        // fades out vs. where the model produces near-zero predictions.
        // 25 latent frames × 1920 hop = 48000 samples = 1 second at 48kHz.
        let nFrames = xt.shape[1]
        let framesPerSec = 25
        let nSecs = (nFrames + framesPerSec - 1) / framesPerSec
        var perSecStats: [(Int, Float)] = []
        for s in 0..<nSecs {
            let startF = s * framesPerSec
            let endF = min(startF + framesPerSec, nFrames)
            // slice [0..., startF..<endF, 0...]
            let chunk = xt[0..., startF..<endF, 0...]
            let chunkAbsMean = MLX.abs(chunk).mean().item(Float.self)
            perSecStats.append((s, chunkAbsMean))
        }
        let statStrs = perSecStats.map { (s, v) in "s\(s)=\(String(format: "%.4f", v))" }.joined(separator: " ")
        diagLogger.info("[DiffusionActor] per-sec latent abs_mean: \(statStrs)")
        AudioDiagnostics.floats("diff/final_latent_abs_mean_per_sec", perSecStats.map(\.1))
        AudioDiagnostics.tensor("diff/generate/finalLatents", xt)
        AudioDiagnostics.memory("diff/generate/end")

        progressContinuation?.finish()
        progressContinuation = nil
        return DiffusionResult(latents: xt, timeCosts: costs)
    }

    public func loadDiTWeights(_ weights: [(String, MLXArray)], modelID: String) async {
        await dit.loadWeights(weights, modelID: modelID)
    }

    public func configureDiT(config: DiTConfig) async {
        await dit.configure(config: config)
    }

    public func unloadDiTWeights() async {
        await dit.unloadWeights()
    }

    // MARK: - LoRA

    public func injectLoRA(config: LoRAConfig) async throws -> [String] {
        try await dit.injectLoRA(config: config)
    }

    public func removeLoRA() async {
        await dit.removeLoRA()
    }

    public func loadLoRAWeights(_ weights: [(String, MLXArray)], adapterID: String) async throws {
        try await dit.loadLoRAWeights(weights, adapterID: adapterID)
    }

    public func addLoRAContribution(
        adapterID: String,
        displayName: String?,
        source: AdapterSource,
        weights: [(String, MLXArray)],
        strength: Float
    ) async {
        await dit.addLoRAContribution(
            adapterID: adapterID,
            displayName: displayName,
            source: source,
            weights: weights,
            strength: strength
        )
    }

    public func clearLoRAContributions() async {
        await dit.clearLoRAContributions()
    }

    public func injectLoKR(config: LoKRConfig) async throws -> [String] {
        try await dit.injectLoKR(config: config)
    }

    public func loadLoKRWeights(_ weights: [(String, MLXArray)], adapterID: String) async throws {
        try await dit.loadLoKRWeights(weights, adapterID: adapterID)
    }

    public func addLoKRContribution(
        adapterID: String,
        displayName: String?,
        source: AdapterSource,
        weights: [(String, MLXArray)],
        strength: Float
    ) async {
        await dit.addLoKRContribution(
            adapterID: adapterID,
            displayName: displayName,
            source: source,
            weights: weights,
            strength: strength
        )
    }

    public func clearLoKRContributions() async {
        await dit.clearLoKRContributions()
    }

    public var injectedAdapterID: String? {
        get async { await dit.injectedAdapterID }
    }

    public func trainLoRA(
        samples: [TrainingSample],
        config: LoRAConfig,
        resumeFrom checkpoint: TrainingCheckpoint? = nil,
        initialWeights: [(String, MLXArray)]? = nil,
        onProgress: @escaping @Sendable (TrainingProgress) -> Void
    ) async throws -> (weights: [(String, MLXArray)], finalLoss: Float?) {
        try await dit.trainLoRA(
            samples: samples,
            config: config,
            resumeFrom: checkpoint,
            initialWeights: initialWeights,
            onProgress: onProgress
        )
    }

    public func trainLoKR(
        samples: [TrainingSample],
        config: LoKRConfig,
        initialWeights: [(String, MLXArray)]? = nil,
        onProgress: @escaping @Sendable (TrainingProgress) -> Void
    ) async throws -> (weights: [(String, MLXArray)], finalLoss: Float?) {
        try await dit.trainLoKR(
            samples: samples,
            config: config,
            initialWeights: initialWeights,
            onProgress: onProgress
        )
    }

    public func cancelTraining() async {
        await dit.cancelTraining()
    }

    public func inpaint(
        startLatents: MLXArray,
        schedule: [Float],
        encoderStates: MLXArray,
        contextLatents: MLXArray,
        encoderMask: MLXArray?,
        config: DiffusionConfig
    ) async throws -> DiffusionResult {
        let totalStart = Date()
        let numSteps = schedule.count

        guard numSteps > 0 else {
            let costs = TimeCosts(diffusionTime: 0, perStepTime: 0, totalTime: 0)
            return DiffusionResult(latents: startLatents, timeCosts: costs)
        }

        var xt = startLatents

        let bsz = xt.shape[0]

        let doCFG = config.guidanceScale > 1.0
        AudioDiagnostics.memory("diff/inpaint/begin")
        AudioDiagnostics.note(
            "diff/inpaint/config",
            "startShape=\(startLatents.shape) steps=\(numSteps) method=\(config.method) guidance=\(config.guidanceScale) cfgInterval=\(config.cfgIntervalStart)...\(config.cfgIntervalEnd) precision=\(config.precision) seed=\(config.seed.map(String.init) ?? "random")"
        )
        AudioDiagnostics.floats("diff/inpaint/schedule", schedule)
        AudioDiagnostics.tensor("diff/inpaint/startLatents", startLatents)
        AudioDiagnostics.tensor("diff/inpaint/encoderStates", encoderStates)
        AudioDiagnostics.tensor("diff/inpaint/contextLatents", contextLatents)
        if let encoderMask {
            AudioDiagnostics.tensor("diff/inpaint/encoderMask", encoderMask)
        } else {
            AudioDiagnostics.note("diff/inpaint/encoderMask", "nil")
        }
        if doCFG {
            AudioDiagnostics.note(
                "diff/inpaint/cfgPlan",
                audioDiagnosticCFGPlan(
                    schedule: schedule,
                    intervalStart: config.cfgIntervalStart,
                    intervalEnd: config.cfgIntervalEnd
                )
            )
        } else {
            AudioDiagnostics.note(
                "diff/inpaint/cfgPlan",
                "disabled guidanceScale=\(audioDiagnosticFloat(config.guidanceScale, format: "%.3f"))"
            )
        }
        var encHS = encoderStates
        var ctx = contextLatents
        var encMask = encoderMask
        if doCFG {
            // Match upstream MLX (acestep/models/mlx/dit_generate.py:188): the
            // CFG unconditional branch uses the LEARNED `null_condition_emb`
            // token broadcast to encoder_states shape, NOT zeros. Zeros are
            // out-of-distribution for the trained DiT, making the unconditional
            // prediction garbage so CFG (cond - uncond) injects noise on top
            // of the conditional pred instead of isolating the prompt direction.
            let nullEmb: MLXArray
            if let nullCond = await dit.nullConditionEmbedding() {
                nullEmb = MLX.broadcast(nullCond, to: encHS.shape).asType(encHS.dtype)
                AudioDiagnostics.tensor("diff/inpaint/nullConditionEmbedding", nullEmb)
            } else {
                nullEmb = MLXArray.zeros(like: encHS)
                AudioDiagnostics.note("diff/inpaint/nullConditionEmbedding", "missing; falling back to zeros")
            }
            encHS = concatenated([encHS, nullEmb], axis: 0)
            ctx = concatenated([ctx, ctx], axis: 0)
            encMask = encoderMask.map { concatenated([$0, $0], axis: 0) }
        }
        let momentum: APGMomentum? = doCFG ? APGMomentum() : nil

        let cache = CrossAttentionCache()
        let diffStart = Date()
        AudioDiagnostics.memory("diff/inpaint/diffusionBegin")

        for stepIdx in 0..<numSteps {
            try Task.checkCancellation()
            let currentT = schedule[stepIdx]
            let logDetail = AudioDiagnostics.isEnabled && audioDiagnosticStep(stepIdx, totalSteps: numSteps)

            let xIn = doCFG ? concatenated([xt, xt], axis: 0) : xt
            if logDetail {
                AudioDiagnostics.tensor("diff/inpaint/step\(stepIdx)/xtBefore", xt)
            }

            let (vt, _) = try await dit.step(
                noisyLatents: xIn,
                timestep: currentT,
                timestepR: currentT,
                encoderStates: encHS,
                contextLatents: ctx,
                encoderAttentionMask: encMask,
                cache: cache,
                useCache: !doCFG
            )
            if logDetail {
                AudioDiagnostics.tensor("diff/inpaint/step\(stepIdx)/vt", vt)
            }

            var velocity = vt

            if doCFG {
                let predCond = vt[0..<bsz]
                let predUncond = vt[bsz...]
                let applyCFG = currentT >= config.cfgIntervalStart && currentT <= config.cfgIntervalEnd
                if applyCFG {
                    velocity = DiffusionSampler.apgForward(
                        predCond: predCond, predUncond: predUncond,
                        guidanceScale: config.guidanceScale, momentum: momentum
                    )
                } else {
                    velocity = predCond
                }
                if logDetail {
                    AudioDiagnostics.note(
                        "diff/inpaint/cfgStats",
                        audioDiagnosticCFGStats(
                            stepIdx: stepIdx,
                            timestep: currentT,
                            guidanceScale: config.guidanceScale,
                            applyCFG: applyCFG,
                            predCond: predCond,
                            predUncond: predUncond,
                            velocity: velocity
                        )
                    )
                    AudioDiagnostics.tensor("diff/inpaint/step\(stepIdx)/predCond", predCond)
                    AudioDiagnostics.tensor("diff/inpaint/step\(stepIdx)/predUncond", predUncond)
                    AudioDiagnostics.tensor("diff/inpaint/step\(stepIdx)/cfgDelta", predCond - predUncond)
                    AudioDiagnostics.tensor("diff/inpaint/step\(stepIdx)/velocity", velocity)
                }
            } else if logDetail {
                AudioDiagnostics.note(
                    "diff/inpaint/cfgStats",
                    "step=\(stepIdx) t=\(audioDiagnosticFloat(currentT, format: "%.4f")) doCFG=false guidance=\(audioDiagnosticFloat(config.guidanceScale, format: "%.3f"))"
                )
                AudioDiagnostics.tensor("diff/inpaint/step\(stepIdx)/velocity", velocity)
            }

            if stepIdx == numSteps - 1 {
                xt = DiffusionSampler.finalStep(xt: xt, vt: velocity, currentT: currentT)
            } else {
                let nextT = schedule[stepIdx + 1]
                switch config.method {
                case .ode:
                    xt = DiffusionSampler.odeStep(xt: xt, vt: velocity, currentT: currentT, nextT: nextT)
                case .sde:
                    xt = DiffusionSampler.sdeStep(
                        xt: xt,
                        vt: velocity,
                        currentT: currentT,
                        nextT: nextT,
                        seed: config.seed,
                        stepIndex: stepIdx
                    )
                }
            }
            eval(xt)
            if logDetail {
                AudioDiagnostics.tensor("diff/inpaint/step\(stepIdx)/xtAfter", xt)
                AudioDiagnostics.memory("diff/inpaint/step\(stepIdx)/after")
            }

            let elapsed = Date().timeIntervalSince(diffStart)
            progressContinuation?.yield(DiffusionProgress(
                currentStep: stepIdx + 1, totalSteps: numSteps, elapsedTime: elapsed
            ))
        }

        let diffTime = Date().timeIntervalSince(diffStart)
        let totalTime = Date().timeIntervalSince(totalStart)
        let costs = TimeCosts(
            diffusionTime: diffTime,
            perStepTime: diffTime / Double(max(numSteps, 1)),
            totalTime: totalTime
        )

        AudioDiagnostics.tensor("diff/inpaint/finalLatents", xt)
        AudioDiagnostics.memory("diff/inpaint/end")

        progressContinuation?.finish()
        progressContinuation = nil
        return DiffusionResult(latents: xt, timeCosts: costs)
    }
}
