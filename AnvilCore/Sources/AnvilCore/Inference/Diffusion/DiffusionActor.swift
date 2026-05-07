import Foundation
import MLX
import MLXRandom
import OSLog

private let diagLogger = Logger(subsystem: "com.pimp.AnvilApp", category: "DiffusionActor")

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
        AudioDiagnostics.note("config", "shift=\(config.shift) steps=\(config.steps ?? -1) method=\(config.method) guidance=\(config.guidanceScale) precision=\(config.precision) seed=\(config.seed.map(String.init) ?? "random")")
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
            } else {
                nullEmb = MLXArray.zeros(like: encHS)
            }
            encHS = concatenated([encHS, nullEmb], axis: 0)
            ctx = concatenated([ctx, ctx], axis: 0)
            encMask = encoderMask.map { concatenated([$0, $0], axis: 0) }
        }
        let momentum: APGMomentum? = doCFG ? APGMomentum() : nil

        let coverSteps = Int(Float(numSteps) * config.audioCoverStrength)

        let cache = CrossAttentionCache()
        let diffStart = Date()

        for stepIdx in 0..<numSteps {
            try Task.checkCancellation()
            let currentT = schedule[stepIdx]

            if stepIdx >= coverSteps {
                cache.reset()
            }

            let xIn = doCFG ? concatenated([xt, xt], axis: 0) : xt

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
            }

            if stepIdx == numSteps - 1 {
                xt = DiffusionSampler.finalStep(xt: xt, vt: velocity, currentT: currentT)
            } else {
                let nextT = schedule[stepIdx + 1]
                switch config.method {
                case .ode:
                    xt = DiffusionSampler.odeStep(xt: xt, vt: velocity, currentT: currentT, nextT: nextT)
                case .sde:
                    xt = DiffusionSampler.sdeStep(xt: xt, vt: velocity, currentT: currentT, nextT: nextT)
                }
            }
            eval(xt)

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

        progressContinuation?.finish()
        progressContinuation = nil
        return DiffusionResult(latents: xt, timeCosts: costs)
    }

    public func loadDiTWeights(_ weights: [(String, MLXArray)], modelID: String) async {
        await dit.loadWeights(weights, modelID: modelID)
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

    public var injectedAdapterID: String? {
        get async { await dit.injectedAdapterID }
    }

    public func trainLoRA(
        samples: [TrainingSample],
        config: LoRAConfig,
        onProgress: @escaping (TrainingProgress) -> Void
    ) async throws -> (weights: [(String, MLXArray)], finalLoss: Float?) {
        try await dit.trainLoRA(samples: samples, config: config, onProgress: onProgress)
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
            } else {
                nullEmb = MLXArray.zeros(like: encHS)
            }
            encHS = concatenated([encHS, nullEmb], axis: 0)
            ctx = concatenated([ctx, ctx], axis: 0)
            encMask = encoderMask.map { concatenated([$0, $0], axis: 0) }
        }
        let momentum: APGMomentum? = doCFG ? APGMomentum() : nil

        let cache = CrossAttentionCache()
        let diffStart = Date()

        for stepIdx in 0..<numSteps {
            try Task.checkCancellation()
            let currentT = schedule[stepIdx]

            let xIn = doCFG ? concatenated([xt, xt], axis: 0) : xt

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
            }

            if stepIdx == numSteps - 1 {
                xt = DiffusionSampler.finalStep(xt: xt, vt: velocity, currentT: currentT)
            } else {
                let nextT = schedule[stepIdx + 1]
                switch config.method {
                case .ode:
                    xt = DiffusionSampler.odeStep(xt: xt, vt: velocity, currentT: currentT, nextT: nextT)
                case .sde:
                    xt = DiffusionSampler.sdeStep(xt: xt, vt: velocity, currentT: currentT, nextT: nextT)
                }
            }
            eval(xt)

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

        progressContinuation?.finish()
        progressContinuation = nil
        return DiffusionResult(latents: xt, timeCosts: costs)
    }
}
