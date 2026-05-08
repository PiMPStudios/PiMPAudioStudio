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
    /// - Parameters:
    ///   - crossCond:  `[B, L, conditioningDim]` T5-base hidden states
    ///   - globalCond: `[B, globalCondDim]` concat(startEmbed, totalEmbed)
    public func sample(
        crossCond: MLXArray,
        globalCond: MLXArray,
        config: SFXDiffusionConfig
    ) async throws -> DiffusionResult {
        let totalStart = Date()

        // SAO uses Karras EDM sigma schedule: σ ∈ [500, …, 0.3] (100 steps, high→low noise)
        var xt = DiffusionSampler.generateNoise(shape: config.latentShape, seed: config.seed)

        // Scale initial noise to σ_max (EDM requires x_T ~ N(0, σ_max²·I))
        let sigmaMax: Float = 500.0
        xt = xt * MLXArray(sigmaMax)

        let sigmas = generateKarrasSigmas(steps: config.steps)
        let doCFG  = config.guidanceScale > 1.0

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
                    crossCond: MLXArray.zeros(like: crossCond),
                    globalCond: globalCond
                )
                vt = uncondPred + config.guidanceScale * (condPred - uncondPred)
            } else {
                vt = try await dit.predict(
                    latents: xt, timestep: timestep,
                    crossCond: crossCond, globalCond: globalCond
                )
            }

            xt = DiffusionSampler.edmEulerStep(xt: xt, vt: vt, sigma: sigma, sigmaNext: sigmaNext)
            eval(xt)

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

        progressContinuation?.finish()
        progressContinuation = nil
        return DiffusionResult(latents: xt, timeCosts: costs)
    }

    /// Inpaint using a partial-noise schedule.  Stable Audio inpainting is
    /// currently gated in `InpaintCoordinator`; this method signature is
    /// preserved for future use.
    public func inpaint(
        startLatents: MLXArray,
        schedule: [Float],
        crossCond: MLXArray,
        globalCond: MLXArray,
        config: SFXDiffusionConfig
    ) async throws -> DiffusionResult {
        let totalStart = Date()

        guard !schedule.isEmpty else {
            let costs = TimeCosts(diffusionTime: 0, perStepTime: 0, totalTime: 0)
            return DiffusionResult(latents: startLatents, timeCosts: costs)
        }

        // `schedule` is expected to be a Karras sigma sequence [σ_high … σ_low].
        var xt = startLatents
        let stepCount = schedule.count
        let doCFG = config.guidanceScale > 1.0

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
                    crossCond: MLXArray.zeros(like: crossCond),
                    globalCond: globalCond
                )
                vt = uncondPred + config.guidanceScale * (condPred - uncondPred)
            } else {
                vt = try await dit.predict(
                    latents: xt, timestep: timestep,
                    crossCond: crossCond, globalCond: globalCond
                )
            }

            xt = DiffusionSampler.edmEulerStep(xt: xt, vt: vt, sigma: sigma, sigmaNext: sigmaNext)
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

    /// Karras et al. sigma schedule for EDM diffusion.
    ///
    /// Produces `steps` values descending from σ_max=500 to σ_min=0.3 using the
    /// power-law interpolation: `σ_i = (σ_max^(1/ρ) + i/(N−1)·(σ_min^(1/ρ) − σ_max^(1/ρ)))^ρ`
    /// where ρ=7 (standard EDM default).
    ///
    /// - Returns: array of length `steps` in descending order [σ_max … σ_min]
    private func generateKarrasSigmas(steps: Int) -> [Float] {
        let sigmaMin: Float = 0.3
        let sigmaMax: Float = 500.0
        let rho:      Float = 7.0
        let minInvRho = powf(sigmaMin, 1.0 / rho)
        let maxInvRho = powf(sigmaMax, 1.0 / rho)
        return (0..<steps).map { i in
            let t = Float(i) / Float(steps - 1)
            return powf(maxInvRho + t * (minInvRho - maxInvRho), rho)
        }
    }
}
