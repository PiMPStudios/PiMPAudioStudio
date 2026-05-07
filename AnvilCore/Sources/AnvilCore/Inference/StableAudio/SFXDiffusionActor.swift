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

    public func sample(
        conditioning: MLXArray,
        config: SFXDiffusionConfig
    ) async throws -> DiffusionResult {
        let totalStart = Date()

        var xt = DiffusionSampler.generateNoise(shape: config.latentShape, seed: config.seed)

        let schedule = generateLinearSchedule(steps: config.steps)
        let doCFG = config.guidanceScale > 1.0

        let diffStart = Date()

        for stepIdx in 0..<config.steps {
            try Task.checkCancellation()

            let currentT = schedule[stepIdx]
            let timestep = MLXArray([currentT])

            var vt: MLXArray
            if doCFG {
                let condPred = try await dit.predict(latents: xt, timestep: timestep, conditioning: conditioning)
                let uncondPred = try await dit.predict(latents: xt, timestep: timestep, conditioning: MLXArray.zeros(like: conditioning))
                vt = uncondPred + config.guidanceScale * (condPred - uncondPred)
            } else {
                vt = try await dit.predict(latents: xt, timestep: timestep, conditioning: conditioning)
            }

            if stepIdx == config.steps - 1 {
                xt = DiffusionSampler.finalStep(xt: xt, vt: vt, currentT: currentT)
            } else {
                let nextT = schedule[stepIdx + 1]
                xt = DiffusionSampler.odeStep(xt: xt, vt: vt, currentT: currentT, nextT: nextT)
            }
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

    public func inpaint(
        startLatents: MLXArray,
        schedule: [Float],
        conditioning: MLXArray,
        config: SFXDiffusionConfig
    ) async throws -> DiffusionResult {
        let totalStart = Date()

        guard !schedule.isEmpty else {
            let costs = TimeCosts(diffusionTime: 0, perStepTime: 0, totalTime: 0)
            return DiffusionResult(latents: startLatents, timeCosts: costs)
        }

        var xt = startLatents
        let stepCount = schedule.count
        let doCFG = config.guidanceScale > 1.0

        let diffStart = Date()

        for stepIdx in 0..<stepCount {
            try Task.checkCancellation()

            let currentT = schedule[stepIdx]
            let timestep = MLXArray([currentT])

            var vt: MLXArray
            if doCFG {
                let condPred = try await dit.predict(latents: xt, timestep: timestep, conditioning: conditioning)
                let uncondPred = try await dit.predict(latents: xt, timestep: timestep, conditioning: MLXArray.zeros(like: conditioning))
                vt = uncondPred + config.guidanceScale * (condPred - uncondPred)
            } else {
                vt = try await dit.predict(latents: xt, timestep: timestep, conditioning: conditioning)
            }

            if stepIdx == stepCount - 1 {
                xt = DiffusionSampler.finalStep(xt: xt, vt: vt, currentT: currentT)
            } else {
                let nextT = schedule[stepIdx + 1]
                xt = DiffusionSampler.odeStep(xt: xt, vt: vt, currentT: currentT, nextT: nextT)
            }
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

    private func generateLinearSchedule(steps: Int) -> [Float] {
        (0..<steps).map { i in
            1.0 - Float(i) / Float(steps)
        }
    }
}
