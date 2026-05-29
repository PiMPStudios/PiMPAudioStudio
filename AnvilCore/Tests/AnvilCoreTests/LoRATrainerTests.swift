import Testing
@testable import AnvilCore

@Test func cosineSchedulerUsesWarmupFraction() {
    let scheduler = CosineScheduler(baseLR: 0.001, warmupFraction: 0.25, totalSteps: 8)

    #expect(scheduler.warmupSteps == 2)
    #expect(scheduler.learningRate(atStep: 0) == 0)
    #expect(abs(scheduler.learningRate(atStep: 1) - 0.0005) < 0.0000001)
    #expect(abs(scheduler.learningRate(atStep: 2) - 0.001) < 0.0000001)
}

@Test func cosineSchedulerClampsWarmupFraction() {
    let scheduler = CosineScheduler(baseLR: 0.001, warmupFraction: 2.0, totalSteps: 10)

    #expect(scheduler.warmupFraction == 1)
    #expect(scheduler.warmupSteps == 10)
    #expect(abs(scheduler.learningRate(atStep: 5) - 0.0005) < 0.0000001)
}

@Test func optimizerStepsPerEpochUsesAccumulationWindows() {
    #expect(LoRATrainer.optimizerStepsPerEpoch(
        datasetSize: 10,
        gradientAccumulationSteps: 4
    ) == 3)
    #expect(LoRATrainer.optimizerStepsPerEpoch(
        datasetSize: 2,
        gradientAccumulationSteps: 8
    ) == 1)
    #expect(LoRATrainer.optimizerStepsPerEpoch(
        datasetSize: 0,
        gradientAccumulationSteps: 4
    ) == 1)
}
