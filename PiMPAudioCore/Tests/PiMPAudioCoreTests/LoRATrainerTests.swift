import Testing
import Foundation
import MLX
@testable import PiMPAudioCore

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

@Test func lokrConfigMirrorsTrainingFieldsForCompatibility() {
    let config = LoKRConfig(
        linearDim: 128,
        alpha: 96,
        targetLayers: [.mlp],
        learningRate: 2e-4,
        maxEpochs: 75,
        warmupFraction: 0.2,
        gradientAccumulationSteps: 4
    )

    let compatible = config.loraCompatibleConfig
    #expect(compatible.rank == 128)
    #expect(compatible.alpha == 96)
    #expect(compatible.targetLayers == [.mlp])
    #expect(compatible.learningRate == 2e-4)
    #expect(compatible.maxEpochs == 75)
    #expect(compatible.warmupFraction == 0.2)
    #expect(compatible.gradientAccumulationSteps == 4)
}

@Test func lokrKroneckerWeightCropsToBaseShape() {
    let w1Values: [Float] = [1, 2, 3, 4]
    let w2Values: [Float] = [10, 20, 30, 40]
    let w1 = MLXArray(w1Values).reshaped([2, 2])
    let w2 = MLXArray(w2Values).reshaped([2, 2])

    let weight = LoKRLinear.kroneckerWeight(
        w1,
        w2,
        outputFeatures: 3,
        inputFeatures: 3
    )

    #expect(weight.shape == [3, 3])
    let values = weight.reshaped([-1]).asArray(Float.self)
    #expect(values == [10, 20, 20, 30, 40, 60, 30, 60, 40])
}

@Test func oldAdapterJSONDecodesAsLoRAKind() throws {
    let json = """
    {
      "id": "old-adapter",
      "name": "Old Adapter",
      "config": {
        "rank": 16,
        "alpha": 16,
        "dropout": 0,
        "targetLayers": ["self_attn"],
        "learningRate": 0.0001,
        "maxEpochs": 10,
        "batchSize": 1,
        "warmupFraction": 0.1,
        "gradientAccumulationSteps": 1
      },
      "baseModelID": "ace-step-xl-sft",
      "baseModelCategory": "aceStepDiT",
      "createdAt": "2026-05-30T12:00:00Z",
      "sampleCount": 2,
      "finalLoss": 0.5
    }
    """

    let decoder = JSONDecoder()
    decoder.dateDecodingStrategy = .iso8601
    let adapter = try decoder.decode(LoRAAdapter.self, from: Data(json.utf8))

    #expect(adapter.kind == .lora)
    #expect(adapter.lokrConfig == nil)
    #expect(adapter.source == .local)
}
