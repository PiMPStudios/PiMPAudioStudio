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

@Test func loraConfigDefaultsAlphaToRank() {
    let rankOnly = LoRAConfig(rank: 64)
    #expect(rankOnly.alpha == 64)

    let explicitAlpha = LoRAConfig(rank: 64, alpha: 16)
    #expect(explicitAlpha.alpha == 16)
}

@Test func loraConfigCanRepresentCloudStyleFullAttentionRecipe() {
    let config = LoRAConfig(
        rank: 96,
        alpha: 192,
        dropout: 0.08,
        exactInjectionPatterns: LoRAConfig.fullAttentionInjectionPatterns,
        learningRate: 0.000075,
        maxEpochs: 150,
        gradientAccumulationSteps: 4,
        trainingLayerCount: 32
    )

    #expect(config.injectionPatterns == [
        "selfAttn.qProj",
        "selfAttn.kProj",
        "selfAttn.vProj",
        "selfAttn.oProj",
        "crossAttn.qProj",
        "crossAttn.kProj",
        "crossAttn.vProj",
        "crossAttn.oProj",
    ])
    #expect(config.alpha == 192)
    #expect(config.dropout == 0.08)
    #expect(config.trainingLayerCount == 32)
}

@Test func lokrConfigMirrorsTrainingFieldsForCompatibility() {
    let config = LoKRConfig(
        linearDim: 128,
        alpha: 96,
        targetLayers: [.mlp],
        learningRate: 2e-4,
        maxEpochs: 75,
        warmupFraction: 0.2,
        gradientAccumulationSteps: 4,
        trainingLayerCount: 12
    )

    let compatible = config.loraCompatibleConfig
    #expect(compatible.rank == 128)
    #expect(compatible.alpha == 96)
    #expect(compatible.targetLayers == [.mlp])
    #expect(compatible.learningRate == 2e-4)
    #expect(compatible.maxEpochs == 75)
    #expect(compatible.warmupFraction == 0.2)
    #expect(compatible.gradientAccumulationSteps == 4)
    #expect(compatible.trainingLayerCount == 12)
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
    #expect(adapter.trainedPaths == nil)
    #expect(adapter.trainedLayerIndices == nil)
    #expect(adapter.config.trainingLayerCount == nil)
}

@Test func adapterMetadataRoundTripsTrainedLayerFootprint() throws {
    let adapter = LoRAAdapter(
        id: "layered",
        name: "Layered",
        config: LoRAConfig(rank: 16, trainingLayerCount: 8),
        kind: .lokr,
        lokrConfig: LoKRConfig(linearDim: 32, trainingLayerCount: 8),
        baseModelID: "ace-step-dit-xl",
        baseModelCategory: .aceStepDiT,
        createdAt: Date(timeIntervalSince1970: 0),
        sampleCount: 10,
        finalLoss: 0.25,
        trainedPaths: [
            "layers.28.selfAttn.qProj",
            "layers.28.selfAttn.vProj",
            "layers.29.selfAttn.qProj",
        ],
        trainedLayerIndices: [28, 29]
    )

    let encoder = JSONEncoder()
    encoder.dateEncodingStrategy = .iso8601
    let data = try encoder.encode(adapter)
    let decoder = JSONDecoder()
    decoder.dateDecodingStrategy = .iso8601
    let decoded = try decoder.decode(LoRAAdapter.self, from: data)

    #expect(decoded.config.trainingLayerCount == 8)
    #expect(decoded.lokrConfig?.trainingLayerCount == 8)
    #expect(decoded.effectiveTrainedLayerIndices == [28, 29])
    #expect(decoded.effectiveTrainedPaths?.count == 3)
}

@Test func adapterDerivesTrainedPathsFromWeightKeys() {
    let paths = LoRAAdapter.trainedPaths(
        fromWeightKeys: [
            "layers.28.selfAttn.qProj.lokrW1",
            "layers.28.selfAttn.qProj.lokrW2",
            "layers.29.crossAttn.vProj.lokrW1",
            "metadata"
        ],
        kind: .lokr
    )

    #expect(paths == [
        "layers.28.selfAttn.qProj",
        "layers.29.crossAttn.vProj",
    ])
    #expect(LoRAAdapter.layerIndices(from: paths) == [28, 29])
}

@Test func adapterCompatibilityRejectsWrongACEVariant() {
    let xlSpec = ModelSpec(
        id: "ace-step-dit-xl-sft",
        displayName: "ACE-Step v1.5 XL SFT",
        huggingFaceRepo: "ACE-Step/acestep-v15-xl-sft",
        files: [],
        category: .aceStepDiT,
        variantFamily: "ace-step-1.5-xl"
    )
    let xlAdapter = LoRAAdapter(
        id: "xl",
        name: "XL",
        config: LoRAConfig(),
        baseModelID: "ace-step-dit-xl",
        baseModelCategory: .aceStepDiT,
        createdAt: Date(timeIntervalSince1970: 0),
        sampleCount: 0,
        finalLoss: nil
    )
    let xlSFTAdapter = LoRAAdapter(
        id: "xl-sft",
        name: "XL SFT",
        config: LoRAConfig(),
        baseModelID: "ace-step-dit-xl-sft",
        baseModelCategory: .aceStepDiT,
        createdAt: Date(timeIntervalSince1970: 0),
        sampleCount: 0,
        finalLoss: nil
    )
    let xlBaseAdapter = LoRAAdapter(
        id: "xl-base",
        name: "XL Base",
        config: LoRAConfig(),
        baseModelID: "ace-step-dit-xl-base",
        baseModelCategory: .aceStepDiT,
        createdAt: Date(timeIntervalSince1970: 0),
        sampleCount: 0,
        finalLoss: nil
    )
    let turboAdapter = LoRAAdapter(
        id: "turbo",
        name: "Turbo",
        config: LoRAConfig(),
        baseModelID: "ace-step-dit-turbo",
        baseModelCategory: .aceStepDiT,
        createdAt: Date(timeIntervalSince1970: 0),
        sampleCount: 0,
        finalLoss: nil
    )

    #expect(xlAdapter.isCompatible(with: xlSpec.id, activeSpec: xlSpec))
    #expect(xlSFTAdapter.isCompatible(with: xlSpec.id, activeSpec: xlSpec))
    #expect(!xlBaseAdapter.isCompatible(with: xlSpec.id, activeSpec: xlSpec))
    #expect(!turboAdapter.isCompatible(with: xlSpec.id, activeSpec: xlSpec))
}
