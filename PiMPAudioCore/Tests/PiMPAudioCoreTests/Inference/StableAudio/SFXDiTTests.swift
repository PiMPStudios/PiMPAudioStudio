// PiMPAudioCore/Tests/PiMPAudioCoreTests/Inference/StableAudio/SFXDiTTests.swift
import MLX
import MLXRandom
@testable import PiMPAudioCore
import Testing

@Suite("SFX DiT")
struct SFXDiTTests {
    /// Small test config — all new fields derive from the five core params.
    let config = SFXDiTConfig(
        hiddenSize: 128,
        numLayers: 1,
        numHeads: 4,
        latentDim: 64,
        conditioningDim: 32
        // Derived: headDim=32, numKVHeads=2, ffIntermediateDim=512,
        //          rotaryDim=16, globalCondDim=64, condTokenDim=32
    )

    @Test("Forward pass produces correct output shape")
    func forwardShape() {
        let model = SFXDiTModel(config: config)
        let latents = MLXRandom.normal([1, 16, 64])
        let timestep = MLXArray([Float(0.5)])
        let crossCond = MLXRandom.normal([1, 4, config.conditioningDim])
        let globalCond = MLXRandom.normal([1, config.globalCondDim])
        let output = model.forward(
            latents: latents, timestep: timestep,
            crossCond: crossCond, globalCond: globalCond
        )
        #expect(output.shape == [1, 16, 64])
    }

    @Test("Actor predict returns correct shape after loading weights")
    func actorPredict() async throws {
        let actor = SFXDiTActor(config: config)
        try await actor.loadWeights(randomDiTWeights())
        let latents = MLXRandom.normal([1, 16, 64])
        let timestep = MLXArray([Float(0.5)])
        let crossCond = MLXRandom.normal([1, 4, config.conditioningDim])
        let globalCond = MLXRandom.normal([1, config.globalCondDim])
        let output = try await actor.predict(
            latents: latents, timestep: timestep,
            crossCond: crossCond, globalCond: globalCond
        )
        #expect(output.shape == [1, 16, 64])
    }

    @Test("Predict throws when weights not loaded")
    func predictThrowsWithoutWeights() async {
        let actor = SFXDiTActor(config: config)
        let latents = MLXRandom.normal([1, 16, 64])
        let timestep = MLXArray([Float(0.5)])
        let crossCond = MLXRandom.normal([1, 4, config.conditioningDim])
        let globalCond = MLXRandom.normal([1, config.globalCondDim])
        await #expect(throws: SFXDiTActorError.weightsNotLoaded) {
            try await actor.predict(
                latents: latents, timestep: timestep,
                crossCond: crossCond, globalCond: globalCond
            )
        }
    }

    @Test("unloadWeights causes subsequent predict to throw")
    func unloadWeightsCausesThrow() async throws {
        let actor = SFXDiTActor(config: config)
        try await actor.loadWeights(randomDiTWeights())
        await actor.unloadWeights()
        let latents = MLXRandom.normal([1, 16, 64])
        let timestep = MLXArray([Float(0.5)])
        let crossCond = MLXRandom.normal([1, 4, config.conditioningDim])
        let globalCond = MLXRandom.normal([1, config.globalCondDim])
        await #expect(throws: SFXDiTActorError.weightsNotLoaded) {
            try await actor.predict(
                latents: latents, timestep: timestep,
                crossCond: crossCond, globalCond: globalCond
            )
        }
    }

    @Test("Actor loadWeights marks as loaded")
    func actorLoadWeights() async throws {
        let actor = SFXDiTActor(config: config)
        #expect(await actor.isLoaded == false)
        try await actor.loadWeights(randomDiTWeights())
        #expect(await actor.isLoaded == true)
    }

    // MARK: - Weight factory

    private func randomDiTWeights() -> [(String, MLXArray)] {
        let H = config.hiddenSize
        let L = config.latentDim
        let halfF = config.timestepFeaturesDim / 2
        let condD = config.conditioningDim
        let globalD = config.globalCondDim
        let condTD = config.condTokenDim
        let nH = config.numHeads
        let nKV = config.numKVHeads
        let hD = config.headDim
        let ffDim = config.ffIntermediateDim
        var w: [(String, MLXArray)] = []

        // Conv1d — already in MLX layout [out, kernel, in]
        w.append(("preprocess_conv.weight", MLXRandom.normal([L, 1, L])))
        w.append(("timestep_features.weight", MLXRandom.normal([halfF, 1])))

        // Embed MLPs
        let embedCfg: [(name: String, inputDim: Int, hiddenDim: Int, outputDim: Int)] = [
            ("to_timestep_embed", config.timestepFeaturesDim, H, H),
            ("to_cond_embed", condTD, H, condTD),
            ("to_global_embed", globalD, H, H),
        ]
        for (name, inputDim, hiddenDim, outputDim) in embedCfg {
            w.append(("\(name).fc1.weight", MLXRandom.normal([hiddenDim, inputDim])))
            w.append(("\(name).fc1.bias", MLXArray.zeros([hiddenDim])))
            w.append(("\(name).fc2.weight", MLXRandom.normal([outputDim, hiddenDim])))
            w.append(("\(name).fc2.bias", MLXArray.zeros([outputDim])))
        }

        // Transformer layers
        for i in 0 ..< config.numLayers {
            let p = "transformer.layers.\(i)"
            w.append(("\(p).pre_norm.weight", MLXArray.ones([H])))
            w.append(("\(p).pre_norm.bias", MLXArray.zeros([H])))
            w.append(("\(p).self_attn.to_qkv.weight", MLXRandom.normal([3 * nH * hD, H])))
            w.append(("\(p).self_attn.to_out.weight", MLXRandom.normal([H, nH * hD])))
            w.append(("\(p).cross_attend_norm.weight", MLXArray.ones([H])))
            w.append(("\(p).cross_attend_norm.bias", MLXArray.zeros([H])))
            w.append(("\(p).cross_attn.to_q.weight", MLXRandom.normal([nH * hD, H])))
            w.append(("\(p).cross_attn.to_kv.weight", MLXRandom.normal([2 * nKV * hD, condD])))
            w.append(("\(p).cross_attn.to_out.weight", MLXRandom.normal([H, nH * hD])))
            w.append(("\(p).ff_norm.weight", MLXArray.ones([H])))
            w.append(("\(p).ff_norm.bias", MLXArray.zeros([H])))
            w.append(("\(p).ff.gate_up.weight", MLXRandom.normal([2 * ffDim, H])))
            w.append(("\(p).ff.gate_up.bias", MLXArray.zeros([2 * ffDim])))
            w.append(("\(p).ff.out.weight", MLXRandom.normal([H, ffDim])))
            w.append(("\(p).ff.out.bias", MLXArray.zeros([H])))
        }

        w.append(("transformer.project_in.weight", MLXRandom.normal([H, L])))
        w.append(("transformer.project_out.weight", MLXRandom.normal([L, H])))
        w.append(("postprocess_conv.weight", MLXRandom.normal([L, 1, L])))

        return w
    }
}
