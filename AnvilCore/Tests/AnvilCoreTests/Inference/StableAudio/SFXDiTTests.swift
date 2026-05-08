// AnvilCore/Tests/AnvilCoreTests/Inference/StableAudio/SFXDiTTests.swift
import Testing
import MLX
import MLXRandom
@testable import AnvilCore

@Suite("SFX DiT")
struct SFXDiTTests {
    // Small test config — all new fields derive from the five core params.
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
        let latents    = MLXRandom.normal([1, 16, 64])
        let timestep   = MLXArray([0.5])
        let crossCond  = MLXRandom.normal([1, 4, config.conditioningDim])
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
        let latents    = MLXRandom.normal([1, 16, 64])
        let timestep   = MLXArray([0.5])
        let crossCond  = MLXRandom.normal([1, 4, config.conditioningDim])
        let globalCond = MLXRandom.normal([1, config.globalCondDim])
        let output = try await actor.predict(
            latents: latents, timestep: timestep,
            crossCond: crossCond, globalCond: globalCond
        )
        #expect(output.shape == [1, 16, 64])
    }

    @Test("Predict throws when weights not loaded")
    func predictThrowsWithoutWeights() async {
        let actor      = SFXDiTActor(config: config)
        let latents    = MLXRandom.normal([1, 16, 64])
        let timestep   = MLXArray([0.5])
        let crossCond  = MLXRandom.normal([1, 4, config.conditioningDim])
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
        let latents    = MLXRandom.normal([1, 16, 64])
        let timestep   = MLXArray([0.5])
        let crossCond  = MLXRandom.normal([1, 4, config.conditioningDim])
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
        let H      = config.hiddenSize
        let L      = config.latentDim
        let halfF  = config.timestepFeaturesDim / 2
        let condD  = config.conditioningDim
        let globalD = config.globalCondDim
        let condTD = config.condTokenDim
        let nH     = config.numHeads
        let nKV    = config.numKVHeads
        let hD     = config.headDim
        let ffDim  = config.ffIntermediateDim
        var w: [(String, MLXArray)] = []

        // Conv1d — already in MLX layout [out, kernel, in]
        w.append(("preprocess_conv.weight",  MLXRandom.normal([H, 1, L])))
        w.append(("timestep_features.weight", MLXRandom.normal([halfF, 1])))

        // Embed MLPs
        let embedCfg: [(String, Int)] = [
            ("to_timestep_embed", config.timestepFeaturesDim),
            ("to_cond_embed",     condTD),
            ("to_global_embed",   globalD),
        ]
        for (name, inDim) in embedCfg {
            w.append(("\(name).fc1.weight", MLXRandom.normal([H, inDim])))
            w.append(("\(name).fc1.bias",   MLXArray.zeros([H])))
            w.append(("\(name).fc2.weight", MLXRandom.normal([H, H])))
            w.append(("\(name).fc2.bias",   MLXArray.zeros([H])))
        }

        // Transformer layers
        for i in 0..<config.numLayers {
            let p = "transformer.layers.\(i)"
            w.append(("\(p).norm1.weight", MLXArray.ones([H])))
            w.append(("\(p).norm1.bias",   MLXArray.zeros([H])))
            w.append(("\(p).self_attn.to_qkv.weight", MLXRandom.normal([3 * nH * hD, H])))
            w.append(("\(p).self_attn.to_out.weight", MLXRandom.normal([H, nH * hD])))
            w.append(("\(p).norm2.weight", MLXArray.ones([H])))
            w.append(("\(p).norm2.bias",   MLXArray.zeros([H])))
            w.append(("\(p).cross_attn.to_q.weight",  MLXRandom.normal([nH * hD, H])))
            w.append(("\(p).cross_attn.to_kv.weight", MLXRandom.normal([2 * nKV * hD, condD])))
            w.append(("\(p).cross_attn.to_out.weight", MLXRandom.normal([H, nH * hD])))
            w.append(("\(p).norm3.weight", MLXArray.ones([H])))
            w.append(("\(p).norm3.bias",   MLXArray.zeros([H])))
            w.append(("\(p).ff.gate_up.weight", MLXRandom.normal([2 * ffDim, H])))
            w.append(("\(p).ff.out.weight",     MLXRandom.normal([H, ffDim])))
        }

        // Final norm + output conv
        w.append(("transformer.norm.weight",  MLXArray.ones([H])))
        w.append(("transformer.norm.bias",    MLXArray.zeros([H])))
        w.append(("postprocess_conv.weight",  MLXRandom.normal([L, 1, H])))

        return w
    }
}
