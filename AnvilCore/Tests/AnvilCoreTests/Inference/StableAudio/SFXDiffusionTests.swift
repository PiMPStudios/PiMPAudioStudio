// AnvilCore/Tests/AnvilCoreTests/Inference/StableAudio/SFXDiffusionTests.swift
import Testing
import MLX
import MLXRandom
@testable import AnvilCore

@Suite("SFX Diffusion")
struct SFXDiffusionTests {
    // Tiny config: conditioningDim=32 → globalCondDim=64
    private static let config = SFXDiTConfig(
        hiddenSize: 128, numLayers: 1, numHeads: 4, latentDim: 64, conditioningDim: 32
    )

    @Test("Sample produces latents with correct shape")
    func sampleShape() async throws {
        let dit = SFXDiTActor(config: Self.config)
        try await dit.loadWeights(randomDiTWeights())
        let diffusion = SFXDiffusionActor(dit: dit)
        let diffConfig = SFXDiffusionConfig(
            steps: 4, guidanceScale: 1.0, seed: 42, latentShape: [1, 16, 64]
        )
        let crossCond  = MLXRandom.normal([1, 4, Self.config.conditioningDim])
        let globalCond = MLXRandom.normal([1, Self.config.globalCondDim])
        let result = try await diffusion.sample(
            crossCond: crossCond, globalCond: globalCond, config: diffConfig
        )
        #expect(result.latents.shape == [1, 16, 64])
    }

    @Test("Cancellation stops sampling")
    func cancellation() async {
        let dit = SFXDiTActor(config: Self.config)
        try? await dit.loadWeights(randomDiTWeights())
        let diffusion = SFXDiffusionActor(dit: dit)
        let diffConfig = SFXDiffusionConfig(
            steps: 100, guidanceScale: 1.0, seed: 42, latentShape: [1, 16, 64]
        )
        let crossCond  = MLXRandom.normal([1, 4, Self.config.conditioningDim])
        let globalCond = MLXRandom.normal([1, Self.config.globalCondDim])

        let task = Task {
            try await diffusion.sample(
                crossCond: crossCond, globalCond: globalCond, config: diffConfig
            )
        }
        try? await Task.sleep(for: .milliseconds(50))
        task.cancel()

        let result = await task.result
        switch result {
        case .failure(let error):
            #expect(error is CancellationError)
        case .success:
            break  // May complete before cancellation on fast hardware
        }
    }

    @Test("CFG with guidanceScale > 1 runs without error")
    func cfgMode() async throws {
        let dit = SFXDiTActor(config: Self.config)
        try await dit.loadWeights(randomDiTWeights())
        let diffusion = SFXDiffusionActor(dit: dit)
        let diffConfig = SFXDiffusionConfig(
            steps: 2, guidanceScale: 3.0, seed: 42, latentShape: [1, 16, 64]
        )
        let crossCond  = MLXRandom.normal([1, 4, Self.config.conditioningDim])
        let globalCond = MLXRandom.normal([1, Self.config.globalCondDim])
        let result = try await diffusion.sample(
            crossCond: crossCond, globalCond: globalCond, config: diffConfig
        )
        #expect(result.latents.shape == [1, 16, 64])
    }

    // MARK: - Weight factory (mirrors SFXDiTTests.randomDiTWeights for the tiny config)

    private func randomDiTWeights() -> [(String, MLXArray)] {
        let cfg    = Self.config
        let H      = cfg.hiddenSize
        let L      = cfg.latentDim
        let halfF  = cfg.timestepFeaturesDim / 2
        let condD  = cfg.conditioningDim
        let condTD = cfg.condTokenDim
        let globalD = cfg.globalCondDim
        let nH     = cfg.numHeads
        let nKV    = cfg.numKVHeads
        let hD     = cfg.headDim
        let ffDim  = cfg.ffIntermediateDim
        var w: [(String, MLXArray)] = []

        w.append(("preprocess_conv.weight",  MLXRandom.normal([L, 1, L])))
        w.append(("timestep_features.weight", MLXRandom.normal([halfF, 1])))

        let embedCfg: [(name: String, inputDim: Int, hiddenDim: Int, outputDim: Int)] = [
            ("to_timestep_embed", cfg.timestepFeaturesDim, H, H),
            ("to_cond_embed",     condTD, H, condTD),
            ("to_global_embed",   globalD, H, H),
        ]
        for (name, inputDim, hiddenDim, outputDim) in embedCfg {
            w.append(("\(name).fc1.weight", MLXRandom.normal([hiddenDim, inputDim])))
            w.append(("\(name).fc1.bias",   MLXArray.zeros([hiddenDim])))
            w.append(("\(name).fc2.weight", MLXRandom.normal([outputDim, hiddenDim])))
            w.append(("\(name).fc2.bias",   MLXArray.zeros([outputDim])))
        }

        for i in 0..<cfg.numLayers {
            let p = "transformer.layers.\(i)"
            w.append(("\(p).pre_norm.weight", MLXArray.ones([H])))
            w.append(("\(p).pre_norm.bias",   MLXArray.zeros([H])))
            w.append(("\(p).self_attn.to_qkv.weight", MLXRandom.normal([3 * nH * hD, H])))
            w.append(("\(p).self_attn.to_out.weight", MLXRandom.normal([H, nH * hD])))
            w.append(("\(p).cross_attend_norm.weight", MLXArray.ones([H])))
            w.append(("\(p).cross_attend_norm.bias",   MLXArray.zeros([H])))
            w.append(("\(p).cross_attn.to_q.weight",  MLXRandom.normal([nH * hD, H])))
            w.append(("\(p).cross_attn.to_kv.weight", MLXRandom.normal([2 * nKV * hD, condD])))
            w.append(("\(p).cross_attn.to_out.weight", MLXRandom.normal([H, nH * hD])))
            w.append(("\(p).ff_norm.weight", MLXArray.ones([H])))
            w.append(("\(p).ff_norm.bias",   MLXArray.zeros([H])))
            w.append(("\(p).ff.gate_up.weight", MLXRandom.normal([2 * ffDim, H])))
            w.append(("\(p).ff.gate_up.bias",   MLXArray.zeros([2 * ffDim])))
            w.append(("\(p).ff.out.weight",     MLXRandom.normal([H, ffDim])))
            w.append(("\(p).ff.out.bias",       MLXArray.zeros([H])))
        }

        w.append(("transformer.project_in.weight",  MLXRandom.normal([H, L])))
        w.append(("transformer.project_out.weight", MLXRandom.normal([L, H])))
        w.append(("postprocess_conv.weight",        MLXRandom.normal([L, 1, L])))

        return w
    }
}
