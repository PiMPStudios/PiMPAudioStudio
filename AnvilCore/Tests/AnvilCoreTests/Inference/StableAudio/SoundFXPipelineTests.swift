// AnvilCore/Tests/AnvilCoreTests/Inference/StableAudio/SoundFXPipelineTests.swift
//
// Tests use SoundFXPipeline.generate(precomputedText:duration:config:) to
// bypass T5 tokenisation and the file-backed AnvilTokenizer, making all
// tests runnable without downloading model weights.

import Testing
import MLX
import MLXRandom
@testable import AnvilCore

@Suite("SoundFXPipeline")
struct SoundFXPipelineTests {
    // Tiny dims: conditioningDim=32, globalCondDim=64.
    // headDim=32, numKVHeads=2, ffIntermediateDim=512, rotaryDim=16.
    private static let ditConfig = SFXDiTConfig(
        hiddenSize: 128, numLayers: 1, numHeads: 4, latentDim: 64, conditioningDim: 32
    )

    @Test("Generate produces AudioResult with correct sample rate")
    func generateProducesResult() async throws {
        let pipeline = try await makeTestPipeline()
        let textHidden = MLXRandom.normal([1, 4, Self.ditConfig.conditioningDim])
        let config = SFXDiffusionConfig(
            steps: 2, guidanceScale: 1.0, seed: 42, latentShape: [1, 16, 64]
        )
        let result = try await pipeline.generate(
            precomputedText: textHidden, duration: 5.0, config: config
        )
        #expect(result.sampleRate == 44100)
        #expect(result.samples.shape[0] == 1)
        #expect(result.duration == 5.0)
    }

    @Test("Duration clamped to 47 seconds max")
    func durationClamped() async throws {
        let pipeline = try await makeTestPipeline()
        let textHidden = MLXRandom.normal([1, 4, Self.ditConfig.conditioningDim])
        let config = SFXDiffusionConfig(
            steps: 2, guidanceScale: 1.0, seed: 42, latentShape: [1, 16, 64]
        )
        let result = try await pipeline.generate(
            precomputedText: textHidden, duration: 60.0, config: config
        )
        #expect(result.duration == 47.0)
    }

    @Test("Progress stream emits events")
    func progressStream() async throws {
        let pipeline = try await makeTestPipeline()
        let progressStream = await pipeline.makeProgressStream()
        let textHidden = MLXRandom.normal([1, 4, Self.ditConfig.conditioningDim])
        let config = SFXDiffusionConfig(
            steps: 2, guidanceScale: 1.0, seed: 42, latentShape: [1, 16, 64]
        )

        let progressTask = Task {
            var count = 0
            for await _ in progressStream {
                count += 1
                if count >= 2 { break }
            }
            return count
        }

        _ = try await pipeline.generate(
            precomputedText: textHidden, duration: 5.0, config: config
        )

        let count = await progressTask.value
        #expect(count >= 1)
    }

    // MARK: - Factory

    private func makeTestPipeline() async throws -> SoundFXPipeline {
        let cfg = Self.ditConfig
        let t5  = T5Encoder(config: T5Config(
            vocabSize: 100, dModel: cfg.conditioningDim, dKV: 8,
            dFF: 64, numLayers: 1, numHeads: 4
        ))
        // Use a conditioner whose outputDim matches conditioningDim so the
        // SAONumberConditioner → globalCond dimension equals globalCondDim.
        let conditioner = SAONumberConditioner(numFrequencies: 4, outputDim: cfg.conditioningDim)
        let dit         = SFXDiTActor(config: cfg)
        try await dit.loadWeights(randomDiTWeights())
        let diffusion   = SFXDiffusionActor(dit: dit)
        let vaeConfig   = SFXVAEConfig(
            latentDim: 64, outputChannels: 2, sampleRate: 44100, upsampleRatios: [2, 2]
        )
        let vae = SFXVAEActor(config: vaeConfig)
        await vae.loadWeights(randomVAEWeights(config: vaeConfig))
        return SoundFXPipeline(
            t5: t5,
            conditioner: conditioner,
            diffusion: diffusion,
            vae: vae
        )
    }

    // MARK: - Weight factories

    private func randomDiTWeights() -> [(String, MLXArray)] {
        let cfg    = Self.ditConfig
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

        w.append(("preprocess_conv.weight",  MLXRandom.normal([H, 1, L])))
        w.append(("timestep_features.weight", MLXRandom.normal([halfF, 1])))

        let embedCfg: [(String, Int)] = [
            ("to_timestep_embed", cfg.timestepFeaturesDim),
            ("to_cond_embed",     condTD),
            ("to_global_embed",   globalD),
        ]
        for (name, inDim) in embedCfg {
            w.append(("\(name).fc1.weight", MLXRandom.normal([H, inDim])))
            w.append(("\(name).fc1.bias",   MLXArray.zeros([H])))
            w.append(("\(name).fc2.weight", MLXRandom.normal([H, H])))
            w.append(("\(name).fc2.bias",   MLXArray.zeros([H])))
        }

        for i in 0..<cfg.numLayers {
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

        w.append(("transformer.norm.weight",  MLXArray.ones([H])))
        w.append(("transformer.norm.bias",    MLXArray.zeros([H])))
        w.append(("postprocess_conv.weight",  MLXRandom.normal([L, 1, H])))

        return w
    }

    /// Random weights for the test VAE config (upsampleRatios: [2, 2]).
    private func randomVAEWeights(config: SFXVAEConfig) -> [(String, MLXArray)] {
        var w: [(String, MLXArray)] = []
        let L = config.latentDim          // 64
        let C = config.outputChannels     // 2

        // Decoder: projIn Linear(L, L)
        w.append(("projIn.weight", MLXRandom.normal([L, L])))
        w.append(("projIn.bias",   MLXArray.zeros([L])))

        // upsampleBlocks: inputDim halves each step; SFXUpsampleBlock.linear
        // has out = outputDim * ratio. For upsampleRatios=[2,2]:
        //   block 0: in=64, out=max(32,2)=32, ratio=2 → Linear(64, 64)
        //   block 1: in=32, out=max(16,2)=16, ratio=2 → Linear(32, 32)
        var curDim = L
        for (i, ratio) in config.upsampleRatios.enumerated() {
            let outDim = max(curDim / 2, C)
            w.append(("upsampleBlocks.\(i).linear.weight",
                      MLXRandom.normal([outDim * ratio, curDim])))
            w.append(("upsampleBlocks.\(i).linear.bias",
                      MLXArray.zeros([outDim * ratio])))
            curDim = outDim
        }

        // projOut: Linear(curDim, C)
        w.append(("projOut.weight", MLXRandom.normal([C, curDim])))
        w.append(("projOut.bias",   MLXArray.zeros([C])))

        // Encoder: encoderProjIn Linear(C, C)
        w.append(("encoderProjIn.weight", MLXRandom.normal([C, C])))
        w.append(("encoderProjIn.bias",   MLXArray.zeros([C])))

        // downsampleBlocks: reversed ratios. SFXDownsampleBlock.linear
        // has in = inputDim * ratio, out = outputDim.
        //   block 0: in=2, out=min(4,64)=4, ratio=2 → Linear(4, 4)
        //   block 1: in=4, out=min(8,64)=8, ratio=2 → Linear(8, 8)
        var encDim = C
        for (i, ratio) in config.upsampleRatios.reversed().enumerated() {
            let outDim = min(encDim * 2, L)
            w.append(("downsampleBlocks.\(i).linear.weight",
                      MLXRandom.normal([outDim, encDim * ratio])))
            w.append(("downsampleBlocks.\(i).linear.bias",
                      MLXArray.zeros([outDim])))
            encDim = outDim
        }

        // encoderProjOut: Linear(encDim, L*2)
        w.append(("encoderProjOut.weight", MLXRandom.normal([L * 2, encDim])))
        w.append(("encoderProjOut.bias",   MLXArray.zeros([L * 2])))

        return w
    }
}
