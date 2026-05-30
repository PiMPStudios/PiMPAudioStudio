// PiMPAudioCore/Tests/PiMPAudioCoreTests/Inference/StableAudio/SoundFXPipelineTests.swift
//
// Tests use SoundFXPipeline.generate(precomputedText:duration:config:) to
// bypass T5 tokenisation and the file-backed PiMPAudioTokenizer, making all
// tests runnable without downloading model weights.

import Testing
import MLX
import MLXRandom
@testable import PiMPAudioCore

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
            latentDim: 64, outputChannels: 2, sampleRate: 44100, upsampleRatios: [8, 8, 4, 4, 2]
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

    /// Random weights for the test VAE config.
    ///
    /// Generates keys matching OobleckDecoder's module layout:
    ///   channels=128, channelMultiples=[1,2,4,8,16]
    ///   cm=[1,1,2,4,8,16] (prepend 1), so cm.last!=16
    ///   decoder.conv1: Conv1d(latentDim → 128*16=2048, k=7)
    ///   decoder.blocks.N: OobleckDecoderBlock with snake1, convT1, 3×resUnit
    ///   decoder.snake1: Snake1d(128)
    ///   decoder.conv2: Conv1d(128 → audioChannels, k=7, no bias)
    private func randomVAEWeights(config: SFXVAEConfig) -> [(String, MLXArray)] {
        var w: [(String, MLXArray)] = []
        let channels = 128
        let cm = [1, 1, 2, 4, 8, 16]
        let latentDim = config.latentDim
        let audioChannels = config.outputChannels
        let ratios = config.upsampleRatios

        w.append(("decoder.conv1.weight", MLXRandom.normal([channels * cm.last!, 7, latentDim])))
        w.append(("decoder.conv1.bias",   MLXArray.zeros([channels * cm.last!])))

        func appendResUnit(prefix: String, dim: Int) {
            w.append(("\(prefix).snake1.alpha", MLXArray.zeros([dim])))
            w.append(("\(prefix).snake1.beta",  MLXArray.zeros([dim])))
            w.append(("\(prefix).conv1.weight",  MLXRandom.normal([dim, 7, dim])))
            w.append(("\(prefix).conv1.bias",    MLXArray.zeros([dim])))
            w.append(("\(prefix).snake2.alpha",  MLXArray.zeros([dim])))
            w.append(("\(prefix).snake2.beta",   MLXArray.zeros([dim])))
            w.append(("\(prefix).conv2.weight",  MLXRandom.normal([dim, 1, dim])))
            w.append(("\(prefix).conv2.bias",    MLXArray.zeros([dim])))
        }

        for (i, stride) in ratios.enumerated() {
            let idx = ratios.count - i
            let inDim = channels * cm[idx]
            let outDim = channels * cm[idx - 1]
            let p = "decoder.blocks.\(i)"
            w.append(("\(p).snake1.alpha", MLXArray.zeros([inDim])))
            w.append(("\(p).snake1.beta",  MLXArray.zeros([inDim])))
            w.append(("\(p).convT1.weight", MLXRandom.normal([outDim, 2 * stride, inDim])))
            w.append(("\(p).convT1.bias",   MLXArray.zeros([outDim])))
            appendResUnit(prefix: "\(p).resUnit1", dim: outDim)
            appendResUnit(prefix: "\(p).resUnit2", dim: outDim)
            appendResUnit(prefix: "\(p).resUnit3", dim: outDim)
        }

        w.append(("decoder.snake1.alpha", MLXArray.zeros([channels])))
        w.append(("decoder.snake1.beta",  MLXArray.zeros([channels])))
        w.append(("decoder.conv2.weight", MLXRandom.normal([audioChannels, 7, channels])))

        return w
    }
}
