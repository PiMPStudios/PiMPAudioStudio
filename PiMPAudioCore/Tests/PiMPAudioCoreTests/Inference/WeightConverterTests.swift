import Testing
import MLX
@testable import PiMPAudioCore

@Suite("Weight Converter")
struct WeightConverterTests {
    @Test("VAE weight_norm fusion produces correct shape")
    func vaeWeightNormFusion() {
        let g = MLXArray.ones([16, 1, 1])
        let v = MLXArray.ones([16, 8, 7])
        let fused = VAENameMap.fuseWeightNorm(weightG: g, weightV: v)
        #expect(fused.shape == [16, 8, 7])
    }

    @Test("VAE Conv1d axis swap: [out, in, K] -> [out, K, in]")
    func vaeConv1dSwap() {
        var data: [Float] = []
        for o in 0..<2 { for i in 0..<3 { for k in 0..<5 {
            data.append(Float(o * 100 + i * 10 + k))
        }}}
        let pt = MLXArray(data, [2, 3, 5])
        let mlx = VAENameMap.conv1dToMLX(pt)
        #expect(mlx.shape == [2, 5, 3])
        eval(mlx)
    }

    @Test("VAE ConvTranspose1d: [in, out, K] -> [out, K, in]")
    func vaeConvTranspose1dSwap() {
        let pt = MLXArray.zeros([3, 2, 5])
        let mlx = VAENameMap.convTranspose1dToMLX(pt)
        #expect(mlx.shape == [2, 5, 3])
    }

    @Test("VAE Snake1d squeeze: [1, C, 1] -> [C]")
    func vaeSnakeSqueeze() {
        let pt = MLXArray.ones([1, 32, 1])
        let mlx = VAENameMap.squeezeSnakeParam(pt)
        #expect(mlx.shape == [32])
    }

    @Test("DiT proj_in key remapping")
    func ditProjInRemap() {
        // Keys require "decoder." prefix; Sequential ".1." wrapper is stripped.
        let key = "decoder.proj_in.1.weight"
        let mapped = TensorNameMap.mapDiTKey(key)
        #expect(mapped == "projIn.weight")
    }

    @Test("DiT proj_out key remapping")
    func ditProjOutRemap() {
        let key = "decoder.proj_out.1.weight"
        let mapped = TensorNameMap.mapDiTKey(key)
        #expect(mapped == "projOut.weight")
    }

    @Test("DiT rotary_emb keys are skipped")
    func ditRotarySkipped() {
        let key = "decoder.layers.0.rotary_emb.inv_freq"
        let mapped = TensorNameMap.mapDiTKey(key)
        #expect(mapped == nil)
    }

    @Test("DiT normal keys pass through")
    func ditPassthrough() {
        // "decoder." prefix stripped; snake_case components camelCased.
        let key = "decoder.layers.0.self_attn.q_proj.weight"
        let mapped = TensorNameMap.mapDiTKey(key)
        #expect(mapped == "layers.0.selfAttn.qProj.weight")
    }

    @Test("WeightConverter converts VAE state dict")
    func convertVAEStateDict() async throws {
        let stateDict: [(String, MLXArray)] = [
            ("encoder.conv1.weight_g", MLXArray.ones([128, 1, 1])),
            ("encoder.conv1.weight_v", MLXArray.ones([128, 2, 7])),
            ("encoder.conv1.bias", MLXArray.zeros([128])),
            ("encoder.block.0.res_unit1.snake1.alpha", MLXArray.zeros([1, 128, 1])),
            ("encoder.block.0.res_unit1.snake1.beta", MLXArray.zeros([1, 128, 1])),
        ]
        let converter = WeightConverter()
        let result = try await converter.convertVAEWeights(stateDict: stateDict)

        let keys = result.map { $0.0 }
        #expect(keys.contains("encoder.conv1.weight"))
        #expect(keys.contains("encoder.conv1.bias"))
        #expect(keys.contains("encoder.blocks.0.resUnit1.snake1.alpha"))
        #expect(!keys.contains("encoder.conv1.weight_g"))
        #expect(!keys.contains("encoder.conv1.weight_v"))

        let fusedWeight = try #require(result.first(where: { $0.0 == "encoder.conv1.weight" })?.1)
        #expect(fusedWeight.shape == [128, 7, 2])

        let alpha = try #require(result.first(where: { $0.0 == "encoder.blocks.0.resUnit1.snake1.alpha" })?.1)
        #expect(alpha.shape == [128])
    }

    @Test("WeightConverter converts DiT state dict")
    func convertDiTStateDict() async throws {
        let stateDict: [(String, MLXArray)] = [
            ("decoder.proj_in.1.weight", MLXArray.ones([2048, 192, 2])),
            ("decoder.proj_in.1.bias", MLXArray.zeros([2048])),
            ("decoder.proj_out.1.weight", MLXArray.ones([2048, 64, 2])),
            ("decoder.proj_out.1.bias", MLXArray.zeros([64])),
            ("decoder.layers.0.self_attn.q_proj.weight", MLXArray.ones([2048, 2048])),
            ("decoder.rotary_emb.inv_freq", MLXArray.ones([64])),
        ]
        let converter = WeightConverter()
        let result = try await converter.convertDiTWeights(stateDict: stateDict)

        let keys = result.map { $0.0 }
        #expect(keys.contains("projIn.weight"))
        #expect(keys.contains("projOut.weight"))
        #expect(keys.contains("layers.0.selfAttn.qProj.weight"))
        #expect(!keys.contains("rotaryEmb.invFreq"))

        let projIn = try #require(result.first(where: { $0.0 == "projIn.weight" })?.1)
        #expect(projIn.shape == [2048, 2, 192])

        let projOut = try #require(result.first(where: { $0.0 == "projOut.weight" })?.1)
        #expect(projOut.shape == [64, 2, 2048])
    }
}
