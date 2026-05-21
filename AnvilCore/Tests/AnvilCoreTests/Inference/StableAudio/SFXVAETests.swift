// AnvilCore/Tests/AnvilCoreTests/Inference/StableAudio/SFXVAETests.swift
import Testing
import MLX
import MLXRandom
@testable import AnvilCore

@Suite("SFX VAE")
struct SFXVAETests {
    let config = SFXVAEConfig(latentDim: 64, outputChannels: 2, sampleRate: 44100, upsampleRatios: [8, 8, 4, 4, 2])

    @Test("Decode produces audio from latents")
    func decodeShape() {
        let model = SFXVAEModel(config: config)
        let latents = MLXRandom.normal([1, 16, 64])
        let audio = model.decode(latents: latents)
        #expect(audio.shape[0] == 1)
        #expect(audio.shape[2] == 2) // stereo
        #expect(audio.shape[1] == 16 * 8 * 8 * 4 * 4 * 2) // upsampled
    }

    @Test("Actor decode returns correct shape after loading weights")
    func actorDecode() async throws {
        let actor = SFXVAEActor(config: config)
        await actor.loadWeights(randomVAEWeights())
        let latents = MLXRandom.normal([1, 16, 64])
        let audio = try await actor.decode(latents: latents)
        #expect(audio.shape[0] == 1)
        #expect(audio.shape[2] == 2)
    }

    @Test("Actor loadWeights marks as loaded")
    func actorLoad() async {
        let actor = SFXVAEActor(config: config)
        let loaded = await actor.isLoaded
        #expect(!loaded)
    }

    @Test("Decode throws when weights not loaded")
    func decodeThrowsWithoutWeights() async {
        let actor = SFXVAEActor(config: config)
        let latents = MLXRandom.normal([1, 16, 64])
        await #expect(throws: SFXVAEActorError.weightsNotLoaded) {
            try await actor.decode(latents: latents)
        }
    }

    // MARK: - Helpers

    /// Produces random-shaped weights matching OobleckDecoder's parameter structure.
    ///
    /// Config: channels=128, channelMultiples=[1,2,4,8,16], latentDim=64,
    /// audioChannels=2, upsamplingRatios=[8,8,4,2].
    ///
    /// cm = [1,1,2,4,8,16] (prepend 1 to channelMultiples).
    ///   decoder.conv1: Conv1d(in=64, out=128*16=2048, k=7)
    ///   decoder.blocks.N: OobleckDecoderBlock — see below
    ///   decoder.snake1: Snake1d(hiddenDim=128)
    ///   decoder.conv2: Conv1d(in=128, out=2, k=7, no bias)
    private func randomVAEWeights() -> [(String, MLXArray)] {
        var w: [(String, MLXArray)] = []
        let channels = 128
        let cm = [1, 1, 2, 4, 8, 16]   // [1] + channelMultiples
        let ratios = config.upsampleRatios  // [8,8,4,2]

        // decoder.conv1: Conv1d([B,T,64] → [B,T,2048])
        // MLX Conv1d weight layout: [outC, kernelSize, inC]
        w.append(("decoder.conv1.weight", MLXRandom.normal([channels * cm.last!, 7, config.latentDim])))
        w.append(("decoder.conv1.bias", MLXArray.zeros([channels * cm.last!])))

        // decoder.blocks.N: OobleckDecoderBlock(inputDim, outputDim, stride)
        // Decoder iterates i in 0..<ratios.count, idx = ratios.count - i
        //   inputDim = channels * cm[idx], outputDim = channels * cm[idx-1]
        func appendResUnit(prefix: String, dim: Int) {
            // snake1
            w.append(("\(prefix).snake1.alpha", MLXArray.zeros([dim])))
            w.append(("\(prefix).snake1.beta", MLXArray.zeros([dim])))
            // conv1: Conv1d(dim, dim, k=7) — [outC, k, inC]
            w.append(("\(prefix).conv1.weight", MLXRandom.normal([dim, 7, dim])))
            w.append(("\(prefix).conv1.bias", MLXArray.zeros([dim])))
            // snake2
            w.append(("\(prefix).snake2.alpha", MLXArray.zeros([dim])))
            w.append(("\(prefix).snake2.beta", MLXArray.zeros([dim])))
            // conv2: Conv1d(dim, dim, k=1) — [outC, k, inC]
            w.append(("\(prefix).conv2.weight", MLXRandom.normal([dim, 1, dim])))
            w.append(("\(prefix).conv2.bias", MLXArray.zeros([dim])))
        }

        for (i, stride) in ratios.enumerated() {
            let idx = ratios.count - i
            let inDim = channels * cm[idx]
            let outDim = channels * cm[idx - 1]
            let p = "decoder.blocks.\(i)"
            // snake1
            w.append(("\(p).snake1.alpha", MLXArray.zeros([inDim])))
            w.append(("\(p).snake1.beta", MLXArray.zeros([inDim])))
            // convT1: ConvTransposed1d(inDim, outDim, k=2*stride) — [outC, k, inC] in MLX
            let kernelSize = 2 * stride
            w.append(("\(p).convT1.weight", MLXRandom.normal([outDim, kernelSize, inDim])))
            w.append(("\(p).convT1.bias", MLXArray.zeros([outDim])))
            // 3 residual units operating on outDim
            appendResUnit(prefix: "\(p).resUnit1", dim: outDim)
            appendResUnit(prefix: "\(p).resUnit2", dim: outDim)
            appendResUnit(prefix: "\(p).resUnit3", dim: outDim)
        }

        // decoder.snake1: Snake1d(hiddenDim=128)
        w.append(("decoder.snake1.alpha", MLXArray.zeros([channels])))
        w.append(("decoder.snake1.beta", MLXArray.zeros([channels])))

        // decoder.conv2: Conv1d(128, 2, k=7, no bias) — [outC, k, inC]
        w.append(("decoder.conv2.weight", MLXRandom.normal([config.outputChannels, 7, channels])))

        return w
    }
}
