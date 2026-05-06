// AnvilCore/Tests/AnvilCoreTests/Inference/StableAudio/CLAPEncoderTests.swift
import Testing
import MLX
@testable import AnvilCore

@Suite("CLAPEncoder")
struct CLAPEncoderTests {
    @Test("Initial state is not loaded")
    func initialState() async {
        let encoder = CLAPEncoder(embeddingDim: 512, numLayers: 4, numHeads: 8, vocabSize: 49408)
        let loaded = await encoder.isLoaded
        #expect(!loaded)
    }

    @Test("Encode without loading throws")
    func encodeWithoutLoad() async {
        let encoder = CLAPEncoder(embeddingDim: 512, numLayers: 4, numHeads: 8, vocabSize: 49408)
        do {
            _ = try await encoder.encode(text: "test", tokenizer: CLAPTokenizer.forTesting())
            Issue.record("Should have thrown")
        } catch {
            #expect(error is CLAPError)
        }
    }

    @Test("Encode after loading produces correct shape")
    func encodeShape() async throws {
        let encoder = CLAPEncoder(embeddingDim: 512, numLayers: 4, numHeads: 8, vocabSize: 49408)
        await encoder.loadWeights(randomWeights(for: encoder))
        let result = try await encoder.encode(text: "thunder", tokenizer: CLAPTokenizer.forTesting())
        #expect(result.shape == [1, 512])
    }

    private func randomWeights(for encoder: CLAPEncoder) -> [(String, MLXArray)] {
        var weights: [(String, MLXArray)] = []
        weights.append(("token_embedding.weight", MLXRandom.normal([49408, 512])))
        weights.append(("position_embedding.weight", MLXRandom.normal([77, 512])))
        for i in 0..<4 {
            weights.append(("layers.\(i).attention.query.weight", MLXRandom.normal([512, 512])))
            weights.append(("layers.\(i).attention.key.weight", MLXRandom.normal([512, 512])))
            weights.append(("layers.\(i).attention.value.weight", MLXRandom.normal([512, 512])))
            weights.append(("layers.\(i).attention.out.weight", MLXRandom.normal([512, 512])))
            weights.append(("layers.\(i).ln1.weight", MLXArray.ones([512])))
            weights.append(("layers.\(i).ln1.bias", MLXArray.zeros([512])))
            weights.append(("layers.\(i).ln2.weight", MLXArray.ones([512])))
            weights.append(("layers.\(i).ln2.bias", MLXArray.zeros([512])))
            weights.append(("layers.\(i).mlp.fc1.weight", MLXRandom.normal([2048, 512])))
            weights.append(("layers.\(i).mlp.fc1.bias", MLXArray.zeros([2048])))
            weights.append(("layers.\(i).mlp.fc2.weight", MLXRandom.normal([512, 2048])))
            weights.append(("layers.\(i).mlp.fc2.bias", MLXArray.zeros([512])))
        }
        weights.append(("final_ln.weight", MLXArray.ones([512])))
        weights.append(("final_ln.bias", MLXArray.zeros([512])))
        weights.append(("projection.weight", MLXRandom.normal([512, 512])))
        return weights
    }
}
