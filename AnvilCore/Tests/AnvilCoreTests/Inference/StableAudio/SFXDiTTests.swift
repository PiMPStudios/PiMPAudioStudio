// AnvilCore/Tests/AnvilCoreTests/Inference/StableAudio/SFXDiTTests.swift
import Testing
import MLX
import MLXRandom
@testable import AnvilCore

@Suite("SFX DiT")
struct SFXDiTTests {
    let config = SFXDiTConfig(
        hiddenSize: 256,
        numLayers: 2,
        numHeads: 4,
        latentDim: 64,
        conditioningDim: 512
    )

    @Test("Forward pass produces correct output shape")
    func forwardShape() {
        let model = SFXDiTModel(config: config)
        let latents = MLXRandom.normal([1, 16, 64])
        let timestep = MLXArray([0.5])
        let conditioning = MLXRandom.normal([1, 512])
        let output = model.forward(latents: latents, timestep: timestep, conditioning: conditioning)
        #expect(output.shape == [1, 16, 64])
    }

    @Test("Actor predict returns correct shape after loading weights")
    func actorPredict() async throws {
        let actor = SFXDiTActor(config: config)
        try await actor.loadWeights(randomDiTWeights())
        let latents = MLXRandom.normal([1, 16, 64])
        let timestep = MLXArray([0.5])
        let conditioning = MLXRandom.normal([1, 512])
        let output = try await actor.predict(latents: latents, timestep: timestep, conditioning: conditioning)
        #expect(output.shape == [1, 16, 64])
    }

    @Test("Predict throws when weights not loaded")
    func predictThrowsWithoutWeights() async {
        let actor = SFXDiTActor(config: config)
        let latents = MLXRandom.normal([1, 16, 64])
        let timestep = MLXArray([0.5])
        let conditioning = MLXRandom.normal([1, 512])
        await #expect(throws: SFXDiTActorError.weightsNotLoaded) {
            try await actor.predict(latents: latents, timestep: timestep, conditioning: conditioning)
        }
    }

    @Test("unloadWeights causes subsequent predict to throw")
    func unloadWeightsCausesThrow() async throws {
        let actor = SFXDiTActor(config: config)
        try await actor.loadWeights(randomDiTWeights())
        await actor.unloadWeights()
        let latents = MLXRandom.normal([1, 16, 64])
        let timestep = MLXArray([0.5])
        let conditioning = MLXRandom.normal([1, 512])
        await #expect(throws: SFXDiTActorError.weightsNotLoaded) {
            try await actor.predict(latents: latents, timestep: timestep, conditioning: conditioning)
        }
    }

    @Test("Actor loadWeights marks as loaded")
    func actorLoadWeights() async throws {
        let actor = SFXDiTActor(config: config)
        let loaded = await actor.isLoaded
        #expect(!loaded)
        try await actor.loadWeights(randomDiTWeights())
        let loadedAfter = await actor.isLoaded
        #expect(loadedAfter)
    }

    private func randomDiTWeights() -> [(String, MLXArray)] {
        var weights: [(String, MLXArray)] = []
        weights.append(("proj_in.weight", MLXRandom.normal([256, 64])))
        weights.append(("proj_in.bias", MLXArray.zeros([256])))
        weights.append(("time_embed.0.weight", MLXRandom.normal([256, 256])))
        weights.append(("time_embed.0.bias", MLXArray.zeros([256])))
        weights.append(("time_embed.2.weight", MLXRandom.normal([256, 256])))
        weights.append(("time_embed.2.bias", MLXArray.zeros([256])))
        weights.append(("cond_proj.weight", MLXRandom.normal([256, 512])))
        weights.append(("cond_proj.bias", MLXArray.zeros([256])))
        for i in 0..<2 {
            let prefix = "blocks.\(i)"
            weights.append(("\(prefix).norm1.weight", MLXArray.ones([256])))
            weights.append(("\(prefix).norm1.bias", MLXArray.zeros([256])))
            weights.append(("\(prefix).attn.qkv.weight", MLXRandom.normal([768, 256])))
            weights.append(("\(prefix).attn.out.weight", MLXRandom.normal([256, 256])))
            weights.append(("\(prefix).norm2.weight", MLXArray.ones([256])))
            weights.append(("\(prefix).norm2.bias", MLXArray.zeros([256])))
            weights.append(("\(prefix).cross_attn.q.weight", MLXRandom.normal([256, 256])))
            weights.append(("\(prefix).cross_attn.kv.weight", MLXRandom.normal([512, 256])))
            weights.append(("\(prefix).cross_attn.out.weight", MLXRandom.normal([256, 256])))
            weights.append(("\(prefix).norm3.weight", MLXArray.ones([256])))
            weights.append(("\(prefix).norm3.bias", MLXArray.zeros([256])))
            weights.append(("\(prefix).mlp.fc1.weight", MLXRandom.normal([1024, 256])))
            weights.append(("\(prefix).mlp.fc1.bias", MLXArray.zeros([1024])))
            weights.append(("\(prefix).mlp.fc2.weight", MLXRandom.normal([256, 1024])))
            weights.append(("\(prefix).mlp.fc2.bias", MLXArray.zeros([256])))
            weights.append(("\(prefix).adaLN.weight", MLXRandom.normal([768, 256])))
            weights.append(("\(prefix).adaLN.bias", MLXArray.zeros([768])))
        }
        weights.append(("proj_out.weight", MLXRandom.normal([64, 256])))
        weights.append(("proj_out.bias", MLXArray.zeros([64])))
        weights.append(("final_norm.weight", MLXArray.ones([256])))
        weights.append(("final_norm.bias", MLXArray.zeros([256])))
        return weights
    }
}
