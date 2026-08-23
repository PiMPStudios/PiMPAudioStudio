import Testing
import MLX
@testable import PiMPAudioCore

@Suite("Diffusion Actor")
struct DiffusionActorTests {
    private func makeSmallDiT() -> DiTActor {
        let decoder = DiTDecoder(
            hiddenSize: 256,
            encoderHiddenSize: 256,
            intermediateSize: 512,
            numHiddenLayers: 2,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            headDim: 64,
            rmsNormEps: 1e-6,
            attentionBias: false,
            inChannels: 192,
            audioAcousticHiddenDim: 64,
            patchSize: 2,
            slidingWindow: 128,
            layerTypes: ["sliding_attention", "full_attention"],
            ropeTheta: 1_000_000.0,
            maxPositionEmbeddings: 32768
        )
        return DiTActor(decoder: decoder)
    }

    @Test("Generate returns latents with correct shape")
    func generateShape() async throws {
        let dit = makeSmallDiT()
        let actor = DiffusionActor(dit: dit)
        let encoderStates = MLXArray.zeros([1, 50, 256])
        let contextLatents = MLXArray.zeros([1, 20, 128])
        let config = DiffusionConfig(seed: 42, steps: 4)

        let result = try await actor.generate(
            encoderStates: encoderStates,
            contextLatents: contextLatents,
            encoderMask: nil,
            shape: (batch: 1, frames: 20, channels: 64),
            config: config
        )
        #expect(result.latents.shape == [1, 20, 64])
        #expect(result.timeCosts.totalTime > 0)
    }

    @Test("Deterministic with same seed")
    func deterministic() async throws {
        let dit = makeSmallDiT()
        let actor = DiffusionActor(dit: dit)
        let encoderStates = MLXArray.zeros([1, 20, 256])
        let contextLatents = MLXArray.zeros([1, 10, 128])
        let config = DiffusionConfig(seed: 123, steps: 2)

        let r1 = try await actor.generate(
            encoderStates: encoderStates, contextLatents: contextLatents,
            encoderMask: nil,
            shape: (batch: 1, frames: 10, channels: 64), config: config
        )
        let r2 = try await actor.generate(
            encoderStates: encoderStates, contextLatents: contextLatents,
            encoderMask: nil,
            shape: (batch: 1, frames: 10, channels: 64), config: config
        )
        eval(r1.latents, r2.latents)
        let diff = abs(r1.latents - r2.latents).sum()
        eval(diff)
        #expect(diff.item(Float.self) < 1e-4)
    }

    @Test("SDE method produces different results each run without seed")
    func sdeVariation() async throws {
        let dit = makeSmallDiT()
        let actor = DiffusionActor(dit: dit)
        let encoderStates = MLXArray.zeros([1, 20, 256])
        let contextLatents = MLXArray.zeros([1, 10, 128])
        let config = DiffusionConfig(seed: nil, method: .sde, steps: 2)

        let r1 = try await actor.generate(
            encoderStates: encoderStates, contextLatents: contextLatents,
            encoderMask: nil,
            shape: (batch: 1, frames: 10, channels: 64), config: config
        )
        let r2 = try await actor.generate(
            encoderStates: encoderStates, contextLatents: contextLatents,
            encoderMask: nil,
            shape: (batch: 1, frames: 10, channels: 64), config: config
        )
        eval(r1.latents, r2.latents)
        let diff = abs(r1.latents - r2.latents).sum()
        eval(diff)
        #expect(diff.item(Float.self) > 0.01)
    }

    @Test("Generate latents stay finite")
    func generateFiniteAbsMean() async throws {
        let dit = makeSmallDiT()
        let actor = DiffusionActor(dit: dit)
        let encoderStates = MLXArray.zeros([1, 20, 256])
        let contextLatents = MLXArray.zeros([1, 10, 128])
        let config = DiffusionConfig(seed: 7, steps: 2, guidanceScale: 3.0)

        let result = try await actor.generate(
            encoderStates: encoderStates,
            contextLatents: contextLatents,
            encoderMask: nil,
            shape: (batch: 1, frames: 10, channels: 64),
            config: config
        )
        eval(result.latents)
        let absMean = abs(result.latents).mean().item(Float.self)
        #expect(result.latents.shape == [1, 10, 64])
        #expect(absMean.isFinite)
        #expect(absMean > 0)
    }

    @Test("Inpaint returns latents with the start shape")
    func inpaintShape() async throws {
        let dit = makeSmallDiT()
        let actor = DiffusionActor(dit: dit)
        let start = MLXArray.zeros([1, 10, 64])
        let encoderStates = MLXArray.zeros([1, 20, 256])
        let contextLatents = MLXArray.zeros([1, 10, 128])
        let schedule = TimestepSchedule.compute(shift: 3.0, timesteps: nil, inferSteps: 2)
        let config = DiffusionConfig(seed: 11, steps: 2, guidanceScale: 3.0)

        let result = try await actor.inpaint(
            startLatents: start,
            schedule: schedule,
            encoderStates: encoderStates,
            contextLatents: contextLatents,
            encoderMask: nil,
            config: config
        )
        eval(result.latents)
        #expect(result.latents.shape == [1, 10, 64])
        #expect(abs(result.latents).mean().item(Float.self).isFinite)
    }

    @Test("Empty inpaint schedule returns the start latents")
    func inpaintEmptySchedule() async throws {
        let dit = makeSmallDiT()
        let actor = DiffusionActor(dit: dit)
        let start = MLXArray.ones([1, 8, 64])
        let encoderStates = MLXArray.zeros([1, 10, 256])
        let contextLatents = MLXArray.zeros([1, 8, 128])

        let result = try await actor.inpaint(
            startLatents: start,
            schedule: [],
            encoderStates: encoderStates,
            contextLatents: contextLatents,
            encoderMask: nil,
            config: DiffusionConfig(seed: 1, steps: 2)
        )
        eval(result.latents, start)
        #expect(abs(result.latents - start).sum().item(Float.self) == 0)
    }
}
