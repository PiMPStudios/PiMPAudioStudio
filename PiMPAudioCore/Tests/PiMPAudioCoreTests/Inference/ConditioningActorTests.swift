import Testing
import MLX
@testable import PiMPAudioCore

@Suite("Conditioning Actor")
struct ConditioningActorTests {
    @Test("ConditioningActor encode produces correct shapes (text-only)")
    func encodeTextOnly() async throws {
        let actor = try ConditioningActor.stub()
        let result = try await actor.encode(
            text: "A dreamy synthwave track",
            lyrics: "Floating through the neon lights",
            referenceAudio: nil,
            duration: 30.0,
            precision: .float32
        )
        #expect(result.encoderStates.shape[0] == 1)
        #expect(result.encoderStates.shape[2] == 2048)
        #expect(result.encoderMask.shape[0] == 1)
        #expect(result.encoderMask.shape[1] == result.encoderStates.shape[1])
        #expect(result.contextLatents.shape[0] == 1)
        #expect(result.contextLatents.shape[2] == 128)
    }

    @Test("ConditioningActor encode with reference audio")
    func encodeWithReference() async throws {
        let actor = try ConditioningActor.stub()
        let refAudio = MLXArray.zeros([1, 4096, 2])
        let result = try await actor.encode(
            text: "A pop song",
            lyrics: "Hey hey hey",
            referenceAudio: refAudio,
            duration: 10.0,
            precision: .float32
        )
        #expect(result.encoderStates.shape[0] == 1)
        #expect(result.encoderStates.shape[2] == 2048)
        #expect(result.contextLatents.shape[2] == 128)
    }

    @Test("Reference-audio context length matches requested duration, not timbreFixFrame")
    func referenceContextMatchesRequestedDuration() async throws {
        let actor = try ConditioningActor.stub()
        let refAudio = MLXArray.zeros([1, 4096, 2])
        let duration = 10.0
        let result = try await actor.encode(
            text: "A pop song",
            lyrics: "",
            referenceAudio: refAudio,
            duration: duration,
            precision: .float32
        )
        let expectedFrames = ConditioningActor.latentFrames(forDuration: duration)
        // Regression: contextLatents was previously built from the
        // 750-frame reference latents, breaking the channel concat in
        // DiTModel for any duration ≠ 30 s.
        #expect(expectedFrames != ConditioningActor.timbreFixFrame)
        #expect(result.contextLatents.shape[1] == expectedFrames)
    }

    @Test("ConditioningActor contextLatents uses silence source and full mask for text-only")
    func contextLatentsTextOnlyPath() async throws {
        let actor = try ConditioningActor.stub()
        let result = try await actor.encode(
            text: "Test",
            lyrics: "",
            referenceAudio: nil,
            duration: 5.0,
            precision: .float32
        )
        let sourceLatents = result.contextLatents[0..., 0..., 0..<64]
        let chunkMask = result.contextLatents[0..., 0..., 64..<128]
        let sourceSum = MLX.sum(MLX.abs(sourceLatents))
        let maskMin = chunkMask.min()
        let maskMax = chunkMask.max()
        eval(sourceSum, maskMin, maskMax)
        #expect(sourceSum.item(Float.self) == 0.0)
        #expect(maskMin.item(Float.self) == ConditioningActor.fullGenerationChunkMaskValue)
        #expect(maskMax.item(Float.self) == ConditioningActor.fullGenerationChunkMaskValue)
    }

    @Test("ConditioningActor cover override pads short source with silence latent")
    func contextLatentsCoverOverridePadsShortSourceWithSilenceLatent() async throws {
        let actor = try ConditioningActor.stub()
        await actor.setSilenceLatent(MLXArray.ones([1, 15000, 64]) * 0.25)

        let sourceFrames = 5
        let targetFrames = 10
        let duration = Double(targetFrames * ConditioningActor.vaeHopLength)
            / Double(ConditioningActor.audioSampleRate)
        let result = try await actor.encode(
            text: "Test",
            lyrics: "",
            referenceAudio: nil,
            duration: duration,
            precision: .float32,
            srcLatentsOverride: MLXArray.zeros([1, sourceFrames, 64])
        )

        let sourceLatents = result.contextLatents[0..., 0..., 0..<64]
        let sourceHead = sourceLatents[0..., 0..<sourceFrames, 0...]
        let sourceTail = sourceLatents[0..., sourceFrames..<targetFrames, 0...]
        let headMax = MLX.abs(sourceHead).max()
        let tailMin = sourceTail.min()
        let tailMax = sourceTail.max()
        eval(headMax, tailMin, tailMax)

        #expect(headMax.item(Float.self) == 0)
        #expect(tailMin.item(Float.self) == 0.25)
        #expect(tailMax.item(Float.self) == 0.25)
    }
}
