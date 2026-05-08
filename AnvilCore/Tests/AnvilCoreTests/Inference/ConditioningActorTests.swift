import Testing
import MLX
@testable import AnvilCore

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
        let refAudio = MLXArray.zeros([1, 800, 64])
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

    @Test("ConditioningActor contextLatents uses zeros for text-only")
    func contextLatentsZerosPath() async throws {
        let actor = try ConditioningActor.stub()
        let result = try await actor.encode(
            text: "Test",
            lyrics: "",
            referenceAudio: nil,
            duration: 5.0,
            precision: .float32
        )
        let sum = MLX.sum(MLX.abs(result.contextLatents))
        eval(sum)
        #expect(sum.item(Float.self) == 0.0)
    }
}
