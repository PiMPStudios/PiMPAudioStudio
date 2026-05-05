import Testing
import MLX
import MLXNN
@testable import AnvilCore

@Suite("Condition Encoder")
struct ConditionEncoderTests {
    @Test("ConditionEncoder concatenates all inputs correctly")
    func conditionEncoderOutputShape() {
        let encoder = ConditionEncoder()
        let textHidden = MLXArray.zeros([1, 50, 1024])
        let lyricHidden = MLXArray.zeros([1, 100, 2048])
        let timbreTokens = MLXArray.zeros([1, 5, 2048])
        let textMask = MLXArray.ones([1, 50])
        let lyricMask = MLXArray.ones([1, 100])

        let (states, mask) = encoder(
            textHiddenStates: textHidden,
            textMask: textMask,
            lyricHiddenStates: lyricHidden,
            lyricMask: lyricMask,
            timbreTokens: timbreTokens
        )
        #expect(states.shape == [1, 155, 2048])
        #expect(mask.shape == [1, 155])
    }

    @Test("ConditionEncoder handles max sequence (256 + 2048 + 5)")
    func conditionEncoderMaxSequence() {
        let encoder = ConditionEncoder()
        let textHidden = MLXArray.zeros([1, 256, 1024])
        let lyricHidden = MLXArray.zeros([1, 2048, 2048])
        let timbreTokens = MLXArray.zeros([1, 5, 2048])
        let textMask = MLXArray.ones([1, 256])
        let lyricMask = MLXArray.ones([1, 2048])

        let (states, mask) = encoder(
            textHiddenStates: textHidden,
            textMask: textMask,
            lyricHiddenStates: lyricHidden,
            lyricMask: lyricMask,
            timbreTokens: timbreTokens
        )
        #expect(states.shape == [1, 2309, 2048])
        #expect(mask.shape == [1, 2309])
    }
}
