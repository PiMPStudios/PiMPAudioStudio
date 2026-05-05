import Foundation
import MLX
import MLXNN

public class ConditionEncoder: Module {
    let textProjection: Linear

    static let textInputDim = 1024
    static let hiddenSize = 2048

    override public init() {
        self.textProjection = Linear(Self.textInputDim, Self.hiddenSize, bias: false)
    }

    public func callAsFunction(
        textHiddenStates: MLXArray,
        textMask: MLXArray?,
        lyricHiddenStates: MLXArray,
        lyricMask: MLXArray?,
        timbreTokens: MLXArray
    ) -> (encoderStates: MLXArray, mask: MLXArray) {
        let textProjected = textProjection(textHiddenStates)
        let b = textProjected.shape[0]
        let tText = textProjected.shape[1]
        let tLyric = lyricHiddenStates.shape[1]
        let tTimbre = timbreTokens.shape[1]

        let encoderStates = concatenated(
            [textProjected, lyricHiddenStates, timbreTokens],
            axis: 1
        )

        let effectiveTextMask = textMask ?? MLXArray.ones([b, tText])
        let effectiveLyricMask = lyricMask ?? MLXArray.ones([b, tLyric])
        let timbreMask = MLXArray.ones([b, tTimbre])

        let combinedMask = concatenated(
            [effectiveTextMask, effectiveLyricMask, timbreMask],
            axis: 1
        )

        return (encoderStates, combinedMask)
    }
}
