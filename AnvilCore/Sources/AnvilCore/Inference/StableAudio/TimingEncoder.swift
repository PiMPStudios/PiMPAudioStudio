import Foundation
import MLX

public struct TimingEncoder: Sendable {
    public let embeddingDim: Int

    public init(embeddingDim: Int) {
        self.embeddingDim = embeddingDim
    }

    public func encode(startSeconds: Float, durationSeconds: Float) -> MLXArray {
        let halfDim = embeddingDim / 4

        let startEmb = sinusoidalEmbedding(value: startSeconds, halfDim: halfDim)
        let durationEmb = sinusoidalEmbedding(value: durationSeconds, halfDim: halfDim)

        let combined = concatenated([startEmb, durationEmb], axis: 1)
        return combined
    }

    private func sinusoidalEmbedding(value: Float, halfDim: Int) -> MLXArray {
        let positions = MLXArray(value)
        let dimRange = MLXArray(Array(0..<halfDim).map { Float($0) })
        let freqs = exp(dimRange * (-log(Float(10000.0)) / Float(halfDim)))
        let angles = positions * freqs
        let sinVals = sin(angles)
        let cosVals = cos(angles)
        let embedding = concatenated(
            [sinVals.reshaped([1, halfDim]), cosVals.reshaped([1, halfDim])],
            axis: 1
        )
        return embedding
    }
}
