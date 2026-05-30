import Foundation
import MLX
import MLXNN
import MLXRandom

/// Linear layer wrapped with an experimental Low-Rank Kronecker adapter.
///
/// The learned delta weight is `kron(W1, W2)` cropped to the base linear
/// weight shape. This keeps the native implementation small and fully MLX
/// differentiable while matching LoKR's core Kronecker-factor idea.
public class LoKRLinear: Linear {
    let baseWeight: MLXArray
    let baseBias: MLXArray?

    @ParameterInfo var lokrW1: MLXArray
    @ParameterInfo var lokrW2: MLXArray

    let scaling: Float
    let dropoutRate: Float
    let outputFeatures: Int
    let inputFeatures: Int
    let factorRows: Int
    let factorCols: Int

    var isTraining: Bool = false
    private var isMerged: Bool = false

    public var contributions: [LoKRContribution] = []

    public init(base: Linear, linearDim: Int, alpha: Float = 64.0, dropout: Float = 0.0) {
        let inFeatures = base.weight.shape[1]
        let outFeatures = base.weight.shape[0]
        let safeDim = max(1, linearDim)
        let rows = min(safeDim, outFeatures)
        let cols = min(safeDim, inFeatures)
        let outerRows = Self.ceilDiv(outFeatures, rows)
        let outerCols = Self.ceilDiv(inFeatures, cols)

        self.baseWeight = base.weight + MLXArray(Float(0))
        self.baseBias = base.bias.map { $0 + MLXArray(Float(0)) }
        self.scaling = alpha / Float(safeDim)
        self.dropoutRate = dropout
        self.outputFeatures = outFeatures
        self.inputFeatures = inFeatures
        self.factorRows = rows
        self.factorCols = cols

        let bound = Float(1.0 / Float(max(1, outerCols)).squareRoot())
        self._lokrW1.wrappedValue = MLXRandom.uniform(-bound ..< bound, [outerRows, outerCols])
        self._lokrW2.wrappedValue = MLXArray.zeros([rows, cols])

        super.init(weight: base.weight, bias: base.bias)
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let baseOut: MLXArray
        if let bias {
            baseOut = addMM(bias, x, weight.T)
        } else {
            baseOut = matmul(x, weight.T)
        }

        if isMerged {
            return baseOut
        }

        if isTraining {
            var adapterInput = x
            if dropoutRate > 0 {
                let mask = MLXRandom.uniform(0 ..< 1, adapterInput.shape) .> dropoutRate
                adapterInput = adapterInput * mask / (1.0 - dropoutRate)
            }
            let delta = Self.kroneckerWeight(
                lokrW1,
                lokrW2,
                outputFeatures: outputFeatures,
                inputFeatures: inputFeatures
            )
            return baseOut + scaling * matmul(adapterInput, delta.T)
        }

        if !contributions.isEmpty {
            var out = baseOut
            for contribution in contributions where contribution.strength != 0.0 {
                let delta = Self.kroneckerWeight(
                    contribution.lokrW1,
                    contribution.lokrW2,
                    outputFeatures: outputFeatures,
                    inputFeatures: inputFeatures
                )
                out = out + MLXArray(contribution.strength * scaling) * matmul(x, delta.T)
            }
            return out
        }

        let delta = Self.kroneckerWeight(
            lokrW1,
            lokrW2,
            outputFeatures: outputFeatures,
            inputFeatures: inputFeatures
        )
        return baseOut + scaling * matmul(x, delta.T)
    }

    public func merge() {
        guard !isMerged else { return }
        let delta = scaling * Self.kroneckerWeight(
            lokrW1,
            lokrW2,
            outputFeatures: outputFeatures,
            inputFeatures: inputFeatures
        )
        weight._updateInternal(baseWeight + delta)
        isMerged = true
    }

    public func unmerge() {
        guard isMerged else { return }
        weight._updateInternal(baseWeight)
        isMerged = false
    }

    public func freezeBase() {
        freeze(recursive: false, keys: ["weight", "bias", "baseWeight", "baseBias"])
    }

    public static func kroneckerWeight(
        _ w1: MLXArray,
        _ w2: MLXArray,
        outputFeatures: Int,
        inputFeatures: Int
    ) -> MLXArray {
        let expanded = (
            w1.reshaped([w1.shape[0], 1, w1.shape[1], 1]) *
                w2.reshaped([1, w2.shape[0], 1, w2.shape[1]])
        ).reshaped([w1.shape[0] * w2.shape[0], w1.shape[1] * w2.shape[1]])
        return expanded[0..<outputFeatures, 0..<inputFeatures]
    }

    private static func ceilDiv(_ lhs: Int, _ rhs: Int) -> Int {
        (lhs + rhs - 1) / rhs
    }
}
