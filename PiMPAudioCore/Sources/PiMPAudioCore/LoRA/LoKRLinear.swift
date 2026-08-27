import Foundation
import MLX
import MLXNN
import MLXRandom

/// Linear layer wrapped with an experimental Low-Rank Kronecker adapter.
///
/// The learned delta weight is `kron(W1, W2)` cropped to the base linear
/// weight shape. This keeps the native implementation small and fully MLX
/// differentiable while matching LoKR's core Kronecker-factor idea.
///
/// FORMAT NOTE: this is a PiMP-native, simplified LoKR. The saved tensors
/// are `.lokrW1`, `.lokrW2`, and optionally `.lokrMagnitude` (DoRA). Local
/// import converts compatible PEFT/Kohya/LyCORIS LoKR checkpoints into this
/// native layout, including direct `lokr_w1`/`lokr_w2` tensors and simple
/// factored `lokr_w*_a`/`lokr_w*_b` pairs. Tucker-style variants still need
/// a richer runtime representation and are rejected at import time.
public class LoKRLinear: Linear {
    let baseWeight: MLXArray
    let baseBias: MLXArray?

    @ParameterInfo var lokrW1: MLXArray
    @ParameterInfo var lokrW2: MLXArray
    @ParameterInfo var lokrMagnitude: MLXArray

    let scaling: Float
    let dropoutRate: Float
    let weightDecompose: Bool
    let outputFeatures: Int
    let inputFeatures: Int
    let factorRows: Int
    let factorCols: Int

    var isTraining: Bool = false
    private var isMerged: Bool = false

    public private(set) var contributions: [LoKRContribution] = []

    public init(
        base: Linear,
        linearDim: Int,
        alpha: Float = 64.0,
        dropout: Float = 0.0,
        weightDecompose: Bool = false
    ) {
        let inFeatures = base.weight.shape[1]
        let outFeatures = base.weight.shape[0]
        let safeDim = max(1, linearDim)
        let rows = min(safeDim, outFeatures)
        let cols = min(safeDim, inFeatures)
        let outerRows = Self.ceilDiv(outFeatures, rows)
        let outerCols = Self.ceilDiv(inFeatures, cols)

        baseWeight = base.weight + MLXArray(Float(0))
        baseBias = base.bias.map { $0 + MLXArray(Float(0)) }
        scaling = alpha / Float(safeDim)
        dropoutRate = dropout
        self.weightDecompose = weightDecompose
        outputFeatures = outFeatures
        inputFeatures = inFeatures
        factorRows = rows
        factorCols = cols

        let bound = Float(1.0 / Float(max(1, outerCols)).squareRoot())
        _lokrW1.wrappedValue = MLXRandom.uniform(-bound ..< bound, [outerRows, outerCols])
        _lokrW2.wrappedValue = MLXArray.zeros([rows, cols])
        _lokrMagnitude.wrappedValue = Self.rowNorm(base.weight)

        super.init(weight: base.weight, bias: base.bias)
    }

    override public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let baseOut: MLXArray = if let bias {
            addMM(bias, x, weight.T)
        } else {
            matmul(x, weight.T)
        }

        if isMerged {
            return baseOut
        }

        if isTraining {
            var adapterInput = x
            if dropoutRate > 0 {
                let mask: MLXArray = if let dropoutKey = AdapterDropoutKeyContext.nextKey() {
                    MLXRandom.uniform(0 ..< 1, adapterInput.shape, key: dropoutKey) .> dropoutRate
                } else {
                    MLXRandom.uniform(0 ..< 1, adapterInput.shape) .> dropoutRate
                }
                adapterInput = adapterInput * mask / (1.0 - dropoutRate)
            }
            let delta = Self.kroneckerWeight(
                lokrW1,
                lokrW2,
                outputFeatures: outputFeatures,
                inputFeatures: inputFeatures
            )
            if weightDecompose {
                return Self.doraLinearOutput(
                    x: x,
                    adapterInput: adapterInput,
                    baseWeight: baseWeight,
                    baseBias: baseBias,
                    delta: delta,
                    scaling: scaling,
                    magnitude: lokrMagnitude
                )
            }
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
                if contribution.weightDecompose {
                    let magnitude = contribution.lokrMagnitude ?? Self.rowNorm(baseWeight)
                    let doraWeight = Self.doraWeight(
                        baseWeight: baseWeight,
                        scaledDelta: contribution.scaling * delta,
                        magnitude: magnitude
                    )
                    let contributionOut: MLXArray = if let baseBias {
                        addMM(baseBias, x, doraWeight.T)
                    } else {
                        matmul(x, doraWeight.T)
                    }
                    let deltaOut = MLXArray(contribution.strength) * (contributionOut - baseOut)
                    AdapterEffectDiagnostics.contribution(
                        kind: .lokr,
                        adapterID: contribution.adapterID,
                        path: contribution.path,
                        strength: contribution.strength,
                        scale: contribution.scaling,
                        baseOut: baseOut,
                        deltaOut: deltaOut
                    )
                    out = out + deltaOut
                } else {
                    let deltaOut = MLXArray(contribution.strength * contribution.scaling) * matmul(x, delta.T)
                    AdapterEffectDiagnostics.contribution(
                        kind: .lokr,
                        adapterID: contribution.adapterID,
                        path: contribution.path,
                        strength: contribution.strength,
                        scale: contribution.scaling,
                        baseOut: baseOut,
                        deltaOut: deltaOut
                    )
                    out = out + deltaOut
                }
            }
            return out
        }

        let delta = Self.kroneckerWeight(
            lokrW1,
            lokrW2,
            outputFeatures: outputFeatures,
            inputFeatures: inputFeatures
        )
        if weightDecompose {
            let doraWeight = Self.doraWeight(
                baseWeight: baseWeight,
                scaledDelta: scaling * delta,
                magnitude: lokrMagnitude
            )
            if let baseBias {
                return addMM(baseBias, x, doraWeight.T)
            }
            return matmul(x, doraWeight.T)
        }
        return baseOut + scaling * matmul(x, delta.T)
    }

    public func merge() {
        guard !isMerged else { return }
        let delta = Self.kroneckerWeight(
            lokrW1,
            lokrW2,
            outputFeatures: outputFeatures,
            inputFeatures: inputFeatures
        )
        let mergedWeight = weightDecompose
            ? Self.doraWeight(baseWeight: baseWeight, scaledDelta: scaling * delta, magnitude: lokrMagnitude)
            : baseWeight + scaling * delta
        weight._updateInternal(mergedWeight)
        isMerged = true
    }

    public func unmerge() {
        guard isMerged else { return }
        weight._updateInternal(baseWeight)
        isMerged = false
    }

    public func appendContribution(_ contribution: LoKRContribution) {
        contributions.append(contribution)
    }

    public func clearContributions() {
        contributions.removeAll()
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
        return expanded[0 ..< outputFeatures, 0 ..< inputFeatures]
    }

    public static func doraWeight(
        baseWeight: MLXArray,
        scaledDelta: MLXArray,
        magnitude: MLXArray
    ) -> MLXArray {
        let directionalWeight = baseWeight + scaledDelta
        let rowScale = magnitude.reshaped([magnitude.shape[0], 1]) /
            (stopGradient(rowNorm(directionalWeight)).reshaped([magnitude.shape[0], 1]) + 1e-6)
        return directionalWeight * rowScale
    }

    public static func rowNorm(_ weight: MLXArray) -> MLXArray {
        sqrt((weight * weight).sum(axis: 1))
    }

    private static func doraLinearOutput(
        x: MLXArray,
        adapterInput: MLXArray,
        baseWeight: MLXArray,
        baseBias: MLXArray?,
        delta: MLXArray,
        scaling: Float,
        magnitude: MLXArray
    ) -> MLXArray {
        // PEFT-parity DoRA training step with dropout: the magnitude-
        // normalised correction term is computed entirely on the (possibly
        // dropout-masked) adapter input, while the plain base output uses
        // the full input:
        //
        //   out = base(x) + (m/‖W+sΔ‖ − 1)·(xd Wᵀ) + (m/‖W+sΔ‖)·s·(xd Δᵀ)
        //
        // where xd is the dropout-masked input. With dropout = 0 this
        // reduces exactly to m/‖W+sΔ‖ · (x Wᵀ + s·x Δᵀ) + bias.
        //
        // The previous form applied the row scale to the full-input base
        // term while the adapter term used the dropped input — not a valid
        // sample of the decomposed weight when dropout > 0.
        let scaledDelta = scaling * delta
        let directionalWeight = baseWeight + scaledDelta
        let outputScale = magnitude / (stopGradient(rowNorm(directionalWeight)) + 1e-6)
        let baseOut: MLXArray = if let baseBias {
            addMM(baseBias, x, baseWeight.T)
        } else {
            matmul(x, baseWeight.T)
        }
        let droppedBase = matmul(adapterInput, baseWeight.T)
        let adapterOut = scaling * matmul(adapterInput, delta.T)
        return baseOut + (outputScale - 1.0) * droppedBase + outputScale * adapterOut
    }

    private static func ceilDiv(_ lhs: Int, _ rhs: Int) -> Int {
        (lhs + rhs - 1) / rhs
    }
}
