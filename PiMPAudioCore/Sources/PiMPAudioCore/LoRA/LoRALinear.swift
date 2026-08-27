import Foundation
import MLX
import MLXNN
import MLXRandom

/// A Linear layer wrapped with one or more low-rank adapter contributions.
///
/// Single adapter (Phase 10a/b):
///   Output = base(x) + scaling * dropout(x @ A^T @ B^T)
///
/// Multi-adapter stacking (Phase 10c):
///   Output = base(x) + sum_i(strength_i * scaling_i * x @ A_i^T @ B_i^T)
///
/// A is initialized with Kaiming uniform, B is initialized to zero,
/// so a fresh adapter starts as identity (no effect on output).
public class LoRALinear: Linear {
    /// The original frozen weight matrix stored separately.
    let baseWeight: MLXArray

    /// The original bias (nil if the base layer had no bias).
    let baseBias: MLXArray?

    /// Low-rank matrix A: shape (rank, inFeatures). Kaiming uniform init.
    /// Used for training (single adapter mode). For stacking, see contributions.
    @ParameterInfo var loraA: MLXArray

    /// Low-rank matrix B: shape (outFeatures, rank). Zero init.
    /// Used for training (single adapter mode). For stacking, see contributions.
    @ParameterInfo var loraB: MLXArray

    /// Scaling factor = alpha / rank.
    let scaling: Float

    /// Dropout rate (applied during training only).
    let dropoutRate: Float

    /// Whether we are in training mode (enables dropout, uses loraA/loraB directly).
    var isTraining: Bool = false

    /// Whether adapter has been merged into base weights.
    private var isMerged: Bool = false

    /// Multiple adapter contributions for inference stacking.
    /// When non-empty, these are used instead of the single loraA/loraB pair.
    /// Each contribution has its own strength multiplier.
    public private(set) var contributions: [LoRAContribution] = []

    public init(base: Linear, rank: Int, alpha: Float = 16.0, dropout: Float = 0.0) {
        let inFeatures = base.weight.shape[1]
        let outFeatures = base.weight.shape[0]

        // Store a copy of the original weight for unmerge.
        // Use identity arithmetic (+ 0) to create a distinct MLXArray holding the same data.
        baseWeight = base.weight + MLXArray(Float(0))
        baseBias = base.bias.map { $0 + MLXArray(Float(0)) }
        scaling = alpha / Float(rank)
        dropoutRate = dropout

        // Kaiming uniform for A: bound = sqrt(1/inFeatures)
        let bound = Float(1.0 / Float(inFeatures).squareRoot())
        _loraA.wrappedValue = MLXRandom.uniform(-bound ..< bound, [rank, inFeatures])

        // Zero init for B so adapter starts as identity
        _loraB.wrappedValue = MLXArray.zeros([outFeatures, rank])

        // Initialize Linear superclass with the base weights
        super.init(weight: base.weight, bias: base.bias)
    }

    override public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Base linear: x @ W^T + bias
        let baseOut: MLXArray = if let bias {
            addMM(bias, x, weight.T)
        } else {
            matmul(x, weight.T)
        }

        if isMerged {
            return baseOut
        }

        // Training mode: use single loraA/loraB pair with dropout
        if isTraining {
            var loraInput = x
            if dropoutRate > 0 {
                let mask: MLXArray = if let dropoutKey = AdapterDropoutKeyContext.nextKey() {
                    MLXRandom.uniform(0 ..< 1, loraInput.shape, key: dropoutKey) .> dropoutRate
                } else {
                    MLXRandom.uniform(0 ..< 1, loraInput.shape) .> dropoutRate
                }
                loraInput = loraInput * mask / (1.0 - dropoutRate)
            }
            let loraOut = scaling * matmul(matmul(loraInput, loraA.T), loraB.T)
            return baseOut + loraOut
        }

        // Inference stacking mode: use contributions if present
        if !contributions.isEmpty {
            var out = baseOut
            for contribution in contributions where contribution.strength != 0.0 {
                let loraOut = matmul(matmul(x, contribution.loraA.T), contribution.loraB.T)
                let deltaOut = MLXArray(contribution.strength * contribution.scaling) * loraOut
                AdapterEffectDiagnostics.contribution(
                    kind: .lora,
                    adapterID: contribution.adapterID,
                    path: contribution.path,
                    strength: contribution.strength,
                    scale: contribution.scaling,
                    baseOut: baseOut,
                    deltaOut: deltaOut
                )
                out = out + deltaOut
            }
            return out
        }

        // Single adapter inference (backward compat with 10a/b usage)
        let loraOut = scaling * matmul(matmul(x, loraA.T), loraB.T)
        return baseOut + loraOut
    }

    /// Fold adapter weights into the base weight matrix for zero-overhead inference.
    public func merge() {
        guard !isMerged else { return }
        let delta = scaling * matmul(loraB, loraA)
        weight._updateInternal(baseWeight + delta)
        isMerged = true
    }

    /// Reverse a merge operation, restoring the original base weight.
    public func unmerge() {
        guard isMerged else { return }
        weight._updateInternal(baseWeight)
        isMerged = false
    }

    public func replaceContributions(_ newContributions: [LoRAContribution]) {
        contributions = newContributions
    }

    public func appendContribution(_ contribution: LoRAContribution) {
        contributions.append(contribution)
    }

    public func clearContributions() {
        contributions.removeAll()
    }

    public func setContributionStrength(adapterID: String, strength: Float) {
        for index in contributions.indices where contributions[index].adapterID == adapterID {
            contributions[index].strength = strength
        }
    }

    public func removeContribution(adapterID: String) {
        contributions.removeAll { $0.adapterID == adapterID }
    }

    /// Mark base parameters as non-trainable, keeping only loraA and loraB trainable.
    public func freezeBase() {
        freeze(recursive: false, keys: ["weight", "bias", "baseWeight", "baseBias"])
    }
}
