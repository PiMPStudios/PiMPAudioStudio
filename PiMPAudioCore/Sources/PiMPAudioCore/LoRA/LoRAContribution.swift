import Foundation
import MLX

/// Represents a single adapter's contribution in a multi-adapter stack.
///
/// Used by LoRALinear to compute additive low-rank contributions:
/// `output += strength * (x @ loraA^T @ loraB^T)`
public struct LoRAContribution: @unchecked Sendable {
    /// ID of the adapter providing this contribution.
    public let adapterID: String

    /// Display name for UI and logging.
    public let displayName: String?

    /// Provenance of this adapter.
    public let source: AdapterSource

    /// Low-rank matrix A: shape [rank, inputDim]. Kaiming uniform init.
    public let loraA: MLXArray

    /// Low-rank matrix B: shape [outputDim, rank]. Zero or trained values.
    public let loraB: MLXArray

    /// Strength multiplier (0.0 = no effect, 1.0 = full effect).
    public var strength: Float

    /// Adapter-specific scaling factor, normally alpha/rank.
    public let scaling: Float

    /// Native module path that this contribution was loaded onto.
    public let path: String?

    public init(
        adapterID: String,
        displayName: String? = nil,
        source: AdapterSource = .local,
        loraA: MLXArray,
        loraB: MLXArray,
        strength: Float = 1.0,
        scaling: Float = 1.0,
        path: String? = nil
    ) {
        self.adapterID = adapterID
        self.displayName = displayName
        self.source = source
        self.loraA = loraA
        self.loraB = loraB
        self.strength = strength
        self.scaling = scaling
        self.path = path
    }
}
