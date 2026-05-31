import Foundation
import MLX

/// Represents a single LoKR adapter contribution in an inference stack.
public struct LoKRContribution: @unchecked Sendable {
    public let adapterID: String
    public let displayName: String?
    public let source: AdapterSource
    public let lokrW1: MLXArray
    public let lokrW2: MLXArray
    public var strength: Float

    public init(
        adapterID: String,
        displayName: String? = nil,
        source: AdapterSource = .local,
        lokrW1: MLXArray,
        lokrW2: MLXArray,
        strength: Float = 1.0
    ) {
        self.adapterID = adapterID
        self.displayName = displayName
        self.source = source
        self.lokrW1 = lokrW1
        self.lokrW2 = lokrW2
        self.strength = strength
    }
}
