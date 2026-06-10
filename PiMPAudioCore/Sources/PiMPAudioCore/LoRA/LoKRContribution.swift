import Foundation
import MLX

/// Represents a single LoKR adapter contribution in an inference stack.
public struct LoKRContribution: @unchecked Sendable {
    public let adapterID: String
    public let displayName: String?
    public let source: AdapterSource
    public let lokrW1: MLXArray
    public let lokrW2: MLXArray
    public let lokrMagnitude: MLXArray?
    public let weightDecompose: Bool
    public var strength: Float
    public let scaling: Float
    public let path: String?

    public init(
        adapterID: String,
        displayName: String? = nil,
        source: AdapterSource = .local,
        lokrW1: MLXArray,
        lokrW2: MLXArray,
        lokrMagnitude: MLXArray? = nil,
        weightDecompose: Bool = false,
        strength: Float = 1.0,
        scaling: Float = 1.0,
        path: String? = nil
    ) {
        self.adapterID = adapterID
        self.displayName = displayName
        self.source = source
        self.lokrW1 = lokrW1
        self.lokrW2 = lokrW2
        self.lokrMagnitude = lokrMagnitude
        self.weightDecompose = weightDecompose
        self.strength = strength
        self.scaling = scaling
        self.path = path
    }
}
