import Foundation
import MLX

public enum TensorParityMetricsError: Error, LocalizedError, Sendable {
    case emptyTensor(String)

    public var errorDescription: String? {
        switch self {
        case .emptyTensor(let name):
            return "Cannot compare empty tensor '\(name)'."
        }
    }
}

public struct TensorComparisonMetrics: Sendable {
    public let comparedElements: Int
    public let lhsShape: [Int]
    public let rhsShape: [Int]
    public let meanAbsoluteError: Float
    public let meanSquaredError: Float
    public let rootMeanSquaredError: Float
    public let maxAbsoluteDifference: Float
    public let lhsMeanAbsoluteValue: Float
    public let rhsMeanAbsoluteValue: Float
    public let cosineSimilarity: Float?

    public var shapesMatch: Bool {
        lhsShape == rhsShape
    }
}

public enum TensorParityMetrics {
    public static func compare(
        name: String,
        lhs rawLHS: MLXArray,
        rhs rawRHS: MLXArray
    ) throws -> TensorComparisonMetrics {
        let lhs = rawLHS.asType(.float32).reshaped([-1])
        let rhs = rawRHS.asType(.float32).reshaped([-1])
        let comparedElements = min(elementCount(rawLHS.shape), elementCount(rawRHS.shape))
        guard comparedElements > 0 else {
            throw TensorParityMetricsError.emptyTensor(name)
        }

        let lhsComparable = lhs[0..<comparedElements]
        let rhsComparable = rhs[0..<comparedElements]
        eval(lhsComparable, rhsComparable)

        let diff = lhsComparable - rhsComparable
        let absDiff = MLX.abs(diff)
        let squaredDiff = diff * diff
        eval(absDiff, squaredDiff)

        let mse = squaredDiff.mean().item(Float.self)
        return TensorComparisonMetrics(
            comparedElements: comparedElements,
            lhsShape: rawLHS.shape,
            rhsShape: rawRHS.shape,
            meanAbsoluteError: absDiff.mean().item(Float.self),
            meanSquaredError: mse,
            rootMeanSquaredError: sqrt(max(mse, 0)),
            maxAbsoluteDifference: absDiff.max().item(Float.self),
            lhsMeanAbsoluteValue: MLX.abs(lhsComparable).mean().item(Float.self),
            rhsMeanAbsoluteValue: MLX.abs(rhsComparable).mean().item(Float.self),
            cosineSimilarity: cosineSimilarity(lhsComparable, rhsComparable)
        )
    }

    public static func elementCount(_ shape: [Int]) -> Int {
        shape.reduce(1, *)
    }

    private static func cosineSimilarity(_ lhs: MLXArray, _ rhs: MLXArray) -> Float? {
        let numerator = (lhs * rhs).sum().item(Float.self)
        let lhsNormSquared = (lhs * lhs).sum().item(Float.self)
        let rhsNormSquared = (rhs * rhs).sum().item(Float.self)
        let denominator = sqrt(max(lhsNormSquared * rhsNormSquared, 0))
        guard denominator > 1e-20 else { return nil }
        return numerator / denominator
    }
}
