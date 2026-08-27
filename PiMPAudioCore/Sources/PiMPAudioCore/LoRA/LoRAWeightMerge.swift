import Foundation
@preconcurrency import MLX

/// Errors from mathematically blending two LoRA adapters.
public enum LoRAWeightMergeError: Error, Sendable, Equatable, LocalizedError {
    case invalidBlendWeight(Float)
    case unpairedWeight(String)
    case shapeMismatch(String)
    case noLoRAWeights

    public var errorDescription: String? {
        switch self {
        case let .invalidBlendWeight(value):
            "Merge blend weight must be between 0 and 1 inclusive (got \(value))."
        case let .unpairedWeight(key):
            "LoRA weight key '\(key)' is missing its A/B pair."
        case let .shapeMismatch(detail):
            "Cannot merge LoRA weights with incompatible shapes: \(detail)."
        case .noLoRAWeights:
            "No LoRA A/B weight pairs were found to merge."
        }
    }
}

/// Blends two LoRA adapters by interpolating **effective deltas**, not the A/B
/// factors independently.
///
/// Independent factor interpolation is incorrect:
/// `(w·B1 + (1-w)·B2) @ (w·A1 + (1-w)·A2)` expands to cross terms that are
/// not equal to `w·(B1@A1) + (1-w)·(B2@A2)`.
///
/// This merger forms the true convex combination of scaled deltas via rank
/// concatenation:
///
/// ```
/// Δ = c1·(B1@A1) + c2·(B2@A2)
///   = [c1·B1 | c2·B2] @ [A1 ; A2]
/// ```
///
/// with `c1 = w · (α1/r1)`, `c2 = (1-w) · (α2/r2)`. The merged adapter uses
/// rank `r1+r2` and scaling `1` (alpha = rank) so the concatenated factors
/// already carry the full delta magnitude.
public enum LoRAWeightMerge {
    public struct Result: @unchecked Sendable {
        public let weights: [(String, MLXArray)]
        /// Maximum rank across merged modules (typically r1+r2).
        public let rank: Int
        /// Alpha chosen so `alpha/rank == 1` for the merged adapter.
        public let alpha: Float
    }

    /// Blend two LoRA weight dictionaries.
    ///
    /// - Parameters:
    ///   - weightsA: First adapter weights (`path.loraA` / `path.loraB`).
    ///   - weightsB: Second adapter weights.
    ///   - weightOnA: Blend weight `w` in `[0, 1]` (`1` = all A, `0` = all B).
    ///   - scalingA: First adapter scaling (`alpha/rank`).
    ///   - scalingB: Second adapter scaling (`alpha/rank`).
    public static func blendDeltas(
        weightsA: [String: MLXArray],
        weightsB: [String: MLXArray],
        weightOnA: Float,
        scalingA: Float,
        scalingB: Float
    ) throws -> Result {
        guard weightOnA >= 0, weightOnA <= 1, weightOnA.isFinite else {
            throw LoRAWeightMergeError.invalidBlendWeight(weightOnA)
        }

        let pairsA = try pairedFactors(from: weightsA)
        let pairsB = try pairedFactors(from: weightsB)
        let paths = Set(pairsA.keys).union(pairsB.keys)
        guard !paths.isEmpty else {
            throw LoRAWeightMergeError.noLoRAWeights
        }

        let c1 = weightOnA * scalingA
        let c2 = (1 - weightOnA) * scalingB

        var merged: [(String, MLXArray)] = []
        var maxRank = 0

        for path in paths.sorted() {
            let pairA = pairsA[path]
            let pairB = pairsB[path]

            switch (pairA, pairB) {
            case let (a?, b?):
                try validateCompatible(path: path, left: a, right: b)
                let aMerged = concatenated([a.loraA, b.loraA], axis: 0)
                let bMerged = concatenated(
                    [MLXArray(c1) * a.loraB, MLXArray(c2) * b.loraB],
                    axis: 1
                )
                let rank = aMerged.shape[0]
                maxRank = max(maxRank, rank)
                merged.append(("\(path).loraA", aMerged))
                merged.append(("\(path).loraB", bMerged))

            case let (a?, nil):
                let rank = a.loraA.shape[0]
                maxRank = max(maxRank, rank)
                merged.append(("\(path).loraA", a.loraA))
                merged.append(("\(path).loraB", MLXArray(c1) * a.loraB))

            case let (nil, b?):
                let rank = b.loraA.shape[0]
                maxRank = max(maxRank, rank)
                merged.append(("\(path).loraA", b.loraA))
                merged.append(("\(path).loraB", MLXArray(c2) * b.loraB))

            case (nil, nil):
                continue
            }
        }

        guard maxRank > 0 else {
            throw LoRAWeightMergeError.noLoRAWeights
        }

        // scaling = alpha/rank = 1 so concatenated factors already encode c1/c2.
        return Result(weights: merged, rank: maxRank, alpha: Float(maxRank))
    }

    // MARK: - Pairing

    private struct FactorPair {
        let loraA: MLXArray // [rank, in]
        let loraB: MLXArray // [out, rank]
    }

    private static func pairedFactors(
        from weights: [String: MLXArray]
    ) throws -> [String: FactorPair] {
        var aByPath: [String: MLXArray] = [:]
        var bByPath: [String: MLXArray] = [:]

        for (key, value) in weights {
            if key.hasSuffix(".loraA") {
                let path = String(key.dropLast(".loraA".count))
                aByPath[path] = value
            } else if key.hasSuffix(".loraB") {
                let path = String(key.dropLast(".loraB".count))
                bByPath[path] = value
            } else {
                throw LoRAWeightMergeError.unpairedWeight(key)
            }
        }

        let paths = Set(aByPath.keys).union(bByPath.keys)
        var pairs: [String: FactorPair] = [:]
        for path in paths {
            guard let loraA = aByPath[path], let loraB = bByPath[path] else {
                let missing = aByPath[path] == nil ? "\(path).loraA" : "\(path).loraB"
                throw LoRAWeightMergeError.unpairedWeight(missing)
            }
            try validatePair(path: path, loraA: loraA, loraB: loraB)
            pairs[path] = FactorPair(loraA: loraA, loraB: loraB)
        }
        return pairs
    }

    private static func validatePair(path: String, loraA: MLXArray, loraB: MLXArray) throws {
        guard loraA.ndim == 2, loraB.ndim == 2 else {
            throw LoRAWeightMergeError.shapeMismatch(
                "\(path): expected 2-D A/B, got A\(loraA.shape) B\(loraB.shape)"
            )
        }
        let rankA = loraA.shape[0]
        let rankB = loraB.shape[1]
        guard rankA == rankB, rankA > 0 else {
            throw LoRAWeightMergeError.shapeMismatch(
                "\(path): rank mismatch A\(loraA.shape) vs B\(loraB.shape)"
            )
        }
    }

    private static func validateCompatible(
        path: String,
        left: FactorPair,
        right: FactorPair
    ) throws {
        let inLeft = left.loraA.shape[1]
        let inRight = right.loraA.shape[1]
        let outLeft = left.loraB.shape[0]
        let outRight = right.loraB.shape[0]
        guard inLeft == inRight, outLeft == outRight else {
            throw LoRAWeightMergeError.shapeMismatch(
                "\(path): in/out features differ A\(left.loraA.shape)/\(left.loraB.shape) vs B\(right.loraA.shape)/\(right.loraB.shape)"
            )
        }
    }
}
