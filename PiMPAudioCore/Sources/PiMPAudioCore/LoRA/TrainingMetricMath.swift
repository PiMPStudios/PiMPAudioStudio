import Foundation
import MLX
import MLXNN

enum TrainingMetricMath {
    static func l2Norm(_ arrays: [MLXArray]) -> Float? {
        guard !arrays.isEmpty else { return nil }
        var total = MLXArray(0.0)
        for array in arrays {
            let values = array.asType(.float32)
            total = total + (values * values).sum()
        }
        return sqrt(max(total.item(Float.self), 0))
    }

    static func l2NormOfDeltas(before: [MLXArray], after: [MLXArray]) -> Float? {
        guard before.count == after.count, !before.isEmpty else { return nil }
        let deltas = zip(before, after).map { oldValue, newValue in
            (newValue - oldValue).asType(.float32)
        }
        return l2Norm(deltas)
    }

    static func clipByGlobalNorm(
        _ arrays: [MLXArray],
        globalNorm: Float?,
        maxNorm: Float
    ) -> [MLXArray] {
        guard let globalNorm,
              globalNorm.isFinite,
              maxNorm > 0,
              globalNorm > maxNorm
        else {
            return arrays
        }
        let scale = maxNorm / (globalNorm + 1e-6)
        return arrays.map { $0 * scale }
    }

    static func materializedSnapshot(_ arrays: [MLXArray]) -> [MLXArray] {
        let snapshot = arrays.map { array in
            // `asType(.float32)` can return the same MLXArray for float32 inputs,
            // so add zero to force a new context before the optimizer mutates parameters.
            array.asType(.float32) + MLXArray(Float(0))
        }
        eval(snapshot)
        return snapshot
    }
}
