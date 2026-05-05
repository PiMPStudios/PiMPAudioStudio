import Foundation
import MLX
import MLXRandom

public enum DiffusionSampler {

    public static func odeStep(xt: MLXArray, vt: MLXArray, currentT: Float, nextT: Float) -> MLXArray {
        let dt = currentT - nextT
        let bsz = xt.shape[0]
        let dtArr = MLXArray.full([bsz, 1, 1], values: MLXArray(dt))
        return xt - vt * dtArr
    }

    public static func sdeStep(xt: MLXArray, vt: MLXArray, currentT: Float, nextT: Float) -> MLXArray {
        let bsz = xt.shape[0]
        let tUnsq = MLXArray.full([bsz, 1, 1], values: MLXArray(currentT))
        let predClean = xt - vt * tUnsq
        let newNoise = MLXRandom.normal(xt.shape)
        let ntUnsq = MLXArray.full([bsz, 1, 1], values: MLXArray(nextT))
        return ntUnsq * newNoise + (1.0 - ntUnsq) * predClean
    }

    public static func finalStep(xt: MLXArray, vt: MLXArray, currentT: Float) -> MLXArray {
        let bsz = xt.shape[0]
        let tUnsq = MLXArray.full([bsz, 1, 1], values: MLXArray(currentT))
        return xt - vt * tUnsq
    }

    public static func apgForward(
        predCond: MLXArray,
        predUncond: MLXArray,
        guidanceScale: Float,
        momentum: APGMomentum?,
        normThreshold: Float = 2.5
    ) -> MLXArray {
        var diff = predCond - predUncond
        if let momentum = momentum {
            diff = diff + momentum.running
            momentum.running = diff
        }

        if normThreshold > 0 {
            let diffNorm = sqrt((diff * diff).sum(axis: 1, keepDims: true))
            let scaleFactor = minimum(MLXArray.ones(like: diffNorm), MLXArray(normThreshold) / (diffNorm + 1e-8))
            diff = diff * scaleFactor
        }

        let condNorm = sqrt((predCond * predCond).sum(axis: 1, keepDims: true)) + 1e-8
        let v1 = predCond / condNorm
        let parallel = (diff * v1).sum(axis: 1, keepDims: true) * v1
        let orthogonal = diff - parallel

        return predCond + (guidanceScale - 1.0) * orthogonal
    }

    public static func generateNoise(shape: [Int], seed: Int?) -> MLXArray {
        if let seed = seed {
            let key = MLXRandom.key(UInt64(seed))
            return MLXRandom.normal(shape, key: key)
        }
        return MLXRandom.normal(shape)
    }
}

public class APGMomentum {
    var running: MLXArray

    public init() {
        self.running = MLXArray(Float(0))
    }
}
