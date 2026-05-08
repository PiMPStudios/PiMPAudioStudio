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

    /// EDM v-prediction Euler step for Stable Audio Open.
    ///
    /// SAO uses the v-objective: `v = α·ε − σ_c·x₀` where
    ///   `α(σ) = 1 / sqrt(1 + σ²)` and `σ_c(σ) = σ · α(σ)`.
    ///
    /// Recovering x₀: `x₀_pred = α · x_t − σ_c · v_pred`
    /// First-order Euler ODE step using the score direction `d = (x_t − x₀_pred) / σ_t`:
    ///   `x_next = x_t + (σ_next − σ_t) · d`
    ///
    /// - Parameters:
    ///   - xt:        noisy latent at current sigma
    ///   - vt:        model-predicted velocity (v-prediction, NOT rectified-flow velocity)
    ///   - sigma:     current noise level σ_t  (range [0.3, 500] for SAO)
    ///   - sigmaNext: next noise level σ_{t+1} (lower); pass 0 for the final step
    /// - Returns:     denoised latent at sigmaNext
    public static func edmEulerStep(
        xt: MLXArray,
        vt: MLXArray,
        sigma: Float,
        sigmaNext: Float
    ) -> MLXArray {
        // EDM signal/noise scaling
        let alpha      = 1.0 / sqrtf(1.0 + sigma * sigma)
        let sigmaCoeff = sigma * alpha  // σ / sqrt(1 + σ²)

        // Recover predicted clean sample from v-prediction
        let x0Pred = MLXArray(alpha) * xt - MLXArray(sigmaCoeff) * vt

        // At the final step (sigmaNext ≈ 0) just return the clean prediction
        if sigmaNext <= 0 {
            return x0Pred
        }

        // Score direction and Euler step
        let d = (xt - x0Pred) / MLXArray(sigma)
        return xt + MLXArray(sigmaNext - sigma) * d
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
