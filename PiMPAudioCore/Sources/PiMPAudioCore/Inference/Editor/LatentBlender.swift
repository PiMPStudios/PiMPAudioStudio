import MLX
import MLXRandom

// MARK: - LatentBlender

/// Pure math utility for injecting noise into latent regions and compositing
/// denoised results back over preserved regions.
///
/// Latents shape: `[1, T, latentDim]` (e.g. `[1, T, 64]` for ACE-Step).
/// Mask shape: `[1, T, 1]` (broadcasts over channels) or `[1, T, latentDim]`.
/// Values are 1.0 (regenerate) or 0.0 (preserve).
public enum LatentBlender {

    // MARK: - Public API

    /// Injects Gaussian noise into the masked region at the given strength level.
    ///
    /// Formula:
    /// ```
    /// noised = (1 - strength) * clean + strength * noise
    /// result = clean * (1 - mask) + noised * mask
    /// ```
    ///
    /// - Parameters:
    ///   - latents:  Input latents of shape `[1, T, D]`.
    ///   - mask:     Binary mask of shape `[1, T, 1]` or `[1, T, D]`.
    ///   - strength: Noise strength in `[0, 1]`. 0 → unchanged; 1 → pure noise.
    ///   - seed:     Optional seed for deterministic noise generation.
    /// - Returns: Latents with noise injected into the masked region.
    public static func injectNoise(
        latents: MLXArray,
        mask: MLXArray,
        strength: Float,
        seed: Int? = nil
    ) -> MLXArray {
        guard strength > 0 else { return latents }

        let noise = seededNoise(shape: latents.shape, seed: seed)
        let noised = (1.0 - strength) * latents + strength * noise
        return latents * (1.0 - mask) + noised * mask
    }

    /// Blends a denoised result with the unmasked original.
    ///
    /// Formula:
    /// ```
    /// output = original * (1 - mask) + denoised * mask
    /// ```
    ///
    /// - Parameters:
    ///   - original:  Original latents of shape `[1, T, D]`.
    ///   - denoised:  Denoised latents of shape `[1, T, D]`.
    ///   - mask:      Binary mask of shape `[1, T, 1]` or `[1, T, D]`.
    /// - Returns: Composited latents preserving the unmasked region.
    public static func composite(
        original: MLXArray,
        denoised: MLXArray,
        mask: MLXArray
    ) -> MLXArray {
        original * (1.0 - mask) + denoised * mask
    }

    // MARK: - Private Helpers

    /// Generates standard normal noise of the given shape.
    /// When a seed is provided the result is deterministic across calls.
    private static func seededNoise(shape: [Int], seed: Int?) -> MLXArray {
        if let seed = seed {
            let key = MLXRandom.key(UInt64(seed))
            return MLXRandom.normal(shape, key: key)
        }
        return MLXRandom.normal(shape)
    }
}
