import Testing
import MLX
import MLXRandom
@testable import AnvilCore

// MARK: - LatentBlender Tests

@Suite("LatentBlender")
struct LatentBlenderTests {

    // Common shapes matching ACE-Step defaults
    let T = 10
    let D = 64

    // MARK: 1 — Zero strength returns original unchanged

    @Test("Zero strength returns original latents unchanged")
    func zeroStrengthReturnsOriginal() {
        let latents = MLXRandom.normal([1, 10, 64])
        let mask = MLXArray(
            [Float](repeating: 1.0, count: 10),
            [1, 10, 1]
        )
        eval(latents, mask)

        let result = LatentBlender.injectNoise(latents: latents, mask: mask, strength: 0.0, seed: 42)
        eval(result)

        let diff = abs(result - latents).sum()
        eval(diff)
        #expect(diff.item(Float.self) < 1e-6)
    }

    // MARK: 2 — Full strength on full mask replaces with noised values (sum > 0)

    @Test("Full strength on full mask produces non-zero noised output")
    func fullStrengthFullMaskProducesNoise() {
        // Start from all-zeros latents so any noise contribution is detectable
        let latents = MLXArray.zeros([1, 10, 64])
        let mask = MLXArray(
            [Float](repeating: 1.0, count: 10),
            [1, 10, 1]
        )
        eval(latents, mask)

        let result = LatentBlender.injectNoise(latents: latents, mask: mask, strength: 1.0, seed: 7)
        eval(result)

        let sumAbs = abs(result).sum()
        eval(sumAbs)
        // Pure noise (strength = 1) from zero latents should produce non-zero output
        #expect(sumAbs.item(Float.self) > 0)
    }

    // MARK: 3 — Unmasked region is preserved exactly

    @Test("Unmasked region is preserved exactly after noise injection")
    func preservesUnmaskedRegion() {
        let latents = MLXRandom.normal([1, 10, 64], key: MLXRandom.key(99))
        // Mask only the first 5 frames; last 5 are unmasked (0)
        var maskData = [Float](repeating: 0.0, count: 10)
        for i in 0..<5 { maskData[i] = 1.0 }
        let mask = MLXArray(maskData, [1, 10, 1])
        eval(latents, mask)

        let result = LatentBlender.injectNoise(latents: latents, mask: mask, strength: 0.8, seed: 1)
        eval(result)

        // Slice unmasked frames [5..10) and compare to original
        let origUnmasked = latents[0, 5..., 0...]
        let resultUnmasked = result[0, 5..., 0...]
        eval(origUnmasked, resultUnmasked)

        let diff = abs(origUnmasked - resultUnmasked).sum()
        eval(diff)
        #expect(diff.item(Float.self) < 1e-6)
    }

    // MARK: 4 — Composite blends correctly

    @Test("Composite: unmasked uses original, masked uses denoised")
    func compositeBlendIsCorrect() {
        // Build known arrays: original all 1s, denoised all 2s
        let original = MLXArray(
            [Float](repeating: 1.0, count: 1 * 4 * 8),
            [1, 4, 8]
        )
        let denoised = MLXArray(
            [Float](repeating: 2.0, count: 1 * 4 * 8),
            [1, 4, 8]
        )
        // Mask: first 2 frames regenerated (1), last 2 preserved (0)
        let maskData: [Float] = [1.0, 1.0, 0.0, 0.0]
        let mask = MLXArray(maskData, [1, 4, 1])
        eval(original, denoised, mask)

        let output = LatentBlender.composite(original: original, denoised: denoised, mask: mask)
        eval(output)

        let flat = output.reshaped([-1]).asArray(Float.self)

        // First 2 frames (masked) → denoised value 2.0
        let maskedValues = Array(flat[0..<(2 * 8)])
        #expect(maskedValues.allSatisfy { abs($0 - 2.0) < 1e-6 })

        // Last 2 frames (unmasked) → original value 1.0
        let unmaskedValues = Array(flat[(2 * 8)...])
        #expect(unmaskedValues.allSatisfy { abs($0 - 1.0) < 1e-6 })
    }

    // MARK: 5 — Seeded noise is deterministic

    @Test("Same seed produces identical noise injection results")
    func seededNoiseIsDeterministic() {
        let latents = MLXArray.zeros([1, 10, 64])
        let mask = MLXArray(
            [Float](repeating: 1.0, count: 10),
            [1, 10, 1]
        )
        eval(latents, mask)

        let result1 = LatentBlender.injectNoise(latents: latents, mask: mask, strength: 0.5, seed: 42)
        let result2 = LatentBlender.injectNoise(latents: latents, mask: mask, strength: 0.5, seed: 42)
        eval(result1, result2)

        let diff = abs(result1 - result2).sum()
        eval(diff)
        #expect(diff.item(Float.self) < 1e-6)
    }
}
