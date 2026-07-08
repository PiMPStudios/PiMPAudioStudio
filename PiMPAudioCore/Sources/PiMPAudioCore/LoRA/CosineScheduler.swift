import Foundation

/// Cosine annealing learning rate scheduler with linear warmup.
///
/// - During warmup (steps 0..<warmupSteps): LR ramps linearly from
///   baseLR / warmupSteps to baseLR
/// - After warmup (steps warmupSteps..<totalSteps): LR decays via cosine from baseLR to 0
public struct CosineScheduler: Sendable {
    public let baseLR: Float
    public let warmupFraction: Float
    public let totalSteps: Int

    /// Number of warmup steps (computed from warmupFraction * totalSteps).
    public var warmupSteps: Int {
        Int(Float(totalSteps) * warmupFraction)
    }

    /// - Parameters:
    ///   - baseLR: Peak learning rate reached at end of warmup.
    ///   - warmupFraction: Fraction of total steps used for linear warmup. Default 0.10.
    ///   - totalSteps: Total number of optimizer steps across all epochs.
    public init(baseLR: Float, warmupFraction: Float = 0.10, totalSteps: Int) {
        self.baseLR = baseLR
        self.warmupFraction = max(0, min(1, warmupFraction))
        self.totalSteps = max(1, totalSteps)
    }

    /// Returns the learning rate for the given global step.
    ///
    /// - Parameter step: Current global step (0-indexed).
    /// - Returns: Learning rate for this step.
    public func learningRate(atStep step: Int) -> Float {
        let clampedStep = max(0, min(step, totalSteps - 1))
        let warmup = warmupSteps

        if warmup > 0 && clampedStep < warmup {
            return baseLR * Float(clampedStep + 1) / Float(warmup)
        }

        // Cosine decay: baseLR → 0
        let decaySteps = totalSteps - warmup
        guard decaySteps > 0 else { return baseLR }

        let progress = Float(clampedStep - warmup) / Float(decaySteps)
        return baseLR * 0.5 * (1.0 + cos(Float.pi * progress))
    }
}
