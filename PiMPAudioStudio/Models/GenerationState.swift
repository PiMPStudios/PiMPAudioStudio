import Foundation

enum GenerationState: Equatable, Sendable {
    case idle
    case downloadingModel(progress: Float)
    case loadingWeights(progress: Float)
    case conditioning
    case tokenizingSource
    case lmAudioCodeConditioning
    case generating(step: Int, totalSteps: Int, eta: TimeInterval)
    case decoding
    case complete
    case error(String)
    case cancelled
    /// Models are being unloaded after cancel/release. Blocks new submissions
    /// until the owning release operation finishes so a new generation cannot
    /// start while the previous run's weights are still being torn down.
    case releasing
}
