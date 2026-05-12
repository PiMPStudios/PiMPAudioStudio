import Foundation

enum GenerationState: Equatable, Sendable {
    case idle
    case downloadingModel(progress: Float)
    case loadingWeights(progress: Float)
    case conditioning
    case tokenizingSource
    case runningLM
    case detokenizing
    case generating(step: Int, totalSteps: Int, eta: TimeInterval)
    case decoding
    case complete
    case error(String)
    case cancelled
}
