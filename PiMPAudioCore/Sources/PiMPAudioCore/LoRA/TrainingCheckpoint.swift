import Foundation

/// Persisted training state for pause/resume support.
///
/// Stores everything needed to reconstruct the exact training state:
/// epoch, step, weights snapshot, optimizer state, and the original config
/// (which includes totalSteps for scheduler reconstruction).
public struct TrainingCheckpoint: Codable, Sendable, Equatable {
    /// Current epoch (1-indexed) when checkpoint was saved.
    public let epoch: Int

    /// Global step counter across all epochs.
    public let globalStep: Int

    /// Total steps planned (epochs * stepsPerEpoch). Used to reconstruct CosineScheduler.
    public let totalSteps: Int

    /// Best loss seen so far (nil if no complete epoch finished).
    public let bestLoss: Float?

    /// Full loss history (one entry per step) for loss curve reconstruction.
    public let lossHistory: [Float]

    /// Validation loss points recorded at epoch boundaries.
    public let validationHistory: [TrainingValidationPoint]

    /// URL to the saved weights snapshot (.safetensors).
    public let weightsURL: URL

    /// URL to the serialized optimizer state.
    public let optimizerStateURL: URL

    /// The full training config used (needed for resume to know rank, targets, etc.).
    public let config: LoRAConfig

    /// Base model ID at time of training (for mismatch detection on resume).
    public let baseModelID: String

    public init(
        epoch: Int,
        globalStep: Int,
        totalSteps: Int,
        bestLoss: Float?,
        lossHistory: [Float],
        validationHistory: [TrainingValidationPoint] = [],
        weightsURL: URL,
        optimizerStateURL: URL,
        config: LoRAConfig,
        baseModelID: String
    ) {
        self.epoch = epoch
        self.globalStep = globalStep
        self.totalSteps = totalSteps
        self.bestLoss = bestLoss
        self.lossHistory = lossHistory
        self.validationHistory = validationHistory
        self.weightsURL = weightsURL
        self.optimizerStateURL = optimizerStateURL
        self.config = config
        self.baseModelID = baseModelID
    }

    enum CodingKeys: String, CodingKey {
        case epoch, globalStep, totalSteps, bestLoss, lossHistory, validationHistory
        case weightsURL, optimizerStateURL, config, baseModelID
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        epoch = try container.decode(Int.self, forKey: .epoch)
        globalStep = try container.decode(Int.self, forKey: .globalStep)
        totalSteps = try container.decode(Int.self, forKey: .totalSteps)
        bestLoss = try container.decodeIfPresent(Float.self, forKey: .bestLoss)
        lossHistory = try container.decode([Float].self, forKey: .lossHistory)
        validationHistory = try container.decodeIfPresent(
            [TrainingValidationPoint].self,
            forKey: .validationHistory
        ) ?? []
        weightsURL = try container.decode(URL.self, forKey: .weightsURL)
        optimizerStateURL = try container.decode(URL.self, forKey: .optimizerStateURL)
        config = try container.decode(LoRAConfig.self, forKey: .config)
        baseModelID = try container.decode(String.self, forKey: .baseModelID)
    }
}

/// Errors related to checkpoint operations.
public enum CheckpointError: Error, LocalizedError {
    case filesMissing(String)
    case modelMismatch(expected: String, actual: String)

    public var errorDescription: String? {
        switch self {
        case .filesMissing(let detail):
            return "Checkpoint files missing: \(detail)"
        case .modelMismatch(let expected, let actual):
            return "Model mismatch: checkpoint was trained on '\(expected)' but current model is '\(actual)'"
        }
    }
}
