import Foundation

/// Configuration for LoRA adapter training and injection.
public struct LoRAConfig: Codable, Sendable, Equatable {
    /// Low-rank dimension. Higher = more expressive, more memory. Typical: 8, 16, 32.
    public let rank: Int

    /// Scaling factor. The adapter contribution is scaled by alpha/rank.
    public let alpha: Float

    /// Dropout rate applied to adapter path during training. 0.0 = no dropout.
    public let dropout: Float

    /// Layer name patterns to inject LoRA into (e.g. "selfAttn.qProj").
    public let targetLayers: [String]

    /// Learning rate for AdamW optimizer.
    public let learningRate: Float

    /// Number of training epochs.
    public let epochs: Int

    /// Training batch size. Default 1 (memory constrained on-device).
    public let batchSize: Int

    public init(
        rank: Int = 16,
        alpha: Float = 16.0,
        dropout: Float = 0.0,
        targetLayers: [String] = LoRAConfig.defaultTargetLayers,
        learningRate: Float = 1e-4,
        epochs: Int = 10,
        batchSize: Int = 1
    ) {
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.targetLayers = targetLayers
        self.learningRate = learningRate
        self.epochs = epochs
        self.batchSize = batchSize
    }

    /// Default LoRA target layers for ACE-Step DiT: q and v projections
    /// in both self-attention and cross-attention across all 24 layers.
    public static let defaultTargetLayers: [String] = [
        "selfAttn.qProj",
        "selfAttn.vProj",
        "crossAttn.qProj",
        "crossAttn.vProj",
    ]
}

/// A saved LoRA adapter with metadata.
public struct LoRAAdapter: Codable, Sendable, Identifiable, Equatable {
    /// Directory name used as identifier.
    public let id: String

    /// User-facing display name.
    public let name: String

    /// The LoRA config used during training.
    public let config: LoRAConfig

    /// ID of the base model this adapter was trained against.
    public let baseModelID: String

    /// Category of the base model for fast compatibility filtering.
    public let baseModelCategory: ModelCategory

    /// When training completed.
    public let createdAt: Date

    /// Number of training samples used.
    public let sampleCount: Int

    /// Final training loss (nil if training was stopped early).
    public let finalLoss: Float?

    public init(
        id: String,
        name: String,
        config: LoRAConfig,
        baseModelID: String,
        baseModelCategory: ModelCategory,
        createdAt: Date,
        sampleCount: Int,
        finalLoss: Float?
    ) {
        self.id = id
        self.name = name
        self.config = config
        self.baseModelID = baseModelID
        self.baseModelCategory = baseModelCategory
        self.createdAt = createdAt
        self.sampleCount = sampleCount
        self.finalLoss = finalLoss
    }
}
