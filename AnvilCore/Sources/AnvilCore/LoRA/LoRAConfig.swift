import Foundation

/// High-level categories of DiT layers that can receive LoRA adapters.
public enum TargetLayer: String, Codable, CaseIterable, Sendable, Hashable {
    case selfAttention = "self_attn"
    case crossAttention = "cross_attn"
    case mlp = "mlp"

    /// Maps to the property name pattern used by LoRAInjector (e.g. "selfAttn").
    public var moduleName: String {
        switch self {
        case .selfAttention: return "selfAttn"
        case .crossAttention: return "crossAttn"
        case .mlp: return "mlp"
        }
    }

    /// Returns the injection patterns for this target layer category.
    /// E.g. selfAttention → ["selfAttn.qProj", "selfAttn.vProj"]
    public var defaultPatterns: [String] {
        switch self {
        case .selfAttention:
            return ["selfAttn.qProj", "selfAttn.vProj"]
        case .crossAttention:
            return ["crossAttn.qProj", "crossAttn.vProj"]
        case .mlp:
            return ["mlp.gateProj", "mlp.downProj"]
        }
    }
}

/// Configuration for LoRA adapter training and injection.
public struct LoRAConfig: Codable, Sendable, Equatable, Hashable {
    /// Low-rank dimension. Higher = more expressive, more memory. Typical: 8, 16, 32.
    public var rank: Int

    /// Scaling factor. The adapter contribution is scaled by alpha/rank.
    public var alpha: Float

    /// Dropout rate applied to adapter path during training. 0.0 = no dropout.
    public var dropout: Float

    /// Which module categories to inject LoRA into.
    public var targetLayers: Set<TargetLayer>

    /// Optional exact module paths to inject.
    ///
    /// Local training keeps this nil and uses the conservative defaults for each
    /// `TargetLayer`. Imported PEFT/cloud adapters can set this to preserve the
    /// exact projection surface they were trained with, such as q/k/v/o on both
    /// self-attention and cross-attention.
    public var exactInjectionPatterns: [String]?

    /// Learning rate for AdamW optimizer (peak LR when using scheduler).
    public var learningRate: Float

    /// Maximum number of training epochs.
    public var maxEpochs: Int

    /// Training batch size. Default 1 (memory constrained on-device).
    public var batchSize: Int

    /// Fraction of total steps used for linear LR warmup. Default 0.10.
    public var warmupFraction: Float

    /// Number of mini-batches to accumulate gradients over before optimizer step. Default 1.
    public var gradientAccumulationSteps: Int

    public init(
        rank: Int = 16,
        alpha: Float = 16.0,
        dropout: Float = 0.0,
        targetLayers: Set<TargetLayer> = [.selfAttention, .crossAttention],
        exactInjectionPatterns: [String]? = nil,
        learningRate: Float = 1e-4,
        maxEpochs: Int = 10,
        batchSize: Int = 1,
        warmupFraction: Float = 0.10,
        gradientAccumulationSteps: Int = 1
    ) {
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.targetLayers = targetLayers
        self.exactInjectionPatterns = exactInjectionPatterns
        self.learningRate = learningRate
        self.maxEpochs = maxEpochs
        self.batchSize = batchSize
        self.warmupFraction = warmupFraction
        self.gradientAccumulationSteps = gradientAccumulationSteps
    }

    /// Expands targetLayers into the full list of injection pattern strings
    /// (e.g. ["selfAttn.qProj", "selfAttn.vProj", "crossAttn.qProj", ...]).
    public var injectionPatterns: [String] {
        if let exactInjectionPatterns, !exactInjectionPatterns.isEmpty {
            return exactInjectionPatterns
        }
        return targetLayers.sorted(by: { $0.rawValue < $1.rawValue })
            .flatMap(\.defaultPatterns)
    }
}

/// A saved LoRA adapter with metadata.
public struct LoRAAdapter: Codable, Sendable, Identifiable, Equatable, Hashable {
    /// Directory name used as identifier.
    public var id: String

    /// User-facing display name.
    public var name: String

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

    /// Where this adapter came from (defaults to .local for backward compat).
    public var source: AdapterSource

    /// Original HuggingFace repo URL if imported from HF.
    public var remoteRepoURL: URL?

    public init(
        id: String,
        name: String,
        config: LoRAConfig,
        baseModelID: String,
        baseModelCategory: ModelCategory,
        createdAt: Date,
        sampleCount: Int,
        finalLoss: Float?,
        source: AdapterSource = .local,
        remoteRepoURL: URL? = nil
    ) {
        self.id = id
        self.name = name
        self.config = config
        self.baseModelID = baseModelID
        self.baseModelCategory = baseModelCategory
        self.createdAt = createdAt
        self.sampleCount = sampleCount
        self.finalLoss = finalLoss
        self.source = source
        self.remoteRepoURL = remoteRepoURL
    }

    // Custom Codable for backward compatibility — existing adapters without
    // `source` field decode as .local.
    enum CodingKeys: String, CodingKey {
        case id, name, config, baseModelID, baseModelCategory
        case createdAt, sampleCount, finalLoss, source, remoteRepoURL
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(String.self, forKey: .id)
        name = try container.decode(String.self, forKey: .name)
        config = try container.decode(LoRAConfig.self, forKey: .config)
        baseModelID = try container.decode(String.self, forKey: .baseModelID)
        baseModelCategory = try container.decode(ModelCategory.self, forKey: .baseModelCategory)
        createdAt = try container.decode(Date.self, forKey: .createdAt)
        sampleCount = try container.decode(Int.self, forKey: .sampleCount)
        finalLoss = try container.decodeIfPresent(Float.self, forKey: .finalLoss)
        source = try container.decodeIfPresent(AdapterSource.self, forKey: .source) ?? .local
        remoteRepoURL = try container.decodeIfPresent(URL.self, forKey: .remoteRepoURL)
    }
}
