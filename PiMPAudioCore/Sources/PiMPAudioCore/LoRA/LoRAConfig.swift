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
    /// Low-rank dimension. Higher = more expressive, more memory. Typical: 8, 16, 32, 64.
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

    /// Number of DiT decoder layers to train from the end of the stack.
    ///
    /// `nil` means the caller did not specify a local training footprint. The
    /// training actor may then use its legacy default, while imported adapters
    /// can still be injected from their saved weight paths.
    public var trainingLayerCount: Int?

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
        gradientAccumulationSteps: Int = 1,
        trainingLayerCount: Int? = nil
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
        self.trainingLayerCount = trainingLayerCount
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

/// Adapter implementation stored in the registry.
public enum AdapterKind: String, Codable, Sendable, Equatable, Hashable {
    case lora
    case lokr
}

/// Configuration for experimental LoKR adapter training and injection.
public struct LoKRConfig: Codable, Sendable, Equatable, Hashable {
    /// Kronecker factor block dimension. Side-Step calls this linear dim.
    public var linearDim: Int

    /// Scaling factor. The adapter contribution is scaled by alpha/linearDim.
    public var alpha: Float

    /// Dropout rate applied to adapter path during training. 0.0 = no dropout.
    public var dropout: Float

    /// Which module categories to inject LoKR into.
    public var targetLayers: Set<TargetLayer>

    /// Optional exact module paths to inject.
    public var exactInjectionPatterns: [String]?

    /// Learning rate for AdamW optimizer.
    public var learningRate: Float

    /// Maximum number of training epochs.
    public var maxEpochs: Int

    /// Training batch size. Default 1 (memory constrained on-device).
    public var batchSize: Int

    /// Fraction of total steps used for linear LR warmup.
    public var warmupFraction: Float

    /// Number of mini-batches to accumulate gradients over before optimizer step.
    public var gradientAccumulationSteps: Int

    /// Number of DiT decoder layers to train from the end of the stack.
    public var trainingLayerCount: Int?

    public init(
        linearDim: Int = 64,
        alpha: Float = 64.0,
        dropout: Float = 0.0,
        targetLayers: Set<TargetLayer> = [.selfAttention, .crossAttention],
        exactInjectionPatterns: [String]? = nil,
        learningRate: Float = 1e-4,
        maxEpochs: Int = 10,
        batchSize: Int = 1,
        warmupFraction: Float = 0.10,
        gradientAccumulationSteps: Int = 1,
        trainingLayerCount: Int? = nil
    ) {
        self.linearDim = linearDim
        self.alpha = alpha
        self.dropout = dropout
        self.targetLayers = targetLayers
        self.exactInjectionPatterns = exactInjectionPatterns
        self.learningRate = learningRate
        self.maxEpochs = maxEpochs
        self.batchSize = batchSize
        self.warmupFraction = warmupFraction
        self.gradientAccumulationSteps = gradientAccumulationSteps
        self.trainingLayerCount = trainingLayerCount
    }

    public var injectionPatterns: [String] {
        if let exactInjectionPatterns, !exactInjectionPatterns.isEmpty {
            return exactInjectionPatterns
        }
        return targetLayers.sorted(by: { $0.rawValue < $1.rawValue })
            .flatMap(\.defaultPatterns)
    }

    public var loraCompatibleConfig: LoRAConfig {
        LoRAConfig(
            rank: linearDim,
            alpha: alpha,
            dropout: dropout,
            targetLayers: targetLayers,
            exactInjectionPatterns: exactInjectionPatterns,
            learningRate: learningRate,
            maxEpochs: maxEpochs,
            batchSize: batchSize,
            warmupFraction: warmupFraction,
            gradientAccumulationSteps: gradientAccumulationSteps,
            trainingLayerCount: trainingLayerCount
        )
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

    /// Adapter implementation used by this registry entry.
    public let kind: AdapterKind

    /// LoKR-specific config when `kind == .lokr`.
    public let lokrConfig: LoKRConfig?

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

    /// Exact module paths that were trained for this adapter.
    ///
    /// Local training fills this with paths such as
    /// `layers.28.selfAttn.qProj`. Older adapters may leave it nil; callers
    /// can derive the same information from saved weight keys.
    public var trainedPaths: [String]?

    /// Decoder layer indices covered by `trainedPaths`.
    public var trainedLayerIndices: [Int]?

    public init(
        id: String,
        name: String,
        config: LoRAConfig,
        kind: AdapterKind = .lora,
        lokrConfig: LoKRConfig? = nil,
        baseModelID: String,
        baseModelCategory: ModelCategory,
        createdAt: Date,
        sampleCount: Int,
        finalLoss: Float?,
        source: AdapterSource = .local,
        remoteRepoURL: URL? = nil,
        trainedPaths: [String]? = nil,
        trainedLayerIndices: [Int]? = nil
    ) {
        self.id = id
        self.name = name
        self.config = config
        self.kind = kind
        self.lokrConfig = lokrConfig
        self.baseModelID = baseModelID
        self.baseModelCategory = baseModelCategory
        self.createdAt = createdAt
        self.sampleCount = sampleCount
        self.finalLoss = finalLoss
        self.source = source
        self.remoteRepoURL = remoteRepoURL
        self.trainedPaths = trainedPaths
        self.trainedLayerIndices = trainedLayerIndices
    }

    // Custom Codable for backward compatibility — existing adapters without
    // `source` field decode as .local.
    enum CodingKeys: String, CodingKey {
        case id, name, config, kind, lokrConfig, baseModelID, baseModelCategory
        case createdAt, sampleCount, finalLoss, source, remoteRepoURL
        case trainedPaths, trainedLayerIndices
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(String.self, forKey: .id)
        name = try container.decode(String.self, forKey: .name)
        config = try container.decode(LoRAConfig.self, forKey: .config)
        kind = try container.decodeIfPresent(AdapterKind.self, forKey: .kind) ?? .lora
        lokrConfig = try container.decodeIfPresent(LoKRConfig.self, forKey: .lokrConfig)
        baseModelID = try container.decode(String.self, forKey: .baseModelID)
        baseModelCategory = try container.decode(ModelCategory.self, forKey: .baseModelCategory)
        createdAt = try container.decode(Date.self, forKey: .createdAt)
        sampleCount = try container.decode(Int.self, forKey: .sampleCount)
        finalLoss = try container.decodeIfPresent(Float.self, forKey: .finalLoss)
        source = try container.decodeIfPresent(AdapterSource.self, forKey: .source) ?? .local
        remoteRepoURL = try container.decodeIfPresent(URL.self, forKey: .remoteRepoURL)
        trainedPaths = try container.decodeIfPresent([String].self, forKey: .trainedPaths)
        trainedLayerIndices = try container.decodeIfPresent([Int].self, forKey: .trainedLayerIndices)
    }
}

public extension LoRAAdapter {
    /// Trained layer indices from metadata, or parsed from trained paths.
    var effectiveTrainedLayerIndices: [Int]? {
        if let trainedLayerIndices, !trainedLayerIndices.isEmpty {
            return Array(Set(trainedLayerIndices)).sorted()
        }
        guard let trainedPaths, !trainedPaths.isEmpty else { return nil }
        let parsed = Self.layerIndices(from: trainedPaths)
        return parsed.isEmpty ? nil : parsed
    }

    /// Exact paths from metadata, if available.
    var effectiveTrainedPaths: [String]? {
        guard let trainedPaths, !trainedPaths.isEmpty else { return nil }
        return trainedPaths.sorted()
    }

    /// Parse `layers.N.*` module paths into sorted unique layer indices.
    static func layerIndices(from paths: [String]) -> [Int] {
        let indices = paths.compactMap { path -> Int? in
            let parts = path.split(separator: ".")
            guard parts.count >= 3, parts[0] == "layers" else { return nil }
            return Int(parts[1])
        }
        return Array(Set(indices)).sorted()
    }

    /// Derive trained module paths from adapter weight keys.
    static func trainedPaths(fromWeightKeys keys: [String], kind: AdapterKind) -> [String] {
        let suffixes: [String]
        switch kind {
        case .lora:
            suffixes = [".loraA", ".loraB"]
        case .lokr:
            suffixes = [".lokrW1", ".lokrW2"]
        }

        var paths = Set<String>()
        for key in keys {
            guard let suffix = suffixes.first(where: { key.hasSuffix($0) }) else {
                continue
            }
            paths.insert(String(key.dropLast(suffix.count)))
        }
        return paths.sorted()
    }

    /// Checks whether this adapter can run against the active base model.
    func isCompatible(with activeBaseID: String?, activeSpec: ModelSpec?) -> Bool {
        guard let activeBaseID else { return true }
        guard baseModelCategory == .aceStepDiT else {
            return baseModelID == activeBaseID
        }
        if baseModelID == activeBaseID { return true }

        guard let adapterVariant = Self.canonicalACEVariant(baseModelID),
              let activeVariant = Self.canonicalACEVariant(activeBaseID, spec: activeSpec) else {
            return false
        }
        return adapterVariant == activeVariant
    }

    private static func canonicalACEVariant(_ id: String, spec: ModelSpec? = nil) -> String? {
        let values = [
            id,
            spec?.id,
            spec?.displayName,
            spec?.huggingFaceRepo,
            spec?.variantFamily,
        ]
        let haystack = values
            .compactMap { $0?.lowercased() }
            .joined(separator: " ")

        if haystack.contains("xl") || haystack.contains("ace-step-1.5-xl") {
            return "ace-step-dit-xl"
        }
        if haystack.contains("turbo") {
            return "ace-step-dit-turbo"
        }
        if haystack.contains("sft") {
            return "ace-step-dit-sft"
        }
        if haystack.contains("base") || haystack.contains("full") {
            return "ace-step-dit-base"
        }
        return nil
    }
}
