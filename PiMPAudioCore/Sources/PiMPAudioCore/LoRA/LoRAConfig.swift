import Foundation

/// High-level categories of DiT layers that can receive LoRA adapters.
public enum TargetLayer: String, Codable, CaseIterable, Sendable, Hashable {
    case selfAttention = "self_attn"
    case crossAttention = "cross_attn"
    case mlp

    /// Maps to the property name pattern used by LoRAInjector (e.g. "selfAttn").
    public var moduleName: String {
        switch self {
        case .selfAttention: "selfAttn"
        case .crossAttention: "crossAttn"
        case .mlp: "mlp"
        }
    }

    /// Returns the injection patterns for this target layer category.
    /// E.g. selfAttention → ["selfAttn.qProj", "selfAttn.vProj"]
    ///
    /// These are deliberately conservative memory-saving defaults for
    /// on-device training (q/v only for attention; gate/down only for MLP —
    /// `mlp.upProj` is intentionally excluded, so the MLP default covers
    /// only part of the SwiGLU block). Upstream ACE-Step training presets
    /// use q/k/v/o attention and no MLP. Use
    /// `LoRAConfig.exactInjectionPatterns` (e.g.
    /// `fullAttentionInjectionPatterns`, or adding `"mlp.upProj"`) for a
    /// larger surface — the injector and PEFT remapper support all seven
    /// projections.
    public var defaultPatterns: [String] {
        switch self {
        case .selfAttention:
            ["selfAttn.qProj", "selfAttn.vProj"]
        case .crossAttention:
            ["crossAttn.qProj", "crossAttn.vProj"]
        case .mlp:
            ["mlp.gateProj", "mlp.downProj"]
        }
    }
}

/// Configuration for LoRA adapter training and injection.
public struct LoRAConfig: Codable, Sendable, Equatable, Hashable {
    public static let defaultTrainingSeed: UInt64 = 42

    /// Low-rank dimension. Higher = more expressive, more memory. Typical: 8, 16, 32, 64, 96.
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

    /// Training batch size for remote/cloud training jobs.
    ///
    /// NOTE: the local Swift trainers (`LoRATrainer`/`LoKRTrainer`) always
    /// process one sample at a time (microbatch = 1) and IGNORE this value;
    /// use `gradientAccumulationSteps` to control the effective batch size
    /// for on-device training.
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

    /// Seed for local trainer shuffling, timestep/noise sampling, and CFG dropout.
    public var trainingSeed: UInt64?

    /// Full attention projection surface used by higher-capacity ACE-Step PEFT
    /// recipes: q/k/v/o on both self-attention and cross-attention.
    public static let fullAttentionInjectionPatterns: [String] = [
        "selfAttn.qProj",
        "selfAttn.kProj",
        "selfAttn.vProj",
        "selfAttn.oProj",
        "crossAttn.qProj",
        "crossAttn.kProj",
        "crossAttn.vProj",
        "crossAttn.oProj",
    ]

    public init(
        rank: Int = 16,
        alpha: Float? = nil,
        dropout: Float = 0.0,
        targetLayers: Set<TargetLayer> = [.selfAttention, .crossAttention],
        exactInjectionPatterns: [String]? = nil,
        learningRate: Float = 1e-4,
        maxEpochs: Int = 10,
        batchSize: Int = 1,
        warmupFraction: Float = 0.10,
        gradientAccumulationSteps: Int = 1,
        trainingLayerCount: Int? = nil,
        trainingSeed: UInt64? = Self.defaultTrainingSeed
    ) {
        self.rank = rank
        self.alpha = alpha ?? Float(rank)
        self.dropout = dropout
        self.targetLayers = targetLayers
        self.exactInjectionPatterns = exactInjectionPatterns
        self.learningRate = learningRate
        self.maxEpochs = maxEpochs
        self.batchSize = batchSize
        self.warmupFraction = warmupFraction
        self.gradientAccumulationSteps = gradientAccumulationSteps
        self.trainingLayerCount = trainingLayerCount
        self.trainingSeed = trainingSeed
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

    /// Runtime contribution scale applied to this adapter's learned delta.
    public var scaling: Float {
        alpha / Float(max(1, rank))
    }

    public var effectiveTrainingSeed: UInt64 {
        trainingSeed ?? Self.defaultTrainingSeed
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

    /// Training batch size for remote/cloud training jobs.
    ///
    /// NOTE: the local Swift trainers always process one sample at a time
    /// (microbatch = 1) and IGNORE this value; use
    /// `gradientAccumulationSteps` to control the effective batch size.
    public var batchSize: Int

    /// Fraction of total steps used for linear LR warmup.
    public var warmupFraction: Float

    /// Number of mini-batches to accumulate gradients over before optimizer step.
    public var gradientAccumulationSteps: Int

    /// Number of DiT decoder layers to train from the end of the stack.
    public var trainingLayerCount: Int?

    /// Seed for local trainer shuffling, timestep/noise sampling, and CFG dropout.
    public var trainingSeed: UInt64?

    /// Enable DoRA-style weight decomposition for LoKR.
    ///
    /// New local LoKR runs default this on to match the ACE-Step 1.5 LoKR
    /// training surface. Adapters saved before this field existed decode as
    /// `false` because their safetensors do not contain magnitude vectors.
    public var weightDecompose: Bool

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
        trainingLayerCount: Int? = nil,
        trainingSeed: UInt64? = LoRAConfig.defaultTrainingSeed,
        weightDecompose: Bool = true
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
        self.trainingSeed = trainingSeed
        self.weightDecompose = weightDecompose
    }

    public init(loraCompatibleConfig config: LoRAConfig, weightDecompose: Bool = false) {
        self.init(
            linearDim: config.rank,
            alpha: config.alpha,
            dropout: config.dropout,
            targetLayers: config.targetLayers,
            exactInjectionPatterns: config.exactInjectionPatterns,
            learningRate: config.learningRate,
            maxEpochs: config.maxEpochs,
            batchSize: config.batchSize,
            warmupFraction: config.warmupFraction,
            gradientAccumulationSteps: config.gradientAccumulationSteps,
            trainingLayerCount: config.trainingLayerCount,
            trainingSeed: config.trainingSeed,
            weightDecompose: weightDecompose
        )
    }

    enum CodingKeys: String, CodingKey {
        case linearDim, alpha, dropout, targetLayers, exactInjectionPatterns
        case learningRate, maxEpochs, batchSize, warmupFraction
        case gradientAccumulationSteps, trainingLayerCount, trainingSeed, weightDecompose
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        linearDim = try container.decode(Int.self, forKey: .linearDim)
        alpha = try container.decode(Float.self, forKey: .alpha)
        dropout = try container.decode(Float.self, forKey: .dropout)
        targetLayers = try container.decode(Set<TargetLayer>.self, forKey: .targetLayers)
        exactInjectionPatterns = try container.decodeIfPresent([String].self, forKey: .exactInjectionPatterns)
        learningRate = try container.decode(Float.self, forKey: .learningRate)
        maxEpochs = try container.decode(Int.self, forKey: .maxEpochs)
        batchSize = try container.decode(Int.self, forKey: .batchSize)
        warmupFraction = try container.decode(Float.self, forKey: .warmupFraction)
        gradientAccumulationSteps = try container.decode(Int.self, forKey: .gradientAccumulationSteps)
        trainingLayerCount = try container.decodeIfPresent(Int.self, forKey: .trainingLayerCount)
        trainingSeed = try container.decodeIfPresent(UInt64.self, forKey: .trainingSeed)
        weightDecompose = try container.decodeIfPresent(Bool.self, forKey: .weightDecompose) ?? false
    }

    public var injectionPatterns: [String] {
        if let exactInjectionPatterns, !exactInjectionPatterns.isEmpty {
            return exactInjectionPatterns
        }
        return targetLayers.sorted(by: { $0.rawValue < $1.rawValue })
            .flatMap(\.defaultPatterns)
    }

    /// Runtime contribution scale applied to this adapter's learned delta.
    public var scaling: Float {
        alpha / Float(max(1, linearDim))
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
            trainingLayerCount: trainingLayerCount,
            trainingSeed: trainingSeed
        )
    }

    public var effectiveTrainingSeed: UInt64 {
        trainingSeed ?? LoRAConfig.defaultTrainingSeed
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

    /// Custom Codable for backward compatibility — existing adapters without
    /// `source` field decode as .local.
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
        let suffixes: [String] = switch kind {
        case .lora:
            [".loraA", ".loraB"]
        case .lokr:
            [".lokrW1", ".lokrW2", ".lokrMagnitude"]
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
        if baseModelID == activeBaseID {
            return true
        }

        guard let adapterVariant = Self.canonicalACEVariant(baseModelID),
              let activeVariant = Self.canonicalACEVariant(activeBaseID, spec: activeSpec) else {
            return false
        }
        if adapterVariant == "ace-step-dit-xl", activeVariant.hasPrefix("ace-step-dit-xl") {
            return true
        }
        if activeVariant == "ace-step-dit-xl", adapterVariant.hasPrefix("ace-step-dit-xl") {
            return true
        }
        return adapterVariant == activeVariant
    }

    /// Returns true when a base model identifier or spec describes an ACE-Step XL DiT.
    ///
    /// This intentionally uses token boundaries instead of raw substring matching so
    /// unrelated names like "flexxl" do not look like XL model variants.
    static func isACEBaseModelXL(_ id: String, spec: ModelSpec? = nil) -> Bool {
        let tokens = Self.variantTokens(id, spec: spec)
        let mentionsACE = tokens.contains("acestep") || (tokens.contains("ace") && tokens.contains("step"))
        return mentionsACE && tokens.contains("xl")
    }

    private static func canonicalACEVariant(_ id: String, spec: ModelSpec? = nil) -> String? {
        let tokens = Self.variantTokens(id, spec: spec)
        let haystack = tokens.joined(separator: " ")

        if isACEBaseModelXL(id, spec: spec) {
            if haystack.contains("turbo") {
                return "ace-step-dit-xl-turbo"
            }
            if haystack.contains("sft") {
                return "ace-step-dit-xl-sft"
            }
            if haystack.contains("base") || haystack.contains("full") {
                return "ace-step-dit-xl-base"
            }
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

    private static func variantTokens(_ id: String, spec: ModelSpec? = nil) -> Set<String> {
        let values = [
            id,
            spec?.id,
            spec?.displayName,
            spec?.huggingFaceRepo,
            spec?.variantFamily,
        ]

        return Set(values
            .compactMap { $0?.lowercased() }
            .flatMap { value in
                value.split { !$0.isLetter && !$0.isNumber }
                    .map(String.init)
            })
    }
}
