import Foundation

public enum ModelCategory: String, Codable, Sendable, Hashable {
    case aceStepDiT
    case aceStepVAE
    /// 5 Hz Audio→Token Language Model for experimental discrete audio-code
    /// workflows. Source-audio cover generation does not require this model;
    /// it uses the DiT bundle's tokenizer/detokenizer path.
    case aceStepLM
    case stableAudio
    case lyricLLM
}

public struct ModelFileSpec: Sendable, Codable, Hashable {
    /// Path inside the HuggingFace repo, e.g. "acestep-v15-turbo/model.safetensors".
    public let remotePath: String
    /// Flat filename used on disk and by consumer code, e.g. "model.safetensors".
    /// Defaults to the basename of `remotePath` when not specified.
    public let localName: String
    public let sha256: String?
    public let sizeBytes: Int64

    public init(
        remotePath: String,
        localName: String? = nil,
        sha256: String? = nil,
        sizeBytes: Int64 = 0
    ) {
        self.remotePath = remotePath
        self.localName = localName ?? (remotePath as NSString).lastPathComponent
        self.sha256 = sha256
        self.sizeBytes = sizeBytes
    }
}

public struct ModelSpec: Sendable, Identifiable, Codable {
    public let id: String
    public let displayName: String
    public let huggingFaceRepo: String
    public let files: [ModelFileSpec]
    public let category: ModelCategory
    /// Groups variants of the same model family for UI presentation, e.g. "ace-step-1.5".
    public let variantFamily: String?
    /// When true, downloads attach `Authorization: Bearer <token>` and gate on token presence.
    public let requiresAuth: Bool
    /// Downloads are allowed, but generation should not load this model yet.
    /// Used for model families the registry can discover before the Swift
    /// runtime can instantiate the matching architecture.
    public let isRunnable: Bool
    public let runtimeNote: String?

    private enum CodingKeys: String, CodingKey {
        case id
        case displayName
        case huggingFaceRepo
        case files
        case category
        case variantFamily
        case requiresAuth
        case isRunnable
        case runtimeNote
    }

    public init(
        id: String,
        displayName: String,
        huggingFaceRepo: String,
        files: [ModelFileSpec],
        category: ModelCategory,
        variantFamily: String? = nil,
        requiresAuth: Bool = false,
        isRunnable: Bool = true,
        runtimeNote: String? = nil
    ) {
        self.id = id
        self.displayName = displayName
        self.huggingFaceRepo = huggingFaceRepo
        self.files = files
        self.category = category
        self.variantFamily = variantFamily
        self.requiresAuth = requiresAuth
        self.isRunnable = isRunnable
        self.runtimeNote = runtimeNote
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.id = try container.decode(String.self, forKey: .id)
        self.displayName = try container.decode(String.self, forKey: .displayName)
        self.huggingFaceRepo = try container.decode(String.self, forKey: .huggingFaceRepo)
        self.files = try container.decode([ModelFileSpec].self, forKey: .files)
        self.category = try container.decode(ModelCategory.self, forKey: .category)
        self.variantFamily = try container.decodeIfPresent(String.self, forKey: .variantFamily)
        self.requiresAuth = try container.decodeIfPresent(Bool.self, forKey: .requiresAuth) ?? false
        self.isRunnable = try container.decodeIfPresent(Bool.self, forKey: .isRunnable) ?? true
        self.runtimeNote = try container.decodeIfPresent(String.self, forKey: .runtimeNote)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(displayName, forKey: .displayName)
        try container.encode(huggingFaceRepo, forKey: .huggingFaceRepo)
        try container.encode(files, forKey: .files)
        try container.encode(category, forKey: .category)
        try container.encodeIfPresent(variantFamily, forKey: .variantFamily)
        try container.encode(requiresAuth, forKey: .requiresAuth)
        try container.encode(isRunnable, forKey: .isRunnable)
        try container.encodeIfPresent(runtimeNote, forKey: .runtimeNote)
    }

    public var totalSizeBytes: Int64 {
        files.reduce(0) { $0 + $1.sizeBytes }
    }

    public var totalSizeFormatted: String {
        let gb = Double(totalSizeBytes) / 1_000_000_000
        return String(format: "%.1f GB", gb)
    }

    public func localDirectory(base: URL) -> URL {
        base.appending(path: id)
    }

    /// Resolves to `<base>/<id>/<localName>`. Code that loads weights uses the
    /// flat localName ("model.safetensors") and is unaware of the remote path.
    public func localPath(for localName: String, base: URL) -> URL {
        localDirectory(base: base).appending(path: localName)
    }
}
