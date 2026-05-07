import Foundation

public enum ModelCategory: String, Codable, Sendable, Hashable {
    case aceStepDiT
    case aceStepVAE
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

    public init(
        id: String,
        displayName: String,
        huggingFaceRepo: String,
        files: [ModelFileSpec],
        category: ModelCategory,
        variantFamily: String? = nil,
        requiresAuth: Bool = false
    ) {
        self.id = id
        self.displayName = displayName
        self.huggingFaceRepo = huggingFaceRepo
        self.files = files
        self.category = category
        self.variantFamily = variantFamily
        self.requiresAuth = requiresAuth
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
