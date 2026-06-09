import Foundation

/// A single adapter entry from the community manifest.
public struct CommunityAdapter: Codable, Sendable, Identifiable, Hashable {
    /// Unique identifier (directory name convention).
    public let id: String

    /// User-facing display name.
    public let name: String

    /// Author or organization name.
    public let author: String

    /// Short description of what this adapter does.
    public let description: String

    /// Category of the base model this adapter targets.
    public let baseModelCategory: ModelCategory

    /// LoRA rank used during training.
    public let rank: Int

    /// URL to download the adapter.safetensors file.
    public let adapterURL: URL

    /// Optional SHA-256 checksum for the adapter.safetensors file.
    public let sha256: String?

    /// URL to download the adapter.json metadata file.
    public let metadataURL: URL

    /// Searchable/filterable tags.
    public let tags: [String]

    /// Whether this adapter is featured/promoted.
    public let featured: Bool

    /// Minimum app version required to use this adapter.
    public let minAppVersion: String

    /// When the adapter was published.
    public let createdAt: Date

    /// Total number of downloads (for sorting).
    public let downloadCount: Int

    public init(
        id: String,
        name: String,
        author: String,
        description: String,
        baseModelCategory: ModelCategory,
        rank: Int,
        adapterURL: URL,
        sha256: String? = nil,
        metadataURL: URL,
        tags: [String],
        featured: Bool,
        minAppVersion: String,
        createdAt: Date,
        downloadCount: Int
    ) {
        self.id = id
        self.name = name
        self.author = author
        self.description = description
        self.baseModelCategory = baseModelCategory
        self.rank = rank
        self.adapterURL = adapterURL
        self.sha256 = sha256
        self.metadataURL = metadataURL
        self.tags = tags
        self.featured = featured
        self.minAppVersion = minAppVersion
        self.createdAt = createdAt
        self.downloadCount = downloadCount
    }
}

/// The top-level manifest structure fetched from the community repo.
public struct AdapterManifest: Codable, Sendable {
    /// Schema version for forward compatibility.
    public let version: Int

    /// All published adapters.
    public let adapters: [CommunityAdapter]

    public init(version: Int, adapters: [CommunityAdapter]) {
        self.version = version
        self.adapters = adapters
    }
}
