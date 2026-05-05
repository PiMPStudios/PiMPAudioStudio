import Foundation

public struct ModelSpec: Sendable, Identifiable {
    public let id: String
    public let huggingFaceRepo: String
    public let files: [String]
    public let sizeBytes: Int64
    /// SHA256 checksums keyed by filename. Entries are optional per file —
    /// files without a corresponding hash skip verification during download.
    /// When checksums are available, every file should have an entry.
    public let sha256: [String: String]

    public init(
        id: String,
        huggingFaceRepo: String,
        files: [String],
        sizeBytes: Int64,
        sha256: [String: String]
    ) {
        self.id = id
        self.huggingFaceRepo = huggingFaceRepo
        self.files = files
        self.sizeBytes = sizeBytes
        self.sha256 = sha256
    }

    public var totalSizeFormatted: String {
        let gb = Double(sizeBytes) / 1_000_000_000
        return String(format: "%.1f GB", gb)
    }

    public func localDirectory(base: URL) -> URL {
        base.appending(path: id)
    }

    public func localPath(for file: String, base: URL) -> URL {
        localDirectory(base: base).appending(path: file)
    }
}
