import Foundation

public enum ModelCategory: String, Codable, Sendable {
    case aceStepDiT
    case aceStepVAE
    case stableAudio
    case lyricLLM
}

public struct ModelSpec: Sendable, Identifiable, Codable {
    public let id: String
    public let huggingFaceRepo: String
    public let files: [String]
    public let sizeBytes: Int64
    /// SHA256 checksums keyed by filename. Empty dictionary means skip
    /// verification (used for user-added models without known hashes).
    public let sha256: [String: String]
    public let category: ModelCategory

    public init(
        id: String,
        huggingFaceRepo: String,
        files: [String],
        sizeBytes: Int64,
        sha256: [String: String] = [:],
        category: ModelCategory
    ) {
        self.id = id
        self.huggingFaceRepo = huggingFaceRepo
        self.files = files
        self.sizeBytes = sizeBytes
        self.sha256 = sha256
        self.category = category
    }

    public var displayName: String {
        let name = huggingFaceRepo.components(separatedBy: "/").last ?? id
        return name.replacingOccurrences(of: "-", with: " ")
            .replacingOccurrences(of: "_", with: " ")
            .capitalized
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

