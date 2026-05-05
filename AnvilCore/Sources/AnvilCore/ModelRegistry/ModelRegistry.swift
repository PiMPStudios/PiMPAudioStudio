import Foundation

public enum ModelStatus: Sendable, Equatable {
    case notDownloaded
    case downloading(progress: Double)
    case verifying
    case ready
    case failed(String)
}

public actor ModelRegistry {
    public let baseDirectory: URL
    private let client: DownloadClient
    private var statuses: [String: ModelStatus] = [:]

    public nonisolated let allSpecs: [ModelSpec] = [
        ModelSpec(
            id: "ace-step-dit",
            huggingFaceRepo: "ACEStep/ACE-Step-v1-5-DiT",
            files: ["model.safetensors", "config.json"],
            sizeBytes: 3_200_000_000,
            sha256: [:]
        ),
        ModelSpec(
            id: "ace-step-vae",
            huggingFaceRepo: "ACEStep/ACE-Step-v1-5-VAE",
            files: ["vae.safetensors", "config.json"],
            sizeBytes: 800_000_000,
            sha256: [:]
        ),
        ModelSpec(
            id: "stable-audio-open",
            huggingFaceRepo: "stabilityai/stable-audio-open-1.0",
            files: ["model.safetensors", "config.json"],
            sizeBytes: 1_200_000_000,
            sha256: [:]
        ),
        ModelSpec(
            id: "lyric-llm",
            huggingFaceRepo: "microsoft/Phi-3-mini-4k-instruct",
            files: ["model.safetensors", "tokenizer.json", "config.json"],
            sizeBytes: 2_400_000_000,
            sha256: [:]
        ),
    ]

    public init(baseDirectory: URL, client: DownloadClient = DownloadClient()) {
        self.baseDirectory = baseDirectory
        self.client = client
    }

    public func status(for modelID: String) -> ModelStatus {
        if let cached = statuses[modelID] {
            return cached
        }
        guard let spec = allSpecs.first(where: { $0.id == modelID }) else {
            return .failed("Unknown model: \(modelID)")
        }
        let allFilesExist = spec.files.allSatisfy { file in
            FileManager.default.fileExists(
                atPath: spec.localPath(for: file, base: baseDirectory).path()
            )
        }
        return allFilesExist ? .ready : .notDownloaded
    }

    public func downloadAll(
        onProgress: @Sendable @escaping (String, Double) -> Void
    ) async throws {
        for spec in allSpecs {
            guard status(for: spec.id) != .ready else { continue }
            statuses[spec.id] = .downloading(progress: 0)

            for file in spec.files {
                let destination = spec.localPath(for: file, base: baseDirectory)
                try await client.download(
                    repo: spec.huggingFaceRepo,
                    file: file,
                    to: destination,
                    onProgress: { [id = spec.id] progress in
                        onProgress(id, progress)
                    }
                )

                if let expectedHash = spec.sha256[file], !expectedHash.isEmpty {
                    statuses[spec.id] = .verifying
                    guard await client.verifySHA256(file: destination, expected: expectedHash) else {
                        statuses[spec.id] = .failed("Checksum mismatch: \(file)")
                        throw DownloadError.checksumMismatch(file: file)
                    }
                }
            }

            statuses[spec.id] = .ready
        }
    }
}
