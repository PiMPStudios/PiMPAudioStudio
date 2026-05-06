import Foundation
import OSLog

private let logger = Logger(subsystem: "com.pimp.AnvilApp", category: "ModelRegistry")

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

    // MARK: - Built-in Models

    public static let builtInSpecs: [ModelSpec] = [
        ModelSpec(
            id: "ace-step-dit",
            huggingFaceRepo: "ACEStep/ACE-Step-v1-5-DiT",
            files: ["model.safetensors", "config.json"],
            sizeBytes: 3_200_000_000,
            sha256: [:],
            category: .aceStepDiT
        ),
        ModelSpec(
            id: "ace-step-vae",
            huggingFaceRepo: "ACEStep/ACE-Step-v1-5-VAE",
            files: ["vae.safetensors", "config.json"],
            sizeBytes: 800_000_000,
            sha256: [:],
            category: .aceStepVAE
        ),
        ModelSpec(
            id: "stable-audio-open",
            huggingFaceRepo: "stabilityai/stable-audio-open-1.0",
            files: ["model.safetensors", "config.json"],
            sizeBytes: 1_200_000_000,
            sha256: [:],
            category: .stableAudio
        ),
        ModelSpec(
            id: "lyric-llm",
            huggingFaceRepo: "microsoft/Phi-3-mini-4k-instruct",
            files: ["model.safetensors", "tokenizer.json", "config.json"],
            sizeBytes: 2_400_000_000,
            sha256: [:],
            category: .lyricLLM
        ),
    ]

    // MARK: - User-Added Models

    private(set) var userSpecs: [ModelSpec] = []

    private var userSpecsFileURL: URL {
        baseDirectory.appending(path: "user-models.json")
    }

    // MARK: - Unified Access

    public var allSpecs: [ModelSpec] {
        Self.builtInSpecs + userSpecs
    }

    /// The ACE-Step model IDs required for initial setup.
    public static let essentialModelIDs: Set<String> = [
        "ace-step-dit",
        "ace-step-vae"
    ]

    // MARK: - Active Model Tracking

    private static let defaultActiveModels: [ModelCategory: String] = [
        .aceStepDiT: "ace-step-dit",
        .aceStepVAE: "ace-step-vae",
        .stableAudio: "stable-audio-open",
        .lyricLLM: "lyric-llm"
    ]

    private(set) var activeModels: [ModelCategory: String] = [:]

    private var activeModelsFileURL: URL {
        baseDirectory.appending(path: "active-models.json")
    }

    // MARK: - Init

    public init(baseDirectory: URL, client: DownloadClient = DownloadClient()) {
        self.baseDirectory = baseDirectory
        self.client = client
        loadUserSpecs()
        loadActiveModels()
    }

    // MARK: - Status

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

    public func isModelReady(_ modelID: String) -> Bool {
        status(for: modelID) == .ready
    }

    private func setStatus(_ modelID: String, to status: ModelStatus) {
        statuses[modelID] = status
    }

    // MARK: - Download

    /// Downloads the specified models with SHA256 verification and auto-retry.
    public func downloadModels(
        _ specs: [ModelSpec],
        onProgress: @Sendable @escaping (String, Double) -> Void
    ) async throws {
        for spec in specs {
            guard status(for: spec.id) != .ready else { continue }
            statuses[spec.id] = .downloading(progress: 0)

            for file in spec.files {
                let destination = spec.localPath(for: file, base: baseDirectory)
                let expectedHash = spec.sha256[file]

                do {
                    try await client.downloadAndVerify(
                        repo: spec.huggingFaceRepo,
                        file: file,
                        to: destination,
                        expectedHash: expectedHash,
                        onProgress: { [id = spec.id] progress in
                            Task { [weak self] in
                                await self?.setStatus(id, to: .downloading(progress: progress))
                            }
                            onProgress(id, progress)
                        },
                        onRetry: { [id = spec.id] in
                            logger.warning("Verification failed for \(file) in \(id), retrying...")
                        }
                    )
                } catch {
                    let message: String
                    if case DownloadError.checksumMismatch = error {
                        message = "Model verification failed. The file may be corrupted or tampered with."
                    } else {
                        message = error.localizedDescription
                    }
                    statuses[spec.id] = .failed(message)
                    throw error
                }
            }

            statuses[spec.id] = .ready
        }
    }

    /// Downloads only the essential (ACE-Step) models for first launch.
    public func downloadEssentialModels(
        onProgress: @Sendable @escaping (String, Double) -> Void
    ) async throws {
        let essentials = Self.builtInSpecs.filter { Self.essentialModelIDs.contains($0.id) }
        try await downloadModels(essentials, onProgress: onProgress)
    }

    /// Downloads all built-in models. Kept for backward compatibility.
    public func downloadAll(
        onProgress: @Sendable @escaping (String, Double) -> Void
    ) async throws {
        try await downloadModels(Self.builtInSpecs, onProgress: onProgress)
    }

    // MARK: - User Model Management (extension points for 9b)

    public func addModel(_ spec: ModelSpec) throws {
        guard !isBuiltIn(id: spec.id) else {
            throw RegistryError.cannotModifyBuiltIn
        }
        guard !userSpecs.contains(where: { $0.id == spec.id }) else {
            throw RegistryError.duplicateModel(spec.id)
        }
        userSpecs.append(spec)
        saveUserSpecs()
    }

    public func removeModel(id: String) throws {
        guard !isBuiltIn(id: id) else {
            throw RegistryError.cannotModifyBuiltIn
        }
        userSpecs.removeAll { $0.id == id }
        statuses.removeValue(forKey: id)
        saveUserSpecs()
    }

    public nonisolated func isBuiltIn(id: String) -> Bool {
        Self.builtInSpecs.contains { $0.id == id }
    }

    // MARK: - Active Model Selection

    public func setActiveModel(id: String, for category: ModelCategory) throws {
        guard let spec = allSpecs.first(where: { $0.id == id }) else {
            throw RegistryError.unknownModel(id)
        }
        guard spec.category == category else {
            throw RegistryError.categoryMismatch(expected: category, got: spec.category)
        }
        guard isModelReady(id) else {
            throw RegistryError.modelNotReady(id)
        }
        activeModels[category] = id
        saveActiveModels()
    }

    public func activeModelID(for category: ModelCategory) -> String? {
        activeModels[category]
    }

    public func activeSpec(for category: ModelCategory) -> ModelSpec? {
        guard let id = activeModels[category] else { return nil }
        return allSpecs.first(where: { $0.id == id })
    }

    public func specs(for category: ModelCategory) -> [ModelSpec] {
        allSpecs.filter { $0.category == category }
    }

    // MARK: - File Management

    public func deleteModelFiles(id: String) throws {
        guard let spec = allSpecs.first(where: { $0.id == id }) else {
            throw RegistryError.unknownModel(id)
        }
        let dir = spec.localDirectory(base: baseDirectory)
        if FileManager.default.fileExists(atPath: dir.path()) {
            try FileManager.default.removeItem(at: dir)
        }
        statuses[id] = .notDownloaded

        // If this was the active model, revert to built-in default
        if activeModels[spec.category] == id {
            let defaultID = Self.builtInSpecs.first(where: { $0.category == spec.category })?.id
            if let defaultID, defaultID != id, isModelReady(defaultID) {
                activeModels[spec.category] = defaultID
            } else if let defaultID {
                activeModels[spec.category] = defaultID
            }
            saveActiveModels()
        }
    }

    // MARK: - Persistence

    private func loadUserSpecs() {
        let url = userSpecsFileURL
        guard FileManager.default.fileExists(atPath: url.path()) else { return }

        do {
            let data = try Data(contentsOf: url)
            userSpecs = try JSONDecoder().decode([ModelSpec].self, from: data)
        } catch {
            logger.error("Failed to load user-models.json: \(error.localizedDescription). Starting with empty list.")
            userSpecs = []
        }
    }

    private func saveUserSpecs() {
        do {
            let data = try JSONEncoder().encode(userSpecs)
            try data.write(to: userSpecsFileURL, options: .atomic)
        } catch {
            logger.error("Failed to save user-models.json: \(error.localizedDescription)")
        }
    }

    private func loadActiveModels() {
        let url = activeModelsFileURL
        guard FileManager.default.fileExists(atPath: url.path()) else {
            // First launch — default to built-in models
            activeModels = Self.defaultActiveModels
            return
        }

        do {
            let data = try Data(contentsOf: url)
            let decoded = try JSONDecoder().decode([String: String].self, from: data)
            activeModels = Dictionary(
                uniqueKeysWithValues: decoded.compactMap { key, value in
                    guard let category = ModelCategory(rawValue: key) else { return nil }
                    return (category, value)
                }
            )
        } catch {
            logger.error("Failed to load active-models.json: \(error.localizedDescription). Using defaults.")
            activeModels = Self.defaultActiveModels
        }
    }

    private func saveActiveModels() {
        let encoded = Dictionary(
            uniqueKeysWithValues: activeModels.map { ($0.key.rawValue, $0.value) }
        )
        do {
            let data = try JSONEncoder().encode(encoded)
            try data.write(to: activeModelsFileURL, options: .atomic)
        } catch {
            logger.error("Failed to save active-models.json: \(error.localizedDescription)")
        }
    }
}

public enum RegistryError: Error, Sendable, LocalizedError {
    case cannotModifyBuiltIn
    case duplicateModel(String)
    case unknownModel(String)
    case categoryMismatch(expected: ModelCategory, got: ModelCategory)
    case modelNotReady(String)

    public var errorDescription: String? {
        switch self {
        case .cannotModifyBuiltIn:
            return "Built-in models cannot be modified."
        case .duplicateModel(let id):
            return "Model '\(id)' already exists."
        case .unknownModel(let id):
            return "Unknown model '\(id)'."
        case .categoryMismatch(let expected, let got):
            return "Category mismatch: expected \(expected.rawValue), got \(got.rawValue)."
        case .modelNotReady(let id):
            return "Model '\(id)' is not downloaded yet."
        }
    }
}
