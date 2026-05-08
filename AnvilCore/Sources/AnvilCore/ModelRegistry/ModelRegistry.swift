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
    private let tokenProvider: (@Sendable () async -> String?)?
    private var statuses: [String: ModelStatus] = [:]

    // MARK: - Built-in Models

    public static let builtInSpecs: [ModelSpec] = [
        ModelSpec(
            id: "ace-step-dit-turbo",
            displayName: "ACE-Step v1.5 Turbo",
            huggingFaceRepo: "ACE-Step/Ace-Step1.5",
            files: [
                ModelFileSpec(
                    remotePath: "acestep-v15-turbo/model.safetensors",
                    localName: "model.safetensors",
                    sizeBytes: 4_787_825_604
                ),
                ModelFileSpec(
                    remotePath: "acestep-v15-turbo/config.json",
                    localName: "config.json",
                    sizeBytes: 1_968
                ),
                ModelFileSpec(
                    remotePath: "acestep-v15-turbo/silence_latent.pt",
                    localName: "silence_latent.pt",
                    sizeBytes: 3_841_215
                )
            ],
            category: .aceStepDiT,
            variantFamily: "ace-step-1.5"
        ),
        ModelSpec(
            id: "ace-step-dit-base",
            displayName: "ACE-Step v1.5 Full",
            huggingFaceRepo: "ACE-Step/acestep-v15-base",
            files: [
                ModelFileSpec(
                    remotePath: "model.safetensors",
                    localName: "model.safetensors",
                    sizeBytes: 4_787_825_604
                ),
                ModelFileSpec(
                    remotePath: "config.json",
                    localName: "config.json",
                    sizeBytes: 1_940
                ),
                ModelFileSpec(
                    remotePath: "silence_latent.pt",
                    localName: "silence_latent.pt",
                    sizeBytes: 3_841_215
                )
            ],
            category: .aceStepDiT,
            variantFamily: "ace-step-1.5"
        ),
        ModelSpec(
            id: "ace-step-dit-sft",
            displayName: "ACE-Step v1.5 SFT",
            huggingFaceRepo: "ACE-Step/acestep-v15-sft",
            files: [
                ModelFileSpec(
                    remotePath: "model.safetensors",
                    localName: "model.safetensors",
                    sizeBytes: 4_787_825_604
                ),
                ModelFileSpec(
                    remotePath: "config.json",
                    localName: "config.json",
                    sizeBytes: 1_940
                ),
                ModelFileSpec(
                    remotePath: "silence_latent.pt",
                    localName: "silence_latent.pt",
                    sizeBytes: 3_841_215
                )
            ],
            category: .aceStepDiT,
            variantFamily: "ace-step-1.5"
        ),
        ModelSpec(
            id: "ace-step-vae",
            displayName: "ACE-Step VAE + Text Encoder",
            huggingFaceRepo: "ACE-Step/Ace-Step1.5",
            files: [
                ModelFileSpec(
                    remotePath: "vae/diffusion_pytorch_model.safetensors",
                    localName: "vae.safetensors",
                    sizeBytes: 337_431_388
                ),
                ModelFileSpec(
                    remotePath: "vae/config.json",
                    localName: "config.json",
                    sizeBytes: 425
                ),
                ModelFileSpec(
                    remotePath: "Qwen3-Embedding-0.6B/tokenizer.json",
                    localName: "tokenizer.json",
                    sizeBytes: 11_423_705
                ),
                ModelFileSpec(
                    remotePath: "Qwen3-Embedding-0.6B/model.safetensors",
                    localName: "qwen3_model.safetensors",
                    sizeBytes: 1_191_586_416
                )
            ],
            category: .aceStepVAE,
            variantFamily: "ace-step-1.5"
        ),
        ModelSpec(
            id: "stable-audio-open",
            displayName: "Stable Audio Open 1.0",
            huggingFaceRepo: "stabilityai/stable-audio-open-1.0",
            files: [
                ModelFileSpec(
                    remotePath: "model.safetensors",
                    localName: "model.safetensors",
                    sizeBytes: 4_853_889_016
                ),
                ModelFileSpec(
                    remotePath: "model_config.json",
                    localName: "config.json",
                    sizeBytes: 4_172
                ),
                ModelFileSpec(
                    remotePath: "vae_model.ckpt",
                    localName: "vae_model.ckpt",
                    sizeBytes: 624_474_947
                ),
                ModelFileSpec(
                    remotePath: "vae_model_config.json",
                    localName: "vae_config.json",
                    sizeBytes: 3_608
                )
            ],
            category: .stableAudio,
            variantFamily: nil,
            requiresAuth: true
        ),
        // 5 Hz Audio→Token Language Models for cover-song generation
        // (audio→audio conditioning). Optional — pick one based on
        // quality/disk/RAM budget. Inference plumbing is a follow-up
        // phase; these specs land first so the Settings UI can show
        // them and users can begin downloading in advance.
        ModelSpec(
            id: "ace-step-lm-0.6b",
            displayName: "ACE-Step LM 0.6B",
            huggingFaceRepo: "ACE-Step/acestep-5Hz-lm-0.6B",
            files: [
                ModelFileSpec(remotePath: "model.safetensors", sizeBytes: 1_325_804_024),
                ModelFileSpec(remotePath: "config.json", sizeBytes: 1_386),
                ModelFileSpec(remotePath: "tokenizer.json", sizeBytes: 24_321_939),
                ModelFileSpec(remotePath: "tokenizer_config.json", sizeBytes: 14_072_925),
                ModelFileSpec(remotePath: "special_tokens_map.json", sizeBytes: 1_824_199),
                ModelFileSpec(remotePath: "added_tokens.json", sizeBytes: 2_217_787),
                ModelFileSpec(remotePath: "merges.txt", sizeBytes: 1_671_853),
                ModelFileSpec(remotePath: "vocab.json", sizeBytes: 2_776_833)
            ],
            category: .aceStepLM,
            variantFamily: "ace-step-lm"
        ),
        ModelSpec(
            id: "ace-step-lm-4b",
            displayName: "ACE-Step LM 4B",
            huggingFaceRepo: "ACE-Step/acestep-5Hz-lm-4B",
            files: [
                ModelFileSpec(remotePath: "model-00001-of-00002.safetensors", sizeBytes: 4_999_914_896),
                ModelFileSpec(remotePath: "model-00002-of-00002.safetensors", sizeBytes: 3_379_239_240),
                ModelFileSpec(remotePath: "model.safetensors.index.json", sizeBytes: 32_818),
                ModelFileSpec(remotePath: "config.json", sizeBytes: 1_562),
                ModelFileSpec(remotePath: "tokenizer.json", sizeBytes: 24_321_939),
                ModelFileSpec(remotePath: "tokenizer_config.json", sizeBytes: 14_072_925),
                ModelFileSpec(remotePath: "special_tokens_map.json", sizeBytes: 1_824_199),
                ModelFileSpec(remotePath: "added_tokens.json", sizeBytes: 2_217_787),
                ModelFileSpec(remotePath: "merges.txt", sizeBytes: 1_671_853),
                ModelFileSpec(remotePath: "vocab.json", sizeBytes: 2_776_833)
            ],
            category: .aceStepLM,
            variantFamily: "ace-step-lm"
        ),
        ModelSpec(
            id: "lyric-llm",
            displayName: "Phi-3 Mini 4k Instruct",
            huggingFaceRepo: "microsoft/Phi-3-mini-4k-instruct",
            files: [
                ModelFileSpec(remotePath: "model.safetensors", sizeBytes: 2_400_000_000),
                ModelFileSpec(remotePath: "tokenizer.json", sizeBytes: 2_000_000),
                ModelFileSpec(remotePath: "config.json", sizeBytes: 1_000)
            ],
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
        "ace-step-dit-turbo",
        "ace-step-vae"
    ]

    // MARK: - Active Model Tracking

    private static let defaultActiveModels: [ModelCategory: String] = [
        .aceStepDiT: "ace-step-dit-turbo",
        .aceStepVAE: "ace-step-vae",
        .stableAudio: "stable-audio-open",
        .lyricLLM: "lyric-llm"
    ]

    private(set) var activeModels: [ModelCategory: String] = [:]

    private var activeModelsFileURL: URL {
        baseDirectory.appending(path: "active-models.json")
    }

    // MARK: - Init

    public init(
        baseDirectory: URL,
        client: DownloadClient = DownloadClient(),
        tokenProvider: (@Sendable () async -> String?)? = nil
    ) {
        self.baseDirectory = baseDirectory
        self.client = client
        self.tokenProvider = tokenProvider
        loadUserSpecs()
        loadActiveModels()
        migrateLegacyModelIDsIfNeeded()
    }

    /// One-shot migration from Phase-1 placeholder spec IDs to Phase-2 real IDs.
    /// Maps legacy `ace-step-dit` active ID to the new Turbo default and sweeps
    /// orphan model directories whose names don't match any current spec ID.
    /// Gated on `UserDefaults` flag so this runs exactly once.
    private func migrateLegacyModelIDsIfNeeded() {
        let migrationKey = "com.pimp.AnvilApp.modelMigration.v1"
        guard !UserDefaults.standard.bool(forKey: migrationKey) else { return }

        if activeModels[.aceStepDiT] == "ace-step-dit" {
            activeModels[.aceStepDiT] = "ace-step-dit-turbo"
            logger.info("Migrated legacy active DiT 'ace-step-dit' → 'ace-step-dit-turbo'")
        }

        let validIDs = Set(allSpecs.map(\.id))
        if let entries = try? FileManager.default.contentsOfDirectory(
            at: baseDirectory, includingPropertiesForKeys: nil
        ) {
            for entry in entries {
                let name = entry.lastPathComponent
                var isDir: ObjCBool = false
                FileManager.default.fileExists(atPath: entry.path(percentEncoded: false), isDirectory: &isDir)
                guard isDir.boolValue, name != "user-models.json" else { continue }
                if !validIDs.contains(name) {
                    try? FileManager.default.removeItem(at: entry)
                    logger.info("Removed orphan model directory: \(name)")
                }
            }
        }

        UserDefaults.standard.set(true, forKey: migrationKey)
    }

    // MARK: - Status

    public func status(for modelID: String) -> ModelStatus {
        if let cached = statuses[modelID] {
            return cached
        }
        guard let spec = allSpecs.first(where: { $0.id == modelID }) else {
            return .failed("Unknown model: \(modelID)")
        }
        let allFilesExist = spec.files.allSatisfy { fileSpec in
            // Use percentEncoded:false so paths containing "Application Support"
            // (with its space) match the filesystem layout, not the percent-
            // encoded URL form.
            FileManager.default.fileExists(
                atPath: spec.localPath(for: fileSpec.localName, base: baseDirectory)
                    .path(percentEncoded: false)
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

            let bearerToken: String?
            if spec.requiresAuth {
                bearerToken = await tokenProvider?()
                if (bearerToken ?? "").isEmpty {
                    statuses[spec.id] = .failed("HuggingFace token required")
                    throw DownloadError.authRequired(repo: spec.huggingFaceRepo)
                }
            } else {
                bearerToken = nil
            }

            for fileSpec in spec.files {
                let destination = spec.localPath(for: fileSpec.localName, base: baseDirectory)

                do {
                    try await client.downloadAndVerify(
                        repo: spec.huggingFaceRepo,
                        file: fileSpec.remotePath,
                        to: destination,
                        expectedHash: fileSpec.sha256,
                        bearerToken: bearerToken,
                        onProgress: { [id = spec.id] progress in
                            Task { [weak self] in
                                await self?.setStatus(id, to: .downloading(progress: progress))
                            }
                            onProgress(id, progress)
                        },
                        onRetry: { [id = spec.id, remote = fileSpec.remotePath] in
                            logger.warning("Verification failed for \(remote) in \(id), retrying...")
                        }
                    )
                } catch {
                    if let httpError = error as? DownloadError,
                       case .httpError(let code) = httpError,
                       (code == 401 || code == 403),
                       bearerToken != nil {
                        statuses[spec.id] = .failed("HuggingFace rejected the token")
                        throw DownloadError.authForbidden(repo: spec.huggingFaceRepo)
                    }
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
        guard spec.isRunnable else {
            throw RegistryError.modelNotRunnable(spec.displayName, spec.runtimeNote)
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
        if FileManager.default.fileExists(atPath: dir.path(percentEncoded: false)) {
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
        guard FileManager.default.fileExists(atPath: url.path(percentEncoded: false)) else { return }

        do {
            let data = try Data(contentsOf: url)
            userSpecs = try JSONDecoder().decode([ModelSpec].self, from: data)
            if refreshRuntimeMetadataForDownloadedACEModels() {
                saveUserSpecs()
            }
        } catch {
            logger.error("Failed to load user-models.json: \(error.localizedDescription). Starting with empty list.")
            userSpecs = []
        }
    }

    private func refreshRuntimeMetadataForDownloadedACEModels() -> Bool {
        var changed = false
        userSpecs = userSpecs.map { spec in
            guard spec.category == .aceStepDiT else { return spec }
            let configURL = spec.localPath(for: "config.json", base: baseDirectory)
            guard FileManager.default.fileExists(atPath: configURL.path(percentEncoded: false)),
                  let config = try? DiTConfig.load(from: configURL) else {
                return spec
            }

            let runtimeNote = config.runtimeSupportNote
            let variantFamily = config.variantFamily
            let isRunnable = runtimeNote == nil
            guard spec.isRunnable != isRunnable
                    || spec.runtimeNote != runtimeNote
                    || spec.variantFamily != variantFamily else {
                return spec
            }

            changed = true
            return ModelSpec(
                id: spec.id,
                displayName: spec.displayName,
                huggingFaceRepo: spec.huggingFaceRepo,
                files: spec.files,
                category: spec.category,
                variantFamily: variantFamily,
                requiresAuth: spec.requiresAuth,
                isRunnable: isRunnable,
                runtimeNote: runtimeNote
            )
        }
        return changed
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
        guard FileManager.default.fileExists(atPath: url.path(percentEncoded: false)) else {
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
    case modelNotRunnable(String, String?)

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
        case .modelNotRunnable(let name, let note):
            return note ?? "\(name) can be downloaded, but this build cannot run it yet."
        }
    }
}
