import Foundation
import OSLog

private let logger = Logger(subsystem: "com.pimpstudios.PiMPAudioStudio", category: "ModelRegistry")

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

    private static func aceStepXLFiles(configSizeBytes: Int64) -> [ModelFileSpec] {
        [
            ModelFileSpec(remotePath: "model.safetensors.index.json", sizeBytes: 71_471),
            ModelFileSpec(remotePath: "model-00001-of-00004.safetensors", sizeBytes: 4_986_971_456),
            ModelFileSpec(remotePath: "model-00002-of-00004.safetensors", sizeBytes: 4_986_942_776),
            ModelFileSpec(remotePath: "model-00003-of-00004.safetensors", sizeBytes: 4_986_942_808),
            ModelFileSpec(remotePath: "model-00004-of-00004.safetensors", sizeBytes: 4_988_483_464),
            ModelFileSpec(remotePath: "config.json", sizeBytes: configSizeBytes),
            ModelFileSpec(remotePath: "silence_latent.pt", sizeBytes: 3_841_215)
        ]
    }

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
            id: "ace-step-dit-xl-turbo",
            displayName: "ACE-Step v1.5 XL Turbo",
            huggingFaceRepo: "ACE-Step/acestep-v15-xl-turbo",
            files: aceStepXLFiles(configSizeBytes: 2_407),
            category: .aceStepDiT,
            variantFamily: "ace-step-1.5-xl"
        ),
        ModelSpec(
            id: "ace-step-dit-xl-base",
            displayName: "ACE-Step v1.5 XL Base",
            huggingFaceRepo: "ACE-Step/acestep-v15-xl-base",
            files: aceStepXLFiles(configSizeBytes: 2_525),
            category: .aceStepDiT,
            variantFamily: "ace-step-1.5-xl"
        ),
        ModelSpec(
            id: "ace-step-dit-xl-sft",
            displayName: "ACE-Step v1.5 XL SFT",
            huggingFaceRepo: "ACE-Step/acestep-v15-xl-sft",
            files: aceStepXLFiles(configSizeBytes: 2_525),
            category: .aceStepDiT,
            variantFamily: "ace-step-1.5-xl"
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
                // DiT — the legacy top-level `model.safetensors` packs the
                // whole pipeline as a single file. The diffusers-style
                // `transformer/diffusion_pytorch_model.safetensors` is the
                // same DiT weights as a separate component. We download
                // both for now: top-level satisfies legacy callers, the
                // per-component path is what the new T5+projection pipeline
                // will read from.
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
                    remotePath: "vae/diffusion_pytorch_model.safetensors",
                    localName: "vae.safetensors",
                    sizeBytes: 624_490_240
                ),
                ModelFileSpec(
                    remotePath: "vae/config.json",
                    localName: "vae_config.json",
                    sizeBytes: 350
                ),
                // T5-base text encoder. Stable Audio Open conditioning
                // pipeline: tokenize → T5 encode (max_length=128, d_model
                // =768) → projection → cross-attn into DiT.
                ModelFileSpec(
                    remotePath: "text_encoder/model.safetensors",
                    localName: "text_encoder.safetensors",
                    sizeBytes: 438_525_864
                ),
                ModelFileSpec(
                    remotePath: "text_encoder/config.json",
                    localName: "text_encoder_config.json",
                    sizeBytes: 1_497
                ),
                // SentencePiece tokenizer used by T5.
                ModelFileSpec(
                    remotePath: "tokenizer/spiece.model",
                    localName: "spiece.model",
                    sizeBytes: 791_656
                ),
                ModelFileSpec(
                    remotePath: "tokenizer/tokenizer.json",
                    localName: "tokenizer.json",
                    sizeBytes: 2_424_064
                ),
                // Projection model — maps T5 hidden states into the DiT
                // conditioning embedding space.
                ModelFileSpec(
                    remotePath: "projection_model/diffusion_pytorch_model.safetensors",
                    localName: "projection.safetensors",
                    sizeBytes: 24_096_080
                ),
                ModelFileSpec(
                    remotePath: "projection_model/config.json",
                    localName: "projection_config.json",
                    sizeBytes: 198
                )
            ],
            category: .stableAudio,
            variantFamily: nil,
            requiresAuth: true,
            isRunnable: true
        ),
        // 5 Hz Audio→Token Language Models for experimental discrete
        // audio-code workflows. Source-audio cover generation uses the
        // DiT bundle's tokenizer/detokenizer path and does not require
        // one of these LM variants.
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
            id: "ace-step-lm-1.7b",
            displayName: "ACE-Step LM 1.7B",
            huggingFaceRepo: "ACE-Step/Ace-Step1.5",
            files: [
                ModelFileSpec(
                    remotePath: "acestep-5Hz-lm-1.7B/model.safetensors",
                    localName: "model.safetensors",
                    sizeBytes: 3_708_521_528
                ),
                ModelFileSpec(
                    remotePath: "acestep-5Hz-lm-1.7B/config.json",
                    localName: "config.json",
                    sizeBytes: 1_385
                ),
                ModelFileSpec(
                    remotePath: "acestep-5Hz-lm-1.7B/tokenizer.json",
                    localName: "tokenizer.json",
                    sizeBytes: 24_321_939
                ),
                ModelFileSpec(
                    remotePath: "acestep-5Hz-lm-1.7B/tokenizer_config.json",
                    localName: "tokenizer_config.json",
                    sizeBytes: 14_072_925
                ),
                ModelFileSpec(
                    remotePath: "acestep-5Hz-lm-1.7B/special_tokens_map.json",
                    localName: "special_tokens_map.json",
                    sizeBytes: 1_824_199
                ),
                ModelFileSpec(
                    remotePath: "acestep-5Hz-lm-1.7B/added_tokens.json",
                    localName: "added_tokens.json",
                    sizeBytes: 2_217_787
                ),
                ModelFileSpec(
                    remotePath: "acestep-5Hz-lm-1.7B/merges.txt",
                    localName: "merges.txt",
                    sizeBytes: 1_671_853
                ),
                ModelFileSpec(
                    remotePath: "acestep-5Hz-lm-1.7B/vocab.json",
                    localName: "vocab.json",
                    sizeBytes: 2_776_833
                ),
                ModelFileSpec(
                    remotePath: "acestep-5Hz-lm-1.7B/chat_template.jinja",
                    localName: "chat_template.jinja",
                    sizeBytes: 4_168
                )
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
        let loadedUserSpecs = Self.loadUserSpecs(from: baseDirectory)
        switch loadedUserSpecs {
        case .missing:
            self.userSpecs = []
        case .loaded(let specs, let shouldPersist):
            self.userSpecs = specs
            if shouldPersist {
                Self.saveUserSpecs(
                    specs,
                    to: baseDirectory.appending(path: "user-models.json")
                )
            }
        case .corrupt:
            // Keep an empty in-memory list; the corrupt index was preserved on disk.
            self.userSpecs = []
        }
        self.activeModels = Self.migratedActiveModelsIfNeeded(
            activeModels: Self.loadActiveModels(from: baseDirectory),
            userSpecs: self.userSpecs,
            baseDirectory: baseDirectory,
            allowOrphanCleanup: loadedUserSpecs.allowsDestructiveOrphanCleanup
        )
    }

    /// Per-models-root marker so migration state is isolated per directory.
    /// (Avoids a global UserDefaults race across app roots and unit tests.)
    private static let modelMigrationMarkerFileName = ".model-migration-v1-complete"
    private static let modelMigrationProtectedDirectoryNames: Set<String> = [
        "adapters",
    ]

    /// One-shot migration from Phase-1 placeholder spec IDs to Phase-2 real IDs.
    /// Maps legacy `ace-step-dit` active ID to the new Turbo default and sweeps
    /// orphan model directories whose names don't match any current spec ID.
    ///
    /// Destructive orphan cleanup is skipped when the user-model index is
    /// corrupt so a decode failure cannot wipe every unrecognized model dir.
    private static func migratedActiveModelsIfNeeded(
        activeModels loadedActiveModels: [ModelCategory: String],
        userSpecs: [ModelSpec],
        baseDirectory: URL,
        allowOrphanCleanup: Bool
    ) -> [ModelCategory: String] {
        guard !isModelMigrationComplete(baseDirectory: baseDirectory) else {
            return loadedActiveModels
        }

        var activeModels = loadedActiveModels

        if activeModels[.aceStepDiT] == "ace-step-dit" {
            activeModels[.aceStepDiT] = "ace-step-dit-turbo"
            logger.info("Migrated legacy active DiT 'ace-step-dit' → 'ace-step-dit-turbo'")
        }

        guard allowOrphanCleanup else {
            logger.error(
                "Skipping destructive model-directory migration because user-models.json did not load successfully."
            )
            // Do not mark migration complete — retry orphan cleanup after the
            // index is restored or replaced with a readable file.
            return activeModels
        }

        var validIDs = Set((Self.builtInSpecs + userSpecs).map(\.id))
        validIDs.formUnion([
            NativeACEDatasetAnnotationBackend.captionerDirectoryName,
            NativeACEDatasetAnnotationBackend.transcriberDirectoryName,
        ])
        if let entries = try? FileManager.default.contentsOfDirectory(
            at: baseDirectory, includingPropertiesForKeys: nil
        ) {
            for entry in entries {
                let name = entry.lastPathComponent
                if name == modelMigrationMarkerFileName { continue }
                if modelMigrationProtectedDirectoryNames.contains(name) {
                    continue
                }
                var isDir: ObjCBool = false
                FileManager.default.fileExists(atPath: entry.path(percentEncoded: false), isDirectory: &isDir)
                guard isDir.boolValue else { continue }
                if !validIDs.contains(name) {
                    try? FileManager.default.removeItem(at: entry)
                    logger.info("Removed orphan model directory: \(name)")
                }
            }
        }

        markModelMigrationComplete(baseDirectory: baseDirectory)
        return activeModels
    }

    private static func modelMigrationMarkerURL(baseDirectory: URL) -> URL {
        baseDirectory.appending(path: modelMigrationMarkerFileName)
    }

    private static func isModelMigrationComplete(baseDirectory: URL) -> Bool {
        FileManager.default.fileExists(
            atPath: modelMigrationMarkerURL(baseDirectory: baseDirectory).path(percentEncoded: false)
        )
    }

    private static func markModelMigrationComplete(baseDirectory: URL) {
        let markerURL = modelMigrationMarkerURL(baseDirectory: baseDirectory)
        do {
            try FileManager.default.createDirectory(
                at: baseDirectory,
                withIntermediateDirectories: true
            )
            try Data().write(to: markerURL, options: .atomic)
        } catch {
            logger.error(
                "Failed to write model migration marker: \(error.localizedDescription, privacy: .public)"
            )
        }
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

    private func setDownloadProgress(_ modelID: String, progress: Double) {
        switch statuses[modelID] {
        case .ready, .failed:
            return
        case .downloading(let current) where progress < current:
            return
        default:
            statuses[modelID] = .downloading(progress: progress)
        }
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
            onProgress(spec.id, 0)

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

            let totalBytes = spec.files.reduce(Int64(0)) { $0 + max($1.sizeBytes, 0) }
            var completedBytes: Int64 = 0
            let fileCount = max(spec.files.count, 1)

            for (index, fileSpec) in spec.files.enumerated() {
                let destination = spec.localPath(for: fileSpec.localName, base: baseDirectory)
                let fileStartBytes = completedBytes
                let fileSizeBytes = max(fileSpec.sizeBytes, 0)
                let fileStartFraction = Double(index) / Double(fileCount)
                let fileWeightFraction = 1.0 / Double(fileCount)

                do {
                    try await client.downloadAndVerify(
                        repo: spec.huggingFaceRepo,
                        file: fileSpec.remotePath,
                        to: destination,
                        expectedHash: fileSpec.sha256,
                        bearerToken: bearerToken,
                        onProgress: { [id = spec.id] fileProgress in
                            let clampedFileProgress = min(max(fileProgress, 0), 1)
                            let modelProgress: Double
                            if totalBytes > 0, fileSizeBytes > 0 {
                                let bytesDone = Double(fileStartBytes) + Double(fileSizeBytes) * clampedFileProgress
                                modelProgress = bytesDone / Double(totalBytes)
                            } else {
                                modelProgress = fileStartFraction + fileWeightFraction * clampedFileProgress
                            }
                            let clampedModelProgress = min(max(modelProgress, 0), 1)
                            Task { [weak self] in
                                await self?.setDownloadProgress(id, progress: clampedModelProgress)
                            }
                            onProgress(id, clampedModelProgress)
                        },
                        onRetry: { [id = spec.id, remote = fileSpec.remotePath] in
                            logger.warning("Verification failed for \(remote) in \(id), retrying...")
                        }
                    )
                    completedBytes += fileSizeBytes
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
            onProgress(spec.id, 1)
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

    /// Outcome of reading `user-models.json`. Corrupt is distinct from missing so
    /// migration never treats a decode failure as "no user models."
    private enum UserSpecsLoadResult: Sendable {
        case missing
        case loaded(specs: [ModelSpec], shouldPersist: Bool)
        case corrupt

        /// Orphan model-directory deletion is only safe when the index is absent
        /// or fully decoded — never when it failed to parse.
        var allowsDestructiveOrphanCleanup: Bool {
            switch self {
            case .missing, .loaded:
                return true
            case .corrupt:
                return false
            }
        }
    }

    private static func loadUserSpecs(from baseDirectory: URL) -> UserSpecsLoadResult {
        let url = baseDirectory.appending(path: "user-models.json")
        guard FileManager.default.fileExists(atPath: url.path(percentEncoded: false)) else {
            return .missing
        }

        do {
            let data = try Data(contentsOf: url)
            let decoded = try JSONDecoder().decode([ModelSpec].self, from: data)
            let refreshed = refreshedRuntimeMetadataForDownloadedACEModels(
                decoded,
                baseDirectory: baseDirectory
            )
            return .loaded(specs: refreshed.specs, shouldPersist: refreshed.changed)
        } catch {
            preserveCorruptUserModelsFile(at: url, cause: error)
            return .corrupt
        }
    }

    private static func preserveCorruptUserModelsFile(at url: URL, cause: Error) {
        let fm = FileManager.default
        let directory = url.deletingLastPathComponent()
        var destination = directory.appending(path: "user-models.json.corrupt")
        if fm.fileExists(atPath: destination.path(percentEncoded: false)) {
            let suffix = Int(Date().timeIntervalSince1970)
            destination = directory.appending(path: "user-models.json.corrupt-\(suffix)")
        }

        do {
            try fm.moveItem(at: url, to: destination)
            logger.error(
                "user-models.json was unreadable and has been preserved at \(destination.lastPathComponent, privacy: .public): \(cause.localizedDescription, privacy: .public)"
            )
        } catch {
            logger.error(
                "Failed to preserve unreadable user-models.json: \(error.localizedDescription, privacy: .public); original load error: \(cause.localizedDescription, privacy: .public)"
            )
        }
    }

    private static func refreshedRuntimeMetadataForDownloadedACEModels(
        _ specs: [ModelSpec],
        baseDirectory: URL
    ) -> (specs: [ModelSpec], changed: Bool) {
        var changed = false
        let refreshedSpecs = specs.map { spec in
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
        return (refreshedSpecs, changed)
    }

    private func saveUserSpecs() {
        Self.saveUserSpecs(userSpecs, to: userSpecsFileURL)
    }

    private static func saveUserSpecs(_ specs: [ModelSpec], to url: URL) {
        do {
            let data = try JSONEncoder().encode(specs)
            try data.write(to: url, options: .atomic)
        } catch {
            logger.error("Failed to save user-models.json: \(error.localizedDescription)")
        }
    }

    private static func loadActiveModels(from baseDirectory: URL) -> [ModelCategory: String] {
        let url = baseDirectory.appending(path: "active-models.json")
        guard FileManager.default.fileExists(atPath: url.path(percentEncoded: false)) else {
            // First launch — default to built-in models
            return Self.defaultActiveModels
        }

        do {
            let data = try Data(contentsOf: url)
            let decoded = try JSONDecoder().decode([String: String].self, from: data)
            return Dictionary(
                uniqueKeysWithValues: decoded.compactMap { key, value in
                    guard let category = ModelCategory(rawValue: key) else { return nil }
                    return (category, value)
                }
            )
        } catch {
            logger.error("Failed to load active-models.json: \(error.localizedDescription). Using defaults.")
            return Self.defaultActiveModels
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
