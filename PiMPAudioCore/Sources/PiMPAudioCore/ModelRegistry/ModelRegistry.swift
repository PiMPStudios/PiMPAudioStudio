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

    /// Shared `silence_latent.pt` across ACE-Step v1.5 DiT repos.
    private static let aceStepSilenceLatentSHA256 =
        "8a2ef60fef2b175d83e420a0eb76c696dcb428494347ecb5ca5866cb4bab09f4"

    /// Shared XL `model.safetensors.index.json` (turbo / base / sft).
    private static let aceStepXLIndexSHA256 =
        "0a0c7e153a7bb6f6061b155739e9502cf83977bb959186d4df688595302a9474"

    private static func aceStepXLFiles(
        configSizeBytes: Int64,
        configSHA256: String,
        shard1: String,
        shard2: String,
        shard3: String,
        shard4: String
    ) -> [ModelFileSpec] {
        [
            ModelFileSpec(
                remotePath: "model.safetensors.index.json",
                sha256: aceStepXLIndexSHA256,
                sizeBytes: 71471
            ),
            ModelFileSpec(
                remotePath: "model-00001-of-00004.safetensors",
                sha256: shard1,
                sizeBytes: 4_986_971_456
            ),
            ModelFileSpec(
                remotePath: "model-00002-of-00004.safetensors",
                sha256: shard2,
                sizeBytes: 4_986_942_776
            ),
            ModelFileSpec(
                remotePath: "model-00003-of-00004.safetensors",
                sha256: shard3,
                sizeBytes: 4_986_942_808
            ),
            ModelFileSpec(
                remotePath: "model-00004-of-00004.safetensors",
                sha256: shard4,
                sizeBytes: 4_988_483_464
            ),
            ModelFileSpec(
                remotePath: "config.json",
                sha256: configSHA256,
                sizeBytes: configSizeBytes
            ),
            ModelFileSpec(
                remotePath: "silence_latent.pt",
                sha256: aceStepSilenceLatentSHA256,
                sizeBytes: 3_841_215
            ),
        ]
    }

    /// Tokenizer/vocab files shared by the 5 Hz LM repos.
    private static func aceStepLMTokenizerFiles(remotePrefix: String = "") -> [ModelFileSpec] {
        let prefix = remotePrefix.isEmpty ? "" : "\(remotePrefix)/"
        return [
            ModelFileSpec(
                remotePath: "\(prefix)tokenizer.json",
                localName: "tokenizer.json",
                sha256: "0823bb309a130de57e419bb1283214fea8e5cf3d375706c1dcbda810feae6aa8",
                sizeBytes: 24_321_939
            ),
            ModelFileSpec(
                remotePath: "\(prefix)tokenizer_config.json",
                localName: "tokenizer_config.json",
                sha256: "4d0ea85ee37d515583a92848d9cbeb5181a6294268201b2ea4317eff18812320",
                sizeBytes: 14_072_925
            ),
            ModelFileSpec(
                remotePath: "\(prefix)special_tokens_map.json",
                localName: "special_tokens_map.json",
                sha256: "76e233bfc357b0b03d7b6e6ba8e799244f358552575a7cdadb78d3d19106b298",
                sizeBytes: 1_824_199
            ),
            ModelFileSpec(
                remotePath: "\(prefix)added_tokens.json",
                localName: "added_tokens.json",
                sha256: "db08b66a515fb5d6acca0b3492d25bb44e0deda6241fc1113ac0679d40558c48",
                sizeBytes: 2_217_787
            ),
            ModelFileSpec(
                remotePath: "\(prefix)merges.txt",
                localName: "merges.txt",
                sha256: "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
                sizeBytes: 1_671_853
            ),
            ModelFileSpec(
                remotePath: "\(prefix)vocab.json",
                localName: "vocab.json",
                sha256: "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
                sizeBytes: 2_776_833
            ),
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
                    sha256: "077d06c5a16c9f9ff327389b66b0dd355fe3b8f7ad2777ef6767cbd9022d2a7c",
                    sizeBytes: 4_787_825_604
                ),
                ModelFileSpec(
                    remotePath: "acestep-v15-turbo/config.json",
                    localName: "config.json",
                    sha256: "74745ff704ea49164c3d2d1c99fc0670f3fc635a869f0aec2d1311e6a52d400a",
                    sizeBytes: 1968
                ),
                ModelFileSpec(
                    remotePath: "acestep-v15-turbo/silence_latent.pt",
                    localName: "silence_latent.pt",
                    sha256: aceStepSilenceLatentSHA256,
                    sizeBytes: 3_841_215
                ),
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
                    sha256: "897a3f6af4bf1d87a88a6de2c93f13e36a446764496a40bc6f7c1d94032b0223",
                    sizeBytes: 4_787_825_604
                ),
                ModelFileSpec(
                    remotePath: "config.json",
                    localName: "config.json",
                    sha256: "9bb4f832f2e5e6c8bf7bddab3c6a6da1b13c01d072efbf0ed3c830536c473359",
                    sizeBytes: 1940
                ),
                ModelFileSpec(
                    remotePath: "silence_latent.pt",
                    localName: "silence_latent.pt",
                    sha256: aceStepSilenceLatentSHA256,
                    sizeBytes: 3_841_215
                ),
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
                    sha256: "dc9cedbc5b0c55559e71047de50bd80060d39f30d7e8af7df00bdb275ab5867e",
                    sizeBytes: 4_787_825_604
                ),
                ModelFileSpec(
                    remotePath: "config.json",
                    localName: "config.json",
                    sha256: "9bb4f832f2e5e6c8bf7bddab3c6a6da1b13c01d072efbf0ed3c830536c473359",
                    sizeBytes: 1940
                ),
                ModelFileSpec(
                    remotePath: "silence_latent.pt",
                    localName: "silence_latent.pt",
                    sha256: aceStepSilenceLatentSHA256,
                    sizeBytes: 3_841_215
                ),
            ],
            category: .aceStepDiT,
            variantFamily: "ace-step-1.5"
        ),
        ModelSpec(
            id: "ace-step-dit-xl-turbo",
            displayName: "ACE-Step v1.5 XL Turbo",
            huggingFaceRepo: "ACE-Step/acestep-v15-xl-turbo",
            files: aceStepXLFiles(
                configSizeBytes: 2407,
                configSHA256: "e3665376d46477cbd3e76203cd4e241f11a6a6bb5f88464c49f3c6aae51a918b",
                shard1: "ec9eadbaca0a158f081202251df17f0705005748894a156e616cdef444e21e36",
                shard2: "18a5eb68cd09c1828948f9e0152c0e2ba88213a7d3b372af7b48e73685c64dbe",
                shard3: "b513523ba4fc2b15a87cfd718448ad3e551133f8a5b5d4c6a8fde1ab8210e354",
                shard4: "facfecb98896340eb5df7e06efdeccafe182408d2ea04ce4a99623bf4bfa5c9e"
            ),
            category: .aceStepDiT,
            variantFamily: "ace-step-1.5-xl"
        ),
        ModelSpec(
            id: "ace-step-dit-xl-base",
            displayName: "ACE-Step v1.5 XL Base",
            huggingFaceRepo: "ACE-Step/acestep-v15-xl-base",
            files: aceStepXLFiles(
                configSizeBytes: 2525,
                configSHA256: "174cf8c4afc2c41212546b4dca9afb11e7958f8b1e1a770d7cd009a82d72a6f1",
                shard1: "1749e9905cb25caa445928249c0f5136b0995e7194c5155d4ffacea7747e4e8d",
                shard2: "03b39bebb8a9e9522b4f53dd3d03d34ff40e9ca6a573057b421c953b4336ae6f",
                shard3: "1d5550d66f5d6078ec7fb479c37bad7284e6b0848231dbef781c6cfc2893b8da",
                shard4: "e66354e8035ff5758b1d5fe4dc14216dc66023a4555d496ea3faa6e63617e382"
            ),
            category: .aceStepDiT,
            variantFamily: "ace-step-1.5-xl"
        ),
        ModelSpec(
            id: "ace-step-dit-xl-sft",
            displayName: "ACE-Step v1.5 XL SFT",
            huggingFaceRepo: "ACE-Step/acestep-v15-xl-sft",
            files: aceStepXLFiles(
                configSizeBytes: 2525,
                configSHA256: "174cf8c4afc2c41212546b4dca9afb11e7958f8b1e1a770d7cd009a82d72a6f1",
                shard1: "0a9cfd65eef84c0f25724e4d88e3e630feb4c82437c5f02df74ad4bdff9f347b",
                shard2: "871f801cb5e3d5b8fb5505234340da458761693d0b1b2d82246c4c49bb19b01d",
                shard3: "e0b2f38d2930844aa128afc3ed7ed075bb817e4befdc7f3a054e28e1eb15262b",
                shard4: "113055326f9855034c33d47eeab1db79eb49acf4fd14924c0956c30a8147b0d0"
            ),
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
                    sha256: "e8d754cd5cb69cf864a2d44a61b39bf61354ebac7fc70680bbfdb8ed49b1bc8b",
                    sizeBytes: 337_431_388
                ),
                ModelFileSpec(
                    remotePath: "vae/config.json",
                    localName: "config.json",
                    sha256: "14e019904df567f26df750317a70e2bd08f9f8f3c40ff4a24c97d1cd3f20ccd2",
                    sizeBytes: 425
                ),
                ModelFileSpec(
                    remotePath: "Qwen3-Embedding-0.6B/tokenizer.json",
                    localName: "tokenizer.json",
                    sha256: "e1c5ae75d50677cf0cce3ad0441fbf25396122838224b9473c03004e92077538",
                    sizeBytes: 11_423_705
                ),
                ModelFileSpec(
                    remotePath: "Qwen3-Embedding-0.6B/model.safetensors",
                    localName: "qwen3_model.safetensors",
                    sha256: "50e3c282187ba05fed66edb58ffbba6f4ce258bd2d3dcda860b8dc125e041bab",
                    sizeBytes: 1_191_586_416
                ),
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
                    sizeBytes: 4172
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
                    sizeBytes: 1497
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
                ),
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
                ModelFileSpec(
                    remotePath: "model.safetensors",
                    sha256: "b7faa8fb8c208849547000dae6d0e6a54fbf4d3f0f236f8c46636d530f709f11",
                    sizeBytes: 1_325_804_024
                ),
                ModelFileSpec(
                    remotePath: "config.json",
                    sha256: "873ca63808b74e1e008ab950114765553b898b489b9e077e80db83348a118384",
                    sizeBytes: 1386
                ),
            ] + aceStepLMTokenizerFiles(),
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
                    sha256: "b2a2c0d5de99b6255e72a519af42ded22fa26f60bdaf408aa659e340c172ab0c",
                    sizeBytes: 3_708_521_528
                ),
                ModelFileSpec(
                    remotePath: "acestep-5Hz-lm-1.7B/config.json",
                    localName: "config.json",
                    sha256: "73515ca4bfd86a5b56f28bf680311d1911e159c85d7f7ceb2188e310007c9010",
                    sizeBytes: 1385
                ),
            ] + aceStepLMTokenizerFiles(remotePrefix: "acestep-5Hz-lm-1.7B") + [
                ModelFileSpec(
                    remotePath: "acestep-5Hz-lm-1.7B/chat_template.jinja",
                    localName: "chat_template.jinja",
                    sha256: "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8",
                    sizeBytes: 4168
                ),
            ],
            category: .aceStepLM,
            variantFamily: "ace-step-lm"
        ),
        ModelSpec(
            id: "ace-step-lm-4b",
            displayName: "ACE-Step LM 4B",
            huggingFaceRepo: "ACE-Step/acestep-5Hz-lm-4B",
            files: [
                ModelFileSpec(
                    remotePath: "model-00001-of-00002.safetensors",
                    sha256: "c9e9c9a05ca539d6f618f2c798c5672dbe53635dd2d6741ede979319429f6dad",
                    sizeBytes: 4_999_914_896
                ),
                ModelFileSpec(
                    remotePath: "model-00002-of-00002.safetensors",
                    sha256: "55088d893e0ea7a8a117bf1cee23640ffb44ccf7c9efdf3e09e6a63e3efac96a",
                    sizeBytes: 3_379_239_240
                ),
                ModelFileSpec(
                    remotePath: "model.safetensors.index.json",
                    sha256: "8a9ec992fbae144ba9b40521f8763a3419546bd7a8864d80b0c3f1e1e3678fcf",
                    sizeBytes: 32818
                ),
                ModelFileSpec(
                    remotePath: "config.json",
                    sha256: "211bb68b2d641360526e4654b524e508f4046c1292016c08555bb877bd686d1b",
                    sizeBytes: 1562
                ),
            ] + aceStepLMTokenizerFiles(),
            category: .aceStepLM,
            variantFamily: "ace-step-lm"
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
        "ace-step-vae",
    ]

    // MARK: - Active Model Tracking

    private static let defaultActiveModels: [ModelCategory: String] = [
        .aceStepDiT: "ace-step-dit-turbo",
        .aceStepVAE: "ace-step-vae",
        .stableAudio: "stable-audio-open",
    ]

    private static let retiredBuiltInModelIDs: Set<String> = [
        "lyric-llm",
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
            userSpecs = []
        case let .loaded(specs, shouldPersist):
            userSpecs = specs
            if shouldPersist {
                Self.saveUserSpecs(
                    specs,
                    to: baseDirectory.appending(path: "user-models.json")
                )
            }
        case .corrupt:
            // Keep an empty in-memory list; the corrupt index was preserved on disk.
            userSpecs = []
        }
        activeModels = Self.strippingRetiredActiveModels(
            Self.migratedActiveModelsIfNeeded(
                activeModels: Self.loadActiveModels(from: baseDirectory),
                userSpecs: userSpecs,
                baseDirectory: baseDirectory,
                allowOrphanCleanup: loadedUserSpecs.allowsDestructiveOrphanCleanup
            )
        )
        Self.removeRetiredBuiltInDirectories(from: baseDirectory)
        let activeURL = baseDirectory.appending(path: "active-models.json")
        if Self.activeModelsFileNeedsRewrite(activeModels, at: activeURL) {
            Self.saveActiveModels(activeModels, to: activeURL)
        }
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
                if name == modelMigrationMarkerFileName {
                    continue
                }
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

    private static func strippingRetiredActiveModels(
        _ models: [ModelCategory: String]
    ) -> [ModelCategory: String] {
        models.filter { !retiredBuiltInModelIDs.contains($0.value) }
    }

    private static func removeRetiredBuiltInDirectories(from baseDirectory: URL) {
        for id in retiredBuiltInModelIDs {
            let url = baseDirectory.appending(path: id)
            var isDir: ObjCBool = false
            let exists = FileManager.default.fileExists(
                atPath: url.path(percentEncoded: false),
                isDirectory: &isDir
            )
            guard exists, isDir.boolValue else { continue }
            try? FileManager.default.removeItem(at: url)
            logger.info("Removed retired built-in model directory: \(id)")
        }
    }

    private static func activeModelsFileNeedsRewrite(
        _ models: [ModelCategory: String],
        at url: URL
    ) -> Bool {
        guard FileManager.default.fileExists(atPath: url.path(percentEncoded: false)) else {
            return false
        }
        guard let data = try? Data(contentsOf: url),
              let decoded = try? JSONDecoder().decode([String: String].self, from: data)
        else {
            return false
        }
        let expected = Dictionary(
            uniqueKeysWithValues: models.map { ($0.key.rawValue, $0.value) }
        )
        return decoded != expected
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
        case let .downloading(current) where progress < current:
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
                       case let .httpError(code) = httpError,
                       code == 401 || code == 403,
                       bearerToken != nil {
                        statuses[spec.id] = .failed("HuggingFace rejected the token")
                        throw DownloadError.authForbidden(repo: spec.huggingFaceRepo)
                    }
                    let message: String = if case DownloadError.checksumMismatch = error {
                        "Model verification failed. The file may be corrupted or tampered with."
                    } else {
                        error.localizedDescription
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
                true
            case .corrupt:
                false
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
        Self.saveActiveModels(activeModels, to: activeModelsFileURL)
    }

    private static func saveActiveModels(
        _ models: [ModelCategory: String],
        to url: URL
    ) {
        let encoded = Dictionary(
            uniqueKeysWithValues: models.map { ($0.key.rawValue, $0.value) }
        )
        do {
            let data = try JSONEncoder().encode(encoded)
            try data.write(to: url, options: .atomic)
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
            "Built-in models cannot be modified."
        case let .duplicateModel(id):
            "Model '\(id)' already exists."
        case let .unknownModel(id):
            "Unknown model '\(id)'."
        case let .categoryMismatch(expected, got):
            "Category mismatch: expected \(expected.rawValue), got \(got.rawValue)."
        case let .modelNotReady(id):
            "Model '\(id)' is not downloaded yet."
        case let .modelNotRunnable(name, note):
            note ?? "\(name) can be downloaded, but this build cannot run it yet."
        }
    }
}
