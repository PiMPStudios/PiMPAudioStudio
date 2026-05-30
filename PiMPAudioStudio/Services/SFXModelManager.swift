// PiMPAudioStudio/Services/SFXModelManager.swift
import Foundation
import PiMPAudioCore
import MLXNN

enum SFXDownloadState: Equatable, Sendable {
    case notDownloaded
    case downloading(progress: Float)
    case downloaded
    case error(String)
}

@MainActor
final class SFXModelManager: ObservableObject {
    @Published var downloadState: SFXDownloadState = .notDownloaded
    @Published var pipeline: SoundFXPipeline?

    /// HF tokenizer for the T5-base text encoder (loads from `tokenizer.json`
    /// shipped under SAO's `tokenizer/` dir). Replaces the previous
    /// CLAPTokenizer wiring.
    private(set) var tokenizer: PiMPAudioTokenizer?

    private let registry: ModelRegistry
    private var loadedModelID: String?
    private var resolvedModelPath: URL?

    /// The model path for the currently active Stable Audio model.
    var modelPath: URL {
        resolvedModelPath ?? Self.baseModelsDirectory()
            .appendingPathComponent("stable-audio-open", isDirectory: true)
    }

    var isReady: Bool { pipeline != nil }

    init(registry: ModelRegistry) {
        self.registry = registry
        if isModelDownloaded() {
            downloadState = .downloaded
        }
    }

    /// Canonical base directory for all PiMP Audio Studio model files.
    private static func baseModelsDirectory() -> URL {
        AppStorageLocations.modelsDirectory
    }

    func downloadIfNeeded() async throws {
        // Resolve the model path first so modelPath is correct even on early return
        let activeSpec = await registry.activeSpec(for: .stableAudio)
        let base = Self.baseModelsDirectory()
        resolvedModelPath = activeSpec.map { base.appendingPathComponent($0.id, isDirectory: true) }
            ?? base.appendingPathComponent("stable-audio-open", isDirectory: true)

        guard downloadState != .downloaded else { return }

        let repoID = activeSpec?.huggingFaceRepo ?? "stabilityai/stable-audio-open-1.0"

        let fm = FileManager.default
        try fm.createDirectory(at: modelPath, withIntermediateDirectories: true)

        downloadState = .downloading(progress: 0)

        let baseURL = "https://huggingface.co/\(repoID)/resolve/main/"
        let session = URLSession.shared

        // Use the registry-driven download path so we honor the spec's
        // remotePath → localName mapping (e.g., the VAE lives at
        // `vae/diffusion_pytorch_model.safetensors` upstream but is saved
        // locally as `vae.safetensors`). The previous parallel
        // hardcoded-list path assumed flat filenames at the repo root —
        // which gave 404s on the safetensors VAE — and didn't carry HF
        // auth headers for gated models.
        if let spec = activeSpec {
            try await registry.downloadModels([spec]) { _, progress in
                Task { @MainActor in
                    self.downloadState = .downloading(progress: Float(progress))
                }
            }
        }
        _ = baseURL  // intentionally unused now — registry handles URLs
        _ = session  // intentionally unused now — registry uses DownloadClient
        _ = fm       // intentionally unused now

        downloadState = .downloaded
    }

    /// Checks if the active model has changed and reloads if needed.
    func ensureActiveModel() async throws {
        let activeID = await registry.activeModelID(for: .stableAudio)
        guard let activeID else { return }

        if activeID != loadedModelID && pipeline != nil {
            // Active model changed — unload current pipeline
            pipeline = nil
        }

        if pipeline == nil {
            let spec = await registry.activeSpec(for: .stableAudio)
            guard let spec else { return }
            let baseDir = await registry.baseDirectory
            let dir = spec.localDirectory(base: baseDir)

            // Resolve and store the model path for synchronous access
            resolvedModelPath = dir

            // Check if model files exist at new location
            let hasFiles = spec.files.allSatisfy { fileSpec in
                FileManager.default.fileExists(
                    atPath: dir.appending(path: fileSpec.localName).path(percentEncoded: false)
                )
            }

            if hasFiles {
                downloadState = .downloaded
                try await loadPipeline()
                loadedModelID = activeID
            }
        }
    }

    func loadPipeline() async throws {
        guard isModelDownloaded() else { return }

        let converter = WeightConverter()
        let ditWeights = try await converter.loadSFXDiTWeights(from: modelPath.appendingPathComponent("model.safetensors"))
        let vaeWeights = try await converter.loadSFXVAEWeights(from: modelPath.appendingPathComponent("vae.safetensors"))

        // T5-base text encoder.
        let t5Weights = try await converter.loadT5Weights(from: modelPath.appendingPathComponent("text_encoder.safetensors"))

        let t5 = T5Encoder(config: .t5Base)
        t5.update(parameters: ModuleParameters.unflattened(t5Weights))

        // SAO number conditioner (seconds_start + seconds_total timing).
        let conditionerWeights = try await converter.loadSAOProjectionWeights(from: modelPath.appendingPathComponent("projection.safetensors"))

        let conditioner = SAONumberConditioner()
        conditioner.update(parameters: ModuleParameters.unflattened(conditionerWeights))

        // DiT — full SAO 1.0 architecture (hiddenSize=1536, GQA 24/12 heads).
        let ditConfig = SFXDiTConfig.sao
        let dit = SFXDiTActor(config: ditConfig)
        try await dit.loadWeights(ditWeights)

        let diffusion = SFXDiffusionActor(dit: dit)

        let vaeConfig = SFXVAEConfig(
            latentDim: 64, outputChannels: 2, sampleRate: 44100, upsampleRatios: [8, 8, 4, 4, 2]
        )
        let vae = SFXVAEActor(config: vaeConfig)
        await vae.loadWeights(vaeWeights)

        // T5 tokenizer ships at `tokenizer/tokenizer.json` upstream; saved
        // locally as `tokenizer.json` alongside the other SAO model files.
        let tokenizerURL = modelPath.appendingPathComponent("tokenizer.json")
        self.tokenizer = try PiMPAudioTokenizer(configURL: tokenizerURL)

        self.pipeline = SoundFXPipeline(
            t5: t5,
            conditioner: conditioner,
            diffusion: diffusion,
            vae: vae
        )
    }

    /// Release the in-memory SFX pipeline, freeing GPU memory.
    /// The downloaded model files remain on disk and can be reloaded via loadPipeline().
    func unloadPipeline() {
        pipeline = nil
        tokenizer = nil
        loadedModelID = nil
    }

    func deleteWeights() {
        try? FileManager.default.removeItem(at: modelPath)
        downloadState = .notDownloaded
        pipeline = nil
        loadedModelID = nil
    }

    private func isModelDownloaded() -> Bool {
        let fm = FileManager.default
        let hasConfig = fm.fileExists(atPath: modelPath.appendingPathComponent("config.json").path)
        let hasModel = fm.fileExists(atPath: modelPath.appendingPathComponent("model.safetensors").path)
        return hasConfig && hasModel
    }

    /// Stricter check than `isModelDownloaded()` — verifies every file
    /// `loadPipeline()` will actually try to read is present. Used by
    /// `GenerationService.shouldUseStableAudio` so Auto-mode routing can
    /// safely fall back to ACE-Step when the SFX install is incomplete
    /// (e.g., the registry-driven download gives us `vae_model.ckpt` but
    /// loadPipeline() looks for `vae.safetensors` — pre-existing wiring
    /// gap that crashes generation on entry to the SFX path).
    func isPipelineLoadable() -> Bool {
        let fm = FileManager.default
        // Match the file list that `loadPipeline()` actually reads. CLAP
        // (`clap_model.safetensors`) is no longer required — replaced by
        // the T5 text encoder + projection model. Until the T5 forward
        // pass is enabled this method returns false until ALL the new
        // files are present, which is what the spec downloader provides
        // when `requiresAuth` HF auth is configured.
        let required = [
            "config.json",
            "model.safetensors",
            "vae.safetensors",
            "text_encoder.safetensors",
            "projection.safetensors",
            "tokenizer.json",
        ]
        return required.allSatisfy { name in
            fm.fileExists(
                atPath: modelPath.appendingPathComponent(name)
                    .path(percentEncoded: false)
            )
        }
    }
}

enum SFXDownloadError: Error, Sendable {
    case downloadFailed(String)
}
