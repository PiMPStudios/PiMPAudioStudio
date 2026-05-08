// AnvilApp/Services/SFXModelManager.swift
import Foundation
import AnvilCore

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

    private(set) var tokenizer: CLAPTokenizer?

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

    /// Canonical base directory for all Anvil model files.
    private static func baseModelsDirectory() -> URL {
        let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        ).first!
        return appSupport
            .appendingPathComponent("Anvil", isDirectory: true)
            .appendingPathComponent("models", isDirectory: true)
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
        let ditWeights  = try await converter.loadSFXDiTWeights(from: modelPath.appendingPathComponent("model.safetensors"))
        let vaeWeights  = try await converter.loadSFXVAEWeights(from: modelPath.appendingPathComponent("vae.safetensors"))
        let clapWeights = try await converter.loadCLAPWeights(from: modelPath.appendingPathComponent("clap_model.safetensors"))

        let clapEncoder = CLAPEncoder(embeddingDim: 512, numLayers: 12, numHeads: 8, vocabSize: 49408)
        await clapEncoder.loadWeights(clapWeights)

        let timingEncoder = TimingEncoder(embeddingDim: 256)

        let ditConfig = SFXDiTConfig(
            hiddenSize: 1024, numLayers: 24, numHeads: 16, latentDim: 64, conditioningDim: 768
        )
        let dit = SFXDiTActor(config: ditConfig)
        try await dit.loadWeights(ditWeights)

        let diffusion = SFXDiffusionActor(dit: dit)

        let vaeConfig = SFXVAEConfig(
            latentDim: 64, outputChannels: 2, sampleRate: 44100, upsampleRatios: [8, 8, 4, 2]
        )
        let vae = SFXVAEActor(config: vaeConfig)
        await vae.loadWeights(vaeWeights)

        let tokenizerURL = modelPath.appendingPathComponent("tokenizer.json")
        let configURL    = modelPath.appendingPathComponent("config.json")
        self.tokenizer = try CLAPTokenizer(vocabularyURL: tokenizerURL, configURL: configURL)

        self.pipeline = SoundFXPipeline(
            clap: clapEncoder,
            diffusion: diffusion,
            vae: vae,
            timingEncoder: timingEncoder
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
        let required = [
            "config.json",
            "model.safetensors",
            "vae.safetensors",
            "clap_model.safetensors",
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
