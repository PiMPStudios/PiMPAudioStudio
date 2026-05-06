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

    private let registry: ModelRegistry
    private var loadedModelID: String?

    /// The model path for the currently active Stable Audio model.
    var modelPath: URL {
        let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        ).first!
        let base = appSupport
            .appendingPathComponent("Anvil", isDirectory: true)
            .appendingPathComponent("models", isDirectory: true)

        // Use active model's directory if available
        // Fall back to built-in stable-audio-open
        return base.appendingPathComponent("stable-audio-open", isDirectory: true)
    }

    var isReady: Bool { pipeline != nil }

    init(registry: ModelRegistry) {
        self.registry = registry
        if isModelDownloaded() {
            downloadState = .downloaded
        }
    }

    func downloadIfNeeded() async throws {
        guard downloadState != .downloaded else { return }

        let fm = FileManager.default
        try fm.createDirectory(at: modelPath, withIntermediateDirectories: true)

        downloadState = .downloading(progress: 0)

        // Get the active spec's repo, or fall back to built-in
        let activeSpec = await registry.activeSpec(for: .stableAudio)
        let repoID = activeSpec?.huggingFaceRepo ?? "stabilityai/stable-audio-open-1.0"
        let baseURL = "https://huggingface.co/\(repoID)/resolve/main/"
        let session = URLSession.shared

        // Metadata files
        let metadataFiles = ["config.json", "tokenizer.json"]
        for (idx, file) in metadataFiles.enumerated() {
            try Task.checkCancellation()
            guard let url = URL(string: baseURL + file) else { continue }
            let (data, response) = try await session.data(from: url)
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                let msg = "Failed to download \(file)"
                downloadState = .error(msg)
                throw SFXDownloadError.downloadFailed(file)
            }
            try data.write(to: modelPath.appendingPathComponent(file))
            downloadState = .downloading(progress: Float(idx + 1) / Float(metadataFiles.count) * 0.1)
        }

        // Weight files (large — use download task)
        let weightFiles = ["model.safetensors", "vae.safetensors", "clap_model.safetensors"]
        for (idx, file) in weightFiles.enumerated() {
            try Task.checkCancellation()
            guard let url = URL(string: baseURL + file) else { continue }
            let dest = modelPath.appendingPathComponent(file)
            let (tempURL, response) = try await session.download(from: url)
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                let msg = "Failed to download \(file)"
                downloadState = .error(msg)
                throw SFXDownloadError.downloadFailed(file)
            }
            if fm.fileExists(atPath: dest.path) {
                try fm.removeItem(at: dest)
            }
            try fm.moveItem(at: tempURL, to: dest)
            let progress = 0.1 + Float(idx + 1) / Float(weightFiles.count) * 0.9
            downloadState = .downloading(progress: progress)
        }

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

            // Check if model files exist at new location
            let hasFiles = spec.files.allSatisfy { file in
                FileManager.default.fileExists(
                    atPath: dir.appending(path: file).path()
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

        let clapEncoder = CLAPEncoder(embeddingDim: 512, numLayers: 12, numHeads: 8, vocabSize: 49408)
        let timingEncoder = TimingEncoder(embeddingDim: 256)
        let ditConfig = SFXDiTConfig(hiddenSize: 1024, numLayers: 24, numHeads: 16, latentDim: 64, conditioningDim: 768)
        let dit = SFXDiTActor(config: ditConfig)
        let diffusion = SFXDiffusionActor(dit: dit)
        let vaeConfig = SFXVAEConfig(latentDim: 64, outputChannels: 2, sampleRate: 44100, upsampleRatios: [8, 8, 4, 2])
        let vae = SFXVAEActor(config: vaeConfig)

        let sfxPipeline = SoundFXPipeline(
            clap: clapEncoder,
            diffusion: diffusion,
            vae: vae,
            timingEncoder: timingEncoder
        )
        self.pipeline = sfxPipeline
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
}

enum SFXDownloadError: Error, Sendable {
    case downloadFailed(String)
}
