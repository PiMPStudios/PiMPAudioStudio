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

    private let modelID = "stable-audio-open-1.0"
    private let repoID = "stabilityai/stable-audio-open-1.0"

    var modelPath: URL {
        let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        ).first!
        return appSupport
            .appendingPathComponent("Anvil", isDirectory: true)
            .appendingPathComponent("models", isDirectory: true)
            .appendingPathComponent(modelID, isDirectory: true)
    }

    var isReady: Bool { pipeline != nil }

    init() {
        if isModelDownloaded() {
            downloadState = .downloaded
        }
    }

    func downloadIfNeeded() async throws {
        guard downloadState != .downloaded else { return }

        let fm = FileManager.default
        try fm.createDirectory(at: modelPath, withIntermediateDirectories: true)

        downloadState = .downloading(progress: 0)

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
