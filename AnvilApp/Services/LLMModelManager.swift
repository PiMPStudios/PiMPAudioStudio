import Foundation
import AnvilCore
import MLX

// MARK: - LLMModelSpec

/// Describes a downloadable LLM model.
struct LLMModelSpec: Identifiable, Codable, Sendable, Equatable {
    let id: String
    let displayName: String
    let repoID: String
    let sizeBytes: Int64
    let isDefault: Bool
}

// MARK: - LLMDownloadState

/// Represents the current download state of a model.
enum LLMDownloadState: Equatable, Sendable {
    case idle
    case downloading(progress: Float)
    case error(String)
}

// MARK: - LLMModelManager

/// Manages downloading, deleting, and activating local LLM models.
@MainActor
final class LLMModelManager: ObservableObject {

    // MARK: Published State

    @Published var availableModels: [LLMModelSpec]
    @Published var activeModel: LLMModelSpec?
    @Published var downloadState: LLMDownloadState = .idle
    @Published var llmActor: LLMActor?

    // MARK: Defaults

    static let defaultModels: [LLMModelSpec] = [
        LLMModelSpec(
            id: "llama-3.2-3b-instruct-4bit",
            displayName: "Llama 3.2 3B Instruct (4-bit)",
            repoID: "mlx-community/Llama-3.2-3B-Instruct-4bit",
            sizeBytes: 1_800_000_000,
            isDefault: true
        )
    ]

    // MARK: Storage

    var modelsDirectory: URL {
        let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        ).first!
        return appSupport
            .appendingPathComponent("Anvil", isDirectory: true)
            .appendingPathComponent("LLM", isDirectory: true)
    }

    private let modelsKey = "LLMModelManager.availableModels"
    private let activeModelKey = "LLMModelManager.activeModelID"

    // MARK: Init

    init() {
        // Load persisted available models or fall back to defaults
        if let data = UserDefaults.standard.data(forKey: modelsKey),
           let decoded = try? JSONDecoder().decode([LLMModelSpec].self, from: data) {
            self.availableModels = decoded
        } else {
            self.availableModels = Self.defaultModels
        }

        // Load persisted active model selection
        if let activeID = UserDefaults.standard.string(forKey: activeModelKey) {
            self.activeModel = availableModels.first { $0.id == activeID }
        }

        // Drop the persisted active selection if its files aren't actually on
        // disk — otherwise the UI shows a stale "active" indicator while a
        // re-download is in flight or after the model was partially deleted.
        // The user can reactivate after the download completes.
        if let active = activeModel, !isModelDownloaded(active) {
            self.activeModel = nil
            UserDefaults.standard.removeObject(forKey: activeModelKey)
        }

        // Auto-activate the default local writer if it is already on disk.
        // Background download can complete before the user opens Settings, and
        // Write Lyrics should work after that without a separate Activate click.
        if activeModel == nil,
           let downloadedDefault = availableModels.first(where: { $0.isDefault && isModelDownloaded($0) }) {
            self.activeModel = downloadedDefault
            UserDefaults.standard.set(downloadedDefault.id, forKey: activeModelKey)
        }

        // Create actor if active model is already downloaded
        if let active = activeModel, isModelDownloaded(active) {
            let path = modelsDirectory.appendingPathComponent(active.id)
            self.llmActor = LLMActor(modelPath: path)
        }
    }

    // MARK: - Public API

    /// Downloads a model's files from HuggingFace.
    func downloadModel(_ spec: LLMModelSpec) async throws {
        let fm = FileManager.default
        let modelDir = modelsDirectory.appendingPathComponent(spec.id)

        try fm.createDirectory(at: modelDir, withIntermediateDirectories: true)

        downloadState = .downloading(progress: 0)

        let baseURL = "https://huggingface.co/\(spec.repoID)/resolve/main/"
        let session = URLSession.shared

        // Required metadata files
        let metadataFiles = ["config.json", "tokenizer.json", "tokenizer_config.json"]

        for (index, filename) in metadataFiles.enumerated() {
            try Task.checkCancellation()
            guard let url = URL(string: baseURL + filename) else {
                downloadState = .error("Invalid URL for \(filename)")
                throw LLMError.missingFile(filename)
            }

            let (data, response) = try await session.data(from: url)
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                // tokenizer_config.json is optional for some models
                if filename == "tokenizer_config.json" { continue }
                let msg = "Failed to download \(filename)"
                downloadState = .error(msg)
                throw LLMError.missingFile(filename)
            }

            let dest = modelDir.appendingPathComponent(filename)
            try data.write(to: dest)

            let progress = Float(index + 1) / Float(metadataFiles.count + 1) * 0.2
            downloadState = .downloading(progress: progress)
        }

        // Download safetensors — try index first, then single file
        try await downloadSafetensors(spec: spec, modelDir: modelDir, baseURL: baseURL, session: session)

        downloadState = .idle
        persistModels()
        if activeModel == nil {
            setActiveModel(spec)
        }
    }

    /// Removes a downloaded model from disk and clears actor if active.
    func deleteModel(_ spec: LLMModelSpec) {
        let modelDir = modelsDirectory.appendingPathComponent(spec.id)
        try? FileManager.default.removeItem(at: modelDir)

        if activeModel == spec {
            activeModel = nil
            llmActor = nil
            UserDefaults.standard.removeObject(forKey: activeModelKey)
        }
    }

    /// Sets the given model as the active model and creates a new LLMActor.
    func setActiveModel(_ spec: LLMModelSpec) {
        activeModel = spec
        let path = modelsDirectory.appendingPathComponent(spec.id)
        llmActor = LLMActor(modelPath: path)
        UserDefaults.standard.set(spec.id, forKey: activeModelKey)
    }

    /// Returns the file URL of the active model's directory, if available.
    func resolveModelPath() -> URL? {
        guard let active = activeModel else { return nil }
        let path = modelsDirectory.appendingPathComponent(active.id)
        return FileManager.default.fileExists(atPath: path.path) ? path : nil
    }

    /// Releases the LLMActor and flushes the MLX GPU allocator pool.
    ///
    /// Call this immediately before starting a heavy generation pass
    /// (DiT + VAE weigh ~5+ GB) so their ~2 GB footprint doesn't pile on
    /// top of the resident LLM weights.  The `$llmActor` publisher fires
    /// automatically, propagating `nil` to PromptEnhancer, IntentRouter,
    /// and LyricWriterManager — disabling inference during generation.
    func evict(clearCache: Bool = true) {
        llmActor = nil
        if clearCache {
            MLX.GPU.clearCache()
        }
    }

    /// Recreates the LLMActor from the active model path (lazy — weights are
    /// NOT loaded until the first inference call).
    ///
    /// Call this after generation completes (success, failure, or cancel) to
    /// restore prompt-enhancement capability.  The `$llmActor` publisher fires
    /// and re-enables PromptEnhancer / IntentRouter / LyricWriterManager.
    func reload() {
        guard let active = activeModel, isModelDownloaded(active) else { return }
        let path = modelsDirectory.appendingPathComponent(active.id)
        llmActor = LLMActor(modelPath: path)
    }

    /// Checks whether a model is fully downloaded — config.json, tokenizer.json,
    /// AND at least one .safetensors body file are all present.
    ///
    /// Previously this only checked the two metadata files, which complete in
    /// the first ~20% of a download (3 small JSON files). The 1.8 GB+ model
    /// body would still be in flight, but `isModelDownloaded` returned true,
    /// so the UI showed the model as ready and the auto-download skip logic
    /// thought it didn't need to run. Now we require an actual safetensors
    /// body file (single or sharded) to be on disk.
    func isModelDownloaded(_ spec: LLMModelSpec) -> Bool {
        let dir = modelsDirectory.appendingPathComponent(spec.id)
        let fm = FileManager.default
        let hasConfig = fm.fileExists(atPath: dir.appendingPathComponent("config.json").path(percentEncoded: false))
        let hasTokenizer = fm.fileExists(atPath: dir.appendingPathComponent("tokenizer.json").path(percentEncoded: false))
        guard hasConfig, hasTokenizer else { return false }

        // Look for any *.safetensors file in the model dir. Single-file models
        // produce `model.safetensors`; sharded models produce
        // `model-00001-of-NNNNN.safetensors` etc. (downloadFile only writes
        // the destination after a successful complete transfer, so any
        // present .safetensors file is fully downloaded).
        guard let entries = try? fm.contentsOfDirectory(atPath: dir.path(percentEncoded: false)) else {
            return false
        }
        return entries.contains { $0.hasSuffix(".safetensors") }
    }

    /// Validates and adds a custom model from a HuggingFace repo ID.
    func addCustomModel(repoID: String) async throws {
        let id = repoID.replacingOccurrences(of: "/", with: "-").lowercased()

        guard !availableModels.contains(where: { $0.id == id }) else { return }

        let baseURL = "https://huggingface.co/\(repoID)/resolve/main/"
        let session = URLSession.shared
        let requiredFiles = ["config.json", "tokenizer.json"]

        for file in requiredFiles {
            guard let url = URL(string: baseURL + file) else {
                throw LLMError.missingFile(file)
            }
            var request = URLRequest(url: url)
            request.httpMethod = "HEAD"
            let (_, response) = try await session.data(for: request)
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                throw LLMError.missingFile(file)
            }
        }

        let displayName = repoID.components(separatedBy: "/").last ?? repoID
        let spec = LLMModelSpec(
            id: id,
            displayName: displayName,
            repoID: repoID,
            sizeBytes: 0,
            isDefault: false
        )
        availableModels.append(spec)
        persistModels()
    }

    /// Removes a custom (non-default) model from the available list and disk.
    func removeCustomModel(_ spec: LLMModelSpec) {
        guard !spec.isDefault else { return }
        deleteModel(spec)
        availableModels.removeAll { $0.id == spec.id }
        persistModels()
    }

    // MARK: - Private Helpers

    private func downloadSafetensors(
        spec: LLMModelSpec,
        modelDir: URL,
        baseURL: String,
        session: URLSession
    ) async throws {
        // Try model.safetensors.index.json first (sharded models)
        let indexName = "model.safetensors.index.json"
        if let indexURL = URL(string: baseURL + indexName) {
            let (indexData, indexResponse) = try await session.data(from: indexURL)
            if let http = indexResponse as? HTTPURLResponse, http.statusCode == 200 {
                try indexData.write(to: modelDir.appendingPathComponent(indexName))

                let shardFiles = try parseShardFiles(from: indexData)
                for (idx, shard) in shardFiles.enumerated() {
                    try Task.checkCancellation()
                    guard let shardURL = URL(string: baseURL + shard) else { continue }
                    let dest = modelDir.appendingPathComponent(shard)
                    try await downloadFile(from: shardURL, to: dest, session: session)

                    let progress = 0.2 + Float(idx + 1) / Float(shardFiles.count) * 0.8
                    downloadState = .downloading(progress: progress)
                }
                return
            }
        }

        // Fallback: single model.safetensors
        let singleName = "model.safetensors"
        guard let singleURL = URL(string: baseURL + singleName) else {
            downloadState = .error("Invalid safetensors URL")
            throw LLMError.missingFile(singleName)
        }

        let dest = modelDir.appendingPathComponent(singleName)
        try await downloadFile(from: singleURL, to: dest, session: session)
        downloadState = .downloading(progress: 1.0)
    }

    private func downloadFile(from url: URL, to destination: URL, session: URLSession) async throws {
        let (tempURL, response) = try await session.download(from: url)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            let name = url.lastPathComponent
            let msg = "Failed to download \(name)"
            downloadState = .error(msg)
            throw LLMError.missingFile(name)
        }
        try FileManager.default.moveItem(at: tempURL, to: destination)
    }

    private func parseShardFiles(from indexData: Data) throws -> [String] {
        struct SafetensorsIndex: Codable {
            let weight_map: [String: String]
        }

        let index = try JSONDecoder().decode(SafetensorsIndex.self, from: indexData)
        // Deduplicate shard filenames while preserving order
        var seen = Set<String>()
        var result: [String] = []
        for file in index.weight_map.values.sorted() {
            if seen.insert(file).inserted {
                result.append(file)
            }
        }
        return result
    }

    private func persistModels() {
        if let data = try? JSONEncoder().encode(availableModels) {
            UserDefaults.standard.set(data, forKey: modelsKey)
        }
    }
}
