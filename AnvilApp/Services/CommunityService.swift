import Foundation
import AnvilCore
import MLX

/// Manages community adapter discovery, downloading, and HuggingFace imports.
@MainActor
final class CommunityService: ObservableObject {

    // MARK: - Published State

    @Published var manifest: AdapterManifest?
    @Published var downloads: [String: DownloadState] = [:]
    @Published var lastRefresh: Date?
    @Published var isRefreshing: Bool = false
    @Published var error: String?

    // MARK: - Types

    enum DownloadState: Equatable {
        case downloading(progress: Double)
        case failed(String)
        case completed

        static func == (lhs: DownloadState, rhs: DownloadState) -> Bool {
            switch (lhs, rhs) {
            case (.downloading(let a), .downloading(let b)): return a == b
            case (.failed(let a), .failed(let b)): return a == b
            case (.completed, .completed): return true
            default: return false
            }
        }
    }

    // MARK: - Configuration

    private let manifestURL = URL(string:
        "https://raw.githubusercontent.com/PiMPStudios/anvil-adapters/main/manifest.json")!
    private let cacheTTL: TimeInterval = 3600  // 1 hour
    private static let maxConcurrentDownloads = 3

    // MARK: - Private State

    private var activeTasks: [String: Task<Void, Never>] = [:]
    private var cachedManifest: AdapterManifest?
    private var cacheTimestamp: Date?

    // MARK: - Manifest

    /// Fetch the community manifest. Respects cache TTL unless force = true.
    func refreshManifest(force: Bool = false) async {
        // Check cache
        if !force, let cached = cachedManifest, let ts = cacheTimestamp,
           Date().timeIntervalSince(ts) < cacheTTL {
            manifest = cached
            return
        }

        isRefreshing = true
        error = nil

        do {
            let (data, response) = try await URLSession.shared.data(from: manifestURL)

            // A 404 means the manifest hasn't been published yet — not an error worth
            // surfacing.  Treat it as an empty catalogue so the UI stays clean.
            if let http = response as? HTTPURLResponse, http.statusCode == 404 {
                let empty = AdapterManifest(version: 1, adapters: [])
                cachedManifest = empty
                cacheTimestamp = Date()
                manifest = empty
                lastRefresh = Date()
                isRefreshing = false
                return
            }

            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            let fetched = try decoder.decode(AdapterManifest.self, from: data)

            cachedManifest = fetched
            cacheTimestamp = Date()
            manifest = fetched
            lastRefresh = Date()
        } catch {
            // Genuine network failure — show cached version if available, otherwise error.
            if cachedManifest != nil {
                self.error = "Offline — showing cached results"
            } else {
                self.error = "Failed to load community adapters: \(error.localizedDescription)"
            }
        }

        isRefreshing = false
    }

    /// Whether a community adapter is already installed locally.
    func isInstalled(_ adapterID: String) -> Bool {
        AdapterRegistry.listAdapters().contains { $0.id == adapterID }
    }

    // MARK: - Downloads

    /// Download a community adapter (safetensors + metadata) and save locally.
    func download(adapter: CommunityAdapter) {
        guard activeTasks.count < Self.maxConcurrentDownloads else {
            // Queue: just mark as downloading at 0
            downloads[adapter.id] = .downloading(progress: 0)
            // Retry after a delay when a slot opens
            let task = Task { [weak self] in
                try? await Task.sleep(for: .seconds(2))
                await self?.download(adapter: adapter)
            }
            activeTasks[adapter.id] = task
            return
        }

        downloads[adapter.id] = .downloading(progress: 0)

        let task = Task { [weak self] in
            guard let self else { return }
            do {
                try await self.performDownload(adapter: adapter, retryCount: 1)
                self.downloads[adapter.id] = .completed
            } catch {
                self.downloads[adapter.id] = .failed(error.localizedDescription)
            }
            self.activeTasks.removeValue(forKey: adapter.id)
        }
        activeTasks[adapter.id] = task
    }

    /// Cancel an in-progress download and clean up partial files.
    func cancelDownload(adapterID: String) {
        activeTasks[adapterID]?.cancel()
        activeTasks.removeValue(forKey: adapterID)
        downloads.removeValue(forKey: adapterID)

        // Clean up any partial files
        let dir = AdapterRegistry.adapterDirectory(id: adapterID)
        try? FileManager.default.removeItem(at: dir)
    }

    /// Retry a failed download.
    func retryDownload(adapter: CommunityAdapter) {
        downloads.removeValue(forKey: adapter.id)
        download(adapter: adapter)
    }

    // MARK: - HuggingFace Import

    /// Import an adapter from a HuggingFace repo URL.
    ///
    /// Supports two repo formats:
    /// - **PEFT repos**: contain `adapter_config.json` + `adapter_model.safetensors`.
    ///   Keys are remapped from PEFT format to AnvilApp format at import time so the
    ///   stored safetensors is always in native format. Metadata is synthesised from
    ///   the PEFT config (rank, alpha, target modules).
    /// - **Native Anvil repos**: contain `adapter.json` + `adapter.safetensors`.
    ///   Metadata is loaded verbatim and weights are stored as-is.
    ///
    /// - Parameter repoURL: The HuggingFace repo URL (e.g. `https://huggingface.co/user/repo`)
    /// - Returns: The installed `LoRAAdapter`.
    func importFromHuggingFace(repoURL: URL) async throws -> LoRAAdapter {
        let baseURL = repoURL.appendingPathComponent("resolve/main")

        // ── PEFT probe ────────────────────────────────────────────────────────
        // Try adapter_config.json first; a 200 response means this is a PEFT repo.
        let peftConfigURL = baseURL.appendingPathComponent("adapter_config.json")
        let (peftConfigData, peftConfigResp) = try await URLSession.shared.data(from: peftConfigURL)
        if let http = peftConfigResp as? HTTPURLResponse, http.statusCode == 200 {
            return try await importPEFTAdapter(
                repoURL: repoURL,
                baseURL: baseURL,
                configData: peftConfigData
            )
        }

        // ── Native Anvil format ───────────────────────────────────────────────
        let metadataURL = baseURL.appendingPathComponent("adapter.json")
        let weightsURL  = baseURL.appendingPathComponent("adapter.safetensors")

        let (metaData, metaResponse) = try await URLSession.shared.data(from: metadataURL)
        guard let httpResponse = metaResponse as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            throw CommunityError.missingFile("adapter_config.json or adapter.json", repoURL)
        }

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        var adapter = try decoder.decode(LoRAAdapter.self, from: metaData)

        if AdapterRegistry.listAdapters().contains(where: { $0.remoteRepoURL == repoURL }) {
            throw CommunityError.alreadyInstalled(adapter.name)
        }

        adapter.source = .huggingface
        adapter.remoteRepoURL = repoURL

        let (weightsData, weightsResponse) = try await URLSession.shared.data(from: weightsURL)
        guard let httpWeightsResponse = weightsResponse as? HTTPURLResponse,
              httpWeightsResponse.statusCode == 200 else {
            throw CommunityError.missingFile("adapter.safetensors", repoURL)
        }

        let dir = AdapterRegistry.adapterDirectory(id: adapter.id)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try metaData.write(to: dir.appending(path: "adapter.json"))
        try weightsData.write(to: dir.appending(path: "adapter.safetensors"))

        return adapter
    }

    // MARK: - PEFT Import

    /// Import a PEFT LoRA adapter, remapping weight keys to AnvilApp format on disk.
    private func importPEFTAdapter(
        repoURL: URL,
        baseURL: URL,
        configData: Data
    ) async throws -> LoRAAdapter {
        // Parse PEFT config
        let peftConfig = try JSONDecoder().decode(PEFTAdapterConfig.self, from: configData)

        guard peftConfig.peftType.lowercased() == "lora" else {
            throw CommunityError.unsupportedPEFTType(peftConfig.peftType)
        }

        // Synthesise adapter name and ID from the repo's last path component.
        // e.g. "ACE-Step-v1.5-chinese-new-year-LoRA" → id "ace-step-v1-5-chinese-new-year-lora"
        let repoSlug = repoURL.lastPathComponent
        let adapterID = repoSlug
            .lowercased()
            .replacingOccurrences(of: "[^a-z0-9\\-]", with: "-", options: .regularExpression)
            .trimmingCharacters(in: CharacterSet(charactersIn: "-"))
        let adapterName = repoSlug

        if AdapterRegistry.listAdapters().contains(where: { $0.remoteRepoURL == repoURL }) {
            throw CommunityError.alreadyInstalled(adapterName)
        }

        // Build LoRAConfig from PEFT metadata
        let targetLayers = PEFTKeyRemapper.inferTargetLayers(from: peftConfig.targetModules)
        let config = LoRAConfig(
            rank: peftConfig.r,
            alpha: peftConfig.loraAlpha,
            dropout: peftConfig.loraDropout,
            targetLayers: targetLayers
        )

        let baseModelID = PEFTKeyRemapper.resolveBaseModelID(
            from: peftConfig.baseModelNameOrPath
        )

        let adapter = LoRAAdapter(
            id: adapterID,
            name: adapterName,
            config: config,
            baseModelID: baseModelID,
            baseModelCategory: .aceStepDiT,
            createdAt: Date(),
            sampleCount: 0,
            finalLoss: nil,
            source: .huggingface,
            remoteRepoURL: repoURL
        )

        // Download PEFT weights (adapter_model.safetensors)
        let weightsURL = baseURL.appendingPathComponent("adapter_model.safetensors")
        let (weightsData, weightsResponse) = try await URLSession.shared.data(from: weightsURL)
        guard let http = weightsResponse as? HTTPURLResponse, http.statusCode == 200 else {
            throw CommunityError.missingFile("adapter_model.safetensors", repoURL)
        }

        // Write to a temp file, load with MLX, remap keys, re-save in AnvilApp format.
        // This ensures the on-disk safetensors always uses native key names so
        // AdapterRegistry.loadWeights → LoRAInjector.addContribution works without changes.
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("safetensors")
        defer { try? FileManager.default.removeItem(at: tmp) }
        try weightsData.write(to: tmp)

        let rawArrays = try MLX.loadArrays(url: tmp)
        let remapped  = PEFTKeyRemapper.remap(rawArrays)

        guard !remapped.isEmpty else {
            throw CommunityError.noMappableWeights(repoURL)
        }

        let remappedWeights = remapped.map { ($0.key, $0.value) }
        try AdapterRegistry.save(adapter: adapter, weights: remappedWeights)

        return adapter
    }

    // MARK: - Private

    private func performDownload(adapter: CommunityAdapter, retryCount: Int) async throws {
        do {
            // Download metadata
            let (metaData, _) = try await URLSession.shared.data(from: adapter.metadataURL)
            downloads[adapter.id] = .downloading(progress: 0.1)

            // Download weights with progress
            let (weightsData, _) = try await URLSession.shared.data(from: adapter.adapterURL)
            downloads[adapter.id] = .downloading(progress: 0.9)

            // Parse metadata to get a LoRAAdapter, update source
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            var localAdapter = try decoder.decode(LoRAAdapter.self, from: metaData)
            localAdapter.source = .community
            localAdapter.remoteRepoURL = nil

            // Save to disk
            let dir = AdapterRegistry.adapterDirectory(id: adapter.id)
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

            // Write updated metadata with source field
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let updatedMeta = try encoder.encode(localAdapter)
            try updatedMeta.write(to: dir.appending(path: "adapter.json"))
            try weightsData.write(to: dir.appending(path: "adapter.safetensors"))

            downloads[adapter.id] = .downloading(progress: 1.0)
        } catch {
            if retryCount > 0 {
                // Auto-retry once
                try await Task.sleep(for: .seconds(1))
                try await performDownload(adapter: adapter, retryCount: retryCount - 1)
            } else {
                // Clean up partial files
                let dir = AdapterRegistry.adapterDirectory(id: adapter.id)
                try? FileManager.default.removeItem(at: dir)
                throw error
            }
        }
    }
}

/// Errors specific to community adapter operations.
enum CommunityError: Error, LocalizedError {
    case missingFile(String, URL)
    case alreadyInstalled(String)
    case incompatibleVersion(String)
    case unsupportedPEFTType(String)
    case noMappableWeights(URL)

    var errorDescription: String? {
        switch self {
        case .missingFile(let file, let url):
            return "Couldn't find \(file) in this repo. Check that the URL points to a valid Anvil or PEFT LoRA adapter. (\(url.absoluteString))"
        case .alreadyInstalled(let name):
            return "\"\(name)\" is already installed. Delete it first to re-download."
        case .incompatibleVersion(let required):
            return "This adapter requires Anvil \(required) or later."
        case .unsupportedPEFTType(let type):
            return "Unsupported PEFT adapter type '\(type)'. Only LoRA adapters are supported."
        case .noMappableWeights(let url):
            return "No compatible LoRA weights found in this repo. The adapter key format may not be supported. (\(url.absoluteString))"
        }
    }
}

// MARK: - PEFT Config Model (private)

/// Minimal decodable representation of a PEFT `adapter_config.json`.
private struct PEFTAdapterConfig: Decodable {
    let peftType: String
    let r: Int
    let loraAlpha: Float
    let loraDropout: Float
    let targetModules: [String]
    let baseModelNameOrPath: String?

    enum CodingKeys: String, CodingKey {
        case peftType           = "peft_type"
        case r
        case loraAlpha          = "lora_alpha"
        case loraDropout        = "lora_dropout"
        case targetModules      = "target_modules"
        case baseModelNameOrPath = "base_model_name_or_path"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        peftType = try c.decode(String.self, forKey: .peftType)
        r        = try c.decodeIfPresent(Int.self, forKey: .r) ?? 16

        // lora_alpha and lora_dropout may be stored as Int or Float in JSON —
        // decode via Double to avoid type-mismatch errors.
        func decodeFloat(_ key: CodingKeys, default defaultVal: Float) -> Float {
            if let v = try? c.decodeIfPresent(Double.self, forKey: key) { return Float(v) }
            if let v = try? c.decodeIfPresent(Int.self,    forKey: key) { return Float(v) }
            return defaultVal
        }
        loraAlpha   = decodeFloat(.loraAlpha,   default: Float(r))
        loraDropout = decodeFloat(.loraDropout, default: 0.0)

        targetModules      = (try? c.decodeIfPresent([String].self, forKey: .targetModules)) ?? []
        baseModelNameOrPath = try? c.decodeIfPresent(String.self, forKey: .baseModelNameOrPath)
    }
}
