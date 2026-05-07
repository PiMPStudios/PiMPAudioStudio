import Foundation
import AnvilCore

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
            let (data, _) = try await URLSession.shared.data(from: manifestURL)
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            let fetched = try decoder.decode(AdapterManifest.self, from: data)

            cachedManifest = fetched
            cacheTimestamp = Date()
            manifest = fetched
            lastRefresh = Date()
        } catch {
            // Show cached version if available
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
    /// Resolves `{repoURL}/resolve/main/adapter.json` and `adapter.safetensors`.
    /// Fetches metadata first for validation, then downloads weights.
    ///
    /// - Parameter repoURL: The HuggingFace repo URL (e.g. https://huggingface.co/user/repo)
    /// - Returns: The installed LoRAAdapter.
    func importFromHuggingFace(repoURL: URL) async throws -> LoRAAdapter {
        let baseURL = repoURL.appendingPathComponent("resolve/main")
        let metadataURL = baseURL.appendingPathComponent("adapter.json")
        let weightsURL = baseURL.appendingPathComponent("adapter.safetensors")

        // 1. Fetch metadata
        let (metaData, metaResponse) = try await URLSession.shared.data(from: metadataURL)
        guard let httpResponse = metaResponse as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            throw CommunityError.missingFile("adapter.json", repoURL)
        }

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        var adapter = try decoder.decode(LoRAAdapter.self, from: metaData)

        // Check for duplicate
        if AdapterRegistry.listAdapters().contains(where: { $0.remoteRepoURL == repoURL }) {
            throw CommunityError.alreadyInstalled(adapter.name)
        }

        // Update provenance
        adapter.source = .huggingface
        adapter.remoteRepoURL = repoURL

        // 2. Download weights
        let (weightsData, weightsResponse) = try await URLSession.shared.data(from: weightsURL)
        guard let httpWeightsResponse = weightsResponse as? HTTPURLResponse,
              httpWeightsResponse.statusCode == 200 else {
            throw CommunityError.missingFile("adapter.safetensors", repoURL)
        }

        // 3. Save to local registry
        let dir = AdapterRegistry.adapterDirectory(id: adapter.id)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try metaData.write(to: dir.appending(path: "adapter.json"))
        try weightsData.write(to: dir.appending(path: "adapter.safetensors"))

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

    var errorDescription: String? {
        switch self {
        case .missingFile(let file, let url):
            return "This repo doesn't follow the Anvil adapter format. Expected \(file) at \(url.absoluteString)"
        case .alreadyInstalled(let name):
            return "\"\(name)\" is already installed. Delete it first to re-download."
        case .incompatibleVersion(let required):
            return "This adapter requires Anvil \(required) or later"
        }
    }
}
