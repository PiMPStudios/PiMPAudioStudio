import Foundation
import PiMPAudioCore
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
    private var pendingDownloads: [String: CommunityAdapter] = [:]
    private var pendingDownloadOrder: [String] = []
    private var cachedManifest: AdapterManifest?
    private var cacheTimestamp: Date?
    private let downloadClient = DownloadClient()

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
        guard activeTasks[adapter.id] == nil else { return }

        if pendingDownloads[adapter.id] != nil {
            downloads[adapter.id] = downloads[adapter.id] ?? .downloading(progress: 0)
            return
        }

        guard activeTasks.count < Self.maxConcurrentDownloads else {
            enqueueDownload(adapter)
            return
        }

        startDownload(adapter)
    }

    private func enqueueDownload(_ adapter: CommunityAdapter) {
        pendingDownloads[adapter.id] = adapter
        pendingDownloadOrder.append(adapter.id)
        downloads[adapter.id] = downloads[adapter.id] ?? .downloading(progress: 0)
    }

    private func startDownload(_ adapter: CommunityAdapter) {
        downloads[adapter.id] = .downloading(progress: 0)

        let task = Task { [weak self] in
            guard let self else { return }
            do {
                try await self.performDownload(adapter: adapter, retryCount: 1)
                if !Task.isCancelled {
                    self.downloads[adapter.id] = .completed
                }
            } catch is CancellationError {
                // Cancellation is user-initiated; cancelDownload already clears UI state.
            } catch {
                if !Task.isCancelled {
                    self.downloads[adapter.id] = .failed(error.localizedDescription)
                }
            }
            self.activeTasks.removeValue(forKey: adapter.id)
            self.startNextDownloadIfPossible()
        }
        activeTasks[adapter.id] = task
    }

    private func startNextDownloadIfPossible() {
        while activeTasks.count < Self.maxConcurrentDownloads,
              let adapterID = pendingDownloadOrder.first {
            pendingDownloadOrder.removeFirst()
            guard let adapter = pendingDownloads.removeValue(forKey: adapterID) else {
                continue
            }
            startDownload(adapter)
        }
    }

    /// Cancel an in-progress download and clean up partial files.
    func cancelDownload(adapterID: String) {
        activeTasks[adapterID]?.cancel()
        activeTasks.removeValue(forKey: adapterID)
        pendingDownloads.removeValue(forKey: adapterID)
        pendingDownloadOrder.removeAll { $0 == adapterID }
        downloads.removeValue(forKey: adapterID)

        // Clean up any partial files
        let dir = AdapterRegistry.adapterDirectory(id: adapterID)
        try? FileManager.default.removeItem(at: dir)
        startNextDownloadIfPossible()
    }

    /// Retry a failed download.
    func retryDownload(adapter: CommunityAdapter) {
        downloads.removeValue(forKey: adapter.id)
        download(adapter: adapter)
    }

    // MARK: - HuggingFace Import

    /// Import an adapter from a HuggingFace repo URL.
    ///
    /// Probes three formats in order, stopping at the first hit:
    ///
    /// 1. **PEFT with config** (`adapter_config.json` + `adapter_model.safetensors`) —
    ///    rank/alpha/target_modules parsed from the config; keys remapped at import time.
    /// 2. **Native PiMP Audio Studio** (`adapter.json` + `adapter.safetensors`) —
    ///    metadata loaded verbatim, weights stored as-is.
    /// 3. **Weights-only PEFT** (`adapter_model.safetensors` only, no config) —
    ///    rank inferred from tensor shape, target modules inferred from key names.
    ///    Handles community repos that ship only the weights file.
    ///
    /// - Parameter repoURL: The HuggingFace repo URL (e.g. `https://huggingface.co/user/repo`)
    /// - Returns: The installed `LoRAAdapter`.
    func importFromHuggingFace(repoURL: URL) async throws -> LoRAAdapter {
        let baseURL = repoURL.appendingPathComponent("resolve/main")

        // ── 1. PEFT with config ───────────────────────────────────────────────
        let peftConfigURL = baseURL.appendingPathComponent("adapter_config.json")
        let (peftConfigData, peftConfigResp) = try await URLSession.shared.data(from: peftConfigURL)
        if let http = peftConfigResp as? HTTPURLResponse, http.statusCode == 200 {
            return try await importPEFTAdapter(
                repoURL: repoURL,
                baseURL: baseURL,
                configData: peftConfigData
            )
        }

        // ── 2. Native PiMP Audio Studio format ────────────────────────────────
        let metadataURL = baseURL.appendingPathComponent("adapter.json")
        let (metaData, metaResponse) = try await URLSession.shared.data(from: metadataURL)
        if let http = metaResponse as? HTTPURLResponse, http.statusCode == 200 {
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            var adapter = try decoder.decode(LoRAAdapter.self, from: metaData)

            if AdapterRegistry.listAdapters().contains(where: { $0.remoteRepoURL == repoURL }) {
                throw CommunityError.alreadyInstalled(adapter.name)
            }

            adapter.source = .huggingface
            adapter.remoteRepoURL = repoURL

            let weightsURL = baseURL.appendingPathComponent("adapter.safetensors")
            let (weightsData, weightsResponse) = try await URLSession.shared.data(from: weightsURL)
            guard let httpW = weightsResponse as? HTTPURLResponse, httpW.statusCode == 200 else {
                throw CommunityError.missingFile("adapter.safetensors", repoURL)
            }
            let expectedSHA256 = adapterSHA256(from: metaData)

            let dir = AdapterRegistry.adapterDirectory(id: adapter.id)
            let weightsDestination = dir.appending(path: "adapter.safetensors")
            do {
                try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
                try metaData.write(to: dir.appending(path: "adapter.json"))
                try weightsData.write(to: weightsDestination)
                try await verifyAdapterWeights(
                    at: weightsDestination,
                    expectedSHA256: expectedSHA256,
                    fileName: "adapter.safetensors"
                )
            } catch {
                try? FileManager.default.removeItem(at: dir)
                throw error
            }
            return adapter
        }

        // ── 3. Weights-only PEFT (no adapter_config.json) ────────────────────
        // Community repos often ship only adapter_model.safetensors. Infer rank
        // and target modules from the tensor shapes and key names.
        let peftWeightsURL = baseURL.appendingPathComponent("adapter_model.safetensors")
        let (peftWeightsData, peftWeightsResp) = try await URLSession.shared.data(from: peftWeightsURL)
        if let http = peftWeightsResp as? HTTPURLResponse, http.statusCode == 200 {
            return try await importWeightsOnlyPEFT(repoURL: repoURL, weightsData: peftWeightsData)
        }

        throw CommunityError.missingFile(
            "adapter_config.json, adapter.json, or adapter_model.safetensors", repoURL
        )
    }

    // MARK: - PEFT Import

    /// Import a PEFT LoRA adapter, remapping weight keys to PiMPAudioStudio format on disk.
    private func importPEFTAdapter(
        repoURL: URL,
        baseURL: URL,
        configData: Data
    ) async throws -> LoRAAdapter {
        let peftConfig = try JSONDecoder().decode(PEFTAdapterConfig.self, from: configData)

        guard peftConfig.peftType.lowercased() == "lora" else {
            throw CommunityError.unsupportedPEFTType(peftConfig.peftType)
        }

        let repoSlug = repoURL.lastPathComponent

        if AdapterRegistry.listAdapters().contains(where: { $0.remoteRepoURL == repoURL }) {
            throw CommunityError.alreadyInstalled(repoSlug)
        }

        // Download PEFT weights (adapter_model.safetensors)
        let weightsURL = baseURL.appendingPathComponent("adapter_model.safetensors")
        let (weightsData, weightsResponse) = try await URLSession.shared.data(from: weightsURL)
        guard let http = weightsResponse as? HTTPURLResponse, http.statusCode == 200 else {
            throw CommunityError.missingFile("adapter_model.safetensors", repoURL)
        }

        // Write to a temp file, load with MLX, remap keys, re-save in PiMPAudioStudio format.
        // This ensures the on-disk safetensors always uses native key names so
        // AdapterRegistry.loadWeights → LoRAInjector.addContribution works without changes.
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("safetensors")
        defer { try? FileManager.default.removeItem(at: tmp) }
        try weightsData.write(to: tmp)
        try await verifyAdapterWeights(
            at: tmp,
            expectedSHA256: adapterSHA256(from: configData),
            fileName: "adapter_model.safetensors"
        )

        let rawArrays = try MLX.loadArrays(url: tmp)
        return try savePEFTAdapter(
            rawArrays: rawArrays,
            peftConfig: peftConfig,
            nameSeed: repoSlug,
            source: .huggingface,
            remoteRepoURL: repoURL,
            errorURL: repoURL
        )
    }

    /// Import a weights-only PEFT repo (no `adapter_config.json`).
    ///
    /// Rank is inferred from the `lora_A` tensor shape; target modules from key names.
    /// This handles community repos that publish only `adapter_model.safetensors`.
    private func importWeightsOnlyPEFT(
        repoURL: URL,
        weightsData: Data
    ) async throws -> LoRAAdapter {
        let repoSlug = repoURL.lastPathComponent

        if AdapterRegistry.listAdapters().contains(where: { $0.remoteRepoURL == repoURL }) {
            throw CommunityError.alreadyInstalled(repoSlug)
        }

        // Load weights from a temp file so MLX can parse the safetensors format
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("safetensors")
        defer { try? FileManager.default.removeItem(at: tmp) }
        try weightsData.write(to: tmp)

        let rawArrays = try MLX.loadArrays(url: tmp)

        guard PEFTKeyRemapper.isPEFT(Array(rawArrays.keys)) else {
            throw CommunityError.noMappableWeights(repoURL)
        }

        guard let inferred = PEFTKeyRemapper.inferConfig(from: rawArrays) else {
            throw CommunityError.noMappableWeights(repoURL)
        }

        return try savePEFTAdapter(
            rawArrays: rawArrays,
            peftConfig: nil,
            inferredConfig: inferred,
            nameSeed: repoSlug,
            source: .huggingface,
            remoteRepoURL: repoURL,
            errorURL: repoURL
        )
    }

    /// Import a local PEFT LoRA directory (`adapter_config.json` +
    /// `adapter_model.safetensors`) and save it in PiMPAudioStudio's native adapter
    /// registry format.
    func importLocalPEFTAdapter(directoryURL: URL) async throws -> LoRAAdapter {
        let didAccess = directoryURL.startAccessingSecurityScopedResource()
        defer {
            if didAccess {
                directoryURL.stopAccessingSecurityScopedResource()
            }
        }

        let configURL = directoryURL.appending(path: "adapter_config.json")
        let weightsURL = directoryURL.appending(path: "adapter_model.safetensors")

        guard FileManager.default.fileExists(atPath: configURL.path(percentEncoded: false)) else {
            throw CommunityError.missingFile("adapter_config.json", directoryURL)
        }
        guard FileManager.default.fileExists(atPath: weightsURL.path(percentEncoded: false)) else {
            throw CommunityError.missingFile("adapter_model.safetensors", directoryURL)
        }

        let configData = try Data(contentsOf: configURL)
        let peftConfig = try JSONDecoder().decode(PEFTAdapterConfig.self, from: configData)

        guard peftConfig.peftType.lowercased() == "lora" else {
            throw CommunityError.unsupportedPEFTType(peftConfig.peftType)
        }

        let rawArrays = try MLX.loadArrays(url: weightsURL)
        return try savePEFTAdapter(
            rawArrays: rawArrays,
            peftConfig: peftConfig,
            nameSeed: localPEFTNameSeed(for: directoryURL),
            source: .imported,
            remoteRepoURL: nil,
            errorURL: directoryURL
        )
    }

    private func savePEFTAdapter(
        rawArrays: [String: MLXArray],
        peftConfig: PEFTAdapterConfig?,
        inferredConfig: PEFTKeyRemapper.InferredConfig? = nil,
        nameSeed: String,
        source: AdapterSource,
        remoteRepoURL: URL?,
        errorURL: URL
    ) throws -> LoRAAdapter {
        let remapped = PEFTKeyRemapper.remap(rawArrays)
        guard !remapped.isEmpty else {
            throw CommunityError.noMappableWeights(errorURL)
        }

        let exactPatterns = PEFTKeyRemapper.inferInjectionPatterns(
            fromRemappedKeys: Array(remapped.keys)
        )
        let targetLayers = PEFTKeyRemapper.inferTargetLayers(
            fromInjectionPatterns: exactPatterns
        )

        let rank = peftConfig?.r ?? inferredConfig?.rank ?? 16
        let alpha = peftConfig?.loraAlpha ?? inferredConfig?.alpha ?? Float(rank)
        let dropout = peftConfig?.loraDropout ?? 0.0

        let config = LoRAConfig(
            rank: rank,
            alpha: alpha,
            dropout: dropout,
            targetLayers: targetLayers,
            exactInjectionPatterns: exactPatterns
        )

        let adapterID = uniqueAdapterID(from: nameSeed)
        let adapterName = nameSeed
        let baseModelID = PEFTKeyRemapper.resolveBaseModelID(
            from: peftConfig?.baseModelNameOrPath
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
            source: source,
            remoteRepoURL: remoteRepoURL
        )

        let remappedWeights = remapped.map { ($0.key, $0.value) }
        try AdapterRegistry.save(adapter: adapter, weights: remappedWeights)

        return adapter
    }

    private func uniqueAdapterID(from seed: String) -> String {
        let baseID = seed
            .lowercased()
            .replacingOccurrences(of: "[^a-z0-9\\-]", with: "-", options: .regularExpression)
            .trimmingCharacters(in: CharacterSet(charactersIn: "-"))
        let normalized = baseID.isEmpty ? "imported-adapter" : baseID
        let existing = Set(AdapterRegistry.listAdapters().map(\.id))
        guard existing.contains(normalized) else { return normalized }

        var suffix = 2
        while existing.contains("\(normalized)-\(suffix)") {
            suffix += 1
        }
        return "\(normalized)-\(suffix)"
    }

    private func localPEFTNameSeed(for directoryURL: URL) -> String {
        let components = directoryURL.pathComponents
        if let cloudJobsIndex = components.firstIndex(of: "cloud-jobs"),
           cloudJobsIndex + 1 < components.count {
            return components[cloudJobsIndex + 1]
        }

        if directoryURL.lastPathComponent == "final" {
            let parent = directoryURL.deletingLastPathComponent().lastPathComponent
            if !parent.isEmpty && parent != "lora" {
                return parent
            }
        }

        return directoryURL.lastPathComponent
    }

    private func adapterSHA256(from metadata: Data) -> String? {
        guard let object = try? JSONSerialization.jsonObject(with: metadata) as? [String: Any] else {
            return nil
        }
        for key in ["sha256", "adapterSHA256", "weightsSHA256"] {
            if let value = object[key] as? String,
               !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                return value
            }
        }
        return nil
    }

    // MARK: - Private

    private func performDownload(adapter: CommunityAdapter, retryCount: Int) async throws {
        do {
            // Download metadata
            let (metaData, _) = try await URLSession.shared.data(from: adapter.metadataURL)
            try Task.checkCancellation()
            downloads[adapter.id] = .downloading(progress: 0.1)

            // Download weights with progress
            let (weightsData, _) = try await URLSession.shared.data(from: adapter.adapterURL)
            try Task.checkCancellation()
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
            let weightsDestination = dir.appending(path: "adapter.safetensors")

            // Write updated metadata with source field
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let updatedMeta = try encoder.encode(localAdapter)
            try updatedMeta.write(to: dir.appending(path: "adapter.json"))
            try weightsData.write(to: weightsDestination)
            downloads[adapter.id] = .downloading(progress: 0.95)
            try await verifyAdapterWeights(
                at: weightsDestination,
                expectedSHA256: adapter.sha256,
                fileName: adapter.adapterURL.lastPathComponent
            )

            downloads[adapter.id] = .downloading(progress: 1.0)
        } catch is CancellationError {
            let dir = AdapterRegistry.adapterDirectory(id: adapter.id)
            try? FileManager.default.removeItem(at: dir)
            throw CancellationError()
        } catch {
            if Task.isCancelled {
                let dir = AdapterRegistry.adapterDirectory(id: adapter.id)
                try? FileManager.default.removeItem(at: dir)
                throw CancellationError()
            }
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

    private func verifyAdapterWeights(
        at url: URL,
        expectedSHA256: String?,
        fileName: String
    ) async throws {
        guard let expectedSHA256 = expectedSHA256?.trimmingCharacters(in: .whitespacesAndNewlines),
              !expectedSHA256.isEmpty else {
            return
        }

        let verified = await downloadClient.verifySHA256(file: url, expected: expectedSHA256)
        guard verified else {
            try? FileManager.default.removeItem(at: url)
            throw DownloadError.checksumMismatch(file: fileName)
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
            return "Couldn't find \(file) in this repo. Check that the URL points to a valid PiMP Audio Studio or PEFT LoRA adapter. (\(url.absoluteString))"
        case .alreadyInstalled(let name):
            return "\"\(name)\" is already installed. Delete it first to re-download."
        case .incompatibleVersion(let required):
            return "This adapter requires PiMP Audio Studio \(required) or later."
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
