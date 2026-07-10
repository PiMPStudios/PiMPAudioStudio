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

    private struct ActiveDownload {
        let runID: UUID
        let task: Task<Void, Never>
    }

    private var activeTasks: [String: ActiveDownload] = [:]
    private var cancellingDownloads: Set<String> = []
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
        if activeTasks[adapter.id] != nil {
            if cancellingDownloads.contains(adapter.id) {
                enqueueDownload(adapter)
            }
            return
        }

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
        if pendingDownloads[adapter.id] == nil {
            pendingDownloadOrder.append(adapter.id)
        }
        pendingDownloads[adapter.id] = adapter
        downloads[adapter.id] = downloads[adapter.id] ?? .downloading(progress: 0)
    }

    private func startDownload(_ adapter: CommunityAdapter) {
        downloads[adapter.id] = .downloading(progress: 0)
        let runID = UUID()

        let task = Task { [weak self] in
            guard let self else { return }
            do {
                try await self.performDownload(adapter: adapter, retryCount: 1)
                if !Task.isCancelled {
                    self.downloads[adapter.id] = .completed
                } else {
                    self.removePartialDownload(adapterID: adapter.id)
                }
            } catch is CancellationError {
                self.removePartialDownload(adapterID: adapter.id)
            } catch {
                if !Task.isCancelled {
                    self.downloads[adapter.id] = .failed(error.localizedDescription)
                }
            }
            self.finishDownload(adapterID: adapter.id, runID: runID)
        }
        activeTasks[adapter.id] = ActiveDownload(runID: runID, task: task)
    }

    private func finishDownload(adapterID: String, runID: UUID) {
        guard activeTasks[adapterID]?.runID == runID else { return }
        activeTasks.removeValue(forKey: adapterID)
        cancellingDownloads.remove(adapterID)
        startNextDownloadIfPossible()
    }

    private func removePartialDownload(adapterID: String) {
        guard let directory = try? AdapterRegistry.adapterDirectory(id: adapterID) else { return }
        try? FileManager.default.removeItem(at: directory)
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

    /// Cancel an in-progress download. Its task cleans up partial files before releasing its slot.
    func cancelDownload(adapterID: String) {
        if let active = activeTasks[adapterID] {
            cancellingDownloads.insert(adapterID)
            active.task.cancel()
        }
        pendingDownloads.removeValue(forKey: adapterID)
        pendingDownloadOrder.removeAll { $0 == adapterID }
        downloads.removeValue(forKey: adapterID)
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
    /// 1. **PEFT with config** (`adapter_config.json` + `adapter_model.safetensors`
    ///    or `lokr_weights.safetensors`) — rank/alpha/target_modules parsed from
    ///    the config; keys remapped at import time.
    /// 2. **Native PiMP Audio Studio** (`adapter.json` + `adapter.safetensors`) —
    ///    metadata loaded verbatim, weights stored as-is.
    /// 3. **Weights-only PEFT/LyCORIS** (`adapter_model.safetensors` or
    ///    `lokr_weights.safetensors` only, no config) — rank/dim inferred from
    ///    tensor shape or safetensors metadata, target modules inferred from key names.
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
            adapter.id = uniqueAdapterID(from: adapter.id)

            let weightsURL = baseURL.appendingPathComponent("adapter.safetensors")
            let (weightsData, weightsResponse) = try await URLSession.shared.data(from: weightsURL)
            guard let httpW = weightsResponse as? HTTPURLResponse, httpW.statusCode == 200 else {
                throw CommunityError.missingFile("adapter.safetensors", repoURL)
            }
            let expectedSHA256 = adapterSHA256(from: metaData)

            let dir = try AdapterRegistry.adapterDirectory(id: adapter.id)
            let weightsDestination = dir.appending(path: "adapter.safetensors")
            do {
                try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
                let encoder = JSONEncoder()
                encoder.dateEncodingStrategy = .iso8601
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                try encoder.encode(adapter).write(to: dir.appending(path: "adapter.json"))
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

        // ── 3. Weights-only LyCORIS LoKR (no adapter_config.json) ────────────
        let lokrWeightsURL = baseURL.appendingPathComponent("lokr_weights.safetensors")
        let (lokrWeightsData, lokrWeightsResp) = try await URLSession.shared.data(from: lokrWeightsURL)
        if let http = lokrWeightsResp as? HTTPURLResponse, http.statusCode == 200 {
            return try await importWeightsOnlyPEFT(
                repoURL: repoURL,
                weightsData: lokrWeightsData,
                fileName: "lokr_weights.safetensors"
            )
        }

        // ── 4. Weights-only PEFT (no adapter_config.json) ────────────────────
        // Community repos often ship only adapter_model.safetensors. Infer rank
        // and target modules from the tensor shapes and key names.
        let peftWeightsURL = baseURL.appendingPathComponent("adapter_model.safetensors")
        let (peftWeightsData, peftWeightsResp) = try await URLSession.shared.data(from: peftWeightsURL)
        if let http = peftWeightsResp as? HTTPURLResponse, http.statusCode == 200 {
            return try await importWeightsOnlyPEFT(
                repoURL: repoURL,
                weightsData: peftWeightsData,
                fileName: "adapter_model.safetensors"
            )
        }

        throw CommunityError.missingFile(
            "adapter_config.json, adapter.json, lokr_weights.safetensors, or adapter_model.safetensors",
            repoURL
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

        let repoSlug = repoURL.lastPathComponent

        if AdapterRegistry.listAdapters().contains(where: { $0.remoteRepoURL == repoURL }) {
            throw CommunityError.alreadyInstalled(repoSlug)
        }

        let type = peftConfig.peftType.lowercased()
        let candidateWeights = type == "lokr"
            ? ["adapter_model.safetensors", "lokr_weights.safetensors"]
            : ["adapter_model.safetensors"]

        guard type == "lora" || type == "lokr" else {
            throw CommunityError.unsupportedPEFTType(peftConfig.peftType)
        }

        let weightsDownload = try await downloadFirstAvailable(
            fileNames: candidateWeights,
            baseURL: baseURL,
            errorURL: repoURL
        )

        // Write to a temp file, load with MLX, remap keys, re-save in PiMPAudioStudio format.
        // This ensures the on-disk safetensors always uses native key names so
        // AdapterRegistry.loadWeights → LoRAInjector.addContribution works without changes.
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("safetensors")
        defer { try? FileManager.default.removeItem(at: tmp) }
        try weightsDownload.data.write(to: tmp)
        try await verifyAdapterWeights(
            at: tmp,
            expectedSHA256: adapterSHA256(from: configData),
            fileName: weightsDownload.fileName
        )

        let rawArrays = try MLX.loadArrays(url: tmp)
        if type == "lokr" {
            let metadata = (try? safetensorsMetadata(from: tmp)) ?? [:]
            return try saveLyCORISLoKRAdapter(
                rawArrays: rawArrays,
                safetensorsMetadata: metadata,
                peftConfig: peftConfig,
                nameSeed: repoSlug,
                source: .huggingface,
                remoteRepoURL: repoURL,
                errorURL: repoURL
            )
        }

        return try savePEFTAdapter(
            rawArrays: rawArrays,
            peftConfig: peftConfig,
            nameSeed: repoSlug,
            source: .huggingface,
            remoteRepoURL: repoURL,
            errorURL: repoURL
        )
    }

    private func downloadFirstAvailable(
        fileNames: [String],
        baseURL: URL,
        errorURL: URL
    ) async throws -> (fileName: String, data: Data) {
        for fileName in fileNames {
            let url = baseURL.appendingPathComponent(fileName)
            let (data, response) = try await URLSession.shared.data(from: url)
            if let http = response as? HTTPURLResponse, http.statusCode == 200 {
                return (fileName, data)
            }
        }
        throw CommunityError.missingFile(fileNames.joined(separator: " or "), errorURL)
    }

    /// Import a weights-only PEFT repo (no `adapter_config.json`).
    ///
    /// Rank is inferred from the `lora_A` tensor shape; target modules from key names.
    /// This handles community repos that publish only `adapter_model.safetensors`.
    private func importWeightsOnlyPEFT(
        repoURL: URL,
        weightsData: Data,
        fileName: String
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
        let metadata = (try? safetensorsMetadata(from: tmp)) ?? [:]

        if PEFTKeyRemapper.isLyCORISLoKR(Array(rawArrays.keys)) {
            return try saveLyCORISLoKRAdapter(
                rawArrays: rawArrays,
                safetensorsMetadata: metadata,
                peftConfig: nil,
                nameSeed: repoSlug,
                source: .huggingface,
                remoteRepoURL: repoURL,
                errorURL: repoURL
            )
        }

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

    /// Import a local adapter directory, accepting native PiMP Audio Studio
    /// folders (`adapter.json` + `adapter.safetensors`), PEFT LoRA/LoKR folders
    /// (`adapter_config.json` + `adapter_model.safetensors` or
    /// `lokr_weights.safetensors`), raw LyCORIS LoKR folders containing only
    /// `lokr_weights.safetensors`, or Side-Step run folders containing `best`
    /// or `final` PEFT LoRA / LoKR adapter weights.
    func importLocalAdapter(directoryURL: URL) async throws -> LoRAAdapter {
        let didAccess = directoryURL.startAccessingSecurityScopedResource()
        defer {
            if didAccess {
                directoryURL.stopAccessingSecurityScopedResource()
            }
        }

        let peftConfigURL = directoryURL.appending(path: "adapter_config.json")
        let peftWeightsURL = directoryURL.appending(path: "adapter_model.safetensors")
        let lokrWeightsURL = directoryURL.appending(path: "lokr_weights.safetensors")
        let hasPEFTConfig = FileManager.default.fileExists(atPath: peftConfigURL.path(percentEncoded: false))
        let hasPEFTWeights = FileManager.default.fileExists(atPath: peftWeightsURL.path(percentEncoded: false))
        let hasLoKRWeights = FileManager.default.fileExists(atPath: lokrWeightsURL.path(percentEncoded: false))
        if hasPEFTConfig && (hasPEFTWeights || hasLoKRWeights) {
            return try importLocalPEFTAdapterUnlocked(directoryURL: directoryURL)
        }

        let nativeMetadataURL = directoryURL.appending(path: "adapter.json")
        let nativeWeightsURL = directoryURL.appending(path: "adapter.safetensors")
        if FileManager.default.fileExists(atPath: nativeMetadataURL.path(percentEncoded: false)),
           FileManager.default.fileExists(atPath: nativeWeightsURL.path(percentEncoded: false)) {
            return try await importLocalNativeAdapterUnlocked(
                directoryURL: directoryURL,
                metadataURL: nativeMetadataURL,
                weightsURL: nativeWeightsURL
            )
        }

        if let sideStepPEFTURL = preferredSideStepPEFTAdapterDirectory(in: directoryURL) {
            return try importLocalPEFTAdapterUnlocked(
                directoryURL: sideStepPEFTURL,
                nameSeedURL: directoryURL
            )
        }

        if hasLoKRWeights {
            return try importLocalLyCORISLoKRAdapterUnlocked(
                directoryURL: directoryURL,
                weightsURL: lokrWeightsURL
            )
        }

        if let sideStepWeightsURL = preferredSideStepLoKRWeightsURL(in: directoryURL) {
            return try importLocalLyCORISLoKRAdapterUnlocked(
                directoryURL: sideStepWeightsURL.deletingLastPathComponent(),
                weightsURL: sideStepWeightsURL,
                nameSeedURL: directoryURL
            )
        }

        throw CommunityError.missingFile(
            "adapter_config.json + adapter_model.safetensors, adapter_config.json + lokr_weights.safetensors, adapter.json + adapter.safetensors, lokr_weights.safetensors, or Side-Step best/final adapter weights",
            directoryURL
        )
    }

    /// Import a local PEFT LoRA/LoKR directory and save it in PiMPAudioStudio's
    /// native adapter registry format.
    func importLocalPEFTAdapter(directoryURL: URL) async throws -> LoRAAdapter {
        let didAccess = directoryURL.startAccessingSecurityScopedResource()
        defer {
            if didAccess {
                directoryURL.stopAccessingSecurityScopedResource()
            }
        }

        return try importLocalPEFTAdapterUnlocked(directoryURL: directoryURL)
    }

    private func importLocalPEFTAdapterUnlocked(
        directoryURL: URL,
        nameSeedURL: URL? = nil
    ) throws -> LoRAAdapter {

        let configURL = directoryURL.appending(path: "adapter_config.json")
        let weightsURL = directoryURL.appending(path: "adapter_model.safetensors")
        let lokrWeightsURL = directoryURL.appending(path: "lokr_weights.safetensors")

        guard FileManager.default.fileExists(atPath: configURL.path(percentEncoded: false)) else {
            throw CommunityError.missingFile("adapter_config.json", directoryURL)
        }

        let configData = try Data(contentsOf: configURL)
        let peftConfig = try JSONDecoder().decode(PEFTAdapterConfig.self, from: configData)

        let type = peftConfig.peftType.lowercased()
        guard type == "lora" || type == "lokr" else {
            throw CommunityError.unsupportedPEFTType(peftConfig.peftType)
        }

        let selectedWeightsURL: URL
        if type == "lokr",
           FileManager.default.fileExists(atPath: lokrWeightsURL.path(percentEncoded: false)) {
            selectedWeightsURL = lokrWeightsURL
        } else {
            guard FileManager.default.fileExists(atPath: weightsURL.path(percentEncoded: false)) else {
                let expected = type == "lokr"
                    ? "adapter_model.safetensors or lokr_weights.safetensors"
                    : "adapter_model.safetensors"
                throw CommunityError.missingFile(expected, directoryURL)
            }
            selectedWeightsURL = weightsURL
        }

        let rawArrays = try MLX.loadArrays(url: selectedWeightsURL)

        if type == "lokr" {
            let metadata = (try? safetensorsMetadata(from: selectedWeightsURL)) ?? [:]
            return try saveLyCORISLoKRAdapter(
                rawArrays: rawArrays,
                safetensorsMetadata: metadata,
                peftConfig: peftConfig,
                nameSeed: localPEFTNameSeed(for: nameSeedURL ?? directoryURL),
                source: .imported,
                remoteRepoURL: nil,
                errorURL: directoryURL
            )
        }

        return try savePEFTAdapter(
            rawArrays: rawArrays,
            peftConfig: peftConfig,
            nameSeed: localPEFTNameSeed(for: nameSeedURL ?? directoryURL),
            source: .imported,
            remoteRepoURL: nil,
            errorURL: directoryURL
        )
    }

    private func importLocalLyCORISLoKRAdapterUnlocked(
        directoryURL: URL,
        weightsURL: URL,
        nameSeedURL: URL? = nil
    ) throws -> LoRAAdapter {
        let rawArrays = try MLX.loadArrays(url: weightsURL)
        let metadata = (try? safetensorsMetadata(from: weightsURL)) ?? [:]
        let trainingConfig = localSideStepTrainingConfig(
            directoryURL: directoryURL,
            weightsURL: weightsURL,
            nameSeedURL: nameSeedURL
        )
        let seedURL = nameSeedURL ?? directoryURL
        return try saveLyCORISLoKRAdapter(
            rawArrays: rawArrays,
            safetensorsMetadata: metadata,
            peftConfig: nil,
            nameSeed: localPEFTNameSeed(for: seedURL),
            baseModelIDOverride: trainingConfig?.resolvedBaseModelID,
            source: .imported,
            remoteRepoURL: nil,
            errorURL: directoryURL
        )
    }

    private func importLocalNativeAdapterUnlocked(
        directoryURL: URL,
        metadataURL: URL,
        weightsURL: URL
    ) async throws -> LoRAAdapter {
        let metadataData = try Data(contentsOf: metadataURL)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        var adapter = try decoder.decode(LoRAAdapter.self, from: metadataData)

        adapter.source = .imported
        adapter.remoteRepoURL = nil

        adapter.id = uniqueAdapterID(from: adapter.id)

        let destination = try AdapterRegistry.adapterDirectory(id: adapter.id)
        let weightsDestination = destination.appending(path: "adapter.safetensors")
        let metadataDestination = destination.appending(path: "adapter.json")

        do {
            try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
            try FileManager.default.copyItem(at: weightsURL, to: weightsDestination)

            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(adapter).write(to: metadataDestination)

            try await verifyAdapterWeights(
                at: weightsDestination,
                expectedSHA256: adapterSHA256(from: metadataData),
                fileName: "adapter.safetensors"
            )
        } catch {
            try? FileManager.default.removeItem(at: destination)
            throw error
        }

        AdapterRegistry.notifyRegistryChanged()
        return adapter
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
        let rawRemapped = PEFTKeyRemapper.remap(rawArrays)
        guard !rawRemapped.isEmpty else {
            throw CommunityError.noMappableWeights(errorURL)
        }

        let rank = peftConfig?.r ?? inferredConfig?.rank ?? 16
        let alpha = peftConfig?.loraAlpha ?? inferredConfig?.alpha ?? Float(rank)
        let remapped = normalizePEFTLoRAScaling(
            rawRemapped,
            peftConfig: peftConfig,
            defaultRank: rank,
            defaultAlpha: alpha
        )

        let exactPatterns = PEFTKeyRemapper.inferInjectionPatterns(
            fromRemappedKeys: Array(remapped.keys)
        )
        let targetLayers = PEFTKeyRemapper.inferTargetLayers(
            fromInjectionPatterns: exactPatterns
        )

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

    private func normalizePEFTLoRAScaling(
        _ remapped: [String: MLXArray],
        peftConfig: PEFTAdapterConfig?,
        defaultRank: Int,
        defaultAlpha: Float
    ) -> [String: MLXArray] {
        guard let peftConfig,
              !peftConfig.alphaPattern.isEmpty || !peftConfig.rankPattern.isEmpty else {
            return remapped
        }

        let alphaByPath = peftConfig.alphaPattern.reduce(into: [String: Float]()) { result, entry in
            if let remappedPath = PEFTKeyRemapper.remapModulePath(entry.key) {
                result[remappedPath] = entry.value
            }
        }
        let rankByPath = peftConfig.rankPattern.reduce(into: [String: Int]()) { result, entry in
            if let remappedPath = PEFTKeyRemapper.remapModulePath(entry.key) {
                result[remappedPath] = entry.value
            }
        }

        guard !alphaByPath.isEmpty || !rankByPath.isEmpty else {
            return remapped
        }

        let globalScale = defaultAlpha / Float(max(1, defaultRank))
        guard globalScale.isFinite, abs(globalScale) > Float.ulpOfOne else {
            return remapped
        }

        var normalized = remapped
        for (key, value) in remapped where key.hasSuffix(".loraB") {
            let path = String(key.dropLast(".loraB".count))
            let pathAlpha = alphaByPath[path] ?? defaultAlpha
            let pathRank = rankByPath[path]
                ?? remapped["\(path).loraA"]?.shape.first
                ?? defaultRank
            let pathScale = pathAlpha / Float(max(1, pathRank))
            let multiplier = pathScale / globalScale
            guard multiplier.isFinite,
                  abs(multiplier - 1.0) > 0.000_001 else {
                continue
            }
            normalized[key] = value * multiplier
        }

        return normalized
    }

    private func saveLyCORISLoKRAdapter(
        rawArrays: [String: MLXArray],
        safetensorsMetadata: [String: String],
        peftConfig: PEFTAdapterConfig?,
        nameSeed: String,
        baseModelIDOverride: String? = nil,
        source: AdapterSource,
        remoteRepoURL: URL?,
        errorURL: URL
    ) throws -> LoRAAdapter {
        let metadataConfig = LyCORISLoKRMetadataConfig(metadata: safetensorsMetadata)
        if metadataConfig.useTucker == true {
            throw CommunityError.unsupportedLoKRFormat(
                "This LoKR checkpoint uses Tucker decomposition (`use_tucker=true`), which PiMP Audio Studio cannot import yet.",
                errorURL
            )
        }

        let rawRemapped: PEFTKeyRemapper.LoKRRemapResult
        do {
            rawRemapped = try PEFTKeyRemapper.remapLyCORISLoKR(rawArrays)
        } catch let error as PEFTKeyRemapper.LoKRRemapError {
            switch error {
            case .noConvertibleWeights:
                throw CommunityError.noMappableWeights(errorURL)
            case .unsupportedVariant(let detail):
                throw CommunityError.unsupportedLoKRFormat(detail, errorURL)
            }
        }
        let remapped = metadataConfig.filtered(rawRemapped)

        let linearDim = max(
            1,
            peftConfig?.loKRLinearDim
                ?? metadataConfig.linearDim
                ?? remapped.inferredLinearDim
        )
        let alpha = peftConfig?.loKRAlpha
            ?? metadataConfig.linearAlpha
            ?? remapped.sourceAlpha
            ?? Float(linearDim)
        let dropout = peftConfig?.loKRDropout ?? 0.0
        let weightDecompose = metadataConfig.weightDecompose ?? remapped.weightDecompose
        let trainingLayerCount = remapped.trainedLayerIndices.isEmpty
            ? nil
            : remapped.trainedLayerIndices.count

        let config = LoRAConfig(
            rank: linearDim,
            alpha: alpha,
            dropout: dropout,
            targetLayers: remapped.targetLayers,
            exactInjectionPatterns: remapped.exactInjectionPatterns,
            trainingLayerCount: trainingLayerCount
        )
        let lokrConfig = LoKRConfig(
            linearDim: linearDim,
            alpha: alpha,
            dropout: dropout,
            targetLayers: remapped.targetLayers,
            exactInjectionPatterns: remapped.exactInjectionPatterns,
            trainingLayerCount: trainingLayerCount,
            weightDecompose: weightDecompose
        )

        let adapterID = uniqueAdapterID(from: nameSeed)
        let adapterName = nameSeed
        let baseModelID = baseModelIDOverride
            ?? (peftConfig?.baseModelNameOrPath != nil
            ? PEFTKeyRemapper.resolveBaseModelID(from: peftConfig?.baseModelNameOrPath)
            : "ace-step-dit-xl-sft")

        let adapter = LoRAAdapter(
            id: adapterID,
            name: adapterName,
            config: config,
            kind: .lokr,
            lokrConfig: lokrConfig,
            baseModelID: baseModelID,
            baseModelCategory: .aceStepDiT,
            createdAt: Date(),
            sampleCount: 0,
            finalLoss: nil,
            source: source,
            remoteRepoURL: remoteRepoURL,
            trainedPaths: remapped.trainedPaths,
            trainedLayerIndices: remapped.trainedLayerIndices
        )

        let remappedWeights = remapped.weights.map { ($0.key, $0.value) }
        try AdapterRegistry.save(adapter: adapter, weights: remappedWeights)

        return adapter
    }

    private func uniqueAdapterID(from seed: String) -> String {
        let normalized = AdapterRegistry.normalizedIdentifier(from: seed)
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

        if directoryURL.lastPathComponent == "final" ||
            directoryURL.lastPathComponent == "best" ||
            directoryURL.lastPathComponent.hasPrefix("epoch_") {
            let parent = directoryURL.deletingLastPathComponent().lastPathComponent
            if !parent.isEmpty && parent != "lora" {
                return parent
            }
        }

        return directoryURL.lastPathComponent
    }

    private func preferredSideStepLoKRWeightsURL(in directoryURL: URL) -> URL? {
        let candidates = [
            directoryURL
                .appending(path: "best", directoryHint: .isDirectory)
                .appending(path: "lokr_weights.safetensors"),
            directoryURL
                .appending(path: "final", directoryHint: .isDirectory)
                .appending(path: "lokr_weights.safetensors"),
        ]
        return candidates.first {
            FileManager.default.fileExists(atPath: $0.path(percentEncoded: false))
        }
    }

    private func preferredSideStepPEFTAdapterDirectory(in directoryURL: URL) -> URL? {
        let candidates = [
            directoryURL.appending(path: "best", directoryHint: .isDirectory),
            directoryURL.appending(path: "final", directoryHint: .isDirectory),
        ]
        return candidates.first { candidate in
            FileManager.default.fileExists(
                atPath: candidate.appending(path: "adapter_config.json").path(percentEncoded: false)
            ) && FileManager.default.fileExists(
                atPath: candidate.appending(path: "adapter_model.safetensors").path(percentEncoded: false)
            )
        }
    }

    private func localSideStepTrainingConfig(
        directoryURL: URL,
        weightsURL: URL,
        nameSeedURL: URL?
    ) -> SideStepTrainingConfig? {
        let parentURL = weightsURL.deletingLastPathComponent().deletingLastPathComponent()
        let candidates = [
            weightsURL.deletingLastPathComponent().appending(path: "training_config.json"),
            directoryURL.appending(path: "training_config.json"),
            parentURL.appending(path: "training_config.json"),
            nameSeedURL?.appending(path: "training_config.json"),
        ].compactMap { $0 }

        let decoder = JSONDecoder()
        for candidate in candidates {
            guard FileManager.default.fileExists(atPath: candidate.path(percentEncoded: false)),
                  let data = try? Data(contentsOf: candidate),
                  let decoded = try? decoder.decode(SideStepTrainingConfig.self, from: data) else {
                continue
            }
            return decoded
        }
        return nil
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

    private func safetensorsMetadata(from url: URL) throws -> [String: String] {
        let data = try Data(contentsOf: url, options: [.mappedIfSafe])
        guard data.count >= 8 else { return [:] }

        var headerLengthLE: UInt64 = 0
        withUnsafeMutableBytes(of: &headerLengthLE) { buffer in
            data.copyBytes(to: buffer, from: 0..<8)
        }

        let headerLength = Int(UInt64(littleEndian: headerLengthLE))
        guard headerLength > 0,
              headerLength <= data.count - 8 else {
            return [:]
        }

        let headerData = data.subdata(in: 8..<(8 + headerLength))
        guard let object = try JSONSerialization.jsonObject(with: headerData) as? [String: Any],
              let metadata = object["__metadata__"] as? [String: Any] else {
            return [:]
        }

        var strings: [String: String] = [:]
        for (key, value) in metadata {
            switch value {
            case let string as String:
                strings[key] = string
            case let number as NSNumber:
                strings[key] = number.stringValue
            default:
                continue
            }
        }
        return strings
    }

    // MARK: - Private

    private func performDownload(adapter: CommunityAdapter, retryCount: Int) async throws {
        let directory = try AdapterRegistry.adapterDirectory(id: adapter.id)
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
            localAdapter.id = adapter.id
            localAdapter.source = .community
            localAdapter.remoteRepoURL = nil

            // Save to disk
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let weightsDestination = directory.appending(path: "adapter.safetensors")

            // Write updated metadata with source field
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let updatedMeta = try encoder.encode(localAdapter)
            try updatedMeta.write(to: directory.appending(path: "adapter.json"))
            try weightsData.write(to: weightsDestination)
            downloads[adapter.id] = .downloading(progress: 0.95)
            try await verifyAdapterWeights(
                at: weightsDestination,
                expectedSHA256: adapter.sha256,
                fileName: adapter.adapterURL.lastPathComponent
            )

            downloads[adapter.id] = .downloading(progress: 1.0)
        } catch is CancellationError {
            try? FileManager.default.removeItem(at: directory)
            throw CancellationError()
        } catch {
            if Task.isCancelled {
                try? FileManager.default.removeItem(at: directory)
                throw CancellationError()
            }
            if retryCount > 0 {
                // Auto-retry once
                try await Task.sleep(for: .seconds(1))
                try await performDownload(adapter: adapter, retryCount: retryCount - 1)
            } else {
                // Clean up partial files
                try? FileManager.default.removeItem(at: directory)
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
    case unsupportedLoKRFormat(String, URL)
    case noMappableWeights(URL)

    var errorDescription: String? {
        switch self {
        case .missingFile(let file, let url):
            return "Couldn't find \(file) in this repo. Check that the URL points to a valid PiMP Audio Studio, PEFT LoRA, or compatible LyCORIS LoKR adapter. (\(url.absoluteString))"
        case .alreadyInstalled(let name):
            return "\"\(name)\" is already installed. Delete it first to re-download."
        case .incompatibleVersion(let required):
            return "This adapter requires PiMP Audio Studio \(required) or later."
        case .unsupportedPEFTType(let type):
            return "Unsupported PEFT adapter type '\(type)'. Only LoRA and compatible full-matrix LoKR adapters are supported."
        case .unsupportedLoKRFormat(let detail, let url):
            return "\(detail) (\(url.absoluteString))"
        case .noMappableWeights(let url):
            return "No compatible LoRA or LoKR weights found in this repo. The adapter key format may not be supported. (\(url.absoluteString))"
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
    let alpha: Float?
    let linearDim: Int?
    let linearAlpha: Float?
    let moduleDropout: Float?
    let targetModules: [String]
    let baseModelNameOrPath: String?
    let alphaPattern: [String: Float]
    let rankPattern: [String: Int]

    var loKRLinearDim: Int {
        max(1, linearDim ?? r)
    }

    var loKRAlpha: Float {
        alpha ?? linearAlpha ?? Float(loKRLinearDim)
    }

    var loKRDropout: Float {
        moduleDropout ?? loraDropout
    }

    enum CodingKeys: String, CodingKey {
        case peftType           = "peft_type"
        case r
        case loraAlpha          = "lora_alpha"
        case loraDropout        = "lora_dropout"
        case alpha
        case linearDim          = "linear_dim"
        case linearAlpha        = "linear_alpha"
        case moduleDropout      = "module_dropout"
        case targetModules      = "target_modules"
        case baseModelNameOrPath = "base_model_name_or_path"
        case alphaPattern       = "alpha_pattern"
        case rankPattern        = "rank_pattern"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        peftType = try c.decode(String.self, forKey: .peftType)
        let decodedLinearDim = try c.decodeIfPresent(Int.self, forKey: .linearDim)
        r = try c.decodeIfPresent(Int.self, forKey: .r) ?? decodedLinearDim ?? 16

        // lora_alpha and lora_dropout may be stored as Int or Float in JSON —
        // decode via Double to avoid type-mismatch errors.
        func decodeOptionalFloat(_ key: CodingKeys) -> Float? {
            if let v = try? c.decodeIfPresent(Double.self, forKey: key) { return Float(v) }
            if let v = try? c.decodeIfPresent(Int.self,    forKey: key) { return Float(v) }
            return nil
        }

        func decodeFloat(_ key: CodingKeys, default defaultVal: Float) -> Float {
            decodeOptionalFloat(key) ?? defaultVal
        }

        loraAlpha   = decodeFloat(.loraAlpha,   default: Float(r))
        loraDropout = decodeFloat(.loraDropout, default: 0.0)
        alpha = decodeOptionalFloat(.alpha)
        linearDim = decodedLinearDim
        linearAlpha = decodeOptionalFloat(.linearAlpha)
        moduleDropout = decodeOptionalFloat(.moduleDropout)

        if let modules = try? c.decodeIfPresent([String].self, forKey: .targetModules) {
            targetModules = modules
        } else if let module = try? c.decodeIfPresent(String.self, forKey: .targetModules) {
            targetModules = [module]
        } else {
            targetModules = []
        }
        baseModelNameOrPath = try? c.decodeIfPresent(String.self, forKey: .baseModelNameOrPath)
        alphaPattern = Self.decodeFloatPattern(from: c, key: .alphaPattern)
        rankPattern = Self.decodeIntPattern(from: c, key: .rankPattern)
    }

    private static func decodeFloatPattern(
        from container: KeyedDecodingContainer<CodingKeys>,
        key: CodingKeys
    ) -> [String: Float] {
        if let values = try? container.decodeIfPresent([String: Double].self, forKey: key) {
            return values.mapValues(Float.init)
        }
        if let values = try? container.decodeIfPresent([String: Int].self, forKey: key) {
            return values.mapValues(Float.init)
        }
        return [:]
    }

    private static func decodeIntPattern(
        from container: KeyedDecodingContainer<CodingKeys>,
        key: CodingKeys
    ) -> [String: Int] {
        if let values = try? container.decodeIfPresent([String: Int].self, forKey: key) {
            return values
        }
        if let values = try? container.decodeIfPresent([String: Double].self, forKey: key) {
            return values.mapValues { max(1, Int($0.rounded())) }
        }
        return [:]
    }
}

private struct LyCORISLoKRMetadataConfig {
    let linearDim: Int?
    let linearAlpha: Float?
    let weightDecompose: Bool?
    let useTucker: Bool?
    let targetModules: [String]?
    let targetMLP: Bool?
    let attentionType: String?

    init(metadata: [String: String]) {
        guard let rawConfig = metadata["lokr_config"],
              let data = rawConfig.data(using: .utf8),
              let decoded = try? JSONDecoder().decode(Decoded.self, from: data) else {
            linearDim = nil
            linearAlpha = nil
            weightDecompose = nil
            useTucker = nil
            targetModules = nil
            targetMLP = nil
            attentionType = nil
            return
        }

        linearDim = decoded.linearDim
        linearAlpha = decoded.linearAlpha
        weightDecompose = decoded.weightDecompose
        useTucker = decoded.useTucker
        targetModules = decoded.targetModules
        targetMLP = decoded.targetMLP
        attentionType = decoded.attentionType
    }

    func filtered(
        _ result: PEFTKeyRemapper.LoKRRemapResult
    ) -> PEFTKeyRemapper.LoKRRemapResult {
        guard let allowedPatterns = allowedInjectionPatterns(existing: result.exactInjectionPatterns) else {
            return result
        }

        let filteredWeights = result.weights.filter { key, _ in
            guard let pattern = injectionPattern(fromRemappedKey: key) else { return false }
            return allowedPatterns.contains(pattern)
        }
        guard !filteredWeights.isEmpty else { return result }

        let exactPatterns = PEFTKeyRemapper.inferLoKRInjectionPatterns(
            fromRemappedKeys: Array(filteredWeights.keys)
        )
        let targetLayers = PEFTKeyRemapper.inferTargetLayers(
            fromInjectionPatterns: exactPatterns
        )
        let trainedPaths = LoRAAdapter.trainedPaths(
            fromWeightKeys: Array(filteredWeights.keys),
            kind: .lokr
        )

        return PEFTKeyRemapper.LoKRRemapResult(
            weights: filteredWeights,
            inferredLinearDim: result.inferredLinearDim,
            sourceAlpha: result.sourceAlpha,
            exactInjectionPatterns: exactPatterns,
            targetLayers: targetLayers,
            trainedPaths: trainedPaths,
            trainedLayerIndices: LoRAAdapter.layerIndices(from: trainedPaths),
            weightDecompose: filteredWeights.keys.contains { $0.hasSuffix(".lokrMagnitude") }
        )
    }

    private func allowedInjectionPatterns(existing: [String]) -> Set<String>? {
        guard let targetModules, !targetModules.isEmpty else {
            guard targetMLP == false else { return nil }
            return Set(existing.filter { !$0.hasPrefix("mlp.") })
        }

        let attentionSelection = attentionType?.lowercased() ?? "both"
        let includeSelfAttention = attentionSelection != "cross"
        let includeCrossAttention = attentionSelection != "self"
        let includeMLP = targetMLP != false
        var patterns = Set<String>()

        for module in targetModules {
            let normalized = module
                .lowercased()
                .replacingOccurrences(of: "-", with: "_")

            if let projection = Self.attentionProjection(from: normalized) {
                if normalized.hasPrefix("self_attn.") {
                    patterns.insert("selfAttn.\(projection)")
                } else if normalized.hasPrefix("cross_attn.") {
                    patterns.insert("crossAttn.\(projection)")
                } else {
                    if includeSelfAttention { patterns.insert("selfAttn.\(projection)") }
                    if includeCrossAttention { patterns.insert("crossAttn.\(projection)") }
                }
                continue
            }

            if includeMLP, let projection = Self.mlpProjection(from: normalized) {
                patterns.insert("mlp.\(projection)")
            }
        }

        return patterns.isEmpty ? nil : patterns
    }

    private func injectionPattern(fromRemappedKey key: String) -> String? {
        let parts = key.split(separator: ".")
        guard parts.count >= 5,
              parts[0] == "layers",
              Int(parts[1]) != nil else {
            return nil
        }
        return "\(parts[2]).\(parts[3])"
    }

    private static func attentionProjection(from module: String) -> String? {
        let leaf = module.split(separator: ".").last.map(String.init) ?? module
        switch leaf {
        case "q_proj", "q": return "qProj"
        case "k_proj", "k": return "kProj"
        case "v_proj", "v": return "vProj"
        case "o_proj", "out_proj", "o": return "oProj"
        default: return nil
        }
    }

    private static func mlpProjection(from module: String) -> String? {
        let leaf = module.split(separator: ".").last.map(String.init) ?? module
        switch leaf {
        case "gate_proj", "w1": return "gateProj"
        case "up_proj", "w3": return "upProj"
        case "down_proj", "w2": return "downProj"
        default: return nil
        }
    }

    private struct Decoded: Decodable {
        let linearDim: Int?
        let linearAlpha: Float?
        let weightDecompose: Bool?
        let useTucker: Bool?
        let targetModules: [String]?
        let targetMLP: Bool?
        let attentionType: String?

        enum CodingKeys: String, CodingKey {
            case linearDim = "linear_dim"
            case linearAlpha = "linear_alpha"
            case weightDecompose = "weight_decompose"
            case useTucker = "use_tucker"
            case targetModules = "target_modules"
            case targetMLP = "target_mlp"
            case attentionType = "attention_type"
        }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            linearDim = try c.decodeIfPresent(Int.self, forKey: .linearDim)
            weightDecompose = try c.decodeIfPresent(Bool.self, forKey: .weightDecompose)
            useTucker = try c.decodeIfPresent(Bool.self, forKey: .useTucker)
            targetMLP = try c.decodeIfPresent(Bool.self, forKey: .targetMLP)
            attentionType = try c.decodeIfPresent(String.self, forKey: .attentionType)
            if let modules = try? c.decodeIfPresent([String].self, forKey: .targetModules) {
                targetModules = modules
            } else if let module = try? c.decodeIfPresent(String.self, forKey: .targetModules) {
                targetModules = [module]
            } else {
                targetModules = nil
            }

            if let v = try? c.decodeIfPresent(Double.self, forKey: .linearAlpha) {
                linearAlpha = Float(v)
            } else if let v = try? c.decodeIfPresent(Int.self, forKey: .linearAlpha) {
                linearAlpha = Float(v)
            } else {
                linearAlpha = nil
            }
        }
    }
}

private struct SideStepTrainingConfig: Decodable {
    let modelVariant: String?
    let baseModel: String?
    let explicitBaseModelID: String?

    var resolvedBaseModelID: String? {
        if let explicitBaseModelID {
            return explicitBaseModelID
        }
        guard let candidate = modelVariant ?? baseModel else {
            return nil
        }
        return PEFTKeyRemapper.resolveBaseModelID(from: candidate)
    }

    enum CodingKeys: String, CodingKey {
        case modelVariant = "model_variant"
        case baseModel = "base_model"
        case explicitBaseModelID = "base_model_id"
    }
}
