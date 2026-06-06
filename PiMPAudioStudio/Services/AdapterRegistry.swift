import Foundation
import PiMPAudioCore
import MLX

struct AdapterRegistry {
    static let trainingRunLogFileName = "training_log.json"

    static var adaptersDirectory: URL {
        AppStorageLocations.adaptersDirectory
    }

    static func listAdapters() -> [LoRAAdapter] {
        let dir = adaptersDirectory
        let fm = FileManager.default

        guard let contents = try? fm.contentsOfDirectory(
            at: dir,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) else {
            return []
        }

        var adapters: [LoRAAdapter] = []
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601

        for itemURL in contents {
            let isDir = (try? itemURL.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory ?? false
            guard isDir else { continue }

            let metadataURL = itemURL.appending(path: "adapter.json")
            let weightsURL = itemURL.appending(path: "adapter.safetensors")

            guard fm.fileExists(atPath: metadataURL.path(percentEncoded: false)),
                  fm.fileExists(atPath: weightsURL.path(percentEncoded: false)) else {
                continue
            }

            guard let data = try? Data(contentsOf: metadataURL),
                  let adapter = try? decoder.decode(LoRAAdapter.self, from: data) else {
                continue
            }

            adapters.append(adapter)
        }

        return adapters.sorted { $0.createdAt > $1.createdAt }
    }

    static func compatibleAdapters(for category: ModelCategory) -> [LoRAAdapter] {
        listAdapters().filter { $0.baseModelCategory == category }
    }

    static func adapterDirectory(id: String) -> URL {
        adaptersDirectory.appending(path: id)
    }

    static func save(adapter: LoRAAdapter, weights: [(String, MLXArray)]) throws {
        let dir = adapterDirectory(id: adapter.id)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        // Save metadata as JSON
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(adapter)
        try data.write(to: dir.appending(path: "adapter.json"))

        // Save weights as safetensors
        let dict = Dictionary(weights, uniquingKeysWith: { _, last in last })
        try MLX.save(arrays: dict, url: dir.appending(path: "adapter.safetensors"))

        notifyRegistryChanged()
    }

    static func loadWeights(adapterID: String) throws -> [(String, MLXArray)] {
        let url = adapterDirectory(id: adapterID).appending(path: "adapter.safetensors")
        let dict = try MLX.loadArrays(url: url)
        return dict.map { ($0.key, $0.value) }
    }

    /// Async-safe wrapper that runs the safetensors read off the calling actor.
    /// Use this from `@MainActor` callers (e.g. `LoRACoordinator`) so the UI
    /// thread is not blocked while a multi-MB adapter file is read from disk.
    static func loadWeights(adapterID: String) async throws -> [(String, MLXArray)] {
        try await Task.detached(priority: .userInitiated) {
            try Self.loadWeights(adapterID: adapterID) as [(String, MLXArray)]
        }.value
    }

    // MARK: - Adapter Management

    /// Errors for adapter management operations.
    enum AdapterError: Error, LocalizedError {
        case nameExists(String)
        case notFound(String)
        case rankMismatch(Int, Int)
        case kindMismatch(AdapterKind, AdapterKind)
        case unsupportedMergeKind(AdapterKind)

        var errorDescription: String? {
            switch self {
            case .nameExists(let name):
                return "An adapter named '\(name)' already exists"
            case .notFound(let id):
                return "Adapter '\(id)' not found"
            case .rankMismatch(let a, let b):
                return "Cannot merge adapters with different ranks (\(a) vs \(b))"
            case .kindMismatch(let a, let b):
                return "Cannot merge \(a.rawValue.uppercased()) and \(b.rawValue.uppercased()) adapters"
            case .unsupportedMergeKind(let kind):
                return "Merging \(kind.rawValue.uppercased()) adapters is not supported yet"
            }
        }
    }

    /// Rename an adapter (updates directory name and metadata).
    static func rename(adapterID: String, to newName: String) throws {
        let fm = FileManager.default
        let sourceDir = adapterDirectory(id: adapterID)

        guard fm.fileExists(atPath: sourceDir.path(percentEncoded: false)) else {
            throw AdapterError.notFound(adapterID)
        }

        // Generate new ID from new name
        let newID = newName
            .lowercased()
            .replacingOccurrences(of: " ", with: "-")
            .replacingOccurrences(of: "[^a-z0-9\\-]", with: "", options: .regularExpression)

        let destDir = adapterDirectory(id: newID)

        // Check collision
        if newID != adapterID && fm.fileExists(atPath: destDir.path(percentEncoded: false)) {
            throw AdapterError.nameExists(newName)
        }

        // Read existing metadata, update name and id
        let metadataURL = sourceDir.appending(path: "adapter.json")
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let data = try Data(contentsOf: metadataURL)
        var adapter = try decoder.decode(LoRAAdapter.self, from: data)
        adapter.name = newName
        adapter.id = newID

        // Move directory first (if ID changed), then write metadata.
        // This order ensures we don't corrupt metadata if the move fails.
        if newID != adapterID {
            try fm.moveItem(at: sourceDir, to: destDir)
        }

        // Write updated metadata to the (possibly moved) directory
        let finalMetadataURL = adapterDirectory(id: newID).appending(path: "adapter.json")
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let updatedData = try encoder.encode(adapter)
        try updatedData.write(to: finalMetadataURL)

        notifyRegistryChanged()
    }

    /// Delete an adapter and all its files.
    static func delete(adapterID: String) throws {
        let fm = FileManager.default
        let dir = adapterDirectory(id: adapterID)

        guard fm.fileExists(atPath: dir.path(percentEncoded: false)) else {
            throw AdapterError.notFound(adapterID)
        }

        try fm.removeItem(at: dir)

        notifyRegistryChanged()
    }

    /// Export an adapter (safetensors + metadata) to a destination directory.
    static func export(adapterID: String, to destination: URL) throws {
        let fm = FileManager.default
        let sourceDir = adapterDirectory(id: adapterID)

        guard fm.fileExists(atPath: sourceDir.path(percentEncoded: false)) else {
            throw AdapterError.notFound(adapterID)
        }

        // Copy safetensors
        let weightsSource = sourceDir.appending(path: "adapter.safetensors")
        let weightsDest = destination.appending(path: "adapter.safetensors")
        try fm.copyItem(at: weightsSource, to: weightsDest)

        // Copy metadata
        let metaSource = sourceDir.appending(path: "adapter.json")
        let metaDest = destination.appending(path: "adapter.json")
        try fm.copyItem(at: metaSource, to: metaDest)

        for sidecarName in ["training_manifest.json", trainingRunLogFileName] {
            let sidecarSource = sourceDir.appending(path: sidecarName)
            guard fm.fileExists(atPath: sidecarSource.path(percentEncoded: false)) else {
                continue
            }
            try fm.copyItem(at: sidecarSource, to: destination.appending(path: sidecarName))
        }
    }

    // MARK: - Training Manifest

    /// Persists a training manifest so the session can be reloaded for retraining.
    /// Silently ignored on failure — the adapter itself is already saved.
    static func saveManifest(_ manifest: TrainingManifest, adapterID: String) {
        let url = adapterDirectory(id: adapterID).appending(path: "training_manifest.json")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        guard let data = try? encoder.encode(manifest) else { return }
        try? data.write(to: url)
    }

    /// Returns the saved training manifest for an adapter, or nil if none exists.
    static func loadManifest(adapterID: String) -> TrainingManifest? {
        let url = adapterDirectory(id: adapterID).appending(path: "training_manifest.json")
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(TrainingManifest.self, from: data)
    }

    // MARK: - Training Run Logs

    static func trainingRunLogURL(adapterID: String) -> URL {
        adapterDirectory(id: adapterID).appending(path: trainingRunLogFileName)
    }

    static func saveTrainingRunLog(_ log: TrainingRunLog, adapterID: String) throws {
        let dir = adapterDirectory(id: adapterID)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(log)
        try data.write(to: trainingRunLogURL(adapterID: adapterID), options: .atomic)
    }

    static func loadTrainingRunLog(adapterID: String) -> TrainingRunLog? {
        guard let data = try? Data(contentsOf: trainingRunLogURL(adapterID: adapterID)) else {
            return nil
        }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try? decoder.decode(TrainingRunLog.self, from: data)
    }

    // MARK: - Training Checkpoints

    static func checkpointURL(adapterID: String) -> URL {
        adapterDirectory(id: adapterID).appending(path: "training_checkpoint.json")
    }

    static func optimizerStateURL(adapterID: String) -> URL {
        adapterDirectory(id: adapterID).appending(path: "optimizer_state.json")
    }

    static func weightsURL(adapterID: String) -> URL {
        adapterDirectory(id: adapterID).appending(path: "adapter.safetensors")
    }

    /// Saves the metadata needed to resume an interrupted training session.
    static func saveCheckpoint(_ checkpoint: TrainingCheckpoint, adapterID: String) throws {
        let dir = adapterDirectory(id: adapterID)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(checkpoint)
        try data.write(to: checkpointURL(adapterID: adapterID))

        if !FileManager.default.fileExists(atPath: checkpoint.optimizerStateURL.path(percentEncoded: false)) {
            try Data("{}".utf8).write(to: checkpoint.optimizerStateURL)
        }
    }

    /// Returns the saved training checkpoint for a paused adapter, or nil.
    static func loadCheckpoint(adapterID: String) -> TrainingCheckpoint? {
        guard let data = try? Data(contentsOf: checkpointURL(adapterID: adapterID)) else {
            return nil
        }
        return try? JSONDecoder().decode(TrainingCheckpoint.self, from: data)
    }

    static func hasCheckpoint(adapterID: String) -> Bool {
        FileManager.default.fileExists(atPath: checkpointURL(adapterID: adapterID).path(percentEncoded: false))
    }

    /// Removes pause/resume metadata after training completes successfully.
    static func deleteCheckpoint(adapterID: String) {
        let fm = FileManager.default
        try? fm.removeItem(at: checkpointURL(adapterID: adapterID))
        try? fm.removeItem(at: optimizerStateURL(adapterID: adapterID))
    }

    /// Merge two adapters with weighted interpolation: alpha * A + (1-alpha) * B.
    ///
    /// Both adapters must have the same rank. Creates a new adapter.
    ///
    /// - Parameters:
    ///   - adapterA: ID of the first adapter.
    ///   - adapterB: ID of the second adapter.
    ///   - alpha: Interpolation weight (1.0 = all A, 0.0 = all B).
    ///   - name: Display name for the merged adapter.
    /// - Returns: The newly created merged adapter.
    @discardableResult
    static func merge(
        adapterA: String,
        adapterB: String,
        alpha: Float,
        name: String
    ) throws -> LoRAAdapter {
        // Load both adapters' metadata
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601

        let metaA = try Data(contentsOf: adapterDirectory(id: adapterA).appending(path: "adapter.json"))
        let infoA = try decoder.decode(LoRAAdapter.self, from: metaA)

        let metaB = try Data(contentsOf: adapterDirectory(id: adapterB).appending(path: "adapter.json"))
        let infoB = try decoder.decode(LoRAAdapter.self, from: metaB)

        guard infoA.kind == infoB.kind else {
            throw AdapterError.kindMismatch(infoA.kind, infoB.kind)
        }
        guard infoA.kind == .lora else {
            throw AdapterError.unsupportedMergeKind(infoA.kind)
        }

        // Validate same rank
        guard infoA.config.rank == infoB.config.rank else {
            throw AdapterError.rankMismatch(infoA.config.rank, infoB.config.rank)
        }

        // Load weights
        let weightsA = try loadWeights(adapterID: adapterA)
        let weightsB = try loadWeights(adapterID: adapterB)

        let dictA = Dictionary(weightsA, uniquingKeysWith: { _, last in last })
        let dictB = Dictionary(weightsB, uniquingKeysWith: { _, last in last })

        // Interpolate: alpha * A + (1-alpha) * B
        let alphaArray = MLXArray(alpha)
        let oneMinusAlpha = MLXArray(1.0 - alpha)

        var mergedWeights: [(String, MLXArray)] = []
        for (key, valueA) in dictA {
            if let valueB = dictB[key] {
                let merged = alphaArray * valueA + oneMinusAlpha * valueB
                mergedWeights.append((key, merged))
            } else {
                // Key only in A — scale by alpha
                mergedWeights.append((key, alphaArray * valueA))
            }
        }
        // Keys only in B
        for (key, valueB) in dictB where dictA[key] == nil {
            mergedWeights.append((key, oneMinusAlpha * valueB))
        }

        // Create merged adapter metadata
        let mergedID = name
            .lowercased()
            .replacingOccurrences(of: " ", with: "-")
            .replacingOccurrences(of: "[^a-z0-9\\-]", with: "", options: .regularExpression)

        let mergedAdapter = LoRAAdapter(
            id: mergedID,
            name: name,
            config: infoA.config,
            baseModelID: infoA.baseModelID,
            baseModelCategory: infoA.baseModelCategory,
            createdAt: Date(),
            sampleCount: infoA.sampleCount + infoB.sampleCount,
            finalLoss: nil
        )

        try save(adapter: mergedAdapter, weights: mergedWeights)
        return mergedAdapter
    }

    private static func notifyRegistryChanged() {
        if Thread.isMainThread {
            NotificationCenter.default.post(name: .adapterRegistryDidChange, object: nil)
        } else {
            DispatchQueue.main.async {
                NotificationCenter.default.post(name: .adapterRegistryDidChange, object: nil)
            }
        }
    }
}

extension Notification.Name {
    static let adapterRegistryDidChange = Notification.Name("adapterRegistryDidChange")
    static let adapterSelectionDidChange = Notification.Name("adapterSelectionDidChange")
    static let mcpAdapterSelectionDidChange = Notification.Name("mcpAdapterSelectionDidChange")
}
