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
            let resourceValues = try? itemURL.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
            guard resourceValues?.isDirectory == true,
                  resourceValues?.isSymbolicLink != true else {
                continue
            }

            let metadataURL = itemURL.appending(path: "adapter.json")
            let weightsURL = itemURL.appending(path: "adapter.safetensors")

            guard fm.fileExists(atPath: metadataURL.path(percentEncoded: false)),
                  fm.fileExists(atPath: weightsURL.path(percentEncoded: false)) else {
                continue
            }

            guard let data = try? Data(contentsOf: metadataURL),
                  let adapter = try? decoder.decode(LoRAAdapter.self, from: data),
                  let expectedDirectory = try? adapterDirectory(id: adapter.id),
                  expectedDirectory.standardizedFileURL == itemURL.standardizedFileURL else {
                continue
            }

            adapters.append(adapter)
        }

        return adapters.sorted { $0.createdAt > $1.createdAt }
    }

    static func compatibleAdapters(for category: ModelCategory) -> [LoRAAdapter] {
        listAdapters().filter { $0.baseModelCategory == category }
    }

    static func adapterDirectory(id: String, baseDirectory: URL? = nil) throws -> URL {
        let base = (baseDirectory ?? adaptersDirectory).standardizedFileURL
        guard !id.isEmpty,
              id != ".",
              id != "..",
              !id.contains("/"),
              !id.contains("\\"),
              id.rangeOfCharacter(from: .controlCharacters) == nil else {
            throw AdapterError.invalidIdentifier(id)
        }

        let candidate = base.appendingPathComponent(id, isDirectory: true).standardizedFileURL
        guard candidate.deletingLastPathComponent() == base else {
            throw AdapterError.invalidIdentifier(id)
        }

        let resolvedBase = base.resolvingSymlinksInPath()
        let resolvedCandidate = candidate.resolvingSymlinksInPath()
        guard resolvedCandidate.deletingLastPathComponent() == resolvedBase else {
            throw AdapterError.invalidIdentifier(id)
        }
        return candidate
    }

    static func normalizedIdentifier(from seed: String) -> String {
        let normalized = seed
            .lowercased()
            .replacingOccurrences(of: "[^a-z0-9\\-]", with: "-", options: .regularExpression)
            .trimmingCharacters(in: CharacterSet(charactersIn: "-"))
        return normalized.isEmpty ? "imported-adapter" : normalized
    }

    static func save(adapter: LoRAAdapter, weights: [(String, MLXArray)]) throws {
        let dir = try adapterDirectory(id: adapter.id)
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
        let url = try adapterDirectory(id: adapterID).appending(path: "adapter.safetensors")
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
        case invalidIdentifier(String)
        case rankMismatch(Int, Int)
        case kindMismatch(AdapterKind, AdapterKind)
        case unsupportedMergeKind(AdapterKind)
        case mergeFailed(String)

        var errorDescription: String? {
            switch self {
            case .nameExists(let name):
                return "An adapter named '\(name)' already exists"
            case .notFound(let id):
                return "Adapter '\(id)' not found"
            case .invalidIdentifier(let id):
                return "Adapter identifier '\(id)' is not a safe directory name"
            case .rankMismatch(let a, let b):
                return "Cannot merge adapters with different ranks (\(a) vs \(b))"
            case .kindMismatch(let a, let b):
                return "Cannot merge \(a.rawValue.uppercased()) and \(b.rawValue.uppercased()) adapters"
            case .unsupportedMergeKind(let kind):
                return "Merging \(kind.rawValue.uppercased()) adapters is not supported yet"
            case .mergeFailed(let message):
                return message
            }
        }
    }

    /// Rename an adapter (updates directory name and metadata).
    static func rename(adapterID: String, to newName: String) throws {
        let fm = FileManager.default
        let sourceDir = try adapterDirectory(id: adapterID)

        guard fm.fileExists(atPath: sourceDir.path(percentEncoded: false)) else {
            throw AdapterError.notFound(adapterID)
        }

        // Generate new ID from new name
        let newID = normalizedIdentifier(from: newName)

        let destDir = try adapterDirectory(id: newID)

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
        let finalMetadataURL = try adapterDirectory(id: newID).appending(path: "adapter.json")
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
        let dir = try adapterDirectory(id: adapterID)

        guard fm.fileExists(atPath: dir.path(percentEncoded: false)) else {
            throw AdapterError.notFound(adapterID)
        }

        try fm.removeItem(at: dir)

        notifyRegistryChanged()
    }

    /// Export an adapter (safetensors + metadata) to a destination directory.
    static func export(adapterID: String, to destination: URL) throws {
        let fm = FileManager.default
        let sourceDir = try adapterDirectory(id: adapterID)

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
        guard let url = try? adapterDirectory(id: adapterID).appending(path: "training_manifest.json") else {
            return
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        guard let data = try? encoder.encode(manifest) else { return }
        try? data.write(to: url)
    }

    /// Returns the saved training manifest for an adapter, or nil if none exists.
    static func loadManifest(adapterID: String) -> TrainingManifest? {
        guard let url = try? adapterDirectory(id: adapterID).appending(path: "training_manifest.json") else {
            return nil
        }
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(TrainingManifest.self, from: data)
    }

    // MARK: - Training Run Logs

    static func trainingRunLogURL(adapterID: String) throws -> URL {
        try adapterDirectory(id: adapterID).appending(path: trainingRunLogFileName)
    }

    static func saveTrainingRunLog(_ log: TrainingRunLog, adapterID: String) throws {
        let dir = try adapterDirectory(id: adapterID)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(log)
        let url = try trainingRunLogURL(adapterID: adapterID)
        try data.write(to: url, options: .atomic)
    }

    static func loadTrainingRunLog(adapterID: String) -> TrainingRunLog? {
        guard let url = try? trainingRunLogURL(adapterID: adapterID),
              let data = try? Data(contentsOf: url) else {
            return nil
        }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try? decoder.decode(TrainingRunLog.self, from: data)
    }

    // MARK: - Training Checkpoints

    static func checkpointURL(adapterID: String) throws -> URL {
        try adapterDirectory(id: adapterID).appending(path: "training_checkpoint.json")
    }

    static func optimizerStateURL(adapterID: String) throws -> URL {
        try adapterDirectory(id: adapterID).appending(path: "optimizer_state.safetensors")
    }

    private static func legacyOptimizerStateURL(adapterID: String) throws -> URL {
        try adapterDirectory(id: adapterID).appending(path: "optimizer_state.json")
    }

    static func weightsURL(adapterID: String) throws -> URL {
        try adapterDirectory(id: adapterID).appending(path: "adapter.safetensors")
    }

    static func saveOptimizerState(_ state: [(String, MLXArray)], adapterID: String) throws {
        guard !state.isEmpty else { return }
        let dir = try adapterDirectory(id: adapterID)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let dict = Dictionary(state, uniquingKeysWith: { _, last in last })
        let url = try optimizerStateURL(adapterID: adapterID)
        try MLX.save(arrays: dict, url: url)
    }

    /// Saves the metadata needed to resume an interrupted training session.
    static func saveCheckpoint(_ checkpoint: TrainingCheckpoint, adapterID: String) throws {
        let dir = try adapterDirectory(id: adapterID)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(checkpoint)
        let url = try checkpointURL(adapterID: adapterID)
        try data.write(to: url)
    }

    /// Returns the saved training checkpoint for a paused adapter, or nil.
    static func loadCheckpoint(adapterID: String) -> TrainingCheckpoint? {
        guard let url = try? checkpointURL(adapterID: adapterID),
              let data = try? Data(contentsOf: url) else {
            return nil
        }
        return try? JSONDecoder().decode(TrainingCheckpoint.self, from: data)
    }

    static func hasCheckpoint(adapterID: String) -> Bool {
        guard let url = try? checkpointURL(adapterID: adapterID) else { return false }
        return FileManager.default.fileExists(atPath: url.path(percentEncoded: false))
    }

    /// Removes pause/resume metadata after training completes successfully.
    static func deleteCheckpoint(adapterID: String) {
        let fm = FileManager.default
        let urls = [
            try? checkpointURL(adapterID: adapterID),
            try? optimizerStateURL(adapterID: adapterID),
            try? legacyOptimizerStateURL(adapterID: adapterID)
        ].compactMap { $0 }
        for url in urls {
            try? fm.removeItem(at: url)
        }
    }

    /// Merge two LoRA adapters by blending effective deltas:
    /// `w · s1 · (B1@A1) + (1-w) · s2 · (B2@A2)` via rank concatenation.
    ///
    /// Independent A/B factor interpolation is intentionally **not** used —
    /// that introduces cross terms and is not a convex combination of the
    /// adapters' weight deltas.
    ///
    /// - Parameters:
    ///   - adapterA: ID of the first adapter.
    ///   - adapterB: ID of the second adapter.
    ///   - alpha: Blend weight on A (1.0 = all A, 0.0 = all B).
    ///   - name: Display name for the merged adapter.
    /// - Returns: The newly created merged adapter (rank typically rA+rB).
    @discardableResult
    static func merge(
        adapterA: String,
        adapterB: String,
        alpha: Float,
        name: String
    ) throws -> LoRAAdapter {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601

        let directoryA = try adapterDirectory(id: adapterA)
        let metaA = try Data(contentsOf: directoryA.appending(path: "adapter.json"))
        let infoA = try decoder.decode(LoRAAdapter.self, from: metaA)

        let directoryB = try adapterDirectory(id: adapterB)
        let metaB = try Data(contentsOf: directoryB.appending(path: "adapter.json"))
        let infoB = try decoder.decode(LoRAAdapter.self, from: metaB)

        guard infoA.kind == infoB.kind else {
            throw AdapterError.kindMismatch(infoA.kind, infoB.kind)
        }
        guard infoA.kind == .lora else {
            throw AdapterError.unsupportedMergeKind(infoA.kind)
        }
        guard infoA.baseModelID == infoB.baseModelID else {
            throw AdapterError.mergeFailed(
                "Cannot merge adapters for different base models ('\(infoA.baseModelID)' vs '\(infoB.baseModelID)')."
            )
        }

        let weightsA = try loadWeights(adapterID: adapterA)
        let weightsB = try loadWeights(adapterID: adapterB)
        let dictA = Dictionary(weightsA, uniquingKeysWith: { _, last in last })
        let dictB = Dictionary(weightsB, uniquingKeysWith: { _, last in last })

        let scaleA = infoA.config.rank > 0
            ? infoA.config.alpha / Float(infoA.config.rank)
            : 1
        let scaleB = infoB.config.rank > 0
            ? infoB.config.alpha / Float(infoB.config.rank)
            : 1

        let blended: LoRAWeightMerge.Result
        do {
            blended = try LoRAWeightMerge.blendDeltas(
                weightsA: dictA,
                weightsB: dictB,
                weightOnA: alpha,
                scalingA: scaleA,
                scalingB: scaleB
            )
        } catch let error as LocalizedError {
            throw AdapterError.mergeFailed(error.errorDescription ?? String(describing: error))
        } catch {
            throw AdapterError.mergeFailed(error.localizedDescription)
        }

        var config = infoA.config
        config.rank = blended.rank
        config.alpha = blended.alpha
        config.targetLayers = infoA.config.targetLayers.union(infoB.config.targetLayers)
        let patterns = Set(
            (infoA.config.exactInjectionPatterns ?? [])
                + (infoB.config.exactInjectionPatterns ?? [])
        )
        if !patterns.isEmpty {
            config.exactInjectionPatterns = patterns.sorted()
        }

        let trainedPaths = Set(
            (infoA.trainedPaths ?? []) + (infoB.trainedPaths ?? [])
        )
        let trainedLayerIndices = Set(
            (infoA.trainedLayerIndices ?? []) + (infoB.trainedLayerIndices ?? [])
        )

        let mergedID = normalizedIdentifier(from: name)
        let mergedAdapter = LoRAAdapter(
            id: mergedID,
            name: name,
            config: config,
            baseModelID: infoA.baseModelID,
            baseModelCategory: infoA.baseModelCategory,
            createdAt: Date(),
            sampleCount: infoA.sampleCount + infoB.sampleCount,
            finalLoss: nil,
            trainedPaths: trainedPaths.isEmpty ? nil : trainedPaths.sorted(),
            trainedLayerIndices: trainedLayerIndices.isEmpty ? nil : trainedLayerIndices.sorted()
        )

        try save(adapter: mergedAdapter, weights: blended.weights)
        return mergedAdapter
    }

    static func notifyRegistryChanged() {
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
