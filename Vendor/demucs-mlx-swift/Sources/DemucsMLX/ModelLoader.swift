import Dispatch
import Foundation
import Hub
import MLX

/// Shared utilities for resolving model directories, loading configs, and loading weights.
enum ModelLoader {
    private static let defaultHubModelRepo = "iky1e/demucs-mlx"

    /// Resolve the directory containing model files (safetensors + config JSON).
    static func resolveModelDirectory(modelName: String, preferred: URL?) throws -> URL {
        let fm = FileManager.default
        var candidates: [URL] = []

        if let preferred {
            candidates.append(preferred.appendingPathComponent(modelName, isDirectory: true))
            candidates.append(preferred)
        }

        if let env = ProcessInfo.processInfo.environment["DEMUCS_MLX_SWIFT_MODEL_DIR"], !env.isEmpty {
            let envURL = URL(fileURLWithPath: env, isDirectory: true)
            candidates.append(envURL.appendingPathComponent(modelName, isDirectory: true))
            candidates.append(envURL)
        }

        // Check standard cache directory
        if let cachesDir = fm.urls(for: .cachesDirectory, in: .userDomainMask).first {
            let cacheModel = cachesDir
                .appendingPathComponent("demucs-mlx-swift-models", isDirectory: true)
                .appendingPathComponent(modelName, isDirectory: true)
            candidates.append(cacheModel)
        }
        let homeCache = URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
            .appendingPathComponent(".cache/demucs-mlx-swift-models/\(modelName)", isDirectory: true)
        candidates.append(homeCache)

        let cwd = URL(fileURLWithPath: fm.currentDirectoryPath, isDirectory: true)
        candidates.append(cwd.appendingPathComponent(".scratch/models/\(modelName)", isDirectory: true))
        candidates.append(cwd.appendingPathComponent("Models/\(modelName)", isDirectory: true))
        candidates.append(cwd.appendingPathComponent(modelName, isDirectory: true))

        for candidate in candidates where fm.fileExists(atPath: candidate.path) {
            if hasRequiredModelFiles(in: candidate, modelName: modelName, fileManager: fm) {
                return candidate
            }
        }

        let repoOverride = ProcessInfo.processInfo.environment["DEMUCS_MLX_SWIFT_MODEL_REPO"]
        let rawRepo = (repoOverride?.isEmpty == false) ? repoOverride! : defaultHubModelRepo
        let resolvedRepo = normalizeHubRepoID(rawRepo)

        do {
            return try downloadModelDirectoryFromHub(modelName: modelName, repoID: resolvedRepo)
        } catch {
            let searched = candidates.map(\.path).joined(separator: ", ")
            throw DemucsError.unsupportedModelBackend(
                "Could not find model directory for \(modelName). Searched local paths: [\(searched)]. "
                + "Also failed downloading from Hugging Face repo '\(resolvedRepo)': \(error.localizedDescription)"
            )
        }
    }

    /// Load and parse the model config JSON.
    static func loadConfig(from directory: URL, modelName: String) throws -> [String: Any] {
        let configURL = directory.appendingPathComponent("\(modelName)_config.json")
        let data = try Data(contentsOf: configURL)
        guard let raw = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw DemucsError.unsupportedModelBackend("Invalid model config JSON")
        }
        return raw
    }

    /// Load all weight arrays from safetensors file.
    static func loadWeights(from directory: URL, modelName: String) throws -> [String: MLXArray] {
        let weightsURL = directory.appendingPathComponent("\(modelName).safetensors")
        return try MLX.loadArrays(url: weightsURL)
    }

    /// Check if required model files exist in a directory.
    static func hasRequiredModelFiles(
        in directory: URL,
        modelName: String,
        fileManager: FileManager = .default
    ) -> Bool {
        let weights = directory.appendingPathComponent("\(modelName).safetensors")
        let config = directory.appendingPathComponent("\(modelName)_config.json")
        return fileManager.fileExists(atPath: weights.path)
            && fileManager.fileExists(atPath: config.path)
    }

    // MARK: - Hub Helpers

    private static func normalizeHubRepoID(_ raw: String) -> String {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.hasPrefix("https://") || trimmed.hasPrefix("http://") {
            if let url = URL(string: trimmed),
               let host = url.host?.lowercased(),
               host.contains("huggingface.co") {
                let parts = url.path.split(separator: "/").map(String.init)
                if parts.count >= 3 && parts[0] == "models" {
                    return "\(parts[1])/\(parts[2])"
                }
                if parts.count >= 2 {
                    return "\(parts[0])/\(parts[1])"
                }
            }
        }
        return trimmed.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
    }

    private static func downloadModelDirectoryFromHub(modelName: String, repoID: String) throws -> URL {
        let globs = [
            "\(modelName).safetensors",
            "\(modelName)_config.json",
            "README.md",
        ]
        let hub = HubApi()
        let downloaded = try runBlocking {
            try await hub.snapshot(from: repoID, matching: globs)
        }
        if hasRequiredModelFiles(in: downloaded, modelName: modelName) {
            return downloaded
        }

        let nested = downloaded.appendingPathComponent(modelName, isDirectory: true)
        if hasRequiredModelFiles(in: nested, modelName: modelName) {
            return nested
        }

        throw DemucsError.unsupportedModelBackend(
            "Downloaded repo '\(repoID)' but missing required files "
            + "'\(modelName).safetensors' and '\(modelName)_config.json'."
        )
    }

    // MARK: - Async Bridge

    private final class BlockingResultBox<Value>: @unchecked Sendable {
        private let lock = NSLock()
        private var result: Result<Value, Error>?

        init() {}

        func set(_ value: Result<Value, Error>) {
            lock.lock()
            result = value
            lock.unlock()
        }

        func get() -> Result<Value, Error>? {
            lock.lock()
            defer { lock.unlock() }
            return result
        }
    }

    private static func runBlocking<T>(_ operation: @escaping @Sendable () async throws -> T) throws -> T {
        let box = BlockingResultBox<T>()
        let semaphore = DispatchSemaphore(value: 0)

        Task.detached(priority: .userInitiated) {
            do {
                box.set(Result<T, Error>.success(try await operation()))
            } catch {
                box.set(Result<T, Error>.failure(error))
            }
            semaphore.signal()
        }

        semaphore.wait()
        guard let result = box.get() else {
            throw DemucsError.unsupportedModelBackend("Hub download failed before producing a result.")
        }
        return try result.get()
    }

    // MARK: - Config Parsing Helpers

    static func int(_ kwargs: [String: Any], _ key: String, _ fallback: Int) -> Int {
        if let v = kwargs[key] as? Int { return v }
        if let v = kwargs[key] as? Double { return Int(v) }
        if let v = kwargs[key] as? String, let i = Int(v) { return i }
        return fallback
    }

    static func bool(_ kwargs: [String: Any], _ key: String, _ fallback: Bool) -> Bool {
        if let v = kwargs[key] as? Bool { return v }
        if let v = kwargs[key] as? Int { return v != 0 }
        if let v = kwargs[key] as? Double { return v != 0 }
        if let v = kwargs[key] as? String {
            return ["1", "true", "yes"].contains(v.lowercased())
        }
        return fallback
    }

    static func double(_ kwargs: [String: Any], _ key: String, _ fallback: Double) -> Double {
        if let v = kwargs[key] as? Double { return v }
        if let v = kwargs[key] as? Int { return Double(v) }
        if let v = kwargs[key] as? String {
            if let d = Double(v) { return d }
            let parts = v.split(separator: "/")
            if parts.count == 2,
               let n = Double(parts[0]),
               let d = Double(parts[1]),
               d != 0 {
                return n / d
            }
        }
        return fallback
    }

    static func sources(_ kwargs: [String: Any]) -> [String] {
        (kwargs["sources"] as? [String]) ?? ["drums", "bass", "other", "vocals"]
    }
}
