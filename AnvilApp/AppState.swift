import SwiftUI
import AnvilCore

public enum BackgroundDownloadPhase: Equatable {
    case idle
    case downloading(modelName: String, progress: Float)
    case complete
    case failed(String)
}

@MainActor
final class AppState: ObservableObject {
    @Published var isSetupComplete: Bool = false
    @Published var downloadProgress: [String: Double] = [:]
    @Published var currentError: String?
    @Published var backgroundDownloadPhase: BackgroundDownloadPhase = .idle

    let registry: ModelRegistry

    convenience init() {
        let modelsDir = FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        )[0].appending(path: "Anvil/models")
        self.init(registry: ModelRegistry(baseDirectory: modelsDir))
    }

    init(registry: ModelRegistry) {
        self.registry = registry
        Task { await checkSetupStatus() }
    }

    func checkSetupStatus() async {
        let baseDir = await registry.baseDirectory
        let allEssentialsReady = ModelRegistry.builtInSpecs
            .filter { ModelRegistry.essentialModelIDs.contains($0.id) }
            .allSatisfy { spec in
                spec.files.allSatisfy { file in
                    FileManager.default.fileExists(
                        atPath: spec.localPath(for: file, base: baseDir).path()
                    )
                }
            }
        isSetupComplete = allEssentialsReady
    }

    /// Downloads only the essential ACE-Step models for first launch.
    func startDownload() async {
        currentError = nil
        do {
            try await registry.downloadEssentialModels { [weak self] modelID, progress in
                Task { @MainActor [weak self] in
                    self?.downloadProgress[modelID] = progress
                }
            }
            isSetupComplete = true
        } catch {
            currentError = error.localizedDescription
        }
    }

    /// Kicks off background downloads for non-essential models (SFX, LLM).
    /// Called after first-launch completes or on normal app launch.
    func startBackgroundDownloads(
        sfxManager: SFXModelManager,
        llmManager: LLMModelManager
    ) async {
        // Download Stable Audio first (more immediately useful than LLM)
        if sfxManager.downloadState == .notDownloaded {
            backgroundDownloadPhase = .downloading(modelName: "Stable Audio Open", progress: 0)
            do {
                try await sfxManager.downloadIfNeeded()
            } catch {
                backgroundDownloadPhase = .failed("Stable Audio download failed. Tap to retry.")
                return
            }
        }

        // Then download LLM if needed
        let defaultLLM = LLMModelManager.defaultModels.first
        if let spec = defaultLLM, llmManager.downloadState == .idle {
            backgroundDownloadPhase = .downloading(modelName: "Lyric Writer", progress: 0)
            do {
                try await llmManager.downloadModel(spec)
            } catch {
                backgroundDownloadPhase = .failed("Lyric Writer download failed. Tap to retry.")
                return
            }
        }

        backgroundDownloadPhase = .complete
    }
}
