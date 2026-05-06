import SwiftUI
import AnvilCore

enum BackgroundDownloadPhase: Equatable {
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
        let allEssentialsReady = await ModelRegistry.essentialModelIDs.allSatisfy { id in
            await registry.isModelReady(id)
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

            // Observe SFX progress and relay to background phase
            let sfxObserver = Task { @MainActor [weak self] in
                for await state in sfxManager.$downloadState.values {
                    guard let self else { return }
                    if case .downloading(let progress) = state {
                        self.backgroundDownloadPhase = .downloading(
                            modelName: "Stable Audio Open", progress: progress
                        )
                    }
                }
            }

            do {
                try await sfxManager.downloadIfNeeded()
            } catch {
                sfxObserver.cancel()
                backgroundDownloadPhase = .failed("Stable Audio download failed. Click to retry.")
                return
            }
            sfxObserver.cancel()
        }

        // Then download LLM if needed
        let defaultLLM = LLMModelManager.defaultModels.first
        if let spec = defaultLLM, llmManager.downloadState == .idle {
            backgroundDownloadPhase = .downloading(modelName: "Lyric Writer", progress: 0)

            // Observe LLM progress and relay to background phase
            let llmObserver = Task { @MainActor [weak self] in
                for await state in llmManager.$downloadState.values {
                    guard let self else { return }
                    if case .downloading(let progress) = state {
                        self.backgroundDownloadPhase = .downloading(
                            modelName: "Lyric Writer", progress: progress
                        )
                    }
                }
            }

            do {
                try await llmManager.downloadModel(spec)
            } catch {
                llmObserver.cancel()
                backgroundDownloadPhase = .failed("Lyric Writer download failed. Click to retry.")
                return
            }
            llmObserver.cancel()
        }

        backgroundDownloadPhase = .complete
    }
}
