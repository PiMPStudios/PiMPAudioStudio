import SwiftUI
import PiMPAudioCore

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
        let modelsDir = AppStorageLocations.modelsDirectory
        self.init(registry: ModelRegistry(baseDirectory: modelsDir))
    }

    convenience init(apiKeyStore: APIKeyStore) {
        let modelsDir = AppStorageLocations.modelsDirectory
        let tokenProvider: @Sendable () async -> String? = { [apiKeyStore] in
            await MainActor.run { apiKeyStore.huggingFaceToken() }
        }
        self.init(registry: ModelRegistry(
            baseDirectory: modelsDir,
            tokenProvider: tokenProvider
        ))
    }

    init(registry: ModelRegistry) {
        self.registry = registry
        Task { await checkSetupStatus() }
    }

    func checkSetupStatus() async {
        var allReady = true
        for id in ModelRegistry.essentialModelIDs {
            if await !registry.isModelReady(id) {
                allReady = false
                break
            }
        }
        isSetupComplete = allReady
    }

    /// Downloads only the essential ACE-Step models for first launch.
    func startDownload() async {
        currentError = nil
        do {
            try await registry.downloadEssentialModels { [weak self] modelID, progress in
                // DispatchQueue.main.async always defers to a future runloop tick,
                // which prevents @Published mutations from interleaving with
                // SwiftUI body evaluations ("Publishing changes from within view
                // updates" warning). Task { @MainActor … } can run on the
                // current tick if the actor is free.
                DispatchQueue.main.async { [weak self] in
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
                    // Don't downgrade a terminal background phase from a
                    // stale event delivered after the download finished.
                    switch self.backgroundDownloadPhase {
                    case .complete, .failed: return
                    default: break
                    }
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

        // Then download LLM if needed. Skip when the model is already on
        // disk OR a download is currently in progress. Previously this
        // checked `downloadState == .idle`, which both (a) failed to skip
        // the trigger when files were already on disk (post-init state is
        // .idle), and (b) skipped retries after a failure (state is .error
        // after failure, not .idle, so retry from MainShellView's onRetry
        // hit nothing).
        let defaultLLM = LLMModelManager.defaultModels.first
        let llmIsDownloading: Bool
        if case .downloading = llmManager.downloadState {
            llmIsDownloading = true
        } else {
            llmIsDownloading = false
        }
        if let spec = defaultLLM,
           !llmIsDownloading,
           !llmManager.isModelDownloaded(spec) {
            backgroundDownloadPhase = .downloading(modelName: "Lyric Writer", progress: 0)

            // Observe LLM progress and relay to background phase
            let llmObserver = Task { @MainActor [weak self] in
                for await state in llmManager.$downloadState.values {
                    guard let self else { return }
                    // Don't downgrade a terminal background phase from a
                    // stale event delivered after the download finished.
                    switch self.backgroundDownloadPhase {
                    case .complete, .failed: return
                    default: break
                    }
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
