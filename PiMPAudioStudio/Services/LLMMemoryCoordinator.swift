import Combine
import Foundation

/// Evicts the on-device lyric/enhance LLM after a write finishes so DiT/VAE
/// get full unified-memory headroom. Lives on AppServices — not in the scene body.
@MainActor
final class LLMMemoryCoordinator {
    private var cancellables = Set<AnyCancellable>()

    func start(
        promptEnhancer: PromptEnhancer,
        lyricWriterManager: LyricWriterManager,
        modelManager: LLMModelManager
    ) {
        cancellables.removeAll()

        promptEnhancer.$isEnhancing
            .dropFirst()
            .sink { [weak modelManager] isEnhancing in
                guard !isEnhancing, modelManager?.llmActor != nil else { return }
                modelManager?.evict()
                modelManager?.reload()
            }
            .store(in: &cancellables)

        lyricWriterManager.$isWriting
            .dropFirst()
            .sink { [weak modelManager] isWriting in
                guard !isWriting, modelManager?.llmActor != nil else { return }
                modelManager?.evict()
                modelManager?.reload()
            }
            .store(in: &cancellables)
    }
}
