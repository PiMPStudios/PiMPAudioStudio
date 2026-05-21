import SwiftUI
import Combine
import AnvilCore

@MainActor
class EditorService: ObservableObject {
    @Published var coordinator: InpaintCoordinator?
    @Published private(set) var loadedAudio: EditorLoadedAudio?

    private var sfxPipelineSubscription: AnyCancellable?

    func setup(
        vaeActor: VAEActor,
        diffusionActor: DiffusionActor,
        conditioningActor: ConditioningActor,
        sfxModelManager: SFXModelManager? = nil
    ) {
        self.coordinator = InpaintCoordinator(
            vaeActor: vaeActor,
            diffusionActor: diffusionActor,
            conditioningActor: conditioningActor
        )

        // Wire the SAO pipeline into the coordinator whenever SAO weights load.
        // `sfxModelManager.$pipeline` fires on the main actor; tokenizer is set
        // in the same `loadPipeline()` call so it's safe to read alongside.
        if let sfxMgr = sfxModelManager {
            sfxPipelineSubscription = sfxMgr.$pipeline
                .sink { [weak self] pipeline in
                    guard let self, let coord = self.coordinator else { return }
                    let tokenizer = sfxMgr.tokenizer
                    Task { await coord.setSFXPipeline(pipeline, tokenizer: tokenizer) }
                }
        }
    }

    func recordLoadedAudio(_ audio: EditorLoadedAudio) {
        loadedAudio = audio
    }
}

struct EditorLoadedAudio: Equatable {
    let source: String
    let historyEntryID: String?
    let fileName: String
    let path: String
    let duration: TimeInterval
    let sampleRate: Int
    let channels: Int
    let frameCount: Int
    let latentFrames: Int
    let originalPrompt: String?
    let loadedAt: Date
}
