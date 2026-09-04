import Combine
import PiMPAudioCore
import SwiftUI

@MainActor
class EditorService: ObservableObject {
    @Published var coordinator: InpaintCoordinator?
    @Published private(set) var loadedAudio: EditorLoadedAudio?
    @Published private(set) var lastEdit: EditorEditSummary?

    private var sfxPipelineSubscription: AnyCancellable?

    func setup(
        vaeActor: VAEActor,
        diffusionActor: DiffusionActor,
        conditioningActor: ConditioningActor,
        sfxModelManager: SFXModelManager? = nil
    ) {
        coordinator = InpaintCoordinator(
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
                    guard let self, let coord = coordinator else { return }
                    let tokenizer = pipeline == nil ? nil : sfxMgr.tokenizer
                    let vae = pipeline == nil ? nil : sfxMgr.sfxVAE
                    let diffusion = pipeline == nil ? nil : sfxMgr.sfxDiffusion
                    Task {
                        await coord.setSFXPipeline(pipeline, tokenizer: tokenizer)
                        await coord.setSFXActors(vae: vae, diffusion: diffusion)
                    }
                }
        }
    }

    /// Propagate the upgraded ACE-Step ConditioningActor (real tokenizer +
    /// weights) into the inpaint coordinator so AI edits don't keep using the
    /// launch-time stub.
    func updateConditioning(_ actor: ConditioningActor) {
        guard let coordinator else { return }
        Task { await coordinator.setConditioningActor(actor) }
    }

    func recordLoadedAudio(_ audio: EditorLoadedAudio) {
        loadedAudio = audio
        lastEdit = nil
    }

    func recordCompletedEdit(_ edit: EditorEditSummary) {
        lastEdit = edit
    }

    func recordCompletedEdit(_ edit: EditorEditSummary, loadedAudio updatedAudio: EditorLoadedAudio) {
        loadedAudio = updatedAudio
        lastEdit = edit
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

    func updating(
        duration: TimeInterval,
        sampleRate: Int,
        channels: Int,
        frameCount: Int,
        latentFrames: Int
    ) -> EditorLoadedAudio {
        EditorLoadedAudio(
            source: source,
            historyEntryID: historyEntryID,
            fileName: fileName,
            path: path,
            duration: duration,
            sampleRate: sampleRate,
            channels: channels,
            frameCount: frameCount,
            latentFrames: latentFrames,
            originalPrompt: originalPrompt,
            loadedAt: loadedAt
        )
    }
}

struct EditorEditSummary: Equatable {
    let kind: String
    let mode: String?
    let regionPrompt: String?
    let originalPrompt: String?
    let pipeline: String?
    let selectionStart: TimeInterval?
    let selectionEnd: TimeInterval?
    let lowCutHz: Float?
    let highCutHz: Float?
    let strength: Float?
    let guidanceScale: Float?
    let seed: Int?
    let dspEffects: [EditorDSPEffectSummary]
    let beforeDuration: TimeInterval?
    let afterDuration: TimeInterval?
    let completedAt: Date

    init(
        mode: String,
        regionPrompt: String,
        originalPrompt: String?,
        pipeline: String,
        selectionStart: TimeInterval,
        selectionEnd: TimeInterval,
        lowCutHz: Float?,
        highCutHz: Float?,
        strength: Float,
        guidanceScale: Float,
        seed: Int?,
        completedAt: Date
    ) {
        kind = "inpaint"
        self.mode = mode
        self.regionPrompt = regionPrompt
        self.originalPrompt = originalPrompt
        self.pipeline = pipeline
        self.selectionStart = selectionStart
        self.selectionEnd = selectionEnd
        self.lowCutHz = lowCutHz
        self.highCutHz = highCutHz
        self.strength = strength
        self.guidanceScale = guidanceScale
        self.seed = seed
        dspEffects = []
        beforeDuration = nil
        afterDuration = nil
        self.completedAt = completedAt
    }

    init(
        dspEffects: [EditorDSPEffectSummary],
        beforeDuration: TimeInterval,
        afterDuration: TimeInterval,
        completedAt: Date
    ) {
        kind = "dsp"
        mode = nil
        regionPrompt = nil
        originalPrompt = nil
        pipeline = nil
        selectionStart = nil
        selectionEnd = nil
        lowCutHz = nil
        highCutHz = nil
        strength = nil
        guidanceScale = nil
        seed = nil
        self.dspEffects = dspEffects
        self.beforeDuration = beforeDuration
        self.afterDuration = afterDuration
        self.completedAt = completedAt
    }
}

struct EditorDSPEffectSummary: Equatable {
    let effect: String
    let displayName: String
    let description: String
    let target: String
    let selectionStart: TimeInterval?
    let selectionEnd: TimeInterval?
    let parameters: [String: Double]
}
