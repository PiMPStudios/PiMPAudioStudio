import Testing
import MLX
@testable import PiMPAudioCore

// MARK: - InpaintCoordinator Tests

@Suite("InpaintCoordinator")
struct InpaintCoordinatorTests {

    // MARK: - Helpers

    private func makeCoordinator() -> InpaintCoordinator {
        InpaintCoordinator(vaeActor: VAEActor())
    }

    // MARK: 1 — hop size

    @Test("hopSize returns aceStepHopSize for .aceStep and .auto")
    func hopSizeAceStep() {
        #expect(InpaintCoordinator.hopSize(for: .aceStep) == InpaintCoordinator.aceStepHopSize)
        #expect(InpaintCoordinator.hopSize(for: .auto) == InpaintCoordinator.aceStepHopSize)
    }

    @Test("hopSize returns sfxHopSize for .stableAudio")
    func hopSizeStableAudio() {
        #expect(InpaintCoordinator.hopSize(for: .stableAudio) == InpaintCoordinator.sfxHopSize)
    }

    @Test("hopSize constants match pipeline VAE strides")
    func hopSizeConstants() {
        #expect(InpaintCoordinator.aceStepHopSize == MusicPipeline.hopLength)
        #expect(InpaintCoordinator.aceStepHopSize == 1920)
        #expect(InpaintCoordinator.sfxHopSize == SoundFXPipeline.hopLength)
        #expect(InpaintCoordinator.sfxHopSize == 2048)
    }

    // MARK: 2 — initial state

    @Test("hasLoadedAudio is false before loadAudio")
    func initialHasLoadedAudio() async {
        let coordinator = makeCoordinator()
        let hasLoaded = await coordinator.hasLoadedAudio
        #expect(hasLoaded == false)
    }

    @Test("canUndo and canRedo are false before loadAudio")
    func initialUndoRedoState() async {
        let coordinator = makeCoordinator()
        let canUndo = await coordinator.canUndo
        let canRedo = await coordinator.canRedo
        #expect(canUndo == false)
        #expect(canRedo == false)
    }

    @Test("getCurrentAudio returns nil before loadAudio")
    func initialCurrentAudio() async {
        let coordinator = makeCoordinator()
        let audio = await coordinator.getCurrentAudio()
        #expect(audio == nil)
    }

    // MARK: 3 — loadAudio stores state and returns correct metadata

    @Test("loadAudio returns correct duration and sampleRate")
    func loadAudioReturnsCorrectMetadata() async throws {
        let coordinator = makeCoordinator()
        let sampleRate = 44100
        let numSamples = sampleRate * 4  // 4 seconds
        let samples = MLXArray.zeros([1, numSamples, 2])
        let result = try await coordinator.loadAudio(samples: samples, sampleRate: sampleRate)
        #expect(result.sampleRate == sampleRate)
        #expect(abs(result.duration - 4.0) < 0.001)
    }

    @Test("loadAudio sets hasLoadedAudio to true")
    func loadAudioSetsHasLoadedAudio() async throws {
        let coordinator = makeCoordinator()
        let samples = MLXArray.zeros([1, 44100, 2])
        _ = try await coordinator.loadAudio(samples: samples, sampleRate: 44100)
        let hasLoaded = await coordinator.hasLoadedAudio
        #expect(hasLoaded == true)
    }

    @Test("loadAudio makes getCurrentAudio return non-nil")
    func loadAudioStoresAudio() async throws {
        let coordinator = makeCoordinator()
        let samples = MLXArray.zeros([1, 44100, 2])
        _ = try await coordinator.loadAudio(samples: samples, sampleRate: 44100)
        let audio = await coordinator.getCurrentAudio()
        #expect(audio != nil)
    }

    // MARK: 4 — canUndo/canRedo after loadAudio

    @Test("canUndo is false and canRedo is false right after loadAudio (single snapshot)")
    func undoRedoAfterLoad() async throws {
        let coordinator = makeCoordinator()
        let samples = MLXArray.zeros([1, 44100, 2])
        _ = try await coordinator.loadAudio(samples: samples, sampleRate: 44100)
        let canUndo = await coordinator.canUndo
        let canRedo = await coordinator.canRedo
        #expect(canUndo == false)
        #expect(canRedo == false)
    }

    @Test("recentEdits has one entry after loadAudio")
    func recentEditsAfterLoad() async throws {
        let coordinator = makeCoordinator()
        let samples = MLXArray.zeros([1, 44100, 2])
        _ = try await coordinator.loadAudio(samples: samples, sampleRate: 44100)
        let edits = await coordinator.recentEdits
        #expect(edits.count == 1)
        #expect(edits[0].description == "Original")
    }

    // MARK: 5 — deferred encode (no VAE on load)

    @Test("loadAudio defers VAE encode and reports zero latentFrames")
    func loadAudioDefersVAEEncode() async throws {
        let coordinator = makeCoordinator()
        let samples = MLXArray.zeros([1, 44100 * 2, 2])  // 2 seconds
        let result = try await coordinator.loadAudio(samples: samples, sampleRate: 44100)
        // Full-track encode is deferred until first AI edit so Open/History→Edit
        // does not park multi-tens-of-GB of MLX GPU cache.
        #expect(result.latentFrames == 0)
        let hasLoaded = await coordinator.hasLoadedAudio
        #expect(hasLoaded == true)
    }

    @Test("replaceAudio invalidates latents without re-encoding")
    func replaceAudioDoesNotEncode() async throws {
        let coordinator = makeCoordinator()
        let samples = MLXArray.zeros([1, 44100, 2])
        _ = try await coordinator.loadAudio(samples: samples, sampleRate: 44100)
        let replaced = MLXArray.zeros([1, 44100 * 2, 2])
        let result = try await coordinator.replaceAudio(
            samples: replaced,
            sampleRate: 44100,
            description: "Normalize"
        )
        #expect(result.latentFrames == 0)
        #expect(abs(result.duration - 2.0) < 0.001)
        let audio = await coordinator.getCurrentAudio()
        #expect(audio != nil)
        #expect(audio?.shape[1] == 44100 * 2)
    }
}
