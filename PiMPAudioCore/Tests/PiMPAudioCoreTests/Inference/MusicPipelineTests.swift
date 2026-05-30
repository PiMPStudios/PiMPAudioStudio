import Testing
import MLX
@testable import PiMPAudioCore

@Suite("MusicPipeline")
struct MusicPipelineTests {
    private func makeSmallPipeline() -> MusicPipeline {
        let ditDecoder = DiTDecoder(
            hiddenSize: 256,
            encoderHiddenSize: 256,
            intermediateSize: 512,
            numHiddenLayers: 2,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            headDim: 64,
            rmsNormEps: 1e-6,
            attentionBias: false,
            inChannels: 192,
            audioAcousticHiddenDim: 64,
            patchSize: 2,
            slidingWindow: 128,
            layerTypes: ["sliding_attention", "full_attention"],
            ropeTheta: 1_000_000.0,
            maxPositionEmbeddings: 32768
        )
        let ditActor = DiTActor(decoder: ditDecoder)
        let vaeActor = VAEActor()
        let diffusionActor = DiffusionActor(dit: ditActor)
        return MusicPipeline(vae: vaeActor, diffusion: diffusionActor)
    }

    @Test("Initial state is ready when constructed with actors")
    func initialState() async {
        let pipeline = makeSmallPipeline()
        let state = await pipeline.state
        #expect(state == .ready)
    }

    @Test("Generate returns audio with correct sample rate")
    func generateAudio() async throws {
        let pipeline = makeSmallPipeline()
        let duration = 0.5
        let frames = MusicPipeline.latentFrames(forDuration: duration)
        let encoderStates = MLXArray.zeros([1, 20, 256])
        let contextLatents = MLXArray.zeros([1, frames, 128])
        let config = DiffusionConfig(seed: 42, steps: 2)

        let result = try await pipeline.generate(
            encoderStates: encoderStates,
            contextLatents: contextLatents,
            encoderMask: nil,
            duration: duration,
            config: config
        )
        #expect(result.sampleRate == MusicPipeline.defaultSampleRate)
        #expect(result.samples.shape[0] == 1)
        #expect(result.samples.shape[2] == 2)
        #expect(result.timeCosts.totalTime > 0)
    }

    @Test("Duration maps to correct number of latent frames")
    func durationToFrames() {
        let frames = MusicPipeline.latentFrames(forDuration: 1.0, sampleRate: 44100, hopLength: 2048)
        #expect(frames == 22)
    }

    @Test("Duration 4 minutes")
    func fourMinutes() {
        let frames = MusicPipeline.latentFrames(forDuration: 240.0, sampleRate: 44100, hopLength: 2048)
        #expect(frames >= 5160 && frames <= 5180)
    }
}
