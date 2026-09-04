import Foundation
import PiMPAudioCore
@testable import PiMPAudioStudio
import Testing

@Suite(.serialized)
@MainActor
struct GenerationServiceTests {
    @Test
    func rejectedGenerationPreservesReleaseGate() throws {
        let generation = try makeGenerationService()
        generation.state = .releasing
        generation.generate(request: GenerationRequest(text: "Review fixture"))

        #expect(generation.state == .releasing)
        #expect(!generation.canSubmitGeneration)
    }
}

@MainActor
private func makeGenerationService() throws -> GenerationService {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let registry = ModelRegistry(baseDirectory: directory)
    let vae = VAEActor()
    let dit = DiTActor(decoder: DiTDecoder(
        hiddenSize: 64,
        encoderHiddenSize: 64,
        intermediateSize: 128,
        numHiddenLayers: 1,
        numAttentionHeads: 2,
        numKeyValueHeads: 1,
        headDim: 32,
        rmsNormEps: 1e-6,
        attentionBias: false,
        inChannels: 192,
        audioAcousticHiddenDim: 64,
        patchSize: 2,
        slidingWindow: 128,
        layerTypes: ["full_attention"],
        ropeTheta: 1_000_000.0,
        maxPositionEmbeddings: 128
    ))
    let diffusion = DiffusionActor(dit: dit)
    let pipeline = MusicPipeline(vae: vae, diffusion: diffusion)
    let conditioning = ConditioningActor.stub()
    let converter = WeightConverter()
    let sfx = SFXModelManager(registry: registry)
    return GenerationService(
        aceStepService: ACEStepGenerationService(
            pipeline: pipeline,
            conditioning: conditioning,
            weightConverter: converter,
            registry: registry
        ),
        sfxService: SFXGenerationService(sfxModelManager: sfx),
        sfxModelManager: sfx,
        loraCoordinator: LoRACoordinator(pipeline: pipeline, registry: registry),
        coverSongCoordinator: CoverSongCoordinator(
            pipeline: pipeline,
            conditioning: conditioning,
            weightConverter: converter,
            registry: registry
        ),
        historyStore: HistoryStore(directory: directory),
        outputDirectoryManager: OutputDirectoryManager(),
        llmManager: LLMModelManager(),
        registry: registry,
        traceMonitor: GenerationTraceMonitor()
    )
}
