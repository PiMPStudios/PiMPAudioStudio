import SwiftUI
import AppKit
import PiMPAudioCore

@main
struct PiMPAudioStudioApp: App {
    @StateObject private var appState: AppState
    @StateObject private var outputManager: OutputDirectoryManager
    @StateObject private var playerService: AudioPlayerService
    @StateObject private var services: AppServices
    @StateObject private var apiKeyStore: APIKeyStore
    @StateObject private var providerAccessStore: ProviderAccessStore
    @StateObject private var runPodService: RunPodService
    @StateObject private var modelManager: LLMModelManager
    @StateObject private var intentRouter: IntentRouter
    @StateObject private var promptEnhancer: PromptEnhancer
    @StateObject private var lyricWriterManager: LyricWriterManager
    @StateObject private var sfxModelManager: SFXModelManager
    @StateObject private var adapterStackManager = AdapterStackManager()
    @StateObject private var generationParameters = GenerationParameters()
    @StateObject private var presetStore = PresetStore()

    init() {
        AppRuntimeMigration.migrateIfNeeded()
        AudioDiagnostics.setLogDirectory(AppStorageLocations.diagnosticsDirectory)
        AudioDiagnostics.refreshFileLoggingState()

        let apiKeys = APIKeyStore()
        let providerAccess = ProviderAccessStore()
        let runPodService = RunPodService(apiKeyStore: apiKeys)
        let appState = AppState(apiKeyStore: apiKeys)
        let outputManager = OutputDirectoryManager()
        let playerService = AudioPlayerService()
        let manager = LLMModelManager()
        let router = IntentRouter(llm: manager.llmActor)
        let enhancer = PromptEnhancer(llm: manager.llmActor)
        let lyricManager = LyricWriterManager(
            apiKeyStore: apiKeys,
            providerAccessStore: providerAccess
        )
        let sfxManager = SFXModelManager(registry: appState.registry)
        let services = AppServices(
            sfxModelManager: sfxManager,
            registry: appState.registry,
            outputManager: outputManager,
            llmManager: manager,
            runPodService: runPodService,
            promptEnhancer: enhancer,
            lyricWriterManager: lyricManager
        )

        _appState = StateObject(wrappedValue: appState)
        _outputManager = StateObject(wrappedValue: outputManager)
        _playerService = StateObject(wrappedValue: playerService)
        _services = StateObject(wrappedValue: services)
        _apiKeyStore = StateObject(wrappedValue: apiKeys)
        _providerAccessStore = StateObject(wrappedValue: providerAccess)
        _runPodService = StateObject(wrappedValue: runPodService)
        _modelManager = StateObject(wrappedValue: manager)
        _intentRouter = StateObject(wrappedValue: router)
        _promptEnhancer = StateObject(wrappedValue: enhancer)
        _lyricWriterManager = StateObject(wrappedValue: lyricManager)
        _sfxModelManager = StateObject(wrappedValue: sfxManager)
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(appState)
                .environmentObject(outputManager)
                .environmentObject(playerService)
                .environmentObject(services.historyStore)
                .environmentObject(services.generationService)
                .environmentObject(apiKeyStore)
                .environmentObject(providerAccessStore)
                .environmentObject(runPodService)
                .environmentObject(modelManager)
                .environmentObject(intentRouter)
                .environmentObject(promptEnhancer)
                .environmentObject(lyricWriterManager)
                .environmentObject(sfxModelManager)
                .environmentObject(adapterStackManager)
                .environmentObject(generationParameters)
                .environmentObject(presetStore)
                .environmentObject(services.traceMonitor)
                .environmentObject(services.editorService)
                .environmentObject(services.trainingService)
                .task(id: outputManager.directory) {
                    services.configure(outputManager: outputManager)
                    services.trainingService.attachStackManager(adapterStackManager)
                    services.generationService.trainingService = services.trainingService
                    services.trainingService.generationService = services.generationService
                }
                .onReceive(modelManager.$llmActor) { llm in
                    intentRouter.updateLLM(llm)
                    promptEnhancer.updateLLM(llm)
                    lyricWriterManager.updateLLM(llm)
                }
                // Evict the LLM after prompt enhancement completes.
                // The enhanced text is already in the prompt field by the time
                // this fires (onReceive dispatches on the next run loop tick),
                // so the result is preserved. The LLM is no longer needed until
                // the user types a new prompt and clicks Enhance again.
                .onReceive(promptEnhancer.$isEnhancing) { isEnhancing in
                    guard !isEnhancing, modelManager.llmActor != nil else { return }
                    modelManager.evict()
                    modelManager.reload()
                }
                // Evict the LLM after lyric writing completes.
                // The lyrics are already streamed into the text field by the time
                // this fires. Memory returns to baseline before the user clicks
                // Generate, so DiT + VAE load into full headroom.
                .onReceive(lyricWriterManager.$isWriting) { isWriting in
                    guard !isWriting, modelManager.llmActor != nil else { return }
                    modelManager.evict()
                    modelManager.reload()
                }
                .onAppear {
                    intentRouter.updateLLM(modelManager.llmActor)
                    promptEnhancer.updateLLM(modelManager.llmActor)
                    lyricWriterManager.updateLLM(modelManager.llmActor)
                    lyricWriterManager.refreshRemoteBackends()
                }
                .onChange(of: apiKeyStore.availableProviders) { _, _ in
                    lyricWriterManager.refreshRemoteBackends()
                }
                .onChange(of: providerAccessStore.statuses) { _, _ in
                    lyricWriterManager.refreshRemoteBackends()
                }
                .onReceive(NotificationCenter.default.publisher(
                    for: NSApplication.willTerminateNotification
                )) { _ in
                    services.prepareForTermination()
                }
        }
        .windowStyle(.titleBar)
        .defaultSize(width: 1200, height: 800)
        .commands {
            HelpCommands()
        }

        WindowGroup("Generation Analyzer", id: "generationAnalyzer") {
            GenerationAnalyzerView()
                .environmentObject(services.traceMonitor)
                .environmentObject(services.historyStore)
                .environmentObject(playerService)
        }
        .defaultSize(width: 860, height: 640)

        WindowGroup("PiMP Audio Studio Help", id: "documentation") {
            DocumentationView()
        }
        .defaultSize(width: 960, height: 700)

        Settings {
            TabView {
                IntelligenceSettingsView()
                    .tabItem { Label("Intelligence", systemImage: "brain") }
                    .environmentObject(modelManager)
                    .environmentObject(apiKeyStore)
                    .environmentObject(providerAccessStore)
                    .environmentObject(lyricWriterManager)

                ModelsSettingsView()
                    .tabItem { Label("Models", systemImage: "cube.box") }
                    .environmentObject(appState)
                    .environmentObject(apiKeyStore)
                    // SwiftUI's `Settings` scene is a separate scene from
                    // `WindowGroup`, so environment objects injected on
                    // ContentView do NOT propagate here. The Release GPU
                    // Memory button accesses `generationService`; without
                    // explicit injection here,
                    // tapping it fatal-errors with "No ObservableObject of
                    // type GenerationService found."
                    .environmentObject(services.generationService)
                    .environmentObject(sfxModelManager)

                MCPSettingsView()
                    .tabItem { Label("MCP", systemImage: "point.3.connected.trianglepath.dotted") }
                    .environmentObject(services.mcpServer)

                AdaptersSettingsView()
                    .tabItem { Label("Adapters", systemImage: "puzzlepiece.extension") }
                    .environmentObject(services.generationService)
                    .environmentObject(services.traceMonitor)
                    .environmentObject(services.trainingService)
                    .environmentObject(services.historyStore)
                    .environmentObject(presetStore)
            }
            .frame(minWidth: 550, minHeight: 450)
        }
    }
}

@MainActor
final class AppServices: ObservableObject {
    @Published var historyStore: HistoryStore
    @Published var generationService: GenerationService
    let traceMonitor: GenerationTraceMonitor
    let mcpServer: PiMPAudioMCPServer

    /// Shared ML actors — created once, reused by both generation and the editor
    /// so weights are never duplicated in GPU memory.
    let editorService: EditorService

    /// LoRA training service — owns the training task and publishes progress.
    /// Created once and injected into every scene so History, Build, and the
    /// Adapters settings tab share the same running state.
    let trainingService: TrainingService

    private var currentOutputManager: OutputDirectoryManager?
    private var currentDirectory: URL?
    private let sfxModelManager: SFXModelManager
    private let registry: ModelRegistry
    private let llmManager: LLMModelManager
    private let vaeActor: VAEActor
    private let diffusionActor: DiffusionActor
    private let conditioningActor: ConditioningActor
    private let commandFacade: PiMPAudioCommandFacade

    init(
        sfxModelManager: SFXModelManager,
        registry: ModelRegistry,
        outputManager: OutputDirectoryManager,
        llmManager: LLMModelManager,
        runPodService: RunPodService,
        promptEnhancer: PromptEnhancer,
        lyricWriterManager: LyricWriterManager
    ) {
        self.sfxModelManager = sfxModelManager
        self.registry = registry
        self.llmManager = llmManager
        self.currentOutputManager = outputManager
        self.currentDirectory = outputManager.directory
        self.traceMonitor = GenerationTraceMonitor()

        // Create shared ML actors once. Weights are loaded lazily on first use,
        // so this is cheap at launch. The editor and generation service both hold
        // references to the same actor instances — no duplicated GPU memory.
        let vae = VAEActor()
        let dit = DiTActor(config: .v15Base)
        let diffusion = DiffusionActor(dit: dit)
        let conditioning: ConditioningActor
        do {
            conditioning = try ConditioningActor.stub()
        } catch {
            fatalError("Failed to create stub ConditioningActor at launch: \(error)")
        }
        self.vaeActor = vae
        self.diffusionActor = diffusion
        self.conditioningActor = conditioning

        // Wire the editor service with the full pipeline (VAE + DiT diffusion +
        // conditioning + SAO pipeline). The SAO pipeline is nil until weights
        // are downloaded; EditorService's Combine subscription wires it in
        // automatically when SFXModelManager.pipeline becomes non-nil.
        let editorSvc = EditorService()
        editorSvc.setup(
            vaeActor: vae,
            diffusionActor: diffusion,
            conditioningActor: conditioning,
            sfxModelManager: sfxModelManager
        )
        self.editorService = editorSvc

        self.trainingService = TrainingService(
            pipeline: MusicPipeline(vae: vae, diffusion: diffusion),
            vae: vae,
            conditioning: conditioning,
            registry: registry
        )

        let dir = outputManager.directory ?? FileManager.default.temporaryDirectory
        let store = HistoryStore(directory: dir)
        self.historyStore = store
        let facade = PiMPAudioCommandFacade(
            registry: registry,
            historyStore: store,
            promptEnhancer: promptEnhancer,
            lyricWriterManager: lyricWriterManager
        )
        facade.configure(editorService: editorSvc)
        facade.configure(trainingService: self.trainingService)
        facade.configure(runPodService: runPodService)
        self.commandFacade = facade
        self.mcpServer = PiMPAudioMCPServer(facade: facade)
        let generationService = Self.makeGenerationService(
            vaeActor: vae,
            diffusionActor: diffusion,
            conditioningActor: conditioning,
            historyStore: store,
            outputManager: outputManager,
            sfxModelManager: sfxModelManager,
            registry: registry,
            llmManager: llmManager,
            trainingService: self.trainingService,
            editorService: editorSvc,
            traceMonitor: traceMonitor
        )
        generationService.trainingService = self.trainingService
        self.trainingService.generationService = generationService
        self.generationService = generationService
        facade.configure(generationService: generationService)
    }

    /// Rebuilds `historyStore` and `generationService` when the output directory
    /// changes. SwiftUI's `.task(id: outputManager.directory)` fires this on the
    /// initial appear (no-op since `init` already wired the real manager) and
    /// again whenever the user picks a new directory.
    func configure(outputManager: OutputDirectoryManager) {
        if currentOutputManager === outputManager && currentDirectory == outputManager.directory {
            return
        }
        currentOutputManager = outputManager
        currentDirectory = outputManager.directory

        let dir = outputManager.directory ?? FileManager.default.temporaryDirectory
        let store = HistoryStore(directory: dir)
        self.historyStore = store
        commandFacade.configure(historyStore: store)
        let generationService = Self.makeGenerationService(
            vaeActor: vaeActor,
            diffusionActor: diffusionActor,
            conditioningActor: conditioningActor,
            historyStore: store,
            outputManager: outputManager,
            sfxModelManager: sfxModelManager,
            registry: registry,
            llmManager: llmManager,
            trainingService: self.trainingService,
            editorService: self.editorService,
            traceMonitor: traceMonitor
        )
        generationService.trainingService = self.trainingService
        self.trainingService.generationService = generationService
        self.generationService = generationService
        commandFacade.configure(generationService: generationService)
    }

    func prepareForTermination() {
        mcpServer.stop()
    }

    private static func makeGenerationService(
        vaeActor: VAEActor,
        diffusionActor: DiffusionActor,
        conditioningActor: ConditioningActor,
        historyStore: HistoryStore,
        outputManager: OutputDirectoryManager,
        sfxModelManager: SFXModelManager,
        registry: ModelRegistry,
        llmManager: LLMModelManager,
        trainingService: TrainingService? = nil,
        editorService: EditorService? = nil,
        traceMonitor: GenerationTraceMonitor
    ) -> GenerationService {
        let pipeline = MusicPipeline(vae: vaeActor, diffusion: diffusionActor)
        let converter = WeightConverter()

        // Propagate the upgraded ConditioningActor (real tokenizer + weights) to
        // TrainingService, CoverSong, and the editor so they don't keep the
        // launch-time stub after the first ACE-Step model load.
        let sfx = SFXGenerationService(sfxModelManager: sfxModelManager)
        let lora = LoRACoordinator(pipeline: pipeline, registry: registry)
        let coverSong = CoverSongCoordinator(
            pipeline: pipeline,
            conditioning: conditioningActor,
            weightConverter: converter,
            registry: registry
        )

        let conditioningUpgradeHandler: @Sendable (ConditioningActor) async -> Void = { newConditioning in
            if let trainingService {
                await trainingService.updateConditioning(newConditioning)
            }
            await coverSong.updateConditioning(newConditioning)
            if let editorService {
                await MainActor.run {
                    editorService.updateConditioning(newConditioning)
                }
            }
        }

        let aceStep = ACEStepGenerationService(
            pipeline: pipeline,
            conditioning: conditioningActor,
            weightConverter: converter,
            registry: registry,
            onConditioningUpgraded: conditioningUpgradeHandler
        )
        trainingService?.setModelLoader {
            llmManager.evict()
            try await aceStep.ensureLoaded()
        }
        trainingService?.setTrainingFinishedHandler {
            llmManager.reload()
        }

        return GenerationService(
            aceStepService: aceStep,
            sfxService: sfx,
            sfxModelManager: sfxModelManager,
            loraCoordinator: lora,
            coverSongCoordinator: coverSong,
            historyStore: historyStore,
            outputDirectoryManager: outputManager,
            llmManager: llmManager,
            registry: registry,
            traceMonitor: traceMonitor
        )
    }
}

struct ContentView: View {
    @EnvironmentObject private var appState: AppState

    var body: some View {
        if appState.isSetupComplete {
            MainShellView()
        } else {
            FirstLaunchView()
        }
    }
}
