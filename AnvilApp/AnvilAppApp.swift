import SwiftUI
import AnvilCore

@main
struct AnvilAppApp: App {
    @StateObject private var appState: AppState
    @StateObject private var outputManager: OutputDirectoryManager
    @StateObject private var playerService: AudioPlayerService
    @StateObject private var services: AppServices
    @StateObject private var apiKeyStore: APIKeyStore
    @StateObject private var modelManager: LLMModelManager
    @StateObject private var intentRouter: IntentRouter
    @StateObject private var promptEnhancer: PromptEnhancer
    @StateObject private var lyricWriterManager: LyricWriterManager
    @StateObject private var sfxModelManager: SFXModelManager
    @StateObject private var adapterStackManager = AdapterStackManager()
    @StateObject private var generationParameters = GenerationParameters()

    init() {
        let apiKeys = APIKeyStore()
        let appState = AppState(apiKeyStore: apiKeys)
        let outputManager = OutputDirectoryManager()
        let playerService = AudioPlayerService()
        let manager = LLMModelManager()
        let router = IntentRouter(llm: manager.llmActor)
        let enhancer = PromptEnhancer(llm: manager.llmActor)
        let lyricManager = LyricWriterManager(apiKeyStore: apiKeys)
        let sfxManager = SFXModelManager(registry: appState.registry)
        let services = AppServices(
            sfxModelManager: sfxManager,
            registry: appState.registry,
            outputManager: outputManager
        )

        _appState = StateObject(wrappedValue: appState)
        _outputManager = StateObject(wrappedValue: outputManager)
        _playerService = StateObject(wrappedValue: playerService)
        _services = StateObject(wrappedValue: services)
        _apiKeyStore = StateObject(wrappedValue: apiKeys)
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
                .environmentObject(modelManager)
                .environmentObject(intentRouter)
                .environmentObject(promptEnhancer)
                .environmentObject(lyricWriterManager)
                .environmentObject(sfxModelManager)
                .environmentObject(adapterStackManager)
                .environmentObject(generationParameters)
                .environmentObject(services.trainingService)
                .task(id: outputManager.directory) {
                    services.configure(outputManager: outputManager)
                }
                .onReceive(modelManager.$llmActor) { llm in
                    intentRouter.updateLLM(llm)
                    promptEnhancer.updateLLM(llm)
                    lyricWriterManager.updateLLM(llm)
                }
                .onAppear {
                    intentRouter.updateLLM(modelManager.llmActor)
                    promptEnhancer.updateLLM(modelManager.llmActor)
                    lyricWriterManager.updateLLM(modelManager.llmActor)
                    lyricWriterManager.refreshAPIBackends()
                }
                .onChange(of: apiKeyStore.availableProviders) { _, _ in
                    lyricWriterManager.refreshAPIBackends()
                }
        }
        .windowStyle(.titleBar)
        .defaultSize(width: 1200, height: 800)

        Settings {
            TabView {
                IntelligenceSettingsView()
                    .tabItem { Label("Intelligence", systemImage: "brain") }
                    .environmentObject(modelManager)
                    .environmentObject(apiKeyStore)
                    .environmentObject(lyricWriterManager)

                ModelsSettingsView()
                    .tabItem { Label("Models", systemImage: "cube.box") }
                    .environmentObject(appState)
                    .environmentObject(apiKeyStore)
                    // SwiftUI's `Settings` scene is a separate scene from
                    // `WindowGroup`, so environment objects injected on
                    // ContentView do NOT propagate here. The Release GPU
                    // Memory button accesses `generationService` and
                    // `sfxModelManager` — without explicit injection here,
                    // tapping it fatal-errors with "No ObservableObject of
                    // type GenerationService found."
                    .environmentObject(services.generationService)
                    .environmentObject(sfxModelManager)

                AdaptersSettingsView()
                    .tabItem { Label("Adapters", systemImage: "puzzlepiece.extension") }
                    .environmentObject(outputManager)
                    .environmentObject(services.historyStore)
                    .environmentObject(services.trainingService)
            }
            .frame(minWidth: 550, minHeight: 450)
        }
    }
}

@MainActor
final class AppServices: ObservableObject {
    @Published var historyStore: HistoryStore
    @Published var generationService: GenerationService

    let trainingService: TrainingService

    private var currentOutputManager: OutputDirectoryManager?
    private var currentDirectory: URL?
    private let sfxModelManager: SFXModelManager
    private let registry: ModelRegistry
    private let vaeActor: VAEActor
    private let diffusionActor: DiffusionActor
    private let conditioningActor: ConditioningActor

    init(
        sfxModelManager: SFXModelManager,
        registry: ModelRegistry,
        outputManager: OutputDirectoryManager
    ) {
        self.sfxModelManager = sfxModelManager
        self.registry = registry
        self.currentOutputManager = outputManager
        self.currentDirectory = outputManager.directory

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

        self.trainingService = TrainingService(
            pipeline: MusicPipeline(vae: vae, diffusion: diffusion),
            vae: vae,
            conditioning: conditioning,
            registry: registry
        )

        let dir = outputManager.directory ?? FileManager.default.temporaryDirectory
        let store = HistoryStore(directory: dir)
        self.historyStore = store
        self.generationService = Self.makeGenerationService(
            vaeActor: vae,
            diffusionActor: diffusion,
            conditioningActor: conditioning,
            historyStore: store,
            outputManager: outputManager,
            sfxModelManager: sfxModelManager,
            registry: registry
        )
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
        self.generationService = Self.makeGenerationService(
            vaeActor: vaeActor,
            diffusionActor: diffusionActor,
            conditioningActor: conditioningActor,
            historyStore: store,
            outputManager: outputManager,
            sfxModelManager: sfxModelManager,
            registry: registry
        )
    }

    private static func makeGenerationService(
        vaeActor: VAEActor,
        diffusionActor: DiffusionActor,
        conditioningActor: ConditioningActor,
        historyStore: HistoryStore,
        outputManager: OutputDirectoryManager,
        sfxModelManager: SFXModelManager,
        registry: ModelRegistry
    ) -> GenerationService {
        let pipeline = MusicPipeline(vae: vaeActor, diffusion: diffusionActor)
        let converter = WeightConverter()

        let aceStep = ACEStepGenerationService(
            pipeline: pipeline,
            conditioning: conditioningActor,
            weightConverter: converter,
            registry: registry
        )
        let sfx = SFXGenerationService(sfxModelManager: sfxModelManager)
        let lora = LoRACoordinator(pipeline: pipeline, registry: registry)

        return GenerationService(
            aceStepService: aceStep,
            sfxService: sfx,
            sfxModelManager: sfxModelManager,
            loraCoordinator: lora,
            historyStore: historyStore,
            outputDirectoryManager: outputManager
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
