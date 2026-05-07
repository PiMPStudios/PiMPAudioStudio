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

    init() {
        let appState = AppState()
        let outputManager = OutputDirectoryManager()
        let playerService = AudioPlayerService()
        let apiKeys = APIKeyStore()
        let manager = LLMModelManager()
        let router = IntentRouter(llm: manager.llmActor)
        let enhancer = PromptEnhancer(llm: manager.llmActor)
        let lyricManager = LyricWriterManager(apiKeyStore: apiKeys)
        let sfxManager = SFXModelManager(registry: appState.registry)
        let services = AppServices(sfxModelManager: sfxManager, registry: appState.registry)

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
                .task(id: outputManager.directory) {
                    services.configure(outputManager: outputManager)
                }
                .onReceive(modelManager.$llmActor) { llm in
                    intentRouter.updateLLM(llm)
                    promptEnhancer.updateLLM(llm)
                    lyricWriterManager.updateLLM(llm)
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

                AdaptersSettingsView()
                    .tabItem { Label("Adapters", systemImage: "puzzlepiece.extension") }
            }
            .frame(minWidth: 550, minHeight: 450)
        }
    }
}

@MainActor
final class AppServices: ObservableObject {
    @Published var historyStore: HistoryStore
    @Published var generationService: GenerationService

    private var currentOutputManager: OutputDirectoryManager?
    private let sfxModelManager: SFXModelManager
    private let registry: ModelRegistry

    init(sfxModelManager: SFXModelManager, registry: ModelRegistry) {
        self.sfxModelManager = sfxModelManager
        self.registry = registry
        let tempDir = FileManager.default.temporaryDirectory
        let store = HistoryStore(directory: tempDir)
        let outputMgr = OutputDirectoryManager()

        self.historyStore = store
        self.generationService = Self.makeGenerationService(
            historyStore: store,
            outputManager: outputMgr,
            sfxModelManager: sfxModelManager,
            registry: registry
        )
    }

    func configure(outputManager: OutputDirectoryManager) {
        guard currentOutputManager !== outputManager || outputManager.directory != nil else {
            currentOutputManager = outputManager
            generationService = Self.makeGenerationService(
                historyStore: historyStore,
                outputManager: outputManager,
                sfxModelManager: sfxModelManager,
                registry: registry
            )
            return
        }
        currentOutputManager = outputManager

        let dir = outputManager.directory ?? FileManager.default.temporaryDirectory
        let store = HistoryStore(directory: dir)
        self.historyStore = store
        self.generationService = Self.makeGenerationService(
            historyStore: store,
            outputManager: outputManager,
            sfxModelManager: sfxModelManager,
            registry: registry
        )
    }

    private static func makeGenerationService(
        historyStore: HistoryStore,
        outputManager: OutputDirectoryManager,
        sfxModelManager: SFXModelManager,
        registry: ModelRegistry
    ) -> GenerationService {
        let vae = VAEActor()
        let dit = DiTActor(decoder: DiTDecoder())
        let diffusion = DiffusionActor(dit: dit)
        let pipeline = MusicPipeline(vae: vae, diffusion: diffusion)
        guard let conditioning = try? ConditioningActor.forTesting(
            tokenizerURL: Bundle.main.url(forResource: "tokenizer", withExtension: "json")
                ?? FileManager.default.temporaryDirectory.appending(path: "tokenizer.json")
        ) else {
            fatalError("Failed to initialize ConditioningActor: tokenizer could not be loaded from bundle")
        }
        let converter = WeightConverter()

        return GenerationService(
            pipeline: pipeline,
            conditioning: conditioning,
            weightConverter: converter,
            historyStore: historyStore,
            outputDirectoryManager: outputManager,
            sfxModelManager: sfxModelManager,
            registry: registry
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
