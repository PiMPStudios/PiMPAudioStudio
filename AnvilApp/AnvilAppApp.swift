import SwiftUI
import AnvilCore

@main
struct AnvilAppApp: App {
    @StateObject private var appState = AppState()
    @StateObject private var outputManager = OutputDirectoryManager()
    @StateObject private var playerService = AudioPlayerService()
    @StateObject private var services = AppServices()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(appState)
                .environmentObject(outputManager)
                .environmentObject(playerService)
                .environmentObject(services.historyStore)
                .environmentObject(services.generationService)
                .task(id: outputManager.directory) {
                    services.configure(outputManager: outputManager)
                }
        }
        .windowStyle(.titleBar)
        .defaultSize(width: 1200, height: 800)

        Settings {
            Text("Settings coming soon")
                .frame(width: 400, height: 300)
        }
    }
}

@MainActor
final class AppServices: ObservableObject {
    @Published var historyStore: HistoryStore
    @Published var generationService: GenerationService

    private var currentOutputManager: OutputDirectoryManager?

    init() {
        let tempDir = FileManager.default.temporaryDirectory
        let store = HistoryStore(directory: tempDir)
        let outputMgr = OutputDirectoryManager()

        self.historyStore = store
        self.generationService = Self.makeGenerationService(
            historyStore: store,
            outputManager: outputMgr
        )
    }

    func configure(outputManager: OutputDirectoryManager) {
        guard currentOutputManager !== outputManager || outputManager.directory != nil else {
            currentOutputManager = outputManager
            generationService = Self.makeGenerationService(
                historyStore: historyStore,
                outputManager: outputManager
            )
            return
        }
        currentOutputManager = outputManager

        let dir = outputManager.directory ?? FileManager.default.temporaryDirectory
        let store = HistoryStore(directory: dir)
        self.historyStore = store
        self.generationService = Self.makeGenerationService(
            historyStore: store,
            outputManager: outputManager
        )
    }

    private static func makeGenerationService(
        historyStore: HistoryStore,
        outputManager: OutputDirectoryManager
    ) -> GenerationService {
        let vae = VAEActor()
        let dit = DiTActor(decoder: DiTDecoder())
        let diffusion = DiffusionActor(dit: dit)
        let pipeline = MusicPipeline(vae: vae, diffusion: diffusion)
        let conditioning = ConditioningActor.forTesting(
            tokenizerURL: Bundle.main.url(forResource: "tokenizer", withExtension: "json")
                ?? FileManager.default.temporaryDirectory.appending(path: "tokenizer.json")
        )
        let converter = WeightConverter()

        return GenerationService(
            pipeline: pipeline,
            conditioning: conditioning,
            weightConverter: converter,
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
