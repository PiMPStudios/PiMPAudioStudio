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
                    services.reconfigure(outputManager: outputManager)
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

/// Holds services that depend on OutputDirectoryManager and need stable identity.
@MainActor
final class AppServices: ObservableObject {
    @Published var historyStore: HistoryStore
    @Published var generationService: GenerationService

    init() {
        let tempDir = FileManager.default.temporaryDirectory
        let store = HistoryStore(directory: tempDir)
        let outputMgr = OutputDirectoryManager()

        let vae = VAEActor()
        let dit = DiTActor(decoder: DiTDecoder())
        let diffusion = DiffusionActor(dit: dit)
        let pipeline = MusicPipeline(vae: vae, diffusion: diffusion)
        let conditioning = ConditioningActor.forTesting(
            tokenizerURL: Bundle.main.url(forResource: "tokenizer", withExtension: "json")
                ?? tempDir.appending(path: "tokenizer.json")
        )
        let converter = WeightConverter()

        self.historyStore = store
        self.generationService = GenerationService(
            pipeline: pipeline,
            conditioning: conditioning,
            weightConverter: converter,
            historyStore: store,
            outputDirectoryManager: outputMgr
        )
    }

    func reconfigure(outputManager: OutputDirectoryManager) {
        guard let dir = outputManager.directory else { return }

        let store = HistoryStore(directory: dir)
        self.historyStore = store

        let vae = VAEActor()
        let dit = DiTActor(decoder: DiTDecoder())
        let diffusion = DiffusionActor(dit: dit)
        let pipeline = MusicPipeline(vae: vae, diffusion: diffusion)
        let conditioning = ConditioningActor.forTesting(
            tokenizerURL: Bundle.main.url(forResource: "tokenizer", withExtension: "json")
                ?? FileManager.default.temporaryDirectory.appending(path: "tokenizer.json")
        )
        let converter = WeightConverter()

        self.generationService = GenerationService(
            pipeline: pipeline,
            conditioning: conditioning,
            weightConverter: converter,
            historyStore: store,
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
