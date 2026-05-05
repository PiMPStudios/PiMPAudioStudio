import SwiftUI
import AnvilCore

@main
struct AnvilAppApp: App {
    @StateObject private var appState = AppState()

    var body: some Scene {
        WindowGroup {
            if appState.isSetupComplete {
                MainShellView()
                    .environmentObject(appState)
            } else {
                FirstLaunchView()
                    .environmentObject(appState)
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
