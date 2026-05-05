import SwiftUI
import AnvilCore

@MainActor
final class AppState: ObservableObject {
    @Published var isSetupComplete: Bool = false
    @Published var downloadProgress: [String: Double] = [:]
    @Published var currentError: String?

    let registry: ModelRegistry

    convenience init() {
        let modelsDir = FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        )[0].appending(path: "Anvil/models")
        self.init(registry: ModelRegistry(baseDirectory: modelsDir))
    }

    init(registry: ModelRegistry) {
        self.registry = registry
        Task { await checkSetupStatus() }
    }

    func checkSetupStatus() async {
        let baseDir = await registry.baseDirectory
        let allReady = registry.allSpecs.allSatisfy { spec in
            spec.files.allSatisfy { file in
                FileManager.default.fileExists(
                    atPath: spec.localPath(for: file, base: baseDir).path()
                )
            }
        }
        isSetupComplete = allReady
    }

    func startDownload() async {
        currentError = nil
        do {
            try await registry.downloadAll { [weak self] modelID, progress in
                Task { @MainActor [weak self] in
                    self?.downloadProgress[modelID] = progress
                }
            }
            isSetupComplete = true
        } catch {
            currentError = error.localizedDescription
        }
    }
}
