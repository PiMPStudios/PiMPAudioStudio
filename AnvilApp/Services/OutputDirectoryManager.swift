import SwiftUI

@MainActor
final class OutputDirectoryManager: ObservableObject {
    @Published var directory: URL?

    private static let bookmarkKey = "outputDirectoryBookmark"

    init() {
        directory = resolveBookmark()
    }

    func pickDirectory() async -> URL? {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false
        panel.message = "Choose where Anvil saves generated audio"
        panel.prompt = "Select Folder"

        let response = panel.runModal()
        guard response == .OK, let url = panel.url else { return nil }

        saveBookmark(for: url)
        directory = url
        return url
    }

    func hasValidDirectory() -> Bool {
        guard let dir = directory else { return false }
        return FileManager.default.isWritableFile(atPath: dir.path())
    }

    private func saveBookmark(for url: URL) {
        do {
            let bookmark = try url.bookmarkData(
                options: [.withSecurityScope],
                includingResourceValuesForKeys: nil,
                relativeTo: nil
            )
            UserDefaults.standard.set(bookmark, forKey: Self.bookmarkKey)
        } catch {
            print("Failed to save bookmark: \(error)")
        }
    }

    private func resolveBookmark() -> URL? {
        guard let data = UserDefaults.standard.data(forKey: Self.bookmarkKey) else {
            return nil
        }
        do {
            var isStale = false
            let url = try URL(
                resolvingBookmarkData: data,
                options: [.withSecurityScope],
                relativeTo: nil,
                bookmarkDataIsStale: &isStale
            )
            guard url.startAccessingSecurityScopedResource() else { return nil }
            if isStale {
                saveBookmark(for: url)
            }
            return url
        } catch {
            return nil
        }
    }
}
