import OSLog
import SwiftUI

@MainActor
final class OutputDirectoryManager: ObservableObject {
    private let logger = Logger(subsystem: "com.pimpstudios.PiMPAudioStudio", category: "OutputDirectory")

    @Published var directory: URL?

    private static let bookmarkKey = AppUserDefaultsKeys.outputDirectoryBookmark
    private static let legacyBookmarkKey = AppUserDefaultsKeys.legacyOutputDirectoryBookmark

    init() {
        directory = resolveBookmark()
    }

    @discardableResult
    func pickDirectory() async -> URL? {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false
        panel.message = "Choose where PiMP Audio Studio saves generated audio"
        panel.prompt = "Select Folder"

        let response = panel.runModal()
        guard response == .OK, let url = panel.url else { return nil }

        directory?.stopAccessingSecurityScopedResource()
        saveBookmark(for: url)
        directory = url
        return url
    }

    func hasValidDirectory() -> Bool {
        guard let dir = directory else { return false }
        return FileManager.default.isWritableFile(atPath: dir.path(percentEncoded: false))
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
            logger.error("Failed to save bookmark: \(error, privacy: .public)")
        }
    }

    private func resolveBookmark() -> URL? {
        guard let data = UserDefaults.standard.data(forKey: Self.bookmarkKey) ??
            UserDefaults.standard.data(forKey: Self.legacyBookmarkKey) else {
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
            } else if UserDefaults.standard.data(forKey: Self.bookmarkKey) == nil {
                UserDefaults.standard.set(data, forKey: Self.bookmarkKey)
            }
            return url
        } catch {
            return nil
        }
    }
}
