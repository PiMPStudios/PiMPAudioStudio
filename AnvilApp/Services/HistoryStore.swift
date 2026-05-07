import Foundation
import OSLog

@MainActor
final class HistoryStore: ObservableObject {
    private let logger = Logger(subsystem: "com.pimp.AnvilApp", category: "HistoryStore")

    @Published var entries: [HistoryEntry] = []

    private let directory: URL
    private var historyFileURL: URL { directory.appending(path: "history.json") }

    init(directory: URL) {
        self.directory = directory
        loadFromDisk()
    }

    func add(_ entry: HistoryEntry) {
        entries.insert(entry, at: 0)
        saveToDisk()
    }

    func delete(_ entry: HistoryEntry) {
        entries.removeAll { $0.id == entry.id }
        let fileURL = directory.appending(path: entry.fileName)
        try? FileManager.default.removeItem(at: fileURL)
        saveToDisk()
    }

    func deleteAll() {
        for entry in entries {
            let fileURL = directory.appending(path: entry.fileName)
            try? FileManager.default.removeItem(at: fileURL)
        }
        entries.removeAll()
        saveToDisk()
    }

    func fileURL(for entry: HistoryEntry) -> URL {
        directory.appending(path: entry.fileName)
    }

    private func loadFromDisk() {
        guard FileManager.default.fileExists(atPath: historyFileURL.path(percentEncoded: false)) else { return }
        do {
            let data = try Data(contentsOf: historyFileURL)
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            entries = try decoder.decode([HistoryEntry].self, from: data)
        } catch {
            entries = []
        }
    }

    private func saveToDisk() {
        do {
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(entries)
            try data.write(to: historyFileURL, options: .atomic)
        } catch {
            logger.error("Failed to save history: \(error, privacy: .public)")
        }
    }
}
