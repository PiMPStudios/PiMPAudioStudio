import Foundation
import OSLog

@MainActor
final class HistoryStore: ObservableObject {
    private let logger = Logger(subsystem: "com.pimpstudios.PiMPAudioStudio", category: "HistoryStore")

    @Published var entries: [HistoryEntry] = []

    private let directory: URL
    var storageDirectory: URL { directory }
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
        removeFiles(for: entry)
        saveToDisk()
    }

    func delete(ids: Set<HistoryEntry.ID>) {
        guard !ids.isEmpty else { return }

        let entriesToDelete = entries.filter { ids.contains($0.id) }
        guard !entriesToDelete.isEmpty else { return }

        for entry in entriesToDelete {
            removeFiles(for: entry)
        }
        entries.removeAll { ids.contains($0.id) }
        saveToDisk()
    }

    func deleteAll() {
        for entry in entries {
            removeFiles(for: entry)
        }
        entries.removeAll()
        saveToDisk()
    }

    /// Resolves a history-relative filename to a URL strictly contained in the History directory.
    ///
    /// Rejects empty names, `.` / `..`, path separators, absolute paths, and any candidate
    /// whose standardized or symlink-resolved parent is not the History directory.
    func resolvedURL(forFileName fileName: String) throws -> URL {
        let name = fileName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard isSafeHistoryFileName(name) else {
            throw HistoryStoreError.invalidFileName(fileName)
        }

        let base = directory.standardizedFileURL
        let candidate = base
            .appendingPathComponent(name, isDirectory: false)
            .standardizedFileURL

        // Compare path strings (not URL equality): URL == can fail on isDirectory
        // metadata when the History folder does not yet exist on disk.
        guard fileSystemPathsEqual(
            candidate.deletingLastPathComponent().standardizedFileURL,
            base
        ) else {
            throw HistoryStoreError.invalidFileName(fileName)
        }

        let resolvedBase = base.resolvingSymlinksInPath().standardizedFileURL
        let resolvedCandidate = candidate.resolvingSymlinksInPath().standardizedFileURL
        guard fileSystemPathsEqual(
            resolvedCandidate.deletingLastPathComponent().standardizedFileURL,
            resolvedBase
        ) else {
            throw HistoryStoreError.invalidFileName(fileName)
        }

        return candidate
    }

    private func fileSystemPathsEqual(_ lhs: URL, _ rhs: URL) -> Bool {
        normalizedFileSystemPath(lhs) == normalizedFileSystemPath(rhs)
    }

    private func normalizedFileSystemPath(_ url: URL) -> String {
        let path = url.path(percentEncoded: false)
        guard path.count > 1 else { return path }
        return path.hasSuffix("/") ? String(path.dropLast()) : path
    }

    func fileURL(for entry: HistoryEntry) throws -> URL {
        try resolvedURL(forFileName: entry.fileName)
    }

    func fileURL(forAudioFileName fileName: String) throws -> URL {
        try resolvedURL(forFileName: fileName)
    }

    func traceURL(for entry: HistoryEntry) -> URL? {
        if let traceFileName = entry.traceFileName {
            return try? resolvedURL(forFileName: traceFileName)
        }

        let inferredName = HistoryEntry.traceFileName(forAudioFileName: entry.fileName)
        guard let inferredURL = try? resolvedURL(forFileName: inferredName) else {
            return nil
        }
        if FileManager.default.fileExists(atPath: inferredURL.path(percentEncoded: false)) {
            return inferredURL
        }
        return nil
    }

    func loadTrace(for entry: HistoryEntry) throws -> GenerationTrace {
        guard let url = traceURL(for: entry) else {
            throw HistoryStoreError.missingTrace
        }

        return try loadTrace(at: url)
    }

    func loadTrace(traceFileName: String) throws -> GenerationTrace {
        try loadTrace(at: resolvedURL(forFileName: traceFileName))
    }

    func saveTraceReview(
        traceFileName: String,
        listeningNote: String?,
        isPreferred: Bool?
    ) throws -> GenerationTrace {
        let url = try resolvedURL(forFileName: traceFileName)
        let trace = try loadTrace(at: url)
        let updated = trace.withReview(
            listeningNote: normalizedNote(listeningNote),
            isPreferred: isPreferred
        )
        try writeTrace(updated, to: url)
        updateHistoryReview(
            traceFileName: traceFileName,
            listeningNote: updated.listeningNote,
            isPreferred: updated.isPreferred
        )
        return updated
    }

    func saveFailureTrace(_ trace: GenerationFailureTrace) throws {
        let url = try resolvedURL(forFileName: trace.traceFileName)
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(trace)
        try data.write(to: url, options: .atomic)
    }

    private func isSafeHistoryFileName(_ name: String) -> Bool {
        guard !name.isEmpty,
              name != ".",
              name != "..",
              !name.hasPrefix("/"),
              !name.contains("/"),
              !name.contains("\\"),
              name.rangeOfCharacter(from: .controlCharacters) == nil else {
            return false
        }
        return true
    }

    private func loadTrace(at url: URL) throws -> GenerationTrace {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode(GenerationTrace.self, from: data)
    }

    private func writeTrace(_ trace: GenerationTrace, to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(trace)
        try data.write(to: url, options: .atomic)
    }

    private func removeFiles(for entry: HistoryEntry) {
        if let audioURL = try? resolvedURL(forFileName: entry.fileName) {
            try? FileManager.default.removeItem(at: audioURL)
        } else {
            logger.error(
                "Skipping delete for escaped or invalid history audio name: \(entry.fileName, privacy: .public)"
            )
        }

        if let traceFileName = entry.traceFileName {
            if let traceURL = try? resolvedURL(forFileName: traceFileName) {
                try? FileManager.default.removeItem(at: traceURL)
            } else {
                logger.error(
                    "Skipping delete for escaped or invalid history trace name: \(traceFileName, privacy: .public)"
                )
            }
        }
    }

    private func updateHistoryReview(
        traceFileName: String,
        listeningNote: String?,
        isPreferred: Bool?
    ) {
        guard let index = entries.firstIndex(where: { entry in
            entry.traceFileName == traceFileName ||
            HistoryEntry.traceFileName(forAudioFileName: entry.fileName) == traceFileName
        }) else {
            return
        }

        entries[index] = entries[index].withReview(
            listeningNote: listeningNote,
            isPreferred: isPreferred
        )
        saveToDisk()
    }

    private func normalizedNote(_ note: String?) -> String? {
        let trimmed = note?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return trimmed.isEmpty ? nil : trimmed
    }

    private func loadFromDisk() {
        guard FileManager.default.fileExists(atPath: historyFileURL.path(percentEncoded: false)) else { return }
        do {
            let data = try Data(contentsOf: historyFileURL)
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            let loaded = try decoder.decode([HistoryEntry].self, from: data)
            let safe = loaded.filter { entry in
                isSafeHistoryFileName(entry.fileName)
                    && (entry.traceFileName.map(isSafeHistoryFileName) ?? true)
            }
            if safe.count != loaded.count {
                logger.error(
                    "Dropped \(loaded.count - safe.count, privacy: .public) history row(s) with unsafe filenames"
                )
                entries = safe
                saveToDisk()
            } else {
                entries = loaded
            }
        } catch {
            preserveCorruptHistoryFile(cause: error)
            entries = []
        }
    }

    private func preserveCorruptHistoryFile(cause: Error) {
        let fm = FileManager.default
        var destination = directory.appending(path: "history.json.corrupt")
        if fm.fileExists(atPath: destination.path(percentEncoded: false)) {
            let suffix = Int(Date().timeIntervalSince1970)
            destination = directory.appending(path: "history.json.corrupt-\(suffix)")
        }

        do {
            try fm.moveItem(at: historyFileURL, to: destination)
            logger.error("History index was unreadable and has been preserved at \(destination.lastPathComponent, privacy: .public): \(cause.localizedDescription, privacy: .public)")
        } catch {
            logger.error("Failed to preserve unreadable history index: \(error.localizedDescription, privacy: .public); original load error: \(cause.localizedDescription, privacy: .public)")
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

enum HistoryStoreError: LocalizedError {
    case missingTrace
    case invalidFileName(String)

    var errorDescription: String? {
        switch self {
        case .missingTrace:
            return "No trace sidecar was found for this history entry."
        case .invalidFileName(let name):
            return "History filename is invalid or escapes the History directory: \(name)."
        }
    }
}
