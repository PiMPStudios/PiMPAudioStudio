import Foundation
import Testing
@testable import PiMPAudioStudio

@Test @MainActor func historyStorePreservesUnreadableIndexBeforeResettingEntries() throws {
    let tempDir = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: tempDir) }

    let historyURL = tempDir.appending(path: "history.json")
    let corruptData = Data("{ definitely not valid json".utf8)
    try corruptData.write(to: historyURL)

    let store = HistoryStore(directory: tempDir)

    #expect(store.entries.isEmpty)
    #expect(!FileManager.default.fileExists(atPath: historyURL.path(percentEncoded: false)))

    let preserved = tempDir.appending(path: "history.json.corrupt")
    #expect(FileManager.default.fileExists(atPath: preserved.path(percentEncoded: false)))
    #expect(try Data(contentsOf: preserved) == corruptData)
}

@Test @MainActor func historyStoreDeletesSelectedEntriesAndFiles() throws {
    let tempDir = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: tempDir) }

    let keep = makeHistoryEntry(fileName: "keep.wav", traceFileName: "keep.trace.json")
    let deleteWithTrace = makeHistoryEntry(fileName: "delete-with-trace.wav", traceFileName: "delete-with-trace.trace.json")
    let deleteAudioOnly = makeHistoryEntry(fileName: "delete-audio-only.wav")

    for entry in [keep, deleteWithTrace, deleteAudioOnly] {
        try Data(entry.fileName.utf8).write(to: tempDir.appending(path: entry.fileName))
        if let traceFileName = entry.traceFileName {
            try Data(traceFileName.utf8).write(to: tempDir.appending(path: traceFileName))
        }
    }

    let store = HistoryStore(directory: tempDir)
    store.add(keep)
    store.add(deleteWithTrace)
    store.add(deleteAudioOnly)

    store.delete(ids: [deleteWithTrace.id, deleteAudioOnly.id])

    #expect(store.entries.map(\.id) == [keep.id])
    #expect(FileManager.default.fileExists(atPath: tempDir.appending(path: keep.fileName).path(percentEncoded: false)))
    #expect(FileManager.default.fileExists(atPath: tempDir.appending(path: keep.traceFileName!).path(percentEncoded: false)))
    #expect(!FileManager.default.fileExists(atPath: tempDir.appending(path: deleteWithTrace.fileName).path(percentEncoded: false)))
    #expect(!FileManager.default.fileExists(atPath: tempDir.appending(path: deleteWithTrace.traceFileName!).path(percentEncoded: false)))
    #expect(!FileManager.default.fileExists(atPath: tempDir.appending(path: deleteAudioOnly.fileName).path(percentEncoded: false)))

    let reloadedStore = HistoryStore(directory: tempDir)
    #expect(reloadedStore.entries.map(\.id) == [keep.id])
}

private func makeHistoryEntry(
    id: UUID = UUID(),
    fileName: String,
    traceFileName: String? = nil
) -> HistoryEntry {
    HistoryEntry(
        id: id,
        createdAt: Date(timeIntervalSince1970: 1_710_000_000),
        prompt: "Test prompt",
        lyrics: "",
        duration: 30,
        seed: 42,
        method: "euler",
        guidanceScale: 7.5,
        precision: "float32",
        fileName: fileName,
        traceFileName: traceFileName,
        pipeline: nil,
        originalPrompt: nil,
        modelCategory: nil,
        modelID: nil,
        modelName: nil,
        supportingModels: nil,
        adapters: nil,
        listeningNote: nil,
        isPreferred: nil
    )
}
