import Foundation
@testable import PiMPAudioStudio
import Testing

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
    let deleteWithTrace = makeHistoryEntry(
        fileName: "delete-with-trace.wav",
        traceFileName: "delete-with-trace.trace.json"
    )
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
    #expect(try FileManager.default
        .fileExists(atPath: tempDir.appending(path: #require(keep.traceFileName)).path(percentEncoded: false)))
    #expect(!FileManager.default
        .fileExists(atPath: tempDir.appending(path: deleteWithTrace.fileName).path(percentEncoded: false)))
    #expect(try !FileManager.default
        .fileExists(atPath: tempDir.appending(path: #require(deleteWithTrace.traceFileName))
            .path(percentEncoded: false)))
    #expect(!FileManager.default
        .fileExists(atPath: tempDir.appending(path: deleteAudioOnly.fileName).path(percentEncoded: false)))

    let reloadedStore = HistoryStore(directory: tempDir)
    #expect(reloadedStore.entries.map(\.id) == [keep.id])
}

@Test @MainActor func historyStoreRejectsPathTraversalFileNames() throws {
    let tempDir = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: tempDir) }

    let store = HistoryStore(directory: tempDir)

    let escapedNames = [
        "../secret.trace.json",
        "..\\secret.trace.json",
        "/tmp/secret.trace.json",
        "subdir/secret.trace.json",
        "subdir\\secret.trace.json",
        ".",
        "..",
        "",
        "   ",
        "foo/../bar.trace.json",
    ]

    for name in escapedNames {
        #expect(throws: HistoryStoreError.self) {
            try store.resolvedURL(forFileName: name)
        }
        #expect(throws: HistoryStoreError.self) {
            try store.loadTrace(traceFileName: name)
        }
        #expect(throws: HistoryStoreError.self) {
            try store.saveTraceReview(traceFileName: name, listeningNote: "x", isPreferred: nil)
        }
    }
}

@Test @MainActor func historyStoreAcceptsSafeSingleComponentFileNames() throws {
    let tempDir = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: tempDir) }

    let store = HistoryStore(directory: tempDir)
    let name = "safe-take.trace.json"
    let url = try store.resolvedURL(forFileName: name)

    #expect(url.lastPathComponent == name)
    #expect(url.deletingLastPathComponent().standardizedFileURL == tempDir.standardizedFileURL)
    #expect(url.path(percentEncoded: false).hasPrefix(tempDir.standardizedFileURL.path(percentEncoded: false)))
}

@Test @MainActor func historyStoreDeleteDoesNotRemoveEscapedFiles() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    let historyDir = root.appendingPathComponent("History", isDirectory: true)
    try FileManager.default.createDirectory(at: historyDir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }

    // File outside History that a naive join of "../outside.wav" would target.
    let outsideURL = root.appendingPathComponent("outside.wav")
    try Data("outside-audio".utf8).write(to: outsideURL)

    let escapedEntry = makeHistoryEntry(
        fileName: "../outside.wav",
        traceFileName: "../outside.trace.json"
    )
    let outsideTrace = root.appendingPathComponent("outside.trace.json")
    try Data("outside-trace".utf8).write(to: outsideTrace)

    let store = HistoryStore(directory: historyDir)
    store.add(escapedEntry)
    store.delete(escapedEntry)

    #expect(store.entries.isEmpty)
    #expect(FileManager.default.fileExists(atPath: outsideURL.path(percentEncoded: false)))
    #expect(FileManager.default.fileExists(atPath: outsideTrace.path(percentEncoded: false)))
}

@Test @MainActor func historyStoreRejectsSymlinkEscapeOnResolve() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    let historyDir = root.appendingPathComponent("History", isDirectory: true)
    try FileManager.default.createDirectory(at: historyDir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }

    let outsideTarget = root.appendingPathComponent("secret.trace.json")
    try Data(#"{"schemaVersion":2}"#.utf8).write(to: outsideTarget)

    let linkName = "escape.trace.json"
    let linkURL = historyDir.appendingPathComponent(linkName)
    try FileManager.default.createSymbolicLink(
        atPath: linkURL.path(percentEncoded: false),
        withDestinationPath: outsideTarget.path(percentEncoded: false)
    )

    let store = HistoryStore(directory: historyDir)
    #expect(throws: HistoryStoreError.self) {
        try store.resolvedURL(forFileName: linkName)
    }
    #expect(throws: HistoryStoreError.self) {
        try store.loadTrace(traceFileName: linkName)
    }
}

@Test @MainActor func historyStoreDropsUnsafeFilenamesOnLoad() throws {
    let tempDir = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: tempDir) }

    let safe = makeHistoryEntry(fileName: "keep.wav", traceFileName: "keep.trace.json")
    let escaped = makeHistoryEntry(fileName: "../escape.wav", traceFileName: "../escape.trace.json")
    let encoder = JSONEncoder()
    encoder.dateEncodingStrategy = .iso8601
    try encoder.encode([safe, escaped]).write(to: tempDir.appending(path: "history.json"))

    let store = HistoryStore(directory: tempDir)
    #expect(store.entries.map(\.fileName) == ["keep.wav"])

    let reloaded = HistoryStore(directory: tempDir)
    #expect(reloaded.entries.map(\.fileName) == ["keep.wav"])
}

@Test func adapterRegistryClampsMergeAlpha() {
    #expect(AdapterRegistry.clampedMergeAlpha(0) == 0)
    #expect(AdapterRegistry.clampedMergeAlpha(1) == 1)
    #expect(AdapterRegistry.clampedMergeAlpha(0.25) == 0.25)
    #expect(AdapterRegistry.clampedMergeAlpha(-2) == 0)
    #expect(AdapterRegistry.clampedMergeAlpha(3) == 1)
    #expect(AdapterRegistry.clampedMergeAlpha(.nan) == 0.5)
    #expect(AdapterRegistry.clampedMergeAlpha(.infinity) == 0.5)
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
