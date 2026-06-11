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
