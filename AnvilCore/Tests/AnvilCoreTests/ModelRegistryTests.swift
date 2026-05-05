import Testing
import Foundation
@testable import AnvilCore

@Test func registryReturnsAllSpecs() async {
    let registry = ModelRegistry(baseDirectory: FileManager.default.temporaryDirectory)
    let specs = registry.allSpecs

    #expect(specs.isEmpty == false)
    #expect(specs.contains(where: { $0.id == "ace-step-dit" }))
    #expect(specs.contains(where: { $0.id == "ace-step-vae" }))
    #expect(specs.contains(where: { $0.id == "lyric-llm" }))
    #expect(specs.contains(where: { $0.id == "stable-audio-open" }))
}

@Test func registryDetectsModelNotDownloaded() async {
    let tempDir = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    let registry = ModelRegistry(baseDirectory: tempDir)

    let status = await registry.status(for: "ace-step-dit")
    #expect(status == .notDownloaded)
}

@Test func registryDetectsModelReady() async throws {
    let tempDir = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: tempDir) }

    let registry = ModelRegistry(baseDirectory: tempDir)
    let spec = registry.allSpecs.first(where: { $0.id == "ace-step-dit" })!

    let modelDir = spec.localDirectory(base: tempDir)
    try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
    for file in spec.files {
        try Data("stub".utf8).write(to: spec.localPath(for: file, base: tempDir))
    }

    let status = await registry.status(for: "ace-step-dit")
    #expect(status == .ready)
}

@Test func registryReturnsFailedForUnknownModel() async {
    let registry = ModelRegistry(baseDirectory: FileManager.default.temporaryDirectory)
    let status = await registry.status(for: "nonexistent-model-id")
    if case .failed(_) = status {
        // expected
    } else {
        Issue.record("Expected .failed status for unknown model ID, got \(status)")
    }
}
