import Testing
import Foundation
@testable import PiMPAudioCore

@Test func registryReturnsAllSpecs() async {
    let registry = ModelRegistry(baseDirectory: FileManager.default.temporaryDirectory)
    let specs = await registry.allSpecs

    #expect(specs.isEmpty == false)
    #expect(specs.contains(where: { $0.id == "ace-step-dit-turbo" }))
    #expect(specs.contains(where: { $0.id == "ace-step-dit-xl-base" }))
    #expect(specs.contains(where: { $0.id == "ace-step-dit-xl-sft" }))
    #expect(specs.contains(where: { $0.id == "ace-step-dit-xl-turbo" }))
    #expect(specs.contains(where: { $0.id == "ace-step-vae" }))
    #expect(specs.contains(where: { $0.id == "ace-step-lm-1.7b" }))
    #expect(specs.contains(where: { $0.id == "stable-audio-open" }))
    #expect(specs.contains(where: { $0.id == "lyric-llm" }) == false)
}

@Test func modelCategoryDropsRetiredLyricLLMCase() {
    #expect(ModelCategory(rawValue: "lyricLLM") == nil)
}

@Test func registryIncludesOfficialACEv15XLVariants() async {
    let registry = ModelRegistry(baseDirectory: FileManager.default.temporaryDirectory)
    let specs = await registry.allSpecs.filter { $0.variantFamily == "ace-step-1.5-xl" }
    let byID = Dictionary(uniqueKeysWithValues: specs.map { ($0.id, $0) })

    #expect(Set(byID.keys) == [
        "ace-step-dit-xl-base",
        "ace-step-dit-xl-sft",
        "ace-step-dit-xl-turbo",
    ])
    #expect(byID["ace-step-dit-xl-base"]?.huggingFaceRepo == "ACE-Step/acestep-v15-xl-base")
    #expect(byID["ace-step-dit-xl-sft"]?.huggingFaceRepo == "ACE-Step/acestep-v15-xl-sft")
    #expect(byID["ace-step-dit-xl-turbo"]?.huggingFaceRepo == "ACE-Step/acestep-v15-xl-turbo")

    for spec in specs {
        let fileNames = Set(spec.files.map(\.localName))
        #expect(spec.category == .aceStepDiT)
        #expect(spec.isRunnable)
        #expect(fileNames.contains("model.safetensors.index.json"))
        #expect(fileNames.contains("config.json"))
        #expect(fileNames.contains("silence_latent.pt"))
        #expect(fileNames.filter { $0.hasSuffix(".safetensors") }.count == 4)
        #expect(spec.totalSizeBytes > 19_900_000_000)
    }
}

@Test func registryIncludesOfficialACE5HzLMVariants() async {
    let tempDir = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    let registry = ModelRegistry(baseDirectory: tempDir)
    let specs = await registry.allSpecs.filter { $0.variantFamily == "ace-step-lm" }
    let byID = Dictionary(uniqueKeysWithValues: specs.map { ($0.id, $0) })

    #expect(Set(byID.keys) == [
        "ace-step-lm-0.6b",
        "ace-step-lm-1.7b",
        "ace-step-lm-4b",
    ])
    #expect(byID["ace-step-lm-0.6b"]?.huggingFaceRepo == "ACE-Step/acestep-5Hz-lm-0.6B")
    #expect(byID["ace-step-lm-1.7b"]?.huggingFaceRepo == "ACE-Step/Ace-Step1.5")
    #expect(byID["ace-step-lm-4b"]?.huggingFaceRepo == "ACE-Step/acestep-5Hz-lm-4B")

    let onePointSeven = byID["ace-step-lm-1.7b"]
    #expect(onePointSeven?.category == .aceStepLM)
    #expect(onePointSeven?.files.contains {
        $0.remotePath == "acestep-5Hz-lm-1.7B/model.safetensors"
            && $0.localName == "model.safetensors"
    } == true)
    #expect(onePointSeven?.files.contains {
        $0.remotePath == "acestep-5Hz-lm-1.7B/chat_template.jinja"
            && $0.localName == "chat_template.jinja"
    } == true)
    #expect((onePointSeven?.totalSizeBytes ?? 0) > 3_700_000_000)
}

@Test func registryDetectsModelNotDownloaded() async {
    let tempDir = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    let registry = ModelRegistry(baseDirectory: tempDir)

    let status = await registry.status(for: "ace-step-dit-turbo")
    #expect(status == .notDownloaded)
}

@Test func registryDetectsModelReady() async throws {
    let tempDir = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: tempDir) }

    let registry = ModelRegistry(baseDirectory: tempDir)
    let spec = await registry.allSpecs.first(where: { $0.id == "ace-step-dit-turbo" })!

    let modelDir = spec.localDirectory(base: tempDir)
    try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
    for file in spec.files {
        try Data("stub".utf8).write(to: spec.localPath(for: file.localName, base: tempDir))
    }

    let status = await registry.status(for: "ace-step-dit-turbo")
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

@Test func registryPreservesCorruptUserModelIndexAndSkipsOrphanDeletion() async throws {
    let tempDir = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: tempDir) }

    let corruptData = Data("{ not valid user-models json".utf8)
    let indexURL = tempDir.appending(path: "user-models.json")
    try corruptData.write(to: indexURL)

    // Directory that only exists in the user index — would be deleted if
    // corrupt load were treated as an empty list during migration.
    let userModelDir = tempDir.appending(path: "my-custom-dit", directoryHint: .isDirectory)
    try FileManager.default.createDirectory(at: userModelDir, withIntermediateDirectories: true)
    try Data("weights".utf8).write(to: userModelDir.appending(path: "model.safetensors"))

    let registry = ModelRegistry(baseDirectory: tempDir)
    let userSpecs = await registry.userSpecs

    #expect(userSpecs.isEmpty)
    #expect(!FileManager.default.fileExists(atPath: indexURL.path(percentEncoded: false)))

    let preserved = tempDir.appending(path: "user-models.json.corrupt")
    #expect(FileManager.default.fileExists(atPath: preserved.path(percentEncoded: false)))
    #expect(try Data(contentsOf: preserved) == corruptData)

    #expect(FileManager.default.fileExists(atPath: userModelDir.path(percentEncoded: false)))
    // Migration must not be marked complete while the index was unreadable.
    let marker = tempDir.appending(path: ".model-migration-v1-complete")
    #expect(!FileManager.default.fileExists(atPath: marker.path(percentEncoded: false)))
}

@Test func registryOrphanCleanupPreservesRegisteredAndReservedDirectories() async throws {
    let tempDir = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: tempDir) }

    let userSpec = ModelSpec(
        id: "my-custom-dit",
        displayName: "Custom DiT",
        huggingFaceRepo: "example/custom",
        files: [ModelFileSpec(remotePath: "model.safetensors", sizeBytes: 1)],
        category: .aceStepDiT
    )
    let indexURL = tempDir.appending(path: "user-models.json")
    try JSONEncoder().encode([userSpec]).write(to: indexURL)

    let keptDir = tempDir.appending(path: "my-custom-dit", directoryHint: .isDirectory)
    try FileManager.default.createDirectory(at: keptDir, withIntermediateDirectories: true)

    let adaptersDir = tempDir.appending(path: "adapters", directoryHint: .isDirectory)
    try FileManager.default.createDirectory(at: adaptersDir, withIntermediateDirectories: true)
    let adapterSentinel = adaptersDir.appending(path: "adapter.safetensors")
    try Data("adapter-weights".utf8).write(to: adapterSentinel)

    let orphanDir = tempDir.appending(path: "legacy-orphan-model", directoryHint: .isDirectory)
    try FileManager.default.createDirectory(at: orphanDir, withIntermediateDirectories: true)

    let registry = ModelRegistry(baseDirectory: tempDir)
    let userIDs = await registry.userSpecs.map(\.id)

    #expect(userIDs == ["my-custom-dit"])
    #expect(FileManager.default.fileExists(atPath: keptDir.path(percentEncoded: false)))
    #expect(FileManager.default.fileExists(atPath: adapterSentinel.path(percentEncoded: false)))
    #expect(!FileManager.default.fileExists(atPath: orphanDir.path(percentEncoded: false)))
    let marker = tempDir.appending(path: ".model-migration-v1-complete")
    #expect(FileManager.default.fileExists(atPath: marker.path(percentEncoded: false)))
}

@Test func builtInACEStepFilesPinSHA256Checksums() {
    for spec in ModelRegistry.builtInSpecs where spec.huggingFaceRepo.hasPrefix("ACE-Step/") {
        for file in spec.files {
            #expect(
                isPinnedSHA256(file.sha256),
                "\(spec.id) \(file.remotePath) must pin a 64-char lowercase SHA-256"
            )
        }
    }
}

@Test func builtInFileChecksumsAreWellFormedWhenPresent() {
    for spec in ModelRegistry.builtInSpecs {
        for file in spec.files {
            guard let hash = file.sha256 else { continue }
            #expect(
                isPinnedSHA256(hash),
                "\(spec.id) \(file.remotePath) has a malformed sha256 pin"
            )
        }
    }
}

@Test func missingChecksumDoesNotInvalidateAModelFileSpec() {
    let file = ModelFileSpec(remotePath: "model.safetensors", sizeBytes: 1)
    #expect(file.sha256 == nil)
}

@Test func gatedBuiltInsMayOmitChecksums() {
    let stable = ModelRegistry.builtInSpecs.first { $0.id == "stable-audio-open" }
    #expect(stable != nil)
    #expect(stable?.files.contains { $0.sha256 == nil } == true)
}

@Test func registryRemovesRetiredLyricLLMDirectoryEvenAfterMigration() async throws {
    let tempDir = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: tempDir) }

    let leftover = tempDir.appending(path: "lyric-llm", directoryHint: .isDirectory)
    try FileManager.default.createDirectory(at: leftover, withIntermediateDirectories: true)
    try Data("stale-phi3".utf8).write(to: leftover.appending(path: "model.safetensors"))
    try Data().write(to: tempDir.appending(path: ".model-migration-v1-complete"))

    _ = ModelRegistry(baseDirectory: tempDir)

    #expect(!FileManager.default.fileExists(atPath: leftover.path(percentEncoded: false)))
}

@Test func registryDropsRetiredLyricLLMActiveEntry() async throws {
    let tempDir = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: tempDir) }

    let activeURL = tempDir.appending(path: "active-models.json")
    try Data(#"{"aceStepDiT":"ace-step-dit-turbo","lyricLLM":"lyric-llm"}"#.utf8)
        .write(to: activeURL)
    try Data().write(to: tempDir.appending(path: ".model-migration-v1-complete"))

    _ = ModelRegistry(baseDirectory: tempDir)

    let decoded = try JSONDecoder().decode(
        [String: String].self,
        from: Data(contentsOf: activeURL)
    )
    #expect(decoded["lyricLLM"] == nil)
    #expect(decoded["aceStepDiT"] == "ace-step-dit-turbo")
}

private func isPinnedSHA256(_ value: String?) -> Bool {
    guard let value, value.count == 64 else { return false }
    return value.unicodeScalars.allSatisfy { CharacterSet(charactersIn: "0123456789abcdef").contains($0) }
}
