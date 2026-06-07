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
    #expect(specs.contains(where: { $0.id == "lyric-llm" }))
    #expect(specs.contains(where: { $0.id == "stable-audio-open" }))
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
