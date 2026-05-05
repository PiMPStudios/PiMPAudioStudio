import Foundation
import Testing
@testable import AnvilCore

@Test func modelSpecStoresProperties() {
    let spec = ModelSpec(
        id: "ace-step-dit",
        huggingFaceRepo: "ACEStep/ACE-Step-v1-5-DiT",
        files: ["model.safetensors", "config.json"],
        sizeBytes: 3_200_000_000,
        sha256: ["model.safetensors": "abc123", "config.json": "def456"]
    )

    #expect(spec.id == "ace-step-dit")
    #expect(spec.files.count == 2)
    #expect(spec.sha256["model.safetensors"] == "abc123")
}

@Test func modelSpecTotalSizeFormatted() {
    let spec = ModelSpec(
        id: "test",
        huggingFaceRepo: "test/repo",
        files: ["a.bin"],
        sizeBytes: 3_200_000_000,
        sha256: [:]
    )

    #expect(spec.totalSizeFormatted == "3.2 GB")
}

@Test func modelSpecLocalDirectory() {
    let spec = ModelSpec(
        id: "ace-step-dit",
        huggingFaceRepo: "ACEStep/ACE-Step-v1-5-DiT",
        files: [],
        sizeBytes: 0,
        sha256: [:]
    )

    let baseURL = URL(filePath: "/tmp/models")
    #expect(spec.localDirectory(base: baseURL).path().contains("ace-step-dit"))
}
