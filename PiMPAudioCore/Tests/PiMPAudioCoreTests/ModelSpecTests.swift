import Foundation
@testable import PiMPAudioCore
import Testing

@Test func modelSpecStoresProperties() {
    let spec = ModelSpec(
        id: "ace-step-dit",
        displayName: "ACE-Step DiT",
        huggingFaceRepo: "ACEStep/ACE-Step-v1-5-DiT",
        files: [
            ModelFileSpec(remotePath: "model.safetensors"),
            ModelFileSpec(remotePath: "config.json"),
        ],
        category: .aceStepDiT
    )

    #expect(spec.id == "ace-step-dit")
    #expect(spec.files.count == 2)
}

@Test func modelSpecTotalSizeFormatted() {
    let spec = ModelSpec(
        id: "test",
        displayName: "Test",
        huggingFaceRepo: "test/repo",
        files: [ModelFileSpec(remotePath: "a.bin", sizeBytes: 3_200_000_000)],
        category: .aceStepDiT
    )

    #expect(spec.totalSizeFormatted == "3.2 GB")
}

@Test func modelSpecLocalDirectory() {
    let spec = ModelSpec(
        id: "ace-step-dit",
        displayName: "ACE-Step DiT",
        huggingFaceRepo: "ACEStep/ACE-Step-v1-5-DiT",
        files: [],
        category: .aceStepDiT
    )

    let baseURL = URL(filePath: "/tmp/models")
    #expect(spec.localDirectory(base: baseURL) == URL(filePath: "/tmp/models/ace-step-dit"))
}

@Test func modelSpecLocalPath() {
    let spec = ModelSpec(
        id: "ace-step-dit",
        displayName: "ACE-Step DiT",
        huggingFaceRepo: "ACEStep/ACE-Step-v1-5-DiT",
        files: [ModelFileSpec(remotePath: "model.safetensors")],
        category: .aceStepDiT
    )
    let baseURL = URL(filePath: "/tmp/models")
    #expect(spec.localPath(for: "model.safetensors", base: baseURL)
        == URL(filePath: "/tmp/models/ace-step-dit/model.safetensors"))
}
