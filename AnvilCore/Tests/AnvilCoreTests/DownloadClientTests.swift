import Testing
import Foundation
@testable import AnvilCore

@Test func downloadClientBuildsCorrectURL() async {
    let client = DownloadClient()
    let url = await client.huggingFaceURL(repo: "ACEStep/ACE-Step-v1-5-DiT", file: "model.safetensors")

    #expect(url.absoluteString == "https://huggingface.co/ACEStep/ACE-Step-v1-5-DiT/resolve/main/model.safetensors")
}

@Test func downloadClientVerifiesSHA256() async throws {
    let client = DownloadClient()
    let tempDir = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: tempDir) }

    let testData = Data("hello world".utf8)
    let testFile = tempDir.appending(path: "test.bin")
    try testData.write(to: testFile)

    // SHA256 of "hello world" verified via: echo -n "hello world" | shasum -a 256
    let expectedHash = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    let result = await client.verifySHA256(file: testFile, expected: expectedHash)
    #expect(result == true)

    let badResult = await client.verifySHA256(file: testFile, expected: "0000")
    #expect(badResult == false)
}
