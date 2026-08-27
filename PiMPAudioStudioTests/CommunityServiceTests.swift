import Foundation
import MLX
import Testing
@testable import PiMPAudioStudio

@Test func communityManifestURLUsesPiMPAdaptersRepo() {
    let url = CommunityService.communityManifestURL.absoluteString
    #expect(url == "https://raw.githubusercontent.com/PiMPStudios/pimp-adapters/main/manifest.json")
}

@Test func huggingfaceStagingURLKeepsSafetensorsExtension() throws {
    let source = try #require(URL(string:
        "https://huggingface.co/user/repo/resolve/main/adapter_model.safetensors"))
    let staged = CommunityService.temporaryDownloadURL(for: source)
    #expect(staged.pathExtension == "safetensors")
    #expect(staged.lastPathComponent != "adapter_model.safetensors")
}

@Test func huggingfaceStagingURLKeepsJSONExtension() throws {
    let source = try #require(URL(string:
        "https://huggingface.co/user/repo/resolve/main/adapter_config.json"))
    let staged = CommunityService.temporaryDownloadURL(for: source)
    #expect(staged.pathExtension == "json")
}

@Test func mlxLoadArraysRequiresSafetensorsExtension() throws {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("adapter-staging-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }

    let named = directory.appendingPathComponent("adapter_model.safetensors")
    try MLX.save(
        arrays: ["lora_A": MLXArray([Float](repeating: 1, count: 4), [2, 2])],
        url: named
    )

    let extensionless = directory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.copyItem(at: named, to: extensionless)
    do {
        _ = try MLX.loadArrays(url: extensionless)
        Issue.record("expected loadArrays to reject an extensionless file")
    } catch {
        #expect(error.localizedDescription.contains("Unknown extension"))
    }

    let source = try #require(URL(string:
        "https://huggingface.co/user/repo/resolve/main/adapter_model.safetensors"))
    let staged = CommunityService.temporaryDownloadURL(for: source)
    try FileManager.default.copyItem(at: named, to: staged)
    defer { try? FileManager.default.removeItem(at: staged) }
    let loaded = try MLX.loadArrays(url: staged)
    #expect(loaded["lora_A"] != nil)
}
