import Foundation
@testable import PiMPAudioCore
import Testing

@Test func communityAdapterDecodesOptionalSHA256() throws {
    let json = Data("""
    {
      "id": "dark-test",
      "name": "Dark Test",
      "author": "PiMP Studios",
      "description": "Test adapter",
      "baseModelCategory": "aceStepDiT",
      "rank": 64,
      "adapterURL": "https://example.com/adapter.safetensors",
      "sha256": "0123456789abcdef",
      "metadataURL": "https://example.com/adapter.json",
      "tags": ["test"],
      "featured": false,
      "minAppVersion": "0.1.0",
      "createdAt": "2026-06-09T12:00:00Z",
      "downloadCount": 12
    }
    """.utf8)

    let decoder = JSONDecoder()
    decoder.dateDecodingStrategy = .iso8601

    let adapter = try decoder.decode(CommunityAdapter.self, from: json)
    #expect(adapter.sha256 == "0123456789abcdef")
}

@Test func communityAdapterDecodesLegacyManifestWithoutSHA256() throws {
    let json = Data("""
    {
      "id": "legacy-test",
      "name": "Legacy Test",
      "author": "PiMP Studios",
      "description": "Legacy adapter",
      "baseModelCategory": "aceStepDiT",
      "rank": 16,
      "adapterURL": "https://example.com/adapter.safetensors",
      "metadataURL": "https://example.com/adapter.json",
      "tags": [],
      "featured": true,
      "minAppVersion": "0.1.0",
      "createdAt": "2026-06-09T12:00:00Z",
      "downloadCount": 0
    }
    """.utf8)

    let decoder = JSONDecoder()
    decoder.dateDecodingStrategy = .iso8601

    let adapter = try decoder.decode(CommunityAdapter.self, from: json)
    #expect(adapter.sha256 == nil)
}
