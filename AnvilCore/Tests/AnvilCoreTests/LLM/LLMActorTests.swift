import Testing
import Foundation
import MLX
@testable import AnvilCore

@Suite("LLMActor Tests")
struct LLMActorTests {
    @Test("init sets modelPath and isLoaded false")
    func testInitialState() async {
        let path = FileManager.default.temporaryDirectory.appending(path: "fake-model")
        let actor = LLMActor(modelPath: path)
        let loaded = await actor.isLoaded
        #expect(loaded == false)
    }

    @Test("loadModel throws when model directory does not exist")
    func testLoadModelMissingDirectory() async {
        let path = FileManager.default.temporaryDirectory.appending(path: "nonexistent-\(UUID())")
        let actor = LLMActor(modelPath: path)
        do {
            try await actor.loadModel()
            Issue.record("Expected error for missing directory")
        } catch {
            // Expected
        }
    }

    @Test("classify returns empty string when model not loaded")
    func testClassifyWithoutLoad() async {
        let path = FileManager.default.temporaryDirectory.appending(path: "fake-model")
        let actor = LLMActor(modelPath: path)
        let result = await actor.classify(systemPrompt: "Classify this.", userPrompt: "test")
        #expect(result == "")
    }

    @Test("generate returns empty stream when model not loaded")
    func testGenerateWithoutLoad() async {
        let path = FileManager.default.temporaryDirectory.appending(path: "fake-model")
        let actor = LLMActor(modelPath: path)
        let stream = await actor.generate(systemPrompt: "Write.", userPrompt: "test")
        var tokens: [String] = []
        for await token in stream { tokens.append(token) }
        #expect(tokens.isEmpty)
    }

    @Test("KV cache tracks per-layer sequence length")
    func testKVCacheTracksSequenceLength() {
        let cache = LLMKeyValueCache()
        #expect(cache.sequenceLength == 0)

        let key0 = MLXArray.zeros([1, 2, 3, 4])
        let value0 = MLXArray.ones([1, 2, 3, 4])
        cache.update(key: key0, value: value0, layer: 0)

        #expect(cache.sequenceLength == 3)
        #expect(cache.get(layer: 0)?.key.shape == [1, 2, 3, 4])
        #expect(cache.get(layer: 0)?.value.shape == [1, 2, 3, 4])

        let key1 = MLXArray.zeros([1, 2, 5, 4])
        let value1 = MLXArray.ones([1, 2, 5, 4])
        cache.update(key: key1, value: value1, layer: 1)

        #expect(cache.sequenceLength == 3)
        #expect(cache.get(layer: 1)?.key.shape == [1, 2, 5, 4])

        cache.reset()
        #expect(cache.sequenceLength == 0)
        #expect(cache.get(layer: 0) == nil)
    }
}
