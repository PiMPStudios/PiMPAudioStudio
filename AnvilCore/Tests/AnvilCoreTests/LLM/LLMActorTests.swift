import Testing
import Foundation
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
}
