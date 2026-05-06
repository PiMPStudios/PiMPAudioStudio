import Foundation
import AnvilCore
import SwiftUI

@MainActor
final class PromptEnhancer: ObservableObject {
    @Published var isEnhancing: Bool = false

    private var llm: LLMActor?
    private var enhanceTask: Task<Void, Never>?

    private static let systemPromptTemplate = """
        You are an audio prompt enhancer. Given a brief description and the generation \
        mode (%@), expand it into a detailed, vivid description that will produce \
        high-quality audio. Include genre, instruments, tempo, mood, texture, and \
        production style where relevant. Output only the enhanced prompt, nothing else. \
        Keep it under 200 words.
        """

    init(llm: LLMActor?) {
        self.llm = llm
    }

    func updateLLM(_ llm: LLMActor?) {
        self.llm = llm
    }

    var isAvailable: Bool {
        llm != nil
    }

    func enhance(prompt: String, mode: GenerationMode) async -> String {
        guard let llm else { return prompt }

        isEnhancing = true
        defer { isEnhancing = false }

        let systemPrompt = String(format: Self.systemPromptTemplate, mode.displayName)

        var result = ""
        let stream = llm.generate(
            systemPrompt: systemPrompt,
            userPrompt: prompt,
            maxTokens: 300,
            temperature: 0.7
        )
        for await token in stream {
            if Task.isCancelled { return prompt }
            result += token
        }

        let enhanced = result.trimmingCharacters(in: .whitespacesAndNewlines)
        return enhanced.isEmpty ? prompt : enhanced
    }

    func cancel() {
        enhanceTask?.cancel()
        isEnhancing = false
    }
}
