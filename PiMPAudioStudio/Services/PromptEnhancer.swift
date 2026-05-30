import Foundation
import PiMPAudioCore
import SwiftUI

@MainActor
final class PromptEnhancer: ObservableObject {
    @Published var isEnhancing: Bool = false

    private var llm: LLMActor?
    private var enhanceTask: Task<String, Never>?

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

        enhanceTask?.cancel()

        var result = prompt
        let task = Task {
            isEnhancing = true
            defer { isEnhancing = false }

            let systemPrompt = String(format: Self.systemPromptTemplate, mode.displayName)

            var accumulated = ""
            let stream = await llm.generate(
                systemPrompt: systemPrompt,
                userPrompt: prompt,
                maxTokens: 300,
                temperature: 0.7
            )
            for await token in stream {
                if Task.isCancelled { return prompt }
                accumulated += token
            }

            let enhanced = accumulated.trimmingCharacters(in: .whitespacesAndNewlines)
            return enhanced.isEmpty ? prompt : enhanced
        }
        enhanceTask = task

        return (try? await task.value) ?? prompt
    }

    func cancel() {
        enhanceTask?.cancel()
        enhanceTask = nil
        isEnhancing = false
    }
}
