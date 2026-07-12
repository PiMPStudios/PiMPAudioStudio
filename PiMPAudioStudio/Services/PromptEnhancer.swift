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
            // Exiting this loop (cancel or finish) terminates the stream and
            // cancels the LLMActor producer via onTermination.
            for await token in stream {
                if Task.isCancelled { break }
                accumulated += token
            }
            if Task.isCancelled { return prompt }

            let enhanced = accumulated.trimmingCharacters(in: .whitespacesAndNewlines)
            return enhanced.isEmpty ? prompt : enhanced
        }
        enhanceTask = task

        return await task.value
    }

    func cancel() {
        enhanceTask?.cancel()
        enhanceTask = nil
        isEnhancing = false
    }
}
