import Foundation
import AnvilCore
import SwiftUI

@MainActor
final class IntentRouter: ObservableObject {
    @Published var detectedMode: GenerationMode = .music
    @Published var isClassifying: Bool = false

    private var llm: LLMActor?
    private var debounceTask: Task<Void, Never>?

    private static let systemPrompt = """
        You are a classifier. Given a user's audio generation prompt, respond with exactly \
        one word: "music", "soundfx", or "ambient". No explanation.
        """

    private static let debounceInterval: Duration = .milliseconds(300)

    init(llm: LLMActor?) {
        self.llm = llm
    }

    func updateLLM(_ llm: LLMActor?) {
        self.llm = llm
    }

    func classify(prompt: String) {
        debounceTask?.cancel()

        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            detectedMode = .music
            return
        }

        guard let llm else { return }

        debounceTask = Task { [weak self] in
            try? await Task.sleep(for: Self.debounceInterval)
            guard !Task.isCancelled else { return }
            guard let self else { return }

            self.isClassifying = true
            let result = await llm.classify(
                systemPrompt: Self.systemPrompt,
                userPrompt: trimmed
            )
            guard !Task.isCancelled else {
                self.isClassifying = false
                return
            }

            if let mode = GenerationMode(fromClassification: result) {
                self.detectedMode = mode
            }
            self.isClassifying = false
        }
    }

    func cancel() {
        debounceTask?.cancel()
        isClassifying = false
    }
}
