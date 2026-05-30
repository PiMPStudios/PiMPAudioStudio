import Foundation
import PiMPAudioCore

struct LocalLyricWriter: LyricWriter {
    let id = "local"
    let displayName = "Local (On-Device)"
    let requiresAPIKey = false

    private let llm: LLMActor

    init(llm: LLMActor) {
        self.llm = llm
    }

    func write(prompt: String, style: String, duration: TimeInterval) -> AsyncStream<String> {
        let plan = LyricWritingPlan.make(duration: duration)
        let userPrompt = plan.userPrompt(prompt: prompt, style: style)

        let actor = llm
        return AsyncStream { continuation in
            let task = Task {
                let stream = await actor.generate(
                    systemPrompt: plan.systemPrompt,
                    userPrompt: userPrompt,
                    maxTokens: plan.maxTokens,
                    temperature: 0.8
                )
                for await chunk in stream {
                    if Task.isCancelled { break }
                    continuation.yield(chunk)
                }
                continuation.finish()
            }

            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }
}
