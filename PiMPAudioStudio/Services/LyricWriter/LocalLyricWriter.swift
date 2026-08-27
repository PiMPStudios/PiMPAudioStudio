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

    func write(prompt: String, style: String, duration: TimeInterval) -> AsyncThrowingStream<String, Error> {
        let plan = LyricWritingPlan.make(duration: duration)
        let userPrompt = plan.userPrompt(prompt: prompt, style: style)
        // Local small models need the shared craft rules plus a short few-shot example.
        let systemPrompt = plan.systemPromptForLocalLLM

        let actor = llm
        return AsyncThrowingStream { continuation in
            let task = Task {
                let stream = await actor.generate(
                    systemPrompt: systemPrompt,
                    userPrompt: userPrompt,
                    maxTokens: plan.maxTokens,
                    // Cooler sampling + hard bans/few-shot reduce cliché padding;
                    // still high enough for variation across regenerations.
                    temperature: 0.6
                )
                for await chunk in stream {
                    if Task.isCancelled {
                        break
                    }
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
