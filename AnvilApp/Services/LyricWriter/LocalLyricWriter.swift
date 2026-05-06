import Foundation
import AnvilCore

struct LocalLyricWriter: LyricWriter {
    let id = "local"
    let displayName = "Local (On-Device)"
    let requiresAPIKey = false

    private let llm: LLMActor

    init(llm: LLMActor) {
        self.llm = llm
    }

    func write(prompt: String, style: String, duration: TimeInterval) -> AsyncStream<String> {
        let linesEstimate = max(Int(duration / 15), 4)
        let systemPrompt = """
            You are a songwriter. Write lyrics for a song based on the user's description. \
            Use section markers: [Verse], [Chorus], [Bridge], [Outro]. \
            Match the mood, genre, and energy of the description. \
            Target duration: \(Int(duration)) seconds (roughly \(linesEstimate) lines per section). \
            Output only the lyrics with section markers. No commentary.
            """

        let userPrompt: String
        if style.isEmpty {
            userPrompt = prompt
        } else {
            userPrompt = "\(prompt)\nStyle: \(style)"
        }

        let actor = llm
        return AsyncStream { continuation in
            let task = Task {
                let stream = await actor.generate(
                    systemPrompt: systemPrompt,
                    userPrompt: userPrompt,
                    maxTokens: 512,
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
