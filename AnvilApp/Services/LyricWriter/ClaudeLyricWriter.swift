import Foundation

struct ClaudeLyricWriter: LyricWriter {
    let id = "anthropic"
    let displayName = "Claude (Anthropic)"
    let requiresAPIKey = true

    private let apiKey: String

    init(apiKey: String) {
        self.apiKey = apiKey
    }

    func write(prompt: String, style: String, duration: TimeInterval) -> AsyncStream<String> {
        AsyncStream { continuation in
            let task = Task {
                let url = URL(string: "https://api.anthropic.com/v1/messages")!
                var request = URLRequest(url: url)
                request.httpMethod = "POST"
                request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                request.setValue(apiKey, forHTTPHeaderField: "x-api-key")
                request.setValue("2023-06-01", forHTTPHeaderField: "anthropic-version")

                let linesPerSection = Int(duration / 15)
                let systemPrompt = """
                    You are a songwriter. Write lyrics for a song based on the user's description. \
                    Use section markers: [Verse], [Chorus], [Bridge], [Outro]. Match the mood, genre, \
                    and energy of the description. Target duration: \(Int(duration)) seconds (roughly \
                    \(linesPerSection) lines per section). Output only the lyrics with section markers. \
                    No commentary.
                    """

                let body: [String: Any] = [
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 1024,
                    "stream": true,
                    "system": systemPrompt,
                    "messages": [
                        ["role": "user", "content": "Style: \(style)\n\n\(prompt)"]
                    ]
                ]

                request.httpBody = try? JSONSerialization.data(withJSONObject: body)

                guard let (bytes, _) = try? await URLSession.shared.bytes(for: request) else {
                    continuation.finish()
                    return
                }

                for try await line in bytes.lines {
                    if Task.isCancelled { break }
                    guard line.hasPrefix("data: ") else { continue }
                    let json = String(line.dropFirst(6))
                    guard json != "[DONE]",
                          let data = json.data(using: .utf8),
                          let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                          let type = obj["type"] as? String,
                          type == "content_block_delta",
                          let delta = obj["delta"] as? [String: Any],
                          let text = delta["text"] as? String
                    else { continue }
                    continuation.yield(text)
                }

                continuation.finish()
            }

            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }
}
